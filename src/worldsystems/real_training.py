"""Resumable physical-state training for trusted WorldDynamics model architectures."""

import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .checkpoint import sha256
from .rssm_adapter import RSSMAdapter
from .worlddynamics import WorldDynamicsDatasetAdapter, WorldDynamicsModelAdapter


@dataclass(frozen=True)
class RealTrainingConfig:
    repository: str
    dataset: str
    architecture_weights: str
    family: str
    run_dir: str
    device: str = "cpu"
    seed: int = 17
    epochs: int = 2
    batch_size: int = 7
    learning_rate: float = 0.001

    def __post_init__(self):
        if self.family not in {"action_predictor", "rssm"}:
            raise ValueError("unsupported family")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("unsupported device")
        if min(self.epochs, self.batch_size) < 1 or self.learning_rate <= 0:
            raise ValueError("invalid training configuration")


def _build(cfg: RealTrainingConfig):
    adapter = (RSSMAdapter(cfg.repository, cfg.architecture_weights)
               if cfg.family == "rssm" else
               WorldDynamicsModelAdapter(cfg.repository, cfg.architecture_weights))
    return adapter


def _save(path: Path, model: nn.Module, optimizer, scheduler, cfg: RealTrainingConfig,
          dataset_sha: str, source_sha: str, epoch: int, next_batch: int | None,
          global_step: int, loss_sum: float, sample_count: int) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temp.mkdir()
    try:
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                  str(temp / "weights.safetensors"))
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
                   temp / "training.pt")
        manifest = {"schema": "worldsystems-real-training-v1", "config": asdict(cfg),
                    "dataset_sha256": dataset_sha, "source_sha256": source_sha,
                    "epoch": epoch, "next_batch": next_batch,
                    "global_step": global_step, "loss_sum": loss_sum,
                    "sample_count": sample_count,
                    "files": {name: sha256(temp / name) for name in
                              ("weights.safetensors", "training.pt")}}
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temp, path)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def verify_real_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "worldsystems-real-training-v1" or set(
            manifest.get("files", {})) != {"weights.safetensors", "training.pt"}:
        raise ValueError("incomplete or unsupported real-model checkpoint")
    for name, expected in manifest["files"].items():
        if sha256(path / name) != expected:
            raise ValueError(f"real-model checkpoint hash mismatch: {name}")
    return manifest


def _timed(device: str, operation):
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = operation()
    if device == "cuda":
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def train_real(cfg: RealTrainingConfig, resume: str | None = None,
               max_steps: int | None = None) -> dict:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dataset = WorldDynamicsDatasetAdapter(cfg.dataset)
    observations, actions, train_rows = dataset.arrays("train")
    val_observations, val_actions, _ = dataset.arrays("val")
    torch.manual_seed(cfg.seed)
    adapter = _build(cfg)
    model = adapter.model
    if resume is None:
        for module in model.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    start_epoch, start_batch, global_step = 0, 0, 0
    prior_sum, prior_count = 0.0, 0
    if resume:
        manifest = verify_real_checkpoint(resume)
        if manifest["config"] != asdict(cfg) or manifest["dataset_sha256"] != dataset.sha256:
            raise ValueError("resume configuration or dataset mismatch")
        if manifest["source_sha256"] != adapter.source_sha256:
            raise ValueError("model source changed since checkpoint")
        model.load_state_dict(load_file(str(Path(resume) / "weights.safetensors"), device=cfg.device))
        state = torch.load(Path(resume) / "training.pt", map_location=cfg.device, weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if cfg.device == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        start_epoch = manifest["epoch"] + (manifest["next_batch"] is None)
        start_batch = manifest["next_batch"] or 0
        global_step = manifest["global_step"]
        prior_sum, prior_count = manifest["loss_sum"], manifest["sample_count"]
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.perf_counter()
    forward_seconds = backward_seconds = optimizer_seconds = 0.0
    processed_samples = 0
    try:
        for epoch in range(start_epoch, cfg.epochs):
            model.train()
            permutation = torch.randperm(len(observations),
                                         generator=torch.Generator().manual_seed(cfg.seed + epoch))
            batches = list(permutation.split(cfg.batch_size))
            loss_sum = prior_sum if epoch == start_epoch else 0.0
            sample_count = prior_count if epoch == start_epoch else 0
            for index, selected in enumerate(batches):
                if epoch == start_epoch and index < start_batch:
                    continue
                x = observations[selected].to(cfg.device)
                u = actions[selected].to(cfg.device)
                optimizer.zero_grad(set_to_none=True)
                loss, duration = _timed(cfg.device, lambda x=x, u=u: model.loss(x, u))
                forward_seconds += duration
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite physical-model training loss")
                _, duration = _timed(cfg.device, loss.backward)
                backward_seconds += duration
                _, duration = _timed(cfg.device, optimizer.step)
                optimizer_seconds += duration
                global_step += 1
                processed_samples += len(selected)
                loss_sum += loss.item() * len(selected)
                sample_count += len(selected)
                if max_steps is not None and global_step >= max_steps:
                    path = run_dir / f"step-{global_step:06d}"
                    _save(path, model, optimizer, scheduler, cfg, dataset.sha256,
                          adapter.source_sha256, epoch, index + 1, global_step,
                          loss_sum, sample_count)
                    return {"status": "interrupted_at_step_boundary", "checkpoint": str(path),
                            "global_step": global_step}
            scheduler.step()
            model.eval()
            with torch.inference_mode():
                validation = model(val_observations.to(cfg.device),
                                   val_actions.to(cfg.device), teacher_forcing=False)["prediction"]
                val_mse = (validation - val_observations[:, 1:].to(cfg.device)).square().mean().item()
            row = {"epoch": epoch, "global_step": global_step,
                   "train_loss": loss_sum / sample_count, "val_free_running_mse": val_mse}
            history.append(row)
            with (run_dir / "train.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            _save(run_dir / f"epoch-{epoch:04d}", model, optimizer, scheduler, cfg,
                  dataset.sha256, adapter.source_sha256, epoch, None,
                  global_step, 0.0, 0)
        final = run_dir / f"epoch-{cfg.epochs - 1:04d}"
        reloaded = (RSSMAdapter(cfg.repository, final / "weights.safetensors", cfg.device)
                    if cfg.family == "rssm" else
                    WorldDynamicsModelAdapter(cfg.repository, final / "weights.safetensors"))
        if cfg.family == "action_predictor":
            reloaded.model.to(cfg.device)
            # ActionPredictor adapter evaluates on CPU; use the identical saved weights there.
            reloaded.model.cpu()
        test = reloaded.evaluate(dataset)
        elapsed = time.perf_counter() - started
        report = {"status": "completed", "config": asdict(cfg),
                  "source_commit": adapter.source_commit,
                  "source_sha256": adapter.source_sha256,
                  "dataset_manifest_sha256": dataset.sha256,
                  "checkpoint": str(final),
                  "checkpoint_sha256": sha256(final / "weights.safetensors"),
                  "parameter_count": sum(p.numel() for p in model.parameters()),
                  "train_samples": len(train_rows), "optimizer_steps": global_step,
                  "processed_sequences_this_process": processed_samples,
                  "effective_batch_size": cfg.batch_size, "wall_seconds": elapsed,
                  "forward_seconds": forward_seconds,
                  "backward_seconds": backward_seconds,
                  "optimizer_seconds": optimizer_seconds,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated()
                  if cfg.device == "cuda" else None,
                  "sequences_per_second": processed_samples / elapsed,
                  "history": history, "test": test}
        with (run_dir / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        return report
    except BaseException as exc:
        with (run_dir / f"failure-{uuid.uuid4().hex}.json").open("x", encoding="utf-8") as stream:
            json.dump({"type": type(exc).__name__, "message": str(exc),
                       "global_step": global_step}, stream, indent=2)
        raise
