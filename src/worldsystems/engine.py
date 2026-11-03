import json
import math
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .config import Config
from .data import SyntheticWorld, validate_batch
from .model import TinyWorld


def build_model(cfg: Config) -> TinyWorld:
    return TinyWorld(cfg.image_size, cfg.action_dim, cfg.hidden_dim)


def loader(cfg: Config, split: str, epoch: int = 0) -> DataLoader:
    count = getattr(cfg, f"{split}_samples")
    dataset = SyntheticWorld(count, cfg.sequence_length, cfg.image_size,
                             cfg.action_dim, cfg.seed, namespace=split)
    generator = torch.Generator().manual_seed(cfg.seed + epoch)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=split == "train",
                      generator=generator, pin_memory=cfg.device == "cuda")


def evaluate(model: TinyWorld, cfg: Config, split: str = "test") -> dict:
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for batch in loader(cfg, split):
            validate_batch(batch, cfg.action_dim)
            video, actions, target = (batch[k].to(cfg.device) for k in
                                      ("video", "actions", "target"))
            prediction, _ = model(video, actions)
            total += torch.nn.functional.mse_loss(prediction, target, reduction="sum").item()
            count += target.numel()
    return {"split": split, "mse": total / count, "samples": len(loader(cfg, split).dataset)}


def train(cfg: Config, resume: str | None = None, max_epochs: int | None = None,
          max_steps: int | None = None) -> dict:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if cfg.precision == "fp16" and cfg.device != "cuda":
        raise ValueError("fp16 training requires CUDA")
    if cfg.precision == "bf16" and cfg.device == "cuda" and not torch.cuda.is_bf16_supported(
            including_emulation=False):
        raise ValueError("native CUDA BF16 is unavailable on this GPU")
    torch.manual_seed(cfg.seed)
    model = build_model(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.precision == "fp16")
    start_epoch = 0
    start_batch = 0
    resumed_total = 0.0
    resumed_count = 0
    global_step = 0
    if resume:
        manifest = load_checkpoint(resume, model, optimizer, scheduler, cfg.device, scaler)
        if manifest["config"] != asdict(cfg):
            raise ValueError("resume configuration mismatch")
        if manifest.get("sampler_schema") not in (None, "sha256-split-v1"):
            raise ValueError("unsupported sampler schema")
        if manifest.get("sampler_schema") is None and cfg.dataset_version == "synthetic-linear-v1":
            raise ValueError("V1 checkpoint sampler differs from current generator; resume blocked")
        if manifest.get("next_batch") is None:
            start_epoch = manifest["epoch"] + 1
        else:
            start_epoch = manifest["epoch"]
            start_batch = manifest["next_batch"]
            resumed_total = manifest["train_total"]
            resumed_count = manifest["train_count"]
        global_step = manifest["global_step"]
    run_dir = Path(cfg.checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        for epoch in range(start_epoch, max_epochs or cfg.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total = resumed_total if epoch == start_epoch else 0.0
            count = resumed_count if epoch == start_epoch else 0
            start = time.perf_counter()
            batches = loader(cfg, "train", epoch)
            for i, batch in enumerate(batches):
                if epoch == start_epoch and i < start_batch:
                    continue
                validate_batch(batch, cfg.action_dim)
                video, actions, target = (batch[k].to(cfg.device, non_blocking=True)
                                          for k in ("video", "actions", "target"))
                with torch.autocast(cfg.device, dtype=torch.bfloat16 if cfg.precision == "bf16"
                                    else torch.float16, enabled=cfg.precision != "fp32"):
                    prediction, _ = model(video, actions)
                    loss = torch.nn.functional.mse_loss(prediction, target)
                if not math.isfinite(loss.item()) or loss.item() > 1e6:
                    raise FloatingPointError(f"invalid loss: {loss.item()}")
                group_start = (i // cfg.grad_accumulation) * cfg.grad_accumulation
                group_size = min(cfg.grad_accumulation, len(batches) - group_start)
                scaler.scale(loss / group_size).backward()
                if (i + 1) % cfg.grad_accumulation == 0 or i + 1 == len(batches):
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("nonfinite gradient norm")
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                total += loss.item() * video.shape[0]
                count += video.shape[0]
                if max_steps is not None and global_step >= max_steps:
                    if (i + 1) % cfg.grad_accumulation != 0 and i + 1 != len(batches):
                        raise RuntimeError("step limit reached with partial gradient accumulation")
                    checkpoint = run_dir / f"step-{global_step:06d}"
                    save_checkpoint(checkpoint, model, optimizer, scheduler, epoch,
                                    global_step, asdict(cfg), scaler=scaler,
                                    next_batch=i + 1, train_total=total, train_count=count)
                    return {"status": "interrupted_at_step_boundary", "records": records,
                            "checkpoint": str(checkpoint)}
            scheduler.step()
            if cfg.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            validation = evaluate(model, cfg, "val")
            record = {"epoch": epoch, "global_step": global_step,
                      "train_mse": total / count, "val_mse": validation["mse"],
                      "sequences_per_second": count / elapsed, "elapsed_seconds": elapsed,
                      "peak_memory_bytes": torch.cuda.max_memory_allocated() if cfg.device == "cuda" else None}
            records.append(record)
            save_checkpoint(run_dir / f"epoch-{epoch:04d}", model, optimizer, scheduler,
                            epoch, global_step, asdict(cfg), scaler=scaler)
            (run_dir / "train.jsonl").open("a", encoding="utf-8").write(json.dumps(record) + "\n")
        return {"status": "completed", "records": records,
                "checkpoint": str(run_dir / f"epoch-{(max_epochs or cfg.epochs) - 1:04d}")}
    except BaseException as exc:
        (run_dir / f"failure-{uuid.uuid4().hex}.json").write_text(json.dumps({"type": type(exc).__name__,
            "message": str(exc), "epoch": epoch if "epoch" in locals() else None,
            "global_step": global_step}, indent=2), encoding="utf-8")
        raise
