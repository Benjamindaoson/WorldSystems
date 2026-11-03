"""Atomic, hash-verified checkpoints. Only load locally produced optimizer state."""

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path: str | Path, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler, epoch: int,
                    global_step: int, config: dict, *, scaler=None,
                    next_batch: int | None = None, train_total: float = 0.0,
                    train_count: int = 0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temp.mkdir()
    try:
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                  str(temp / "weights.safetensors"))
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else {},
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "epoch": epoch, "global_step": global_step}, temp / "training.pt")
        manifest = {"schema_version": 1, "model_family": "tiny-action-temporal",
                    "epoch": epoch, "global_step": global_step, "config": config,
                    "sampler_schema": "sha256-split-v1",
                    "next_batch": next_batch, "train_total": train_total,
                    "train_count": train_count,
                    "files": {name: sha256(temp / name) for name in
                              ("weights.safetensors", "training.pt")}}
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temp, path)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return path


def verify_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("model_family") != "tiny-action-temporal":
        raise ValueError("unsupported checkpoint schema or family")
    if set(manifest.get("files", {})) != {"weights.safetensors", "training.pt"}:
        raise ValueError("incomplete checkpoint manifest")
    for name, expected in manifest["files"].items():
        if sha256(path / name) != expected:
            raise ValueError(f"checkpoint checksum mismatch: {name}")
    return manifest


def load_checkpoint(path: str | Path, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None, scheduler=None,
                    device: str = "cpu", scaler=None) -> dict:
    manifest = verify_checkpoint(path)
    cfg = manifest["config"]
    if (model.image_size, model.action_dim, model.temporal.hidden_size) != (
            cfg["image_size"], cfg["action_dim"], cfg["hidden_dim"]):
        raise ValueError("model architecture mismatch")
    model.load_state_dict(load_file(str(Path(path) / "weights.safetensors"), device=device))
    if optimizer is not None:
        state = torch.load(Path(path) / "training.pt", map_location=device, weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        if scaler is not None:
            scaler.load_state_dict(state.get("scaler", {}))
        torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available() and state["cuda_rng"]:
            torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])
    return manifest
