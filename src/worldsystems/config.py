from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    seed: int = 17
    device: str = "cpu"
    precision: str = "fp32"
    epochs: int = 2
    batch_size: int = 8
    train_samples: int = 128
    val_samples: int = 32
    test_samples: int = 32
    sequence_length: int = 6
    image_size: int = 8
    action_dim: int = 3
    hidden_dim: int = 32
    learning_rate: float = 0.002
    weight_decay: float = 0.01
    grad_accumulation: int = 1
    grad_clip: float = 1.0
    checkpoint_dir: str = "artifacts/tiny_world"
    dataset_version: str = "synthetic-linear-v1.1"

    def __post_init__(self):
        if self.device not in {"cpu", "cuda"} or self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("unsupported device or precision")
        if min(self.epochs, self.batch_size, self.sequence_length, self.image_size,
               self.action_dim, self.hidden_dim, self.grad_accumulation) < 1:
            raise ValueError("dimensions and training counts must be positive")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.train_samples < 1 or self.val_samples < 1 or self.test_samples < 1:
            raise ValueError("all splits need samples")
        if self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError("learning_rate and grad_clip must be positive")


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("configuration must be a mapping")
    allowed = {f.name for f in fields(Config)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    return Config(**raw)
