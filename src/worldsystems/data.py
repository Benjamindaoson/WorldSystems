"""Deterministic synthetic dynamics for software verification only."""

import hashlib

import torch
from torch.utils.data import Dataset


class SyntheticWorld(Dataset):
    def __init__(self, count: int, sequence_length: int, image_size: int,
                 action_dim: int, seed: int, namespace: str = "default"):
        self.count = count
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.action_dim = action_dim
        self.seed = seed
        self.namespace = namespace

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        key = f"{self.seed}:{self.namespace}:{index}".encode()
        sample_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63)
        g = torch.Generator().manual_seed(sample_seed)
        h = self.image_size
        frames = torch.empty(self.sequence_length + 1, 3, h, h)
        frames[0] = torch.rand(3, h, h, generator=g)
        actions = torch.randn(self.sequence_length, self.action_dim, generator=g) * 0.25
        for t in range(self.sequence_length):
            shift = actions[t].mean()
            frames[t + 1] = (0.92 * frames[t] + 0.08 * shift).clamp(0, 1)
        return {"video": frames[:-1], "actions": actions,
                "target": frames[1:], "episode_id": torch.tensor(index)}


def validate_batch(batch: dict, action_dim: int | None = None) -> None:
    video, actions, target = (batch[k] for k in ("video", "actions", "target"))
    if video.ndim != 5 or actions.ndim != 3 or target.shape != video.shape:
        raise ValueError("expected video/target [B,T,3,H,W], actions [B,T,A]")
    if video.shape[2] != 3 or video.shape[:2] != actions.shape[:2]:
        raise ValueError("video and action dimensions disagree")
    if action_dim is not None and actions.shape[-1] != action_dim:
        raise ValueError("action dimension mismatch")
    if not all(torch.isfinite(x).all() for x in (video, actions, target)):
        raise ValueError("nonfinite input")
