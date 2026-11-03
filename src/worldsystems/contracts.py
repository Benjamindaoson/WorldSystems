"""Versioned interoperability metadata; external code is never imported by a manifest."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int | None, ...]
    dtype: str
    unit: str | None = None
    coordinate_frame: str | None = None


@dataclass(frozen=True)
class ModelManifest:
    schema_version: int
    model_id: str
    family: Literal["tiny-action-temporal", "worlddynamics", "worldpolicy"]
    architecture: str
    model_version: str
    source_repository: str
    source_commit: str
    checkpoint_id: str
    weight_sha256: str
    video: TensorSpec | None
    actions: TensorSpec
    output: TensorSpec
    precision: str
    dataset_version: str
    prediction_horizon: int
    capabilities: tuple[str, ...]
    evidence_status: str

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported manifest version")
        if len(self.weight_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.weight_sha256):
            raise ValueError("invalid weight SHA256")
        if self.actions.shape.__len__() != 3 or self.actions.shape[-1] is None:
            raise ValueError("actions require [B,T,A] with fixed action dimension")
        if self.family == "worldpolicy" and self.actions.unit is None:
            raise ValueError("policy action units required")
        if self.prediction_horizon < 1:
            raise ValueError("prediction horizon must be positive")
