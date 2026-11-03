"""Read-only adapter for a trusted local WorldDynamics ActionPredictor export."""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class WorldDynamicsDatasetAdapter:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != "worlddynamics-v11/v1" or self.manifest.get("backend") != "MUJOCO":
            raise ValueError("unsupported WorldDynamics dataset manifest")
        self.records = []
        for path in sorted(self.root.glob("cube-*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            expected = record.pop("artifact_sha256", None)
            raw = json.dumps(record, sort_keys=True, allow_nan=False).encode()
            if not expected or hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"dataset record hash mismatch: {path.name}")
            if record.get("rgb_file"):
                rgb_path = self.root / record["rgb_file"]
                if digest(rgb_path) != record["rgb_sha256"]:
                    raise ValueError(f"RGB hash mismatch: {rgb_path.name}")
            self.records.append(record)
        if len(self.records) != self.manifest["records"]:
            raise ValueError("dataset record count mismatch")
        scenes: dict[str, tuple[str, str]] = {}
        for record in self.records:
            scene = record["scene_id"]
            key = (record["split"], record["initial_state_sha256"])
            if scene in scenes and scenes[scene] != key:
                raise ValueError("scene split or initial-state leakage")
            scenes[scene] = key
        split_scenes = {split: {r["scene_id"] for r in self.records if r["split"] == split}
                        for split in ("train", "val", "test")}
        if any(split_scenes[a] & split_scenes[b] for a, b in
               (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError("scene leakage across frozen splits")
        initial_by_split: dict[str, str] = {}
        physics_by_split: dict[str, str] = {}
        for scene, (split, initial_hash) in scenes.items():
            if initial_hash in initial_by_split and initial_by_split[initial_hash] != split:
                raise ValueError(f"initial-state leakage across splits: {scene}")
            initial_by_split[initial_hash] = split
        for record in self.records:
            physics = json.dumps(record["physics"], sort_keys=True)
            split = record["split"]
            if physics in physics_by_split and physics_by_split[physics] != split:
                raise ValueError("physics configuration leakage across splits")
            physics_by_split[physics] = split
        self.sha256 = digest(self.manifest_path)

    def arrays(self, split: str) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
        if split not in {"train", "val", "test"}:
            raise ValueError("invalid split")
        rows = [r for r in self.records if r["split"] == split]
        if not rows:
            raise ValueError(f"empty {split} split")
        states = np.asarray([r["state"] for r in rows], dtype=np.float32)
        actions = np.asarray([r["action_sequence"] for r in rows], dtype=np.float32)
        if states.ndim != 3 or actions.ndim != 3 or states.shape[1] != actions.shape[1] + 1:
            raise ValueError("invalid trajectory shapes")
        if states.shape[-1] != 3 or actions.shape[-1] != 3:
            raise ValueError("expected 3D position and 3D force")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("nonfinite physical trajectory")
        return torch.from_numpy(states), torch.from_numpy(actions), rows


class WorldDynamicsModelAdapter:
    def __init__(self, repository: str | Path, weights: str | Path):
        self.repository = Path(repository).resolve()
        self.weights = Path(weights).resolve()
        if self.weights.suffix == ".pt":
            if not self.weights.is_relative_to(self.repository / "artifacts"):
                raise ValueError("PyTorch weights must be under trusted repository artifacts")
        elif self.weights.suffix != ".safetensors":
            raise ValueError("unsupported weight format")
        source = self.repository / "src" / "iwm" / "models.py"
        if subprocess.run(["git", "-C", str(self.repository), "diff", "--quiet", "--",
                           "src/iwm/models.py"], check=False).returncode != 0:
            raise ValueError("WorldDynamics model source has uncommitted changes")
        self.source_commit = subprocess.check_output(
            ["git", "-C", str(self.repository), "rev-parse", "HEAD"], text=True).strip()
        self.source_sha256 = digest(source)
        spec = importlib.util.spec_from_file_location("worldsystems_trusted_wd_models", source)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load WorldDynamics source")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        state = (load_file(str(self.weights), device="cpu") if self.weights.suffix == ".safetensors"
                 else torch.load(self.weights, map_location="cpu", weights_only=True))
        if not isinstance(state, dict) or "net.0.weight" not in state:
            raise ValueError("unsupported ActionPredictor weight format")
        hidden, input_dim = state["net.0.weight"].shape
        if input_dim != 6 or state["net.4.weight"].shape[0] != 3:
            raise ValueError("ActionPredictor input/output dimension mismatch")
        self.model = module.ActionPredictor(obs_dim=3, action_dim=3, hidden=hidden)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.weight_sha256 = digest(self.weights)
        self.hidden = hidden

    @torch.inference_mode()
    def predict(self, states: torch.Tensor, actions: torch.Tensor,
                teacher_forcing: bool = False) -> torch.Tensor:
        return self.model(states, actions, teacher_forcing=teacher_forcing)["prediction"]

    def evaluate(self, dataset: WorldDynamicsDatasetAdapter, split: str = "test") -> dict:
        if split not in {"val", "test"}:
            raise ValueError("evaluation requires frozen val or test split")
        states, actions, rows = dataset.arrays(split)
        forced = self.predict(states, actions, teacher_forcing=True)
        free = self.predict(states, actions, teacher_forcing=False)
        actual = states[:, 1:]
        horizon_mse = (free - actual).square().mean(dim=(0, 2)).tolist()
        return {"status": "measured", "workload": "mujoco-physical-state-prediction",
                "split": split, "samples": len(rows),
                "scenes": len({r["scene_id"] for r in rows}),
                "horizon": actions.shape[1], "state_dim": 3, "action_dim": 3,
                "architecture": "WorldDynamics ActionPredictor", "hidden_dim": self.hidden,
                "parameter_count": sum(p.numel() for p in self.model.parameters()),
                "source_commit": self.source_commit, "source_sha256": self.source_sha256,
                "weight_sha256": self.weight_sha256, "dataset_manifest_sha256": dataset.sha256,
                "dataset_source_commit": dataset.manifest["source_revision"],
                "one_step_mse": torch.mean((forced - actual).square()).item(),
                "free_running_mse": torch.mean((free - actual).square()).item(),
                "final_horizon_mse": torch.mean((free[:, -1] - actual[:, -1]).square()).item(),
                "horizon_mse": horizon_mse}
