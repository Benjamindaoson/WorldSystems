"""Read-only RSSM export adapter with a causal future-observation test."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from .worlddynamics import WorldDynamicsDatasetAdapter, digest


class RSSMAdapter:
    def __init__(self, repository: str | Path, weights: str | Path, device: str = "cpu"):
        self.repository = Path(repository).resolve()
        self.weights = Path(weights).resolve()
        self.device = device
        if self.weights.suffix == ".pt":
            if not self.weights.is_relative_to(self.repository / "artifacts"):
                raise ValueError("external PyTorch weights must be under trusted artifacts")
            sidecar = self.weights.parent / "manifest.json"
            self.export_manifest = json.loads(sidecar.read_text(encoding="utf-8"))
            if self.export_manifest.get("model") != "rssm":
                raise ValueError("checkpoint model family is not RSSM")
        elif self.weights.suffix == ".safetensors":
            self.export_manifest = {"model": "rssm"}
            local_manifest = json.loads((self.weights.parent / "manifest.json").read_text(encoding="utf-8"))
            if digest(self.weights) != local_manifest["files"]["weights.safetensors"]:
                raise ValueError("RSSM checkpoint hash mismatch")
            architecture = Path(local_manifest["config"]["architecture_weights"]).resolve()
            if not architecture.is_relative_to(self.repository / "artifacts"):
                raise ValueError("architecture provenance outside trusted repository")
            self.export_manifest["source_revision"] = json.loads(
                (architecture.parent / "manifest.json").read_text(encoding="utf-8"))["source_revision"]
        else:
            raise ValueError("unsupported RSSM weight format")
        source = self.repository / "src" / "iwm" / "models.py"
        if subprocess.run(["git", "-C", str(self.repository), "diff", "--quiet", "--",
                           "src/iwm/models.py"], check=False).returncode != 0:
            raise ValueError("uncommitted WorldDynamics model source")
        self.source_sha256 = digest(source)
        self.source_commit = subprocess.check_output(
            ["git", "-C", str(self.repository), "rev-parse", "HEAD"], text=True).strip()
        training_commit = self.export_manifest.get("source_revision")
        source_bytes = source.read_bytes()
        if training_commit:
            historical = subprocess.check_output(
                ["git", "-C", str(self.repository), "show",
                 f"{training_commit}:src/iwm/models.py"])
            source_bytes = historical
            self.source_sha256 = hashlib.sha256(historical).hexdigest()
            self.source_commit = training_commit
            if self.weights.suffix == ".safetensors" and self.source_sha256 != local_manifest["source_sha256"]:
                raise ValueError("RSSM pinned source hash mismatch")
        spec = importlib.util.spec_from_file_location("worldsystems_trusted_wd_rssm", source)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot import trusted RSSM source")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        # Execute only source from the trusted local Git repository, pinned above.
        exec(compile(source_bytes, str(source), "exec"), module.__dict__)  # noqa: S102
        state = (load_file(str(self.weights), device="cpu") if self.weights.suffix == ".safetensors"
                 else torch.load(self.weights, map_location="cpu", weights_only=True))
        if not isinstance(state, dict) or "encoder.0.weight" not in state or "prior.weight" not in state:
            raise ValueError("unsupported RSSM state dictionary")
        hidden, observation_dim = state["encoder.0.weight"].shape
        latent = state["prior.weight"].shape[0] // 2
        action_dim = state["recurrent.weight_ih"].shape[1] - latent
        if observation_dim != 3 or action_dim != 3 or state["prior.weight"].shape[0] % 2:
            raise ValueError("RSSM observation, action or latent dimension mismatch")
        self.model = module.RSSM(obs_dim=3, action_dim=3, hidden=hidden, latent=latent)
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.weight_sha256 = digest(self.weights)
        self.hidden = hidden
        self.latent = latent

    @torch.inference_mode()
    def predict(self, observations: torch.Tensor, actions: torch.Tensor,
                teacher_forcing: bool = False) -> torch.Tensor:
        if observations.ndim != 3 or actions.ndim != 3 or observations.shape != (
                actions.shape[0], actions.shape[1] + 1, 3) or actions.shape[-1] != 3:
            raise ValueError("RSSM expects observations [B,T+1,3], actions [B,T,3]")
        return self.model(observations.to(self.device), actions.to(self.device),
                          teacher_forcing=teacher_forcing)["prediction"]

    def evaluate(self, dataset: WorldDynamicsDatasetAdapter, split: str = "test") -> dict:
        if split not in {"test", "val"}:
            raise ValueError("frozen evaluation requires test or val")
        observations, actions, rows = dataset.arrays(split)
        forced = self.predict(observations, actions, teacher_forcing=True)
        free = self.predict(observations, actions, teacher_forcing=False)
        target = observations[:, 1:].to(self.device)
        future_perturbed = observations.clone()
        future_perturbed[:, 1:] += 100.0
        perturbed = self.predict(future_perturbed, actions, teacher_forcing=False)
        future_error = (free - perturbed).abs().max().item()
        if future_error > 1e-6:
            raise ValueError("RSSM free-running output depends on future observations")
        horizon_error = (free - target).square().mean(dim=(0, 2)).cpu().tolist()
        return {"status": "measured", "family": "WorldDynamics RSSM",
                "split": split, "samples": len(rows),
                "scenes": len({row["scene_id"] for row in rows}),
                "horizon": actions.shape[1], "observation_shape": list(observations.shape),
                "action_shape": list(actions.shape), "hidden": self.hidden, "latent": self.latent,
                "parameter_count": sum(p.numel() for p in self.model.parameters()),
                "source_commit": self.source_commit,
                "training_source_commit": self.export_manifest.get("source_revision"),
                "source_sha256": self.source_sha256, "checkpoint_sha256": self.weight_sha256,
                "dataset_manifest_sha256": dataset.sha256,
                "dataset_source_commit": dataset.manifest["source_revision"],
                "teacher_forced_mse": (forced - target).square().mean().item(),
                "free_running_mse": (free - target).square().mean().item(),
                "first_step_mse": horizon_error[0], "final_step_mse": horizon_error[-1],
                "horizon_mse": horizon_error, "future_perturbation_max_diff": future_error}
