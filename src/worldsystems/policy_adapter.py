"""Read-only inference for the locally audited WorldPolicy ACT checkpoint."""

import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import torch


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ACTPolicyAdapter:
    def __init__(self, repository: str | Path, checkpoint: str | Path,
                 training_record: str | Path, device: str = "cpu"):
        repo = Path(repository).resolve()
        path = Path(checkpoint).resolve()
        record_path = Path(training_record).resolve()
        if not path.is_relative_to(repo / "artifacts") or not record_path.is_relative_to(repo):
            raise ValueError("checkpoint and record must be in the trusted WorldPolicy repository")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        self.checkpoint_sha256 = _sha(path)
        if self.checkpoint_sha256 != record["checkpoint_sha256"]:
            raise ValueError("WorldPolicy checkpoint hash mismatch")
        source = repo / "src/worldpolicy/policies.py"
        self.source_commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--",
                                    str(source)], text=True).strip():
            raise ValueError("WorldPolicy model source has uncommitted changes")
        self.source_sha256 = _sha(source)
        spec = importlib.util.spec_from_file_location("worldpolicy_readonly_policies", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.ACT(8, 2, 4, hidden=64, latent=16, layers=2)
        # WorldPolicy's checkpoint embeds NumPy/Python RNG tuples and cannot use
        # weights_only=True on PyTorch 2.5. Load only after checking its recorded hash.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["architecture"] != "ACT" or payload["steps"] != record["steps"]:
            raise ValueError("WorldPolicy checkpoint metadata mismatch")
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def predict_chunk(self, state: torch.Tensor, deadline_ms: int = 1000) -> dict:
        if state.ndim != 2 or state.shape[-1] != 8 or not torch.isfinite(state).all():
            raise ValueError("expected finite state [batch, 8]")
        if deadline_ms < 1:
            raise ValueError("deadline must be positive")
        started = time.perf_counter()
        prediction = self.model(state.to(self.device))[0]
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        if elapsed > deadline_ms:
            raise TimeoutError("policy inference deadline exceeded")
        return {"actions": prediction.cpu(), "latency_ms": elapsed,
                "timestamp_unix": time.time(), "state_dim": 8, "action_dim": 2,
                "horizon": 4, "checkpoint_sha256": self.checkpoint_sha256}
