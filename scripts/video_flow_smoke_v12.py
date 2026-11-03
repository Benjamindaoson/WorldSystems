"""Bounded GPU training probe on a verified frozen MuJoCo RGB trajectory."""

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    repo = Path("../WorldDynamics")
    dataset = repo / "artifacts/v11-mujoco-32scene-h32"
    output = Path("results/v12/video-flow-gpu-smoke.json")
    record_path = next(p for p in sorted(dataset.glob("cube-*.json"))
                       if json.loads(p.read_text(encoding="utf-8"))["split"] == "train")
    row = json.loads(record_path.read_text(encoding="utf-8"))
    rgb_path = dataset / row["rgb_file"]
    if sha(rgb_path) != row["rgb_sha256"]:
        raise ValueError("frozen RGB hash mismatch")
    source = repo / "src/iwm/visual_models.py"
    spec = importlib.util.spec_from_file_location("worlddynamics_visual_readonly", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.VideoFlow(width=32).cuda()
    frames = torch.from_numpy(np.load(rgb_path)["rgb"][:5].copy()).permute(0, 3, 1, 2)
    frames = frames.float().div(255).unsqueeze(0).cuda()
    actions = torch.tensor(row["action_sequence"][:4], dtype=torch.float32).unsqueeze(0).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    torch.manual_seed(17)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    try:
        for step in range(2):
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(frames, actions)
            torch.cuda.synchronize()
            forward = time.perf_counter() - start
            start = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            backward = time.perf_counter() - start
            start = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize()
            update = time.perf_counter() - start
            rows.append({"step": step + 1, "loss": loss.item(),
                         "forward_s": forward, "backward_s": backward,
                         "optimizer_s": update})
        status = "GPU_VERIFIED"
        failure = None
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        status, failure = "BLOCKED", f"{type(exc).__name__}: {exc}"
    report = {"status": status, "scope": "2-step smoke only; no held-out quality claim",
              "model": "WorldDynamics VideoFlow", "source_sha256": sha(source),
              "dataset_manifest_sha256": sha(dataset / "manifest.json"),
              "rgb_sha256": sha(rgb_path), "record": record_path.name,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "batch_size": 1, "horizon": 4, "resolution": [32, 32],
              "precision": "fp32", "device": torch.cuda.get_device_name(0),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "steps": rows, "failure": failure}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
