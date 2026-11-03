"""Build a local hash inventory of V1.2 raw reports without copying large weights."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path("results/v12")
    raw = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "evidence-manifest.json")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True, check=False)
    report = {"schema": "worldsystems-v12-evidence-v1", "git_head_at_generation": head,
              "git_branch": subprocess.check_output(["git", "branch", "--show-current"],
                                                     text=True).strip(),
              "python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "gpu": gpu.stdout.strip() if gpu.returncode == 0 else "unavailable",
              "dataset_version": "worlddynamics-v11/v1",
              "dataset_manifest_sha256": sha(Path("../WorldDynamics/artifacts/"
                                                  "v11-mujoco-32scene-h32/manifest.json")),
              "raw_files": [{"path": p.as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
                            for p in raw]}
    (root / "evidence-manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"indexed {len(raw)} raw files")


if __name__ == "__main__":
    main()
