"""Record the exact local software and GPU environment for a run."""

import json
import platform
import subprocess
from pathlib import Path

import torch


def capture(output: str = "results/v11/environment.json"):
    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
         "--format=csv,noheader"], capture_output=True, text=True, check=False)
    record = {"python": platform.python_version(), "system": platform.platform(),
              "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(),
              "device_count": torch.cuda.device_count(),
              "gpu_query": gpu_query.stdout.strip() if gpu_query.returncode == 0 else None,
              "gpu_query_error": gpu_query.stderr.strip() if gpu_query.returncode else None,
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                    text=True).strip()}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    capture()
