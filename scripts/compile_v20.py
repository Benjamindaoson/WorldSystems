"""Isolated real-model Inductor execution probes, with complete stderr receipts."""

import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch


def main():
    root = Path("results/v20/compile-01")
    root.mkdir(parents=True, exist_ok=False)
    environment = {"os": platform.platform(), "python": sys.version, "torch": torch.__version__,
                   "cuda": torch.version.cuda, "capability": torch.cuda.get_device_capability(),
                   "triton_available": importlib.util.find_spec("triton") is not None}
    reports = []
    for family in ["rssm", "action_predictor"]:
        constructor = ("RSSMAdapter('../WorldDynamics',"
                       "'results/v12/rssm-gpu-s17/epoch-0001/weights.safetensors','cuda')"
                       if family == "rssm" else "WorldDynamicsModelAdapter('../WorldDynamics',"
                       "'../WorldDynamics/artifacts/v11-baselines-fair/conditioned-s17/best.pt')")
        code = f"""import torch
from worldsystems.rssm_adapter import RSSMAdapter
from worldsystems.worlddynamics import WorldDynamicsModelAdapter
a={constructor}
a.model.to('cuda').eval()
x=torch.zeros(1,5,3,device='cuda'); u=torch.zeros(1,4,3,device='cuda')
f=torch.compile(a.model, backend='inductor')
with torch.inference_mode():
 y=f(x,u,teacher_forcing=False)['prediction']; torch.cuda.synchronize()
 print(y.shape)
"""
        try:
            done = subprocess.run([sys.executable, "-c", code], text=True,
                                  capture_output=True, timeout=90, check=False)
            (root / f"{family}-stderr.txt").write_text(done.stderr, encoding="utf-8")
            reports.append({"family": family, "status": "GPU_VERIFIED" if done.returncode == 0
                            else "BLOCKED", "returncode": done.returncode, "stdout": done.stdout})
        except subprocess.TimeoutExpired:
            reports.append({"family": family, "status": "BLOCKED", "reason": "90s timeout"})
    (root / "report.json").write_text(json.dumps({"environment": environment, "models": reports},
                                                 indent=2), encoding="utf-8")
    print(reports)


if __name__ == "__main__":
    main()
