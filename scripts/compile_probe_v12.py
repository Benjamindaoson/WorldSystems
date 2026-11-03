"""Isolated torch.compile/Inductor compatibility probe for a trained RSSM."""

import json
import subprocess
import sys
from pathlib import Path

CODE = """import json, time, torch
from worldsystems.rssm_adapter import RSSMAdapter
a=RSSMAdapter('../WorldDynamics',
 'results/v12/rssm-gpu-s17/epoch-0001/weights.safetensors','cuda')
x=torch.zeros(1,33,3,device='cuda'); u=torch.zeros(1,32,3,device='cuda')
start=time.perf_counter()
f=torch.compile(lambda x,u: a.model(x,u,teacher_forcing=False)['prediction'],backend='inductor')
y=f(x,u); torch.cuda.synchronize()
print(json.dumps({'status':'measured','compile_and_first_call_ms':(time.perf_counter()-start)*1000,
 'output_shape':list(y.shape)}))
"""


def main():
    output = Path("results/v12/rssm-compile-probe.json")
    try:
        done = subprocess.run([sys.executable, "-c", CODE], capture_output=True,
                              text=True, timeout=90, check=False)
        if done.returncode:
            report = {"status": "BLOCKED", "returncode": done.returncode,
                      "stderr_tail": done.stderr[-3000:]}
        else:
            report = json.loads(done.stdout)
    except subprocess.TimeoutExpired as exc:
        report = {"status": "BLOCKED", "reason": f"90-second timeout: {exc}"}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
