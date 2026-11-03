"""Reproduce capture failure in isolation, then audit full-forward graph equivalence."""

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from worldsystems.graph_runtime import StaticGraph, capture_ready_model
from worldsystems.rssm_adapter import RSSMAdapter
from worldsystems.rssm_fastpath import predict_prior_means
from worldsystems.worlddynamics import WorldDynamicsDatasetAdapter

REPO = "../WorldDynamics"
WEIGHTS = "results/v12/rssm-gpu-s17/epoch-0001/weights.safetensors"
DATA = REPO + "/artifacts/v11-mujoco-32scene-h32"


def snapshot():
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.free,utilization.gpu,driver_version",
                           "--format=csv,noheader"], capture_output=True, text=True, check=False).stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if args.probe:
        try:
            x = torch.zeros(1, device="cuda")
            with torch.cuda.graph(torch.cuda.CUDAGraph()):
                torch.distributions.Normal(x, x + 1)
        except RuntimeError:
            (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return
    subprocess.run([sys.executable, __file__, "--output", str(output / "normal-probe"),
                    "--probe"], check=True, timeout=30)
    before = snapshot()
    adapter = RSSMAdapter(REPO, WEIGHTS, "cuda")
    original = adapter.model
    full = capture_ready_model(original)
    dataset = WorldDynamicsDatasetAdapter(DATA)
    x_all, u_all, _ = dataset.arrays("test")
    equality = []
    rows = []
    captures = []
    with torch.inference_mode():
        for batch, horizon in [(1, 4), (2, 8), (1, 16), (1, 32)]:
            x = x_all[:batch, :horizon + 1].cuda()
            u = u_all[:batch, :horizon].cuda()
            def full_call(x, u):
                return full(x, u, teacher_forcing=False)["prediction"]
            def fast_call(x, u):
                return predict_prior_means(original, x, u)
            full_graph = StaticGraph(full_call, x, u)
            fast_graph = StaticGraph(fast_call, x, u)
            captures.append({"batch": batch, "horizon": horizon,
                             "full_ms": full_graph.capture_ms, "fast_ms": fast_graph.capture_ms})
            methods = {"original_eager": lambda x, u: original(x, u, teacher_forcing=False)["prediction"],
                       "full_unchecked_eager": full_call, "fast_eager": fast_call,
                       "full_graph": full_graph, "fast_graph": fast_graph}
            for seed in [17, 23, 41]:
                torch.manual_seed(seed)
                altered_x = x + torch.randn_like(x) * .01
                altered_u = u + torch.randn_like(u) * .1
                reference = methods["original_eager"](altered_x, altered_u)
                for name, fn in methods.items():
                    error = (fn(altered_x, altered_u) - reference).abs().max().item()
                    if error > 1e-6:
                        raise AssertionError((name, error))
                    equality.append({"batch": batch, "horizon": horizon, "seed": seed,
                                     "method": name, "max_error": error})
            for index in range(args.runs):
                order = list(methods)
                order = order[index % len(order):] + order[:index % len(order)]
                for name in order:
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    begin.record()
                    prediction = methods[name](x, u)
                    end.record()
                    launch_done = time.perf_counter()
                    torch.cuda.synchronize()
                    synchronized = time.perf_counter()
                    prediction.cpu()
                    finished = time.perf_counter()
                    rows.append({"batch": batch, "horizon": horizon, "run": index,
                                 "method": name, "wall_ms": (synchronized-started)*1000,
                                 "launch_ms": (launch_done-started)*1000,
                                 "sync_wait_ms": (synchronized-launch_done)*1000,
                                 "d2h_ms": (finished-synchronized)*1000,
                                 "readback_total_ms": (finished-started)*1000,
                                 "cuda_event_ms": begin.elapsed_time(end)})
    summary = []
    for shape in captures:
        for name in methods:
            selected = [r for r in rows if r["batch"] == shape["batch"] and
                        r["horizon"] == shape["horizon"] and r["method"] == name]
            summary.append({"batch": shape["batch"], "horizon": shape["horizon"], "method": name,
                            "p50_ms": statistics.median(r["wall_ms"] for r in selected),
                            "event_p50_ms": statistics.median(r["cuda_event_ms"] for r in selected)})
    with (output / "raw.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"status": "GPU_VERIFIED", "scope": "MODEL_CALL", "summary": summary,
              "equality": equality, "capture": captures, "runs_per_cell": args.runs,
              "checkpoint_sha256": adapter.weight_sha256, "dataset_sha256": dataset.sha256,
              "source_sha256": adapter.source_sha256, "python": platform.python_version(),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu_before": before, "gpu_after": snapshot(),
              "performance_status": "exploratory; exclusive GPU ownership not established",
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "note": "graphs include input copy, finite checks, output clone; events include host gaps; "
                      "full path retains distribution scale arithmetic but disables internal validation"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
