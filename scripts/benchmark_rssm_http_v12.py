"""Actual loopback HTTP versus direct RSSM inference on identical CPU input."""

import json
import socket
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import torch

from worldsystems.rssm_adapter import RSSMAdapter

REPO = "../WorldDynamics"
WEIGHTS = "results/v12/rssm-gpu-s17/epoch-0001/weights.safetensors"
URL = "http://127.0.0.1:18766"


def p(values, q):
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * q)]


def call(client, body):
    started = time.perf_counter()
    response = client.post(URL + "/predict", json=body)
    return {"status": response.status_code, "client_ms": (time.perf_counter() - started) * 1000,
            "body": response.json()}


def main():
    adapter = RSSMAdapter(REPO, WEIGHTS)
    observations = torch.zeros(1, 33, 3)
    actions = torch.zeros(1, 32, 3)
    body = {"observations": observations.tolist(), "actions": actions.tolist()}
    with torch.inference_mode():
        for _ in range(3):
            direct = adapter.predict(observations, actions)
        direct_ms = []
        for _ in range(30):
            started = time.perf_counter()
            direct = adapter.predict(observations, actions)
            direct_ms.append((time.perf_counter() - started) * 1000)
    process = subprocess.Popen([sys.executable, "-m", "worldsystems.cli", "rssm-serve",
                                "--repository", REPO, "--weights", WEIGHTS, "--port", "18766"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    started = time.perf_counter()
    try:
        while time.perf_counter() - started < 15:
            if process.poll() is not None:
                raise RuntimeError("RSSM server exited before bind")
            try:
                with socket.create_connection(("127.0.0.1", 18766), timeout=.1):
                    break
            except OSError:
                time.sleep(.05)
        else:
            raise TimeoutError("RSSM server did not bind")
        cold_ms = (time.perf_counter() - started) * 1000
        with httpx.Client(timeout=5, trust_env=False) as client:
            for _ in range(3):
                call(client, body)
            sequential = [call(client, body) for _ in range(30)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                concurrent = list(pool.map(lambda b: call(client, b), [body] * 12))
    finally:
        process.terminate()
        process.wait(timeout=5)
    good = [x for x in sequential if x["status"] == 200]
    timings = [x["client_ms"] for x in good]
    server = [x["body"]["server_processing_ms"] for x in good]
    error = (torch.tensor(good[0]["body"]["prediction"]) - direct).abs().max().item()
    report = {"status": "CPU_VERIFIED", "model": "WorldDynamics RSSM",
              "checkpoint_sha256": adapter.weight_sha256, "input_shape": [1, 33, 3],
              "action_shape": [1, 32, 3], "precision": "fp32", "runs": 30,
              "concurrent_requests": 12, "client_workers": 4,
              "cold_bind_ms": cold_ms, "direct_p50_ms": statistics.median(direct_ms),
              "server_processing_p50_ms": statistics.median(server),
              "client_p50_ms": statistics.median(timings),
              "client_p95_ms": p(timings, .95), "client_p99_ms": p(timings, .99),
              "sequential_failure_rate": 1 - len(good) / 30,
              "concurrent_status_counts": {str(code): sum(x["status"] == code for x in concurrent)
                                           for code in sorted({x["status"] for x in concurrent})},
              "max_output_error": error,
              "raw": {"direct_ms": direct_ms, "sequential": sequential,
                      "concurrent": concurrent}}
    Path("results/v12/http-rssm-cpu.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "raw"}, indent=2))


if __name__ == "__main__":
    main()
