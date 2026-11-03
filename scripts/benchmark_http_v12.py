"""Loopback HTTP client versus direct TinyWorld inference, with raw timings."""

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

from worldsystems.config import load_config
from worldsystems.runtime import load_model

HOST, PORT = "127.0.0.1", 18765
CONFIG = "configs/training/tiny_world.yaml"
CHECKPOINT = "artifacts/tiny_world/epoch-0001"


def percentile(values, q):
    sorted_values = sorted(values)
    return sorted_values[int((len(sorted_values) - 1) * q)]


def start_server():
    code = ("import uvicorn; from worldsystems.config import load_config; "
            "from worldsystems.server import create_app; "
            f"uvicorn.run(create_app(load_config('{CONFIG}'),'{CHECKPOINT}'),"
            f"host='{HOST}',port={PORT},log_level='error')")
    process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, text=True)
    start = time.perf_counter()
    while time.perf_counter() - start < 15:
        if process.poll() is not None:
            raise RuntimeError(f"server exited: {process.stderr.read()[-1000:]}")
        try:
            with socket.create_connection((HOST, PORT), timeout=0.1):
                return process, (time.perf_counter() - start) * 1000
        except OSError:
            time.sleep(0.05)
    process.terminate()
    raise TimeoutError("server did not bind loopback port")


def request(client, payload):
    start = time.perf_counter()
    response = client.post(f"http://{HOST}:{PORT}/predict", json=payload)
    return {"status": response.status_code, "client_ms": (time.perf_counter() - start) * 1000,
            "body": response.json()}


def main():
    cfg = load_config(CONFIG)
    model = load_model(cfg, CHECKPOINT)
    video = torch.zeros(1, cfg.sequence_length, 3, cfg.image_size, cfg.image_size)
    actions = torch.zeros(1, cfg.sequence_length, cfg.action_dim)
    payload = {"video": video.tolist(), "actions": actions.tolist()}
    with torch.inference_mode():
        for _ in range(10):
            direct, _ = model(video, actions)
        direct_ms = []
        for _ in range(50):
            start = time.perf_counter()
            direct, _ = model(video, actions)
            direct_ms.append((time.perf_counter() - start) * 1000)
    server, cold_ms = start_server()
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            for _ in range(5):
                request(client, payload)
            sequential = [request(client, payload) for _ in range(50)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                concurrent = list(pool.map(lambda body: request(client, body), [payload] * 20))
        restarted = False
    finally:
        server.terminate()
        server.wait(timeout=5)
    second, restart_ms = start_server()
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            restarted = request(client, payload)["status"] == 200
    finally:
        second.terminate()
        second.wait(timeout=5)
    good = [x for x in sequential if x["status"] == 200]
    latency = [x["client_ms"] for x in good]
    processing = [x["body"]["server_processing_ms"] for x in good]
    dispatch = [x["body"]["compute_and_dispatch_ms"] for x in good]
    output = torch.tensor(good[0]["body"]["prediction"])
    report = {"status": "CPU_VERIFIED", "model": "TinyWorld", "checkpoint": CHECKPOINT,
              "binding": HOST, "server_cold_start_ms": cold_ms,
              "server_restart_ms": restart_ms, "restart_success": restarted,
              "sequential_runs": 50, "concurrent_requests": 20, "workers": 4,
              "direct_p50_ms": statistics.median(direct_ms),
              "compute_and_dispatch_p50_ms": statistics.median(dispatch),
              "server_processing_p50_ms": statistics.median(processing),
              "client_p50_ms": statistics.median(latency),
              "client_p95_ms": percentile(latency, .95),
              "client_p99_ms": percentile(latency, .99),
              "sequential_failure_rate": 1 - len(good) / len(sequential),
              "concurrent_status_counts": {str(code): sum(x["status"] == code for x in concurrent)
                                           for code in sorted({x["status"] for x in concurrent})},
              "max_output_error": (output - direct).abs().max().item(),
              "raw": {"direct_ms": direct_ms, "sequential": sequential,
                      "concurrent": concurrent}}
    Path("results/v12/http-tiny-cpu.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "raw"}, indent=2))


if __name__ == "__main__":
    main()
