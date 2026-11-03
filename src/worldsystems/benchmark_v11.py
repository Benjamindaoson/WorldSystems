"""Raw per-call benchmark with separate CUDA event and synchronized wall times."""

import csv
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

from .checkpoint import verify_checkpoint
from .config import Config
from .data import SyntheticWorld
from .engine import evaluate
from .runtime import load_model


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def benchmark_v11(cfg: Config, checkpoint: str, output_dir: str,
                  runs: int = 500, warmup: int = 30) -> dict:
    if runs < 5 or warmup < 1:
        raise ValueError("need at least five timed runs and one warmup")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    load_start = time.perf_counter()
    model = load_model(cfg, checkpoint)
    cold_load_ms = (time.perf_counter() - load_start) * 1000
    sample = SyntheticWorld(1, cfg.sequence_length, cfg.image_size,
                            cfg.action_dim, cfg.seed, namespace="test")[0]
    video = sample["video"].unsqueeze(0).to(cfg.device)
    actions = sample["actions"].unsqueeze(0).to(cfg.device)
    if cfg.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    warmup_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(warmup):
            expected, _ = model(video, actions)
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        warmup_ms = (time.perf_counter() - warmup_start) * 1000
        if cfg.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
        rows = []
        for index in range(runs):
            start = time.perf_counter()
            if cfg.device == "cuda":
                start_event.record()
            actual, _ = model(video, actions)
            if cfg.device == "cuda":
                end_event.record()
                torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - start) * 1000
            rows.append({"run": index, "wall_ms": wall_ms,
                         "cuda_event_ms": start_event.elapsed_time(end_event)
                         if cfg.device == "cuda" else None})
    difference = (actual - expected).abs().max().item()
    if difference > 1e-6:
        raise ValueError("benchmark output mismatch")
    times = [row["wall_ms"] for row in rows]
    events = [row["cuda_event_ms"] for row in rows if row["cuda_event_ms"] is not None]
    manifest = verify_checkpoint(checkpoint)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    gpu = torch.cuda.get_device_name(0) if cfg.device == "cuda" else None
    report = {"status": "measured", "workload": "synthetic-software-verification",
              "method": "CUDA synchronized wall clock; CUDA events span model call and may include host gaps",
              "device": cfg.device, "gpu": gpu, "torch": torch.__version__,
              "cuda_runtime": torch.version.cuda, "python": platform.python_version(),
              "cpu_threads": torch.get_num_threads(), "git_commit": revision,
              "checkpoint": checkpoint, "checkpoint_sha256": manifest["files"]["weights.safetensors"],
              "input_shape": list(video.shape), "action_shape": list(actions.shape),
              "precision": cfg.precision, "warmup": warmup, "runs": runs,
              "cold_model_load_ms": cold_load_ms, "warmup_ms": warmup_ms,
              "p50_wall_ms": _percentile(times, 0.5),
              "p95_wall_ms": _percentile(times, 0.95),
              "p99_wall_ms": _percentile(times, 0.99),
              "mean_wall_ms": statistics.mean(times),
              "p50_cuda_event_ms": _percentile(events, 0.5) if events else None,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated()
              if cfg.device == "cuda" else None,
              "max_output_difference": difference,
              "test_mse": evaluate(model, cfg, "test")["mse"]}
    with (output / "raw.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "wall_ms", "cuda_event_ms"])
        writer.writeheader()
        writer.writerows(rows)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
