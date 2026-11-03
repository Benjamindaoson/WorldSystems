"""Matched eager/CUDA Graph benchmark on a trained physical RSSM."""

import csv
import json
import statistics
import time
from pathlib import Path

import torch

from .rssm_adapter import RSSMAdapter
from .rssm_fastpath import predict_prior_means
from .worlddynamics import WorldDynamicsDatasetAdapter


def _p(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def benchmark_rssm(repository: str, weights: str, dataset_path: str,
                   output_dir: str, runs: int = 100, warmup: int = 10,
                   fastpath: bool = False) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if runs < 20 or warmup < 3:
        raise ValueError("insufficient benchmark repetitions")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    load_start = time.perf_counter()
    adapter = RSSMAdapter(repository, weights, "cuda")
    model = adapter.model
    def predict(obs, act):
        return (predict_prior_means(model, obs, act) if fastpath else
                model(obs, act, teacher_forcing=False)["prediction"])
    load_ms = (time.perf_counter() - load_start) * 1000
    dataset = WorldDynamicsDatasetAdapter(dataset_path)
    states, actions, _ = dataset.arrays("test")
    observations = states[:1].cuda()
    actions = actions[:1].cuda()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        original = model(observations, actions, teacher_forcing=False)["prediction"]
        candidate = predict(observations, actions)
        path_error = (candidate - original).abs().max().item()
        if path_error > 1e-6:
            raise ValueError(f"optimized RSSM path differs from original: {path_error}")
        warm_start = time.perf_counter()
        for _ in range(warmup):
            expected = predict(observations, actions)
        torch.cuda.synchronize()
        warmup_ms = (time.perf_counter() - warm_start) * 1000
        static_observations = torch.empty_like(observations)
        static_actions = torch.empty_like(actions)
        static_observations.copy_(observations)
        static_actions.copy_(actions)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                predict(static_observations, static_actions)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        graph_status = "measured"
        graph_error = None
        capture_ms = None
        try:
            start = time.perf_counter()
            with torch.cuda.graph(graph):
                static_prediction = predict(static_observations, static_actions)
            torch.cuda.synchronize()
            capture_ms = (time.perf_counter() - start) * 1000
            graph.replay()
            torch.cuda.synchronize()
            graph_error = (static_prediction - expected).abs().max().item()
            alternate = actions + 0.1
            alternate_expected = predict(observations, alternate)
            static_actions.copy_(alternate)
            graph.replay()
            torch.cuda.synchronize()
            graph_error = max(graph_error,
                              (static_prediction - alternate_expected).abs().max().item())
            if graph_error > 1e-5 or torch.allclose(alternate_expected, expected):
                raise ValueError(f"CUDA Graph output mismatch: {graph_error}")
            static_actions.copy_(actions)
            graph.replay()
            torch.cuda.synchronize()
        except (RuntimeError, ValueError) as exc:
            # A failed capture can poison this CUDA context; exit before more kernels.
            blocked = {"status": "blocked", "family": "WorldDynamics RSSM",
                       "checkpoint_sha256": adapter.weight_sha256,
                       "dataset_manifest_sha256": dataset.sha256,
                       "graph_status": "blocked",
                       "fastpath": fastpath, "fastpath_max_error": path_error,
                       "graph_error": f"{type(exc).__name__}: {exc}",
                       "model_load_ms": load_ms, "warmup_ms": warmup_ms,
                       "input_shape": list(observations.shape),
                       "action_shape": list(actions.shape)}
            (output / "report.json").write_text(json.dumps(blocked, indent=2), encoding="utf-8")
            return blocked
        rows = []
        for index in range(runs):
            methods = ["eager", "graph"] if graph_status == "measured" else ["eager"]
            if index % 2:
                methods.reverse()
            for method in methods:
                torch.cuda.synchronize()
                start = time.perf_counter()
                if method == "eager":
                    predict(observations, actions)
                else:
                    static_observations.copy_(observations)
                    static_actions.copy_(actions)
                    graph.replay()
                torch.cuda.synchronize()
                rows.append({"run": index, "method": method,
                             "wall_ms": (time.perf_counter() - start) * 1000})
    eager = [r["wall_ms"] for r in rows if r["method"] == "eager"]
    graphed = [r["wall_ms"] for r in rows if r["method"] == "graph"]
    report = {"status": "measured", "family": "WorldDynamics RSSM",
              "fastpath": fastpath, "fastpath_max_error": path_error,
              "weights": weights, "checkpoint_sha256": adapter.weight_sha256,
              "source_commit": adapter.source_commit,
              "source_sha256": adapter.source_sha256,
              "dataset_manifest_sha256": dataset.sha256,
              "input_shape": list(observations.shape), "action_shape": list(actions.shape),
              "precision": "fp32", "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "runs": runs, "warmup": warmup, "model_load_ms": load_ms,
              "warmup_ms": warmup_ms, "capture_ms": capture_ms,
              "graph_status": graph_status, "graph_error": graph_error,
              "eager_p50_ms": statistics.median(eager),
              "eager_p95_ms": _p(eager, 0.95), "eager_p99_ms": _p(eager, 0.99),
              "graph_p50_ms": statistics.median(graphed) if graphed else None,
              "graph_p95_ms": _p(graphed, 0.95) if graphed else None,
              "graph_p99_ms": _p(graphed, 0.99) if graphed else None,
              "speedup_p50": statistics.median(eager) / statistics.median(graphed)
              if graphed else None,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "method": "paired alternating synchronized wall clock; graph includes input copy"}
    with (output / "raw.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "method", "wall_ms"])
        writer.writeheader()
        writer.writerows(rows)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def benchmark_rssm_eager(repository: str, weights: str, dataset_path: str,
                         output_dir: str, runs: int = 100, warmup: int = 10) -> dict:
    if not torch.cuda.is_available() or runs < 20:
        raise ValueError("CUDA and at least 20 runs required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    load_start = time.perf_counter()
    adapter = RSSMAdapter(repository, weights, "cuda")
    load_ms = (time.perf_counter() - load_start) * 1000
    dataset = WorldDynamicsDatasetAdapter(dataset_path)
    states, actions, _ = dataset.arrays("test")
    observations, controls = states[:1].cuda(), actions[:1].cuda()
    model = adapter.model
    with torch.inference_mode():
        warm_start = time.perf_counter()
        for _ in range(warmup):
            expected = model(observations, controls, teacher_forcing=False)["prediction"]
        torch.cuda.synchronize()
        warmup_ms = (time.perf_counter() - warm_start) * 1000
        rows = []
        for index in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            actual = model(observations, controls, teacher_forcing=False)["prediction"]
            torch.cuda.synchronize()
            rows.append({"run": index, "wall_ms": (time.perf_counter() - start) * 1000})
    error = (actual - expected).abs().max().item()
    if error > 1e-6:
        raise ValueError("eager RSSM output changed during benchmark")
    times = [row["wall_ms"] for row in rows]
    report = {"status": "measured", "family": "WorldDynamics RSSM",
              "method": "eager synchronized wall clock", "device": torch.cuda.get_device_name(0),
              "checkpoint_sha256": adapter.weight_sha256,
              "dataset_manifest_sha256": dataset.sha256,
              "input_shape": list(observations.shape), "action_shape": list(controls.shape),
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "runs": runs, "warmup": warmup, "model_load_ms": load_ms,
              "warmup_ms": warmup_ms, "p50_ms": statistics.median(times),
              "p95_ms": _p(times, 0.95), "p99_ms": _p(times, 0.99),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "max_output_difference": error}
    with (output / "raw.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "wall_ms"])
        writer.writeheader()
        writer.writerows(rows)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
