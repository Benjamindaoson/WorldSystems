"""Static-shape CUDA Graph experiment for the TinyWorld compute region."""

import json
import statistics
import time
from pathlib import Path

import torch

from .config import Config
from .data import SyntheticWorld, validate_batch
from .runtime import load_model


def compare_cuda_graph(cfg: Config, checkpoint: str, output_dir: str,
                       runs: int = 200) -> dict:
    if cfg.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA Graph comparison requires CUDA")
    if runs < 20:
        raise ValueError("at least 20 measurements required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    model = load_model(cfg, checkpoint)
    sample = SyntheticWorld(1, cfg.sequence_length, cfg.image_size,
                            cfg.action_dim, cfg.seed, namespace="test")[0]
    video = sample["video"].unsqueeze(0).cuda()
    actions = sample["actions"].unsqueeze(0).cuda()
    validate_batch({"video": video, "actions": actions, "target": video}, cfg.action_dim)
    static_video = torch.empty_like(video)
    static_actions = torch.empty_like(actions)
    static_video.copy_(video)
    static_actions.copy_(actions)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(side):
        for _ in range(10):
            model.forward_compute(static_video, static_actions)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        capture_start = time.perf_counter()
        with torch.inference_mode(), torch.cuda.graph(graph):
            static_output, _ = model.forward_compute(static_video, static_actions)
        torch.cuda.synchronize()
        capture_ms = (time.perf_counter() - capture_start) * 1000
    except Exception as exc:  # noqa: BLE001 - capture can fail in backend-specific ways
        result = {"status": "blocked", "reason": f"{type(exc).__name__}: {exc}",
                  "checkpoint": checkpoint, "shape": list(video.shape)}
        (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    with torch.inference_mode():
        eager_output, _ = model.forward_compute(video, actions)
        static_video.copy_(video)
        static_actions.copy_(actions)
        graph.replay()
        torch.cuda.synchronize()
        error = (static_output - eager_output).abs().max().item()
        alternate_actions = actions + 0.1
        alternate_expected, _ = model.forward_compute(video, alternate_actions)
        static_actions.copy_(alternate_actions)
        graph.replay()
        torch.cuda.synchronize()
        error = max(error, (static_output - alternate_expected).abs().max().item())
        if torch.allclose(alternate_expected, eager_output):
            raise ValueError("alternate action did not change the prediction")
        static_actions.copy_(actions)
        graph.replay()
        torch.cuda.synchronize()
        if error > 1e-5:
            raise ValueError(f"CUDA Graph numerical mismatch: {error}")
        timings = {key: [] for key in ("eager_compute_ms", "graph_compute_ms",
                                      "eager_full_ms", "graph_full_ms")}
        for index in range(runs):
            labels = list(timings)
            if index % 2:
                labels.reverse()
            for label in labels:
                torch.cuda.synchronize()
                start = time.perf_counter()
                if label == "eager_compute_ms":
                    model.forward_compute(video, actions)
                elif label == "graph_compute_ms":
                    static_video.copy_(video)
                    static_actions.copy_(actions)
                    graph.replay()
                elif label == "eager_full_ms":
                    model(video, actions)
                else:
                    validate_batch({"video": video, "actions": actions, "target": video},
                                   cfg.action_dim)
                    static_video.copy_(video)
                    static_actions.copy_(actions)
                    graph.replay()
                torch.cuda.synchronize()
                timings[label].append((time.perf_counter() - start) * 1000)
    medians = {key: statistics.median(values) for key, values in timings.items()}
    result = {"status": "measured", "scope": "validated TinyWorld compute region",
              "checkpoint": checkpoint, "shape": list(video.shape), "runs": runs,
              "capture_ms": capture_ms, "max_abs_error": error,
              "p50_ms": medians,
              "speedup_compute_region": medians["eager_compute_ms"] / medians["graph_compute_ms"],
              "speedup_full_model_call": medians["eager_full_ms"] / medians["graph_full_ms"],
              "measurement": "alternating paired synchronized wall clock; graph includes input copy"}
    (output / "raw.json").write_text(json.dumps(timings), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
