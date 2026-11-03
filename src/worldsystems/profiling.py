"""PyTorch operator and CUDA trace for the verified reference checkpoint."""

import json
from pathlib import Path

import torch

from .config import Config
from .data import SyntheticWorld
from .runtime import load_model


def profile_inference(cfg: Config, checkpoint: str, output_dir: str) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    model = load_model(cfg, checkpoint)
    sample = SyntheticWorld(1, cfg.sequence_length, cfg.image_size,
                            cfg.action_dim, cfg.seed, namespace="test")[0]
    video = sample["video"].unsqueeze(0).to(cfg.device)
    actions = sample["actions"].unsqueeze(0).to(cfg.device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if cfg.device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.inference_mode():
        for _ in range(3):
            model(video, actions)
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        with torch.profiler.profile(activities=activities, record_shapes=True,
                                    profile_memory=True) as profiler:
            for _ in range(10):
                model(video, actions)
            if cfg.device == "cuda":
                torch.cuda.synchronize()
    profiler.export_chrome_trace(str(output / "trace.json"))
    sort_key = "self_cuda_time_total" if cfg.device == "cuda" else "self_cpu_time_total"
    summary = profiler.key_averages().table(sort_by=sort_key, row_limit=20)
    (output / "operators.txt").write_text(summary, encoding="utf-8")
    result = {"status": "measured", "device": cfg.device, "checkpoint": checkpoint,
              "activities": [str(activity) for activity in activities],
              "profiled_calls": 10, "trace": str(output / "trace.json"),
              "operator_summary": str(output / "operators.txt")}
    (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
