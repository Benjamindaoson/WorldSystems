import json
import statistics
import time
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .config import Config
from .data import SyntheticWorld
from .engine import build_model, evaluate


def load_model(cfg: Config, checkpoint: str):
    model = build_model(cfg).to(cfg.device)
    load_checkpoint(checkpoint, model, device=cfg.device)
    model.eval()
    return model


def infer(cfg: Config, checkpoint: str) -> dict:
    model = load_model(cfg, checkpoint)
    sample = SyntheticWorld(1, cfg.sequence_length, cfg.image_size,
                            cfg.action_dim, cfg.seed, namespace="test")[0]
    with torch.inference_mode():
        prediction, _ = model(sample["video"].unsqueeze(0).to(cfg.device),
                              sample["actions"].unsqueeze(0).to(cfg.device))
    return {"shape": list(prediction.shape), "mean": prediction.mean().item(),
            "checkpoint": checkpoint}


def benchmark(cfg: Config, checkpoint: str, runs: int = 30, warmup: int = 5) -> dict:
    model = load_model(cfg, checkpoint)
    sample = SyntheticWorld(1, cfg.sequence_length, cfg.image_size,
                            cfg.action_dim, cfg.seed, namespace="test")[0]
    video = sample["video"].unsqueeze(0).to(cfg.device)
    actions = sample["actions"].unsqueeze(0).to(cfg.device)
    with torch.inference_mode():
        for _ in range(warmup):
            baseline, _ = model(video, actions)
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            output, _ = model(video, actions)
            if cfg.device == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    if not torch.allclose(output, baseline, atol=1e-6, rtol=1e-6):
        raise ValueError("inference output changed during benchmark")
    values = sorted(timings)
    percentile = lambda p: values[min(len(values) - 1, int((len(values) - 1) * p))]
    result = {"status": "measured", "device": cfg.device, "precision": cfg.precision,
              "checkpoint": checkpoint, "torch_version": torch.__version__,
              "method": "synchronized wall-clock inference", "runs": runs,
              "warmup": warmup, "p50_ms": statistics.median(values),
              "p95_ms": percentile(0.95), "p99_ms": percentile(0.99),
              "output_equivalent": True,
              "test_mse": evaluate(model, cfg, "test")["mse"]}
    out = Path(cfg.checkpoint_dir) / "benchmark.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
