"""Small MuJoCo state-prediction workload using frozen WorldDynamics trajectories."""

import json
import math
import os
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from .worlddynamics import WorldDynamicsDatasetAdapter, WorldDynamicsModelAdapter


def train_physical(repository: str, dataset_path: str, architecture_weights: str,
                   output_dir: str, epochs: int = 5, seed: int = 17) -> dict:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    dataset = WorldDynamicsDatasetAdapter(dataset_path)
    adapter = WorldDynamicsModelAdapter(repository, architecture_weights)
    torch.manual_seed(seed)
    for module in adapter.model.modules():
        if isinstance(module, nn.Linear):
            module.reset_parameters()
    model = adapter.model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_states, train_actions, train_rows = dataset.arrays("train")
    val_states, val_actions, val_rows = dataset.arrays("val")
    generator = torch.Generator().manual_seed(seed)
    begin = time.perf_counter()
    history = []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(train_states), generator=generator)
        total = 0.0
        for indices in permutation.split(14):
            states = train_states[indices]
            actions = train_actions[indices]
            loss = model.loss(states, actions)
            if not torch.isfinite(loss):
                raise FloatingPointError("physical training loss is nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(indices)
        model.eval()
        with torch.inference_mode():
            val_mse = torch.mean((model(val_states, val_actions)["prediction"]
                                  - val_states[:, 1:]).square()).item()
        history.append({"epoch": epoch, "train_mse": total / len(train_states),
                        "val_mse": val_mse})
    elapsed = time.perf_counter() - begin
    weights = output / "weights.safetensors"
    temporary = output / "weights.tmp.safetensors"
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
              str(temporary))
    os.replace(temporary, weights)
    reloaded = WorldDynamicsModelAdapter(repository, weights)
    test = reloaded.evaluate(dataset, "test")
    report = {"workload": "mujoco-physical-state-prediction", "initialization": "random",
              "seed": seed, "epochs": epochs,
              "optimizer_steps": epochs * math.ceil(len(train_states) / 14),
              "train_samples": len(train_rows), "train_scenes": len({r["scene_id"] for r in train_rows}),
              "val_samples": len(val_rows), "test_samples": test["samples"],
              "physics_configurations": dataset.manifest["physics_by_split"],
              "dataset_manifest_sha256": dataset.sha256,
              "dataset_source_commit": dataset.manifest["source_revision"],
              "source_commit": adapter.source_commit, "source_sha256": adapter.source_sha256,
              "architecture": "WorldDynamics ActionPredictor", "hidden_dim": adapter.hidden,
              "parameter_count": test["parameter_count"], "checkpoint": str(weights),
              "weight_sha256": reloaded.weight_sha256,
              "training_seconds": elapsed, "history": history, "test": test}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
