# WorldSystems — Scalable World Model Training and Real-Time Inference

Research engineering for training, recovering, evaluating and serving action-conditioned world models on a constrained GPU. **V2.0 development is in progress.** This is the first public research snapshot, not a completed production platform.

## Verified capabilities

| Capability | Evidence and boundary |
| --- | --- |
| RSSM GPU training | 200 updates on frozen MuJoCo trajectories; independent free-running test MSE 0.00830517 |
| ActionPredictor GPU training | 200 updates; test free-running MSE 0.00371720; different architecture and objective |
| Checkpoint save/resume | Atomic safetensors, SHA checks, optimizer/scheduler/RNG/cursor; V1.2 RSSM cross-process step-7 recovery matched uninterrupted final weights exactly |
| CUDA Graph | Full deterministic RSSM free rollout after moving distribution value checks outside capture; multi-shape and mutated-input equivalence tests |
| GPU streaming sessions | Resident recurrent state, isolated sessions, reset, TTL and deadline-checked commits; HTTP integration pending |
| HTTP serving | Loopback TinyWorld/RSSM endpoints, load/unload/reload, validation, deadline and single-worker backpressure; CPU HTTP client measurements |
| WorldDynamics adapter | Real RSSM/ActionPredictor weights and frozen data; RSSM source revision pinning and future-observation leakage tests |
| WorldPolicy adapter | Trusted ACT checkpoint loading and action-chunk inference; no motor control |
| Tests | Recorded CUDA: 21 passed. CPU: 17 passed, 4 skipped. External-artifact/device tests skip when unavailable |

Measured on GTX 1650 4GB, PyTorch 2.5.1+cu118, FP32. Batch1/horizon32 full RSSM Graph P50 was 2.265 ms. GPU exclusivity was not established, so this is exploratory model-call performance, not HTTP latency. Earlier RSSM CPU HTTP P50 was 9.801 ms over 30 requests; this sample count does not support a stable tail-latency claim. See [public evidence](public_results/v20-summary.json) and [Graph audit](docs/V20_CUDA_GRAPH_AUDIT.md).

## Standalone quick start

Python 3.12 is locally tested. Install the PyTorch wheel appropriate for your device. The dependency lower bounds do not mean all versions have been validated.

```bash
python -m venv .venv
# Activate the environment (PowerShell: .venv\Scripts\Activate.ps1).
python -m pip install -e ".[dev]"
python -m worldsystems.cli doctor
python -m worldsystems.cli train --config configs/training/tiny_world.yaml
python -m worldsystems.cli checkpoint verify --path artifacts/tiny_world/epoch-0001
python -m worldsystems.cli evaluate --config configs/training/tiny_world.yaml --checkpoint artifacts/tiny_world/epoch-0001
python -m pytest -q
```

The standalone model uses synthetic data for software verification. Use a fresh checkpoint directory for each new run. A clean checkout without external artifacts will skip integration tests and will not reproduce the full local pass count.

## Real world-model workflows

Place authorized WorldDynamics and WorldPolicy checkouts beside WorldSystems, or edit the example configuration paths. External source, datasets and checkpoints are **not bundled**. Obtain matching artifacts from their owners under their applicable terms. [Public evidence](public_results/v20-summary.json) preserves source, dataset and weight hashes; do not substitute artifacts when comparing reported metrics.

```bash
python -m worldsystems.cli real-train --config configs/training/rssm_cuda_v12.yaml
python -m worldsystems.cli real-train --config configs/training/rssm_cuda_resume_v12.yaml --max-steps 7
python -m worldsystems.cli real-train --config configs/training/rssm_cuda_resume_v12.yaml --resume results/v12/rssm-gpu-resume-s17/step-000007
python -m worldsystems.cli rssm-serve --repository ../WorldDynamics --weights results/v12/rssm-gpu-s17/epoch-0001/weights.safetensors --device cpu
```

RSSM HTTP defaults to `127.0.0.1:8001`; `/health`, `/predict`, `/unload` and `/reload` are available. There is one active worker: excess requests receive 429. Dynamic batching and a bounded waiting queue are not implemented. The ACT adapter uses a trusted local pickle checkpoint after checking its recorded hash; do not supply arbitrary downloaded pickle files.

`scripts/train_v20.py` runs the two 200-update experiments. `scripts/audit_graph_v20.py --output results/v20/new-graph-audit --runs 100` runs the Graph audit after the prerequisite V1.2 checkpoint is available. `scripts/compile_v20.py` executes actual Inductor probes; both failed locally because working Triton was unavailable.

## Status and provenance

- [Delivery report](docs/WORLDSYSTEMS_FINAL_DELIVERY_REPORT.md), [implementation status](docs/V20_IMPLEMENTATION_STATUS.md), [continuation state](execution/v20-state.json).
- [Training](docs/V20_WORLD_MODEL_TRAINING.md), [streaming](docs/V20_STREAMING_INFERENCE.md), [compiler diagnostics](docs/V20_TORCH_COMPILE.md), [audit](docs/V20_V12_AUDIT.md).
- This public snapshot derives from development commit `10aeddf`, with validated implementation `f849eae`. Clean initial public history excludes private author metadata; development history remains local.
- FSDP2, DeepSpeed, multi-GPU scaling, formal video training, dynamic batching and the model registry remain unfinished. CPU Gloo is functional validation, not multi-GPU scaling. L1 is partial; L2/L3 are not met. Standard CUDA Graph is not claimed as original research.
- Raw results, weights, datasets and GPU traces remain local. Published summaries include SHA256 associations without private machine paths.
- No explicit redistribution license is currently included. Public visibility does not grant an additional license to external code, models or datasets. No external source code or artifacts are redistributed here.
