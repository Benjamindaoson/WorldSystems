# Implementation status

| Capability | State | Evidence / limit |
| --- | --- | --- |
| Tiny action-conditioned world model | Implemented | Causal GRU and frame decoder |
| CPU train → checkpoint → resume → infer → evaluate → benchmark | Tested | Synthetic software verification only; `pytest` checks exact resume equivalence |
| Local FastAPI service | Tested | Health, session creation, recurrent step, deletion, stale session rejection; loopback only |
| Recurrent streaming | Tested at runtime level | Per-session GRU state, reset and TTL; no video transport protocol |
| Single GPU CUDA | Not run | Installed PyTorch capability must be checked |
| DDP / FSDP2 / DeepSpeed | Not implemented | Multi-GPU Linux environment unavailable |
| CUDA Graph / compile / profiler | Not implemented | No validated GPU benchmark |
| WorldDynamics / WorldPolicy adapters | Not implemented | Versioned external checkpoint protocols needed |
| Physical world-model quality | Not evaluated | No real dataset or simulation benchmark |

Checkpoint restore is supported only at completed epoch boundaries. Checkpoint files are verified before loading. Optimizer state is loaded with PyTorch `weights_only=True` and should be treated as a local artifact, not an arbitrary remote model package.

## Verified local run

- Environment: Python 3.12.10, PyTorch 2.15.0+cpu, Windows, GTX 1650 physically present but CUDA unavailable to the installed PyTorch wheel.
- Training configuration: `configs/training/tiny_world.yaml`, synthetic-linear-v1, seed 17, 128 train / 32 validation / 32 test samples, 2 epochs.
- Epoch 1: training MSE 0.0514249, validation MSE 0.0510851, 32 optimizer steps total. Independent test MSE 0.0496494.
- CPU inference benchmark: 30 measured runs after 5 warmups, synchronized wall clock, p50 0.4437 ms, p95 0.5634 ms, p99 0.5737 ms. This is a local software measurement, not a GPU or production latency result.
- Tests: 4 passed. Ruff check passed. An initial run failed to save weights because NumPy was absent; the failure receipt remains in ignored `artifacts/tiny_world/failure.json`.

## Next experiments

1. Install a compatible CUDA PyTorch build in an isolated environment and benchmark actual GTX 1650 capacity without interrupting other GPU users.
2. Implement DDP with a Linux CUDA multi-GPU runner and rank-safe distributed checkpointing; verify scaling before reporting efficiency.
3. Add validated FSDP2 and DeepSpeed backends based on the actual installed PyTorch and DeepSpeed APIs.
4. Measure eager versus compile versus CUDA Graph with identical weights, input, precision and output tolerance.
5. Integrate externally versioned WorldDynamics and WorldPolicy model exports after their manifests and checkpoint formats are available.
