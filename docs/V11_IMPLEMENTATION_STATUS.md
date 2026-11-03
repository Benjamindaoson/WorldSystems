# V1.1 implementation status

| Capability | Status | Evidence / limit |
| --- | --- | --- |
| Synthetic CPU train and cross-process mid-epoch resume | CPU_VERIFIED | Exact model, optimizer, scheduler and loss trajectory in test |
| FP32 / FP16 single-GPU training | GPU_VERIFIED | Real CUDA updates, hashes, load and independent evaluation |
| FP16 GradScaler and CUDA RNG recovery | GPU_VERIFIED | Step-5 restart matched uninterrupted final parameters exactly here |
| WorldDynamics ActionPredictor adapter and frozen MuJoCo test | CPU_VERIFIED | External pretrained weights and hashed dataset; 42 test trajectories |
| WorldSystems physical state-model training | CPU_VERIFIED | Random initialization, 50 updates, reloadable safetensors |
| Two-process CPU DDP | CPU_VERIFIED | Gloo, world size 2, rank-zero checkpoint |
| Multi-GPU DDP / FSDP2 / DeepSpeed | NOT_RUN / NOT_IMPLEMENTED | One local GPU; FSDP2 and DeepSpeed code absent |
| CUDA Graph TinyWorld compute region | GPU_VERIFIED | Paired 200-run comparison, original/altered input equivalence |
| CUDA profiler | GPU_VERIFIED | Operator summary and trace |
| torch.compile | NOT_RUN | No matched performance or quality comparison |
| WorldDynamics RSSM/video and WorldPolicy | NOT_IMPLEMENTED | No validated adapter yet |
| API concurrency and cancellation | TESTED_PARTIALLY | Versioned state commit and busy worker; no stress or external-client test |

L1: substantial requirements met, but serving stress and physical workload recovery remain open. L2: partial GPU and DDP evidence, no multi-GPU/FSDP2/DeepSpeed or external video-model optimization. L3: NOT_MET.

Verification on 2026-11-03: `.venv` CPU suite 9 passed, 2 skipped (CUDA and this wheel's Windows Gloo); `.venv-cuda` suite 11 passed, 0 skipped. Ruff passed. The test client emitted one upstream Starlette deprecation warning in each suite; no test failed. Historical V1 run artifacts and the missing-NumPy failure receipt remain intact.
