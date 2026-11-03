# Real world-model GPU training

Status: **GPU_VERIFIED** for random-initialized RSSM and ActionPredictor. GTX 1650 4 GB, driver 457.49, PyTorch 2.5.1+cu118, FP32, batch 7, seed 17, frozen MuJoCo 140/42/42 trajectories, horizon 32. `real-train` uses deterministic epoch permutations, timed synchronized forward/backward/AdamW phases, atomic safetensors and SHA-verified optimizer/scheduler/RNG/cursor checkpoints. It reloads the final weights for independent test evaluation.

| Run | Parameters | Updates | Wall | Peak allocated | Test free-running MSE | Checkpoint SHA256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RSSM | 10,117 | 40 | 11.82 s | 19,081,728 B | 0.00918270 | `da5238546a46e6e763efd8c07fef0fcbb6c55fb6461681407ee3e5d5ed348f57` |
| ActionPredictor | 10,275 | 40 | see raw report | see raw report | 0.128748 | `f26a08a704c59cb8a551fa4a7222526cb055dd9f68aca23a56fc92ad825710d0` |

RSSM 7-step process exit followed by a new-process resume produced **bit-identical** final weights to uninterrupted training (maximum parameter difference 0.0; same final SHA256). Optimizer, scheduler, CUDA RNG, and sampler cursor are restored by code and checked by final equality. The original resumed raw report's throughput divides full-run sample count by continuation-only time and is invalid; code now reports `processed_sequences_this_process` for future runs. Do not cite that original throughput.

VideoFlow: **GPU_VERIFIED smoke only**. Two FP32 updates, batch 1, 32×32 RGB, horizon 4, 162,979 parameters on a hash-verified frozen train trajectory; peak allocated 20,427,264 B. The first forward includes CUDA initialization and is not a steady-state throughput number. No held-out video-quality result or resumable VideoFlow run exists. Raw: `results/v12/video-flow-gpu-smoke.json`.
