# Checkpoint reliability

The V1.1 writer saves model weights in safetensors and optimizer, scheduler, GradScaler, CPU/CUDA RNG and progress state in a local `training.pt`. A manifest records SHA256 hashes. A sibling temporary directory is atomically renamed only after all files are complete. Resume verifies hashes and configuration before continuing.

`train --max-steps K` stops only after an optimizer step, stores `next_batch` and partial epoch loss totals, then returns a step checkpoint. The per-epoch shuffle is reconstructed from its epoch seed; the loader skips already consumed batches. This is deterministic at optimizer boundaries and is not arbitrary mid-batch recovery.

`tests/test_v11_reliability.py` launches uninterrupted and interrupted/resumed training in separate Python processes. It checks exact final parameters, optimizer tensors, scheduler state and loss trajectory on CPU. A separate FP16 CUDA run stopped after step 5 and resumed to step 32; its final parameters matched uninterrupted training with maximum absolute difference 0.0 in this local run. This does not establish bitwise recovery across hardware or driver versions.

Negative tests cover partial manifests, weight corruption and resume configuration mismatch. A CUDA checkpoint load initially failed because RNG tensors were mapped to CUDA; loading them back to CPU before `torch.cuda.set_rng_state_all` fixed it. Local optimizer state still uses PyTorch serialization with `weights_only=True`; do not load arbitrary external optimizer packages.

Historical V1 `synthetic-linear-v1` checkpoints used a different split seed rule. V1.1 refuses to resume those checkpoints under the new sampler, preserving their inference compatibility without presenting a mixed-sampler run as reproducible training.
