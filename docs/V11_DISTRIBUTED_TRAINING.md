# Distributed status

`ddp_train` uses Gloo on CPU or NCCL on compatible Linux CUDA, a `DistributedSampler`, synchronized gradients, all-reduced training metrics, rank-zero checkpoint writing and process-group cleanup. `scripts/run_ddp_cpu.py` ran two Gloo ranks on this Windows host using a file rendezvous store. Both ranks reported epoch MSE `0.0662085` then `0.0625991`, and the rank-zero checkpoint verified. This proves a two-process CPU contract, not multi-GPU scaling.

The first `torchrun --standalone` attempt failed because this Windows PyTorch build requested libuv without support. A later `env://` TCPStore attempt failed with `unmatched '}' in format string`. The file-store runner succeeded. [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html) treats Windows support as prototype and excludes NCCL on Windows.

FSDP2 and DeepSpeed are NOT_IMPLEMENTED / NOT_RUN. Multi-GPU efficiency is NOT_AVAILABLE: the host has one GPU. FSDP2 needs an implementation using `fully_shard` and distributed checkpoint APIs, followed by a Linux multi-GPU verification run; configuration alone would not be evidence.
