# GPU environment

Captured on 2026-11-03 in `results/v11/environment.json`: Windows 11, Python 3.12.10, NVIDIA GeForce GTX 1650 (4,096 MiB), driver 457.49. The untouched CPU `.venv` uses PyTorch 2.15.0+cpu. Project-local `.venv-cuda` uses PyTorch 2.5.1+cu118 and reports CUDA available, one device, compute capability 7.5. `torch.cuda.is_bf16_supported(including_emulation=False)` is false, so native BF16 training is blocked; FP16 uses GradScaler.

The [PyTorch Windows install selector](https://pytorch.org/get-started/locally/) provides CUDA 11.8 binaries. [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) lists the CUDA 11.x driver family minimum; compatibility was confirmed here by actual CUDA execution. No driver or global environment was changed.

FP32 CUDA training: 2 epochs, 32 optimizer steps, 16 test samples, MSE `0.05141558`; second-epoch measured training throughput `660.41 sequences/s`, peak tensor allocation `25,995,776` bytes. FP16: 32 optimizer steps, test MSE `0.05141565`, second-epoch throughput `467.06 sequences/s`, peak tensor allocation `25,937,408` bytes. The tiny synthetic workload and first-epoch initialization effects make these unsuitable as general precision performance conclusions. GPU context and allocator reserved memory exceed `max_memory_allocated`.
