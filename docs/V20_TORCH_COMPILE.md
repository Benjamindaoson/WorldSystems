# Real-model Inductor probes — BLOCKED

Command: `.venv-cuda/Scripts/python.exe scripts/compile_v20.py`.
Results: `results/v20/compile-01/report.json` and separate full stderr files for RSSM and ActionPredictor. Both executed the compiled callable and failed with `Cannot find a working triton installation`. No eager fallback or cache creation counts as compilation success. Environment: Windows, Python 3.12, PyTorch 2.5.1+cu118, GTX1650; exact capability and version snapshot is in the report. No global environment or driver was changed. A separately provisioned compatible Linux CUDA/Triton environment remains required for a measured Inductor comparison. Linux packaging and compile+Graph are NOT_RUN.
