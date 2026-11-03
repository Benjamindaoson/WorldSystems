# Extended physical training — GPU_VERIFIED

Command: `.venv-cuda/Scripts/python.exe scripts/train_v20.py`.
Independent random initialization, seed17, AdamW lr0.001, FP32, batch7, ten epochs/200 updates per model on frozen 140 train trajectories. Validation and test each contain 42 trajectories, horizon32. Each epoch records validation and a hashed checkpoint; final weights are reloaded for independent test.

RSSM test free MSE 0.0083051734 (teacher-forced 0.0078870654), wall41.19s; checkpoint SHA `18981086a7201470b85162fd7976c504b19d316d3829195e434053c3aed0da3b`. ActionPredictor test free MSE0.0037172048, one-step8.48441e-06, final horizon0.01197363, wall9.69s; checkpoint SHA `e33e49318130acc36abe3c3381e6a28e8d86b5220fe11877c5ed73987f9aa5dd`. Reports, ten-point validation curves and optimizer checkpoints: `results/v20/training-01/{rssm,action_predictor}`.

Both use the same data, seed, batch and update budget, but distinct architectures, objectives and initialization streams; this is not a multiseed architecture ranking. RSSM source is pinned. ActionPredictor report binds the source file hash observed at loading; external repository commits changed during this work, so further adapters need equivalent revision pinning. Training wall includes validation/checkpoint/test; it is not pure GPU compute time. Formal video training, video recovery and precision/scaling matrices remain NOT_RUN.
