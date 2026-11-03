# WorldDynamics integration

`WorldDynamicsDatasetAdapter` reads and verifies the frozen MuJoCo `worlddynamics-v11/v1` manifest and record hashes, RGB hashes, scene split isolation, initial-state identity and physics split isolation. It does not write to WorldDynamics. `WorldDynamicsModelAdapter` loads the tracked `iwm/models.py` implementation from the local trusted repository and accepts a `weights_only=True` ActionPredictor `best.pt` state dict or WorldSystems safetensors. It rejects altered model source, unsupported dimensions and arbitrary `.pt` paths.

External pretrained ActionPredictor: `artifacts/v11-baselines-fair/conditioned-s17/best.pt`, SHA256 `e1e99e2cc99d3be022213a708cba2efafb1a7180f08a709effcca335ca06a48c`; model source SHA256 `e05ebd9ed666d97527a3777361d7f5d6c8eaca75b72569e41c97add4a6b17679`. Dataset manifest SHA256 `83576fe2cf07df02f7df318f4d9fb5089e2f8f17270d0b50e121d16ba01b874d`, generated from WorldDynamics commit `f6cb467715ced5caa40617a220b998209cb06f21`.

Frozen test split: 6 scenes, 42 trajectories, 32-step horizon. Independently measured one-step MSE `7.4994e-06`, free-running MSE `0.0033115`, final-horizon MSE `0.0111576`. Raw report: `results/v11/worlddynamics-action-s17-test.json`.

WorldSystems also trained the same 10,275-parameter architecture **from random initialization** on 140 train trajectories for 5 epochs / 50 updates. Frozen validation: 42 trajectories; frozen test: 42. Test one-step MSE `5.3411e-05`, free-running MSE `0.0270796`, final-horizon MSE `0.0925352`. The test physics configurations differ from train; the dataset is still small and this run is not a strong physical generalization result. The checkpoint reloads from safetensors. Raw report: `results/v11/physical-random-s17/report.json`.

RSSM, video flow and visual models are not integrated. The full WorldDynamics `last.pt` contains NumPy objects that `weights_only=True` rejects; this adapter deliberately uses the safe weights-only export.

Reproduction from `.`:

```powershell
.\.venv\Scripts\python.exe -m worldsystems.cli external-evaluate --repository ..\WorldDynamics --dataset ..\WorldDynamics\artifacts\v11-mujoco-32scene-h32 --weights ..\WorldDynamics\artifacts\v11-baselines-fair\conditioned-s17\best.pt --output results/v11/recheck-external.json
.\.venv\Scripts\python.exe -m worldsystems.cli physical-train --repository ..\WorldDynamics --dataset ..\WorldDynamics\artifacts\v11-mujoco-32scene-h32 --architecture-weights ..\WorldDynamics\artifacts\v11-baselines-fair\conditioned-s17\best.pt --output-dir results/v11/recheck-physical --epochs 5 --seed 17
```

Use a new output filename or directory for each run; commands refuse to overwrite existing reports.
