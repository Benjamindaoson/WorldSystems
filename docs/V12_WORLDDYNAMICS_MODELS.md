# WorldDynamics model contracts

Status: **CPU_VERIFIED** for external RSSM and ActionPredictor inference; **GPU_VERIFIED** for training on their architectures. `RSSMAdapter` verifies source revision, model source SHA, checkpoint path and hash, state dimensions, frozen dataset hashes, then loads `best.pt` weights with `weights_only=True` or local safetensors. Input is `[B,T+1,3]` state and `[B,T,3]` action; output is `[B,T,3]` prior prediction. Teacher-forced evaluation is separate from free-running evaluation; free-running never reads observations after time zero. Its future-observation perturbation difference is exactly zero.

External seed-17 RSSM `best.pt` SHA256 `2d7ac257d7a35967368f495336bc7e6fc001867329e2ccb96ddaf5bc28ea47d9`: test teacher-forced MSE `0.00777498`, free-running MSE `0.00820792`, 42 trajectories, horizon 32. External ActionPredictor: one-step MSE `7.4994e-06`, free-running MSE `0.00331152`, final-horizon MSE `0.01115757`. Both reports include per-horizon curves in `results/v12/`.

VisualPredictor and JointWorldAction: **NOT_IMPLEMENTED** in WorldSystems. VideoFlow architecture: **GPU_VERIFIED smoke**, with no exported inference adapter or held-out quality evaluation. No missing checkpoint is represented as loaded. A unified reset/predict/rollout interface across every family is **NOT_IMPLEMENTED**; adapter-specific contracts above are the supported API.
