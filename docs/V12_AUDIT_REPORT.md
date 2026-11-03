# V1.2 V1.1 audit report

Status: **TESTED** on 2026-11-03. See also `V12_V11_AUDIT.md`.

V1.1 source and raw reports remain unchanged. The V1.1 audit, GPU environment, benchmark, inference, reliability, distributed, and WorldDynamics reports were checked against code and local artifacts. V1.1's two-process Gloo result proves CPU parameter synchronization only; the 1.96× CUDA Graph result is for TinyWorld, not RSSM or HTTP. V1.1 mid-epoch recovery compares exact weights, optimizer, scheduler, and RNG; V1.2 repeats a separate real RSSM restart.

WorldDynamics `models.py` at HEAD `569e6ae0dee6e6e0e4f09a89e19408835cd849b4` has SHA256 `e05ebd9ed666d97527a3777361d7f5d6c8eaca75b72569e41c97add4a6b17679`, identical to the model file at checkpoint revision `815387b5cbb0fcc39a258cafd39550d42b777f79`. The prior-before-future fix entered at `f6cb467`; current code predicts from the prior before seeing the next observation. V1.2 additionally perturbs all future observations by +100 and observes zero change to free-running RSSM output.

Frozen dataset manifest SHA256: `83576fe2cf07df02f7df318f4d9fb5089e2f8f17270d0b50e121d16ba01b874d`; the MuJoCo record split is 140 train, 42 validation, 42 test trajectories, with scene/physics isolation and horizon 32. ActionPredictor external seed 17 `best.pt` uses the same frozen test split, MSE over state elements, and 32-step free rollout as the WorldDynamics test report. Its V1.1 external free-running MSE `0.0033115` is a **single seed, one checkpoint**; it cannot be compared as a multiseed statistic or to another checkpoint. RSSM `best.pt` V1.2 free MSE `0.00820792` is not the WorldDynamics `last.pt` test MSE `0.00822510`. Both raw reports name their checkpoint and protocol.

V1.1 CPU/GPU training parameters and hashes are recorded in `results/v11` manifests and `docs/V11_*`. Historical `results/v11` and `artifacts/v11-*` have not been overwritten. V1.2 raw evidence lives only under `results/v12` and is ignored by Git.
