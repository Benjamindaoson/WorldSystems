# V1.1 benchmark report

Raw data live in ignored `results/v11/` and were not copied from V1. The 500-run CUDA eager report at `cuda-eager-408c728/report.json` and `raw.csv` used 30 warmups, identical input and weights, synchronized wall clock, and CUDA events. P50/P95/P99 wall latency: `1.2207 / 1.8098 / 1.9782 ms`; P50 CUDA event span `1.1720 ms`; output difference 0; test MSE `0.05141558`. The event span may include host scheduling gaps and is not a pure kernel sum. Cold model loading was `1544.64 ms` in that run. The earlier 30-call V1-style result and first V1.1 event-allocation result are preserved and are not interchangeable with this protocol.

Graph comparison uses the same checkpoint, input, precision, GPU and validation behavior for each paired call. The trace and raw event data are stored beside the reports. Model-call latency is still below full HTTP request latency; client-observed API latency was NOT_MEASURED.

The cited benchmark JSON captures the tested code commit `408c728754efac3933cae1b37520a84856874ab4`. Earlier pre-commit raw results remain preserved but their HEAD field alone does not identify the uncommitted source used at the time.

`scripts/write_v11_manifests.py` generated `results/v11/tiny-model-manifest.json` and `physical-model-manifest.json` from verified files. The environment snapshot is `results/v11/environment.json`; source and checkpoint SHA256 values are in the model manifests. Failure analysis, including the original missing-NumPy event, the CUDA RNG restore error and Windows torchrun rendezvous failures, is in the V1.1 audit and recovery/distributed reports.
