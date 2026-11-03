# Inference optimization

CUDA eager inference and `torch.profiler` were measured on the verified TinyWorld FP32 checkpoint. `results/v11/cuda-profile/trace.json` and `operators.txt` contain ten profiled CUDA calls. The top operator table includes input validation (`isfinite`, `all`, `item`) and cuDNN GRU, motivating a capture-safe compute region with boundary validation.

`TinyWorld.forward_compute` performs exactly the same frame/action encoding, GRU and decoding as `forward`, after validation. A fixed `[1,6,3,8,8]` input was captured with a side-stream warmup and stable input/output buffers. Replay was checked against eager for the original and changed action input, with maximum absolute output difference 0.0. Graph capture is restricted to that shape and the loaded weights; dynamic shapes and other model families are unsupported.

The committed-code paired 200-run comparison at `results/v11/cuda-graph-408c728/report.json` measured median compute-region eager `0.5754 ms` versus graph `0.1480 ms` (3.89×). Including the same input validation on both paths and graph input copy, median model-call eager `1.1684 ms` versus graph `0.5969 ms` (1.96×). Capture cost was `39.58 ms`. This is a local TinyWorld measurement, **not** client-observed serving latency or a WorldDynamics/Video DiT result. Earlier comparison runs remain preserved. `torch.compile` comparison is NOT_RUN.
