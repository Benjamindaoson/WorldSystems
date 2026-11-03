# Benchmark protocol

The current inference benchmark loads a verified checkpoint, uses one deterministic synthetic test input, performs five warmups and thirty timed eager calls, synchronizes CUDA if selected, checks output equivalence, and writes `benchmark.json` with p50/p95/p99 latency and test MSE. Times include Python dispatch and model execution; they are not GPU kernel timings or client-observed service latency.

No optimization speedup is claimed. Future comparisons must match checkpoint, input, precision, hardware, generation length, warmup count and quality tolerance, and report startup/capture cost separately.
