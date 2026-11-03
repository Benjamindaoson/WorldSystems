# V1.2 implementation status

| Capability | Status | Evidence / limit |
| --- | --- | --- |
| V1.1 provenance and comparability | TESTED | `V12_AUDIT_REPORT.md` |
| External RSSM causal evaluation | CPU_VERIFIED | `results/v12/rssm-external-s17-test.json` |
| RSSM and ActionPredictor GPU training | GPU_VERIFIED | `results/v12/*-gpu-s17/report.json` |
| Real RSSM mid-epoch restart | GPU_VERIFIED | identical final SHA and parameters; original resumed throughput field invalid |
| VideoFlow short training | GPU_VERIFIED smoke | two updates, no independent quality |
| RSSM fastpath CUDA Graph | GPU_VERIFIED | paired 100-call benchmark; original full forward Graph blocked |
| RSSM `torch.compile` Inductor | BLOCKED | Triton absent on this Windows CUDA environment |
| TinyWorld HTTP client/concurrency/restart | CPU_VERIFIED | `results/v12/http-tiny-cpu.json` |
| Real RSSM HTTP serving | CPU_VERIFIED | actual loopback client, unload/reload; `results/v12/http-rssm-cpu.json` |
| WorldPolicy ACT inference | CPU_VERIFIED | `results/v12/policy-act-cpu.json` |
| CPU Gloo DDP | CPU_VERIFIED (V1.1) | no new V1.2 parity run |
| FSDP2, DeepSpeed, multi-GPU scaling | NOT_IMPLEMENTED / BLOCKED | one local GPU |
| Long-sequence/width scaling matrix | NOT_RUN | no quality-controlled matrix |

L1: **partially met**; actual real-model GPU updates, frozen evaluation, exact recovery and real-model CPU HTTP serving pass, but complete failure-injection and GPU deployment coverage remain open. L2: **not met**; no FSDP2 execution, multi-GPU scaling or controlled size matrix. L3: **not met**; no new independently validated method/ablation or research paper. All raw V1.2 results are local under ignored `results/v12`; code and docs are tracked. No remote push was performed.
