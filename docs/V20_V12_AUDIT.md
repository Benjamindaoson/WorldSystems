# V1.2 audit — TESTED

Base commit: `5c231a4`. Command: `.venv-cuda/Scripts/python.exe scripts/audit_v12_evidence.py`.
All 39 files listed by the historical manifest matched SHA256; receipt: `results/v20/v12-integrity.json`.
Reviewed training, adapter, Graph and service code against the reports. Training performs loss/backward/AdamW, uses hash-verified frozen MuJoCo splits and serializes optimizer/scheduler/RNG/cursor. The resumed V1.2 throughput caveat remains valid; historical data are unchanged. State training metrics and video smoke are distinct workloads.

Found and fixed: RSSM safetensors inference previously did not verify its weight manifest and loaded current external HEAD. WorldDynamics has advanced since V1.2; current file hash differs even though the factory change does not alter this RSSM. The adapter now verifies local weights and executes Git source pinned by the architecture checkpoint revision and expected source hash. External best.pt also loads its recorded revision. No external file is modified.

Found and fixed: synchronous FastAPI unload/reload handlers could race asynchronous admission in another thread. They now run on the event loop. Body size is limited while consuming chunks, before assembling the entire body. Empty batch/horizon is rejected. V1.2's 11/12 rejections reflect a one-active-request gate, not high-throughput serving; no queue is implemented yet. Existing model-failure injection test does run despite V1.2 documentation saying NOT_RUN; actual GPU OOM injection remains pending.

Graph failure was caused by distribution value validation, detailed in V20_CUDA_GRAPH_AUDIT.md. V1.2 fastpath omitted unused scale arithmetic and only compared eager-fastpath to graph-fastpath. It did not prove full-model or HTTP speedup. Historical CPU timing called compute+dispatch before output CPU transfer, so it is not an isolated GPU compute metric.
