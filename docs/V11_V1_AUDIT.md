# V1 audit and V1.1 corrections

Audited `README.md`, V1 status, all source modules, four V1 tests, and preserved artifacts under `artifacts/tiny_world` on 2026-11-03. Original V1 commit: `180db18fc2579e04359c91c7406ede0d51b5ca98`.

| Question | Finding | V1.1 action |
| --- | --- | --- |
| Split isolation | V1 used numeric seed offsets; counts above 100,000 could overlap split seed ranges. | Hash `(seed, split, index)` into per-sample seeds. Test split uniqueness. |
| Target leakage | Each target is the next generated frame, not an input at the same step. Validation and test episode indices differ from training. | Kept temporal target contract. |
| MSE | Validation/test use squared-error sum divided by all target elements. Train averages batch mean weighted by batch sample count; shapes are constant, so this equals element MSE. | Preserved. |
| Updates | V1 backward, clipped gradient, AdamW step verified by test. | Added cross-process resume check. |
| Resume | V1 stored optimizer, scheduler and torch RNG at epoch boundaries only; no batch cursor or FP16 GradScaler state. | Added next-batch cursor at optimizer boundaries, partial epoch aggregates, GradScaler state, CUDA RNG CPU conversion. |
| Failure receipt | V1 used one `failure.json`, so later failures could overwrite it. | New failures use unique filenames; original receipt is preserved. |
| Streaming | Per-session GRU state isolated. V1 synchronous compute under `asyncio.timeout` could finish after deadline and commit state. | Split preparation from version-checked commit; timed-out background work never commits. Busy worker rejects new requests. |
| CPU benchmark | Five warmups and 30 timed calls with a synchronized wall clock; too few for a stable P99, no raw samples or hardware manifest. | Added V1.1 raw 500-call benchmark and environment manifest. |

The V1 synthetic quality numbers remain historical software results. Changed split seeding means V1.1 results are a new dataset version and must not be compared as an optimization speedup.
