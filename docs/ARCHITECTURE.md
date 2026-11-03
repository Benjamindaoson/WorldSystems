# Architecture

`SyntheticWorld` produces deterministic RGB/action trajectories. `TinyWorld` encodes each frame and action, propagates causal state through a GRU, and predicts the next RGB frame. `engine.train` performs AdamW updates, validation, epoch-boundary checkpointing, and failure receipts. `runtime` loads verified weights for evaluation, inference and CPU latency measurement. `server` exposes loopback HTTP inference and bounded recurrent sessions.

The reference model establishes a software path only. No physical realism or external model compatibility is claimed.
