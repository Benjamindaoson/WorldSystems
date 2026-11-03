# Resident recurrent sessions — CPU_VERIFIED / GPU_VERIFIED

`RSSMStreaming` retains h/z on the model device, admits at most32 sessions, expires idle state, accepts finite initial observation and bounded action chunks, resets/closes sessions and serializes model access. It checks the deadline after GPU completion before committing new recurrent state; expired work cannot advance a session. A returned chunk is produced after the requested steps finish; no video token/frame transport is implemented.

Command: `.venv-cuda/Scripts/python.exe -m pytest tests/test_v20_streaming.py -q`.
CPU and GPU tests compare 3+5-step chunks to the original eight-step free rollout, compare two isolated sessions, reset, capacity and closed-session rejection. Both passed. This is runtime correctness evidence, not HTTP streaming throughput. Observation-update assimilation, cancellation during execution, bounded waiting queue, dynamic batching, video streaming and policy chunk scheduling are still pending.
