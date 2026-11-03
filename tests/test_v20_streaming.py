from pathlib import Path

import pytest
import torch

from worldsystems.rssm_adapter import RSSMAdapter
from worldsystems.rssm_streaming import RSSMStreaming


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_recurrent_sessions(device):
    weights = Path("results/v12/rssm-gpu-s17/epoch-0001/weights.safetensors")
    if not weights.exists() or (device == "cuda" and not torch.cuda.is_available()):
        pytest.skip("local checkpoint/device unavailable")
    model = RSSMAdapter("../WorldDynamics", weights, device).model
    runtime = RSSMStreaming(model, max_sessions=2)
    x, u = torch.randn(1, 9, 3, device=device), torch.randn(1, 8, 3, device=device)
    first, second = runtime.start(x[:, 0]), runtime.start(x[:, 0])
    with pytest.raises(RuntimeError, match="capacity"):
        runtime.start(x[:, 0])
    output = torch.cat([runtime.predict(first, u[:, :3]), runtime.predict(first, u[:, 3:])], 1)
    with torch.inference_mode():
        expected = model(x, u, teacher_forcing=False)["prediction"]
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(runtime.predict(second, u), expected, atol=1e-6, rtol=0)
    runtime.reset(first, x[:, 0])
    torch.testing.assert_close(runtime.predict(first, u), expected, atol=1e-6, rtol=0)
    runtime.close(first)
    with pytest.raises(KeyError):
        runtime.predict(first, u)
