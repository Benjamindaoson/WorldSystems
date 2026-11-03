from pathlib import Path

import pytest
import torch

from worldsystems.graph_runtime import StaticGraph, capture_ready_model
from worldsystems.rssm_adapter import RSSMAdapter

WEIGHTS = Path("results/v12/rssm-gpu-s17/epoch-0001/weights.safetensors")


@pytest.mark.skipif(not WEIGHTS.exists(), reason="local real RSSM checkpoint unavailable")
def test_source_pinning_and_full_arithmetic():
    adapter = RSSMAdapter("../WorldDynamics", WEIGHTS)
    assert adapter.source_sha256 == "e05ebd9ed666d97527a3777361d7f5d6c8eaca75b72569e41c97add4a6b17679"
    original = adapter.model
    adapted = capture_ready_model(original)
    x, u = torch.randn(2, 9, 3), torch.randn(2, 8, 3)
    with torch.inference_mode():
        for teacher_forcing in [False, True]:
            reference = original(x, u, teacher_forcing=teacher_forcing)
            actual = adapted(x, u, teacher_forcing=teacher_forcing)
            for name in reference:
                torch.testing.assert_close(reference[name], actual[name], atol=0, rtol=0)
    original.train()
    with pytest.raises(ValueError, match="eval"):
        capture_ready_model(original)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available() or not WEIGHTS.exists(),
                    reason="CUDA and local RSSM checkpoint required")
def test_full_graph_replay_mutation_and_output_ownership():
    original = RSSMAdapter("../WorldDynamics", WEIGHTS, "cuda").model
    adapted = capture_ready_model(original)
    def forward(x, u):
        return adapted(x, u, teacher_forcing=False)["prediction"]
    with torch.inference_mode():
        x, u = torch.randn(2, 9, 3, device="cuda"), torch.randn(2, 8, 3, device="cuda")
        graph = StaticGraph(forward, x, u)
        first = graph(x, u)
        saved = first.clone()
        for seed in [1, 2, 3]:
            torch.manual_seed(seed)
            x.normal_()
            u.normal_()
            expected = original(x, u, teacher_forcing=False)["prediction"]
            torch.testing.assert_close(graph(x, u), expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(first, saved, atol=0, rtol=0)
        with pytest.raises(ValueError, match="shape"):
            graph(x[:, :2], u[:, :1])
        with pytest.raises(ValueError, match="nonfinite"):
            graph(x * float("nan"), u)
