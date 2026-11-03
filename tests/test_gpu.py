import pytest
import torch

from worldsystems.checkpoint import verify_checkpoint
from worldsystems.config import Config
from worldsystems.cuda_graph import compare_cuda_graph
from worldsystems.engine import train
from worldsystems.runtime import load_model


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable in this environment")
def test_cuda_training_and_graph(tmp_path):
    cfg = Config(device="cuda", precision="fp32", epochs=1, train_samples=8,
                 val_samples=4, test_samples=4, batch_size=4,
                 checkpoint_dir=str(tmp_path / "train"))
    result = train(cfg)
    assert result["records"][0]["global_step"] == 2
    assert result["records"][0]["peak_memory_bytes"] > 0
    verify_checkpoint(result["checkpoint"])
    assert next(load_model(cfg, result["checkpoint"]).parameters()).device.type == "cuda"
    graph = compare_cuda_graph(cfg, result["checkpoint"], str(tmp_path / "graph"), runs=20)
    assert graph["status"] == "measured"
    assert graph["max_abs_error"] <= 1e-5
