from dataclasses import replace

import pytest
import torch
from fastapi.testclient import TestClient

from worldsystems.checkpoint import verify_checkpoint
from worldsystems.config import Config
from worldsystems.data import SyntheticWorld, validate_batch
from worldsystems.engine import build_model, evaluate, train
from worldsystems.runtime import benchmark, infer, load_model
from worldsystems.server import create_app
from worldsystems.streaming import StreamingRuntime


def test_contract_and_backward():
    cfg = Config()
    sample = SyntheticWorld(2, cfg.sequence_length, cfg.image_size, cfg.action_dim, cfg.seed)
    batch = {key: torch.stack([sample[0][key], sample[1][key]])
             for key in ("video", "actions", "target")}
    validate_batch(batch, cfg.action_dim)
    model = build_model(cfg)
    prediction, _ = model(batch["video"], batch["actions"])
    torch.nn.functional.mse_loss(prediction, batch["target"]).backward()
    assert model.decoder.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        validate_batch({**batch, "actions": batch["actions"][..., :2]}, cfg.action_dim)


def test_stream_sessions_are_isolated():
    cfg = Config()
    model = build_model(cfg)
    stream = StreamingRuntime(model)
    a, b = stream.start(), stream.start()
    frame = torch.rand(1, 1, 3, 8, 8)
    action = torch.zeros(1, 1, 3)
    first = stream.step(a, frame, action)
    torch.testing.assert_close(first, stream.step(b, frame, action))
    second = stream.step(a, frame, action)
    assert not torch.allclose(first, second)
    stream.reset(a)
    torch.testing.assert_close(first, stream.step(a, frame, action))
    stream.reset(a)
    preview, pending_state, version = stream.prepare_step(a, frame, action)
    torch.testing.assert_close(first, preview)
    stream.reset(a)
    with pytest.raises(RuntimeError, match="session changed"):
        stream.commit_step(a, pending_state, version)
    torch.testing.assert_close(first, stream.step(a, frame, action))
    stream.close(a)
    with pytest.raises(KeyError):
        stream.step(a, frame, action)


def test_full_pipeline_and_resume(tmp_path):
    base = Config(train_samples=16, val_samples=8, test_samples=8,
                  checkpoint_dir=str(tmp_path / "full"))
    full = train(base)
    initial = replace(base, checkpoint_dir=str(tmp_path / "resumed"))
    partial = train(initial, max_epochs=1)
    resumed = train(initial, resume=partial["checkpoint"])
    full_model = load_model(base, full["checkpoint"])
    resumed_model = load_model(initial, resumed["checkpoint"])
    for a, b in zip(full_model.parameters(), resumed_model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert evaluate(full_model, base)["mse"] < 1
    assert infer(base, full["checkpoint"])["shape"] == [1, 6, 3, 8, 8]
    assert benchmark(base, full["checkpoint"], runs=3, warmup=1)["output_equivalent"]
    weights = tmp_path / "full" / "epoch-0001" / "weights.safetensors"
    with weights.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        verify_checkpoint(full["checkpoint"])


def test_api(tmp_path):
    cfg = Config(train_samples=8, val_samples=4, test_samples=4, epochs=1,
                 checkpoint_dir=str(tmp_path / "api"))
    checkpoint = train(cfg)["checkpoint"]
    client = TestClient(create_app(cfg, checkpoint))
    assert client.get("/health").json()["status"] == "ready"
    session_id = client.post("/sessions").json()["session_id"]
    sample = SyntheticWorld(1, 6, 8, 3, 17)[0]
    frame = sample["video"][:1].unsqueeze(0).tolist()
    action = sample["actions"][:1].unsqueeze(0).tolist()
    response = client.post(f"/sessions/{session_id}/step",
                           json={"video": frame, "actions": action})
    assert response.status_code == 200
    assert response.json()["valid"]
    assert client.post(f"/sessions/{session_id}/step",
                       json={"video": frame, "actions": action,
                             "deadline_ms": 0}).status_code == 400
    assert client.delete(f"/sessions/{session_id}").status_code == 200
    assert client.post(f"/sessions/{session_id}/step",
                       json={"video": frame, "actions": action}).status_code == 404
