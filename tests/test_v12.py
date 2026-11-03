import json
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from worldsystems.config import Config
from worldsystems.engine import train
from worldsystems.policy_adapter import ACTPolicyAdapter
from worldsystems.real_training import verify_real_checkpoint
from worldsystems.rssm_adapter import RSSMAdapter
from worldsystems.rssm_fastpath import predict_prior_means
from worldsystems.rssm_server import create_rssm_app
from worldsystems.server import create_app
from worldsystems.worlddynamics import WorldDynamicsDatasetAdapter


@pytest.fixture
def serving(tmp_path):
    cfg = Config(train_samples=8, val_samples=2, test_samples=2, epochs=1,
                 image_size=4, hidden_dim=8, checkpoint_dir=str(tmp_path / "tiny"))
    checkpoint = train(cfg)["checkpoint"]
    return cfg, checkpoint, create_app(cfg, checkpoint)


def _request(cfg):
    return {"video": torch.zeros(1, cfg.sequence_length, 3, cfg.image_size,
                                 cfg.image_size).tolist(),
            "actions": torch.zeros(1, cfg.sequence_length, cfg.action_dim).tolist()}


def test_http_limits_sessions_and_recovery(serving):
    cfg, _, app = serving
    payload = _request(cfg)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ready"
        a = client.post("/sessions").json()["session_id"]
        b = client.post("/sessions").json()["session_id"]
        assert a != b
        assert client.post("/predict", json=payload).status_code == 200
        assert client.post("/predict", json={**payload, "session_id": "unknown"}).status_code == 404
        assert client.post("/predict", json={**payload, "deadline_ms": 0}).status_code == 400
        assert client.post("/predict", json={**payload, "video": [0] * 65}).status_code == 413
        assert client.post("/predict", content=b"x" * (256 * 1024 + 1)).status_code == 413
        step = {"video": torch.zeros(1, 1, 3, cfg.image_size, cfg.image_size).tolist(),
                "actions": torch.zeros(1, 1, cfg.action_dim).tolist()}
        assert client.post(f"/sessions/{a}/step", json=step).status_code == 200
        assert app.state.stream.sessions[a][2] == 1
        assert app.state.stream.sessions[b][2] == 0
        assert client.delete(f"/sessions/{a}").status_code == 200
        assert client.post(f"/sessions/{a}/step", json=step).status_code == 404
        assert client.post("/predict", json=payload).status_code == 200


def test_http_backpressure_timeout_and_failure(serving, monkeypatch):
    cfg, _, app = serving
    original = app.state.model.forward

    def slow(*args, **kwargs):
        time.sleep(0.06)
        return original(*args, **kwargs)

    monkeypatch.setattr(app.state.model, "forward", slow)
    with TestClient(app) as client:
        response = client.post("/predict", json={**_request(cfg), "deadline_ms": 5})
        assert response.status_code == 504
        assert client.post("/predict", json=_request(cfg)).status_code == 429
        time.sleep(0.1)
        assert client.post("/predict", json=_request(cfg)).status_code == 200
    monkeypatch.setattr(app.state.model, "forward", original)


@pytest.mark.skipif(not Path("../WorldDynamics/artifacts/v11-mujoco-32scene-h32").exists(),
                    reason="WorldDynamics frozen dataset unavailable")
def test_rssm_causal_fastpath_and_checkpoint():
    repo = Path("../WorldDynamics")
    dataset = WorldDynamicsDatasetAdapter(repo / "artifacts/v11-mujoco-32scene-h32")
    weights = repo / "artifacts/v11-baselines-815387b/rssm-s17/best.pt"
    adapter = RSSMAdapter(repo, weights)
    states, actions, _ = dataset.arrays("test")
    x, u = states[:2], actions[:2]
    with torch.inference_mode():
        baseline = adapter.model(x, u, teacher_forcing=False)["prediction"]
        fast = predict_prior_means(adapter.model, x, u)
        future_changed = x.clone()
        future_changed[:, 1:] += 100
        causal = predict_prior_means(adapter.model, future_changed, u)
    torch.testing.assert_close(fast, baseline, rtol=0, atol=0)
    torch.testing.assert_close(causal, baseline, rtol=0, atol=0)
    result = adapter.evaluate(dataset, "test")
    assert result["free_running_mse"] > result["teacher_forced_mse"]
    assert result["future_perturbation_max_diff"] == 0


@pytest.mark.skipif(not Path("../WorldDynamics/artifacts/v11-mujoco-32scene-h32").exists(),
                    reason="WorldDynamics frozen dataset unavailable")
def test_real_rssm_http_lifecycle(monkeypatch):
    repo = Path("../WorldDynamics")
    weights = repo / "artifacts/v11-baselines-815387b/rssm-s17/best.pt"
    app = create_rssm_app(str(repo), str(weights))
    payload = {"observations": torch.zeros(1, 5, 3).tolist(),
               "actions": torch.zeros(1, 4, 3).tolist()}
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ready"
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert torch.tensor(response.json()["prediction"]).shape == (1, 4, 3)
        assert client.post("/predict", json={**payload, "deadline_ms": 0}).status_code == 400
        assert client.post("/predict", json={**payload, "actions": []}).status_code == 422
        assert client.post("/unload").status_code == 200
        assert client.post("/predict", json=payload).status_code == 503
        assert client.post("/reload").status_code == 200
        assert client.post("/predict", json=payload).json()["prediction"] == response.json()["prediction"]
        original = app.state.adapter.predict

        def failed(*args, **kwargs):
            raise RuntimeError("injected inference failure")

        monkeypatch.setattr(app.state.adapter, "predict", failed)
        assert client.post("/predict", json=payload).status_code == 503
        monkeypatch.setattr(app.state.adapter, "predict", original)
        assert client.post("/predict", json=payload).status_code == 200


def test_real_checkpoint_hash_validation(tmp_path):
    checkpoint = tmp_path / "bad"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(json.dumps({"schema": "worldsystems-real-training-v1",
        "files": {"weights.safetensors": "0", "training.pt": "0"}}), encoding="utf-8")
    (checkpoint / "weights.safetensors").write_bytes(b"corrupt")
    (checkpoint / "training.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_real_checkpoint(checkpoint)


@pytest.mark.skipif(not Path("../WorldPolicy/artifacts/v11/mujoco_reach/act/checkpoint.pt").exists(),
                    reason="WorldPolicy ACT checkpoint unavailable")
def test_worldpolicy_act_contract():
    repo = Path("../WorldPolicy")
    adapter = ACTPolicyAdapter(repo, repo / "artifacts/v11/mujoco_reach/act/checkpoint.pt",
                               repo / "artifacts/v11/mujoco_reach/act/train.json")
    result = adapter.predict_chunk(torch.zeros(1, 8))
    assert result["actions"].shape == (1, 4, 2)
    assert torch.isfinite(result["actions"]).all()
    with pytest.raises(ValueError, match="state"):
        adapter.predict_chunk(torch.zeros(1, 3))
