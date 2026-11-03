import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from worldsystems.checkpoint import load_checkpoint, verify_checkpoint
from worldsystems.config import Config
from worldsystems.engine import build_model, loader, train
from worldsystems.physical import train_physical
from worldsystems.worlddynamics import WorldDynamicsDatasetAdapter, WorldDynamicsModelAdapter


def _run(config: Config, resume: str | None = None, max_steps: int | None = None):
    code = ("from worldsystems.config import Config; from worldsystems.engine import train; "
            f"print(train(Config(**{asdict(config)!r}), {resume!r}, max_steps={max_steps!r}))")
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_split_samples_are_disjoint():
    cfg = Config(train_samples=200001)
    samples = [loader(cfg, split).dataset[0]["video"] for split in ("train", "val", "test")]
    assert not torch.equal(samples[0], samples[1])
    assert not torch.equal(samples[1], samples[2])
    assert not torch.equal(samples[0], samples[2])


def test_cross_process_mid_epoch_resume(tmp_path):
    base = {"train_samples": 19, "val_samples": 4, "test_samples": 4, "epochs": 2,
            "batch_size": 4, "grad_accumulation": 2}
    full = Config(**base, checkpoint_dir=str(tmp_path / "full"))
    resumed = Config(**base, checkpoint_dir=str(tmp_path / "resumed"))
    _run(full)
    _run(resumed, max_steps=1)
    mid = tmp_path / "resumed" / "step-000001"
    manifest = verify_checkpoint(mid)
    assert manifest["next_batch"] == 2
    _run(resumed, str(mid))
    a = build_model(full)
    b = build_model(resumed)
    oa = torch.optim.AdamW(a.parameters())
    ob = torch.optim.AdamW(b.parameters())
    sa = torch.optim.lr_scheduler.CosineAnnealingLR(oa, T_max=2)
    sb = torch.optim.lr_scheduler.CosineAnnealingLR(ob, T_max=2)
    load_checkpoint(tmp_path / "full" / "epoch-0001", a, oa, sa)
    load_checkpoint(tmp_path / "resumed" / "epoch-0001", b, ob, sb)
    for left, right in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert sa.state_dict() == sb.state_dict()
    for key, left in oa.state_dict()["state"].items():
        for name, value in left.items():
            torch.testing.assert_close(value, ob.state_dict()["state"][key][name], rtol=0, atol=0)
    full_log = [json.loads(s) for s in (tmp_path / "full" / "train.jsonl").read_text().splitlines()]
    resumed_log = [json.loads(s) for s in (tmp_path / "resumed" / "train.jsonl").read_text().splitlines()]
    for x, y in zip(full_log, resumed_log):
        assert x["train_mse"] == y["train_mse"]
        assert x["val_mse"] == y["val_mse"]


def test_bad_resume_and_partial_checkpoint(tmp_path):
    cfg = Config(train_samples=8, val_samples=4, test_samples=4, epochs=1,
                 checkpoint_dir=str(tmp_path / "run"))
    checkpoint = train(cfg)["checkpoint"]
    with pytest.raises(ValueError, match="configuration mismatch"):
        train(Config(**{**asdict(cfg), "seed": 99}), resume=checkpoint)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "manifest.json").write_text(json.dumps({"schema_version": 1,
        "model_family": "tiny-action-temporal", "files": {}}))
    with pytest.raises(ValueError, match="incomplete"):
        verify_checkpoint(partial)
    legacy = tmp_path / "legacy"
    shutil.copytree(checkpoint, legacy)
    old = json.loads((legacy / "manifest.json").read_text(encoding="utf-8"))
    old.pop("sampler_schema")
    old["config"]["dataset_version"] = "synthetic-linear-v1"
    (legacy / "manifest.json").write_text(json.dumps(old), encoding="utf-8")
    old_cfg = Config(**{**asdict(cfg), "dataset_version": "synthetic-linear-v1"})
    with pytest.raises(ValueError, match="sampler differs"):
        train(old_cfg, resume=str(legacy))


@pytest.mark.skipif(not Path("../WorldDynamics/artifacts/v11-mujoco-32scene-h32").exists(),
                    reason="WorldDynamics local artifacts unavailable")
def test_worlddynamics_adapter():
    repo = Path("../WorldDynamics")
    dataset = WorldDynamicsDatasetAdapter(repo / "artifacts/v11-mujoco-32scene-h32")
    model = WorldDynamicsModelAdapter(
        repo, repo / "artifacts/v11-baselines-fair/conditioned-s17/best.pt")
    result = model.evaluate(dataset)
    assert result["samples"] == 42
    assert result["one_step_mse"] < result["free_running_mse"]


@pytest.mark.skipif(not Path("../WorldDynamics/artifacts/v11-mujoco-32scene-h32").exists(),
                    reason="WorldDynamics local artifacts unavailable")
def test_physical_training_checkpoint_load(tmp_path):
    repo = Path("../WorldDynamics")
    report = train_physical(
        str(repo), str(repo / "artifacts/v11-mujoco-32scene-h32"),
        str(repo / "artifacts/v11-baselines-fair/conditioned-s17/best.pt"),
        str(tmp_path / "physical"), epochs=1)
    adapter = WorldDynamicsModelAdapter(repo, report["checkpoint"])
    dataset = WorldDynamicsDatasetAdapter(repo / "artifacts/v11-mujoco-32scene-h32")
    assert adapter.evaluate(dataset)["one_step_mse"] == report["test"]["one_step_mse"]
