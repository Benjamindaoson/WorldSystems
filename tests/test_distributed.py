import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import yaml

from worldsystems.checkpoint import verify_checkpoint
from worldsystems.config import Config


@pytest.mark.distributed
@pytest.mark.skipif(platform.system() == "Windows" and torch.__version__.startswith("2.15.0+cpu"),
                    reason="this Windows CPU wheel fails FileStore initialization")
def test_two_rank_cpu_gloo(tmp_path):
    cfg = Config(device="cpu", epochs=1, train_samples=16, val_samples=4,
                 test_samples=4, batch_size=4, checkpoint_dir=str(tmp_path / "ddp"))
    config_path = tmp_path / "ddp.yaml"
    config_path.write_text(yaml.safe_dump(asdict(cfg)), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_ddp_cpu.py"
    run = subprocess.run([sys.executable, str(script), "--config", str(config_path)],
                         cwd=script.parent.parent, capture_output=True, text=True, check=True)
    assert "'world_size': 2" in run.stdout
    manifest = verify_checkpoint(tmp_path / "ddp" / "ddp-epoch-0000")
    assert manifest["config"]["world_size"] == 2
