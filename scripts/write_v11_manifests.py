"""Create machine-readable model lineage from verified local artifacts."""

import json
import subprocess
from pathlib import Path

from worldsystems.checkpoint import verify_checkpoint


def write_manifests(output_dir: str = "results/v11") -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tiny_path = Path("artifacts/v11-cuda-fp32/epoch-0001")
    tiny = verify_checkpoint(tiny_path)
    cfg = tiny["config"]
    reference = {"schema_version": 1, "model_id": "worldsystems-tiny-v11-cuda-fp32",
                 "model_family": tiny["model_family"], "source_repository": "WorldSystems",
                 "source_commit": revision, "checkpoint": str(tiny_path),
                 "weight_sha256": tiny["files"]["weights.safetensors"],
                 "dataset_version": cfg["dataset_version"], "precision": cfg["precision"],
                 "input_video_shape": [None, cfg["sequence_length"], 3,
                                       cfg["image_size"], cfg["image_size"]],
                 "input_action_shape": [None, cfg["sequence_length"], cfg["action_dim"]],
                 "output_shape": [None, cfg["sequence_length"], 3,
                                  cfg["image_size"], cfg["image_size"]],
                 "prediction_horizon": cfg["sequence_length"],
                 "evidence_status": "GPU_VERIFIED_SYNTHETIC_SOFTWARE"}
    physical = json.loads((output / "physical-random-s17" / "report.json").read_text(
        encoding="utf-8"))
    physical_manifest = {"schema_version": 1, "model_id": "worldsystems-mujoco-action-s17",
                         "model_family": "WorldDynamics ActionPredictor",
                         "source_repository": "WorldDynamics",
                         "source_commit": physical["source_commit"],
                         "source_sha256": physical["source_sha256"],
                         "dataset_source_commit": physical["dataset_source_commit"],
                         "dataset_manifest_sha256": physical["dataset_manifest_sha256"],
                         "checkpoint": physical["checkpoint"],
                         "weight_sha256": physical["weight_sha256"],
                         "input_state_shape": [None, 33, 3],
                         "input_action_shape": [None, 32, 3],
                         "output_shape": [None, 32, 3], "prediction_horizon": 32,
                         "evidence_status": "CPU_VERIFIED_MUJOCO_SMALL_DATASET"}
    for name, content in (("tiny-model-manifest.json", reference),
                          ("physical-model-manifest.json", physical_manifest)):
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(content, stream, indent=2)


if __name__ == "__main__":
    write_manifests()
