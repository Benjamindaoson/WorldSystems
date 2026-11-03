"""Matched 200-update physical-model runs; output paths cannot be reused."""

import json
from pathlib import Path

from worldsystems.real_training import RealTrainingConfig, train_real


def main():
    root = Path("results/v20/training-01")
    root.mkdir(parents=True, exist_ok=False)
    for family, weights in [
        ("rssm", "v11-baselines-815387b/rssm-s17/best.pt"),
        ("action_predictor", "v11-baselines-fair/conditioned-s17/best.pt"),
    ]:
        cfg = RealTrainingConfig(
            repository="../WorldDynamics",
            dataset="../WorldDynamics/artifacts/v11-mujoco-32scene-h32",
            architecture_weights="../WorldDynamics/artifacts/" + weights,
            family=family, run_dir=str(root / family), device="cuda", epochs=10,
            batch_size=7, seed=17)
        report = train_real(cfg)
        print(json.dumps({"family": family, "steps": report["optimizer_steps"],
                          "test": report["test"], "wall_seconds": report["wall_seconds"]}))


if __name__ == "__main__":
    main()
