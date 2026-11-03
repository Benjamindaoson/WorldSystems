import argparse
import json
import platform

import torch

from .checkpoint import verify_checkpoint
from .config import load_config
from .data import SyntheticWorld, validate_batch
from .engine import evaluate, train
from .runtime import benchmark, infer, load_model


def main() -> None:
    parser = argparse.ArgumentParser(prog="worldsystems")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    ddp = sub.add_parser("ddp-train")
    ddp.add_argument("--config", required=True)
    external = sub.add_parser("external-evaluate")
    external.add_argument("--repository", required=True)
    external.add_argument("--dataset", required=True)
    external.add_argument("--weights", required=True)
    external.add_argument("--output", required=True)
    external.add_argument("--split", choices=["val", "test"], default="test")
    rssm = sub.add_parser("rssm-evaluate")
    rssm.add_argument("--repository", required=True)
    rssm.add_argument("--dataset", required=True)
    rssm.add_argument("--weights", required=True)
    rssm.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    rssm.add_argument("--output", required=True)
    real_train = sub.add_parser("real-train")
    real_train.add_argument("--config", required=True)
    real_train.add_argument("--resume")
    real_train.add_argument("--max-steps", type=int)
    rssm_bench = sub.add_parser("rssm-benchmark")
    rssm_bench.add_argument("--repository", required=True)
    rssm_bench.add_argument("--dataset", required=True)
    rssm_bench.add_argument("--weights", required=True)
    rssm_bench.add_argument("--output-dir", required=True)
    rssm_bench.add_argument("--runs", type=int, default=100)
    rssm_bench.add_argument("--fastpath", action="store_true")
    rssm_eager = sub.add_parser("rssm-eager-benchmark")
    rssm_eager.add_argument("--repository", required=True)
    rssm_eager.add_argument("--dataset", required=True)
    rssm_eager.add_argument("--weights", required=True)
    rssm_eager.add_argument("--output-dir", required=True)
    rssm_eager.add_argument("--runs", type=int, default=100)
    rssm_serve = sub.add_parser("rssm-serve")
    rssm_serve.add_argument("--repository", required=True)
    rssm_serve.add_argument("--weights", required=True)
    rssm_serve.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    rssm_serve.add_argument("--port", type=int, default=8001)
    physical = sub.add_parser("physical-train")
    physical.add_argument("--repository", required=True)
    physical.add_argument("--dataset", required=True)
    physical.add_argument("--architecture-weights", required=True)
    physical.add_argument("--output-dir", required=True)
    physical.add_argument("--epochs", type=int, default=5)
    physical.add_argument("--seed", type=int, default=17)
    benchmark_new = sub.add_parser("benchmark-v11")
    benchmark_new.add_argument("--config", required=True)
    benchmark_new.add_argument("--checkpoint", required=True)
    benchmark_new.add_argument("--output-dir", required=True)
    benchmark_new.add_argument("--runs", type=int, default=500)
    benchmark_new.add_argument("--warmup", type=int, default=30)
    profile = sub.add_parser("profile-v11")
    profile.add_argument("--config", required=True)
    profile.add_argument("--checkpoint", required=True)
    profile.add_argument("--output-dir", required=True)
    graph = sub.add_parser("cuda-graph-v11")
    graph.add_argument("--config", required=True)
    graph.add_argument("--checkpoint", required=True)
    graph.add_argument("--output-dir", required=True)
    graph.add_argument("--runs", type=int, default=200)
    for name in ("train", "evaluate", "infer", "benchmark", "serve"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name != "train":
            p.add_argument("--checkpoint", required=True)
        else:
            p.add_argument("--resume")
            p.add_argument("--max-steps", type=int)
    check = sub.add_parser("checkpoint")
    check.add_argument("action", choices=["verify"])
    check.add_argument("--path", required=True)
    data = sub.add_parser("data")
    data.add_argument("action", choices=["validate"])
    data.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.command == "doctor":
        result = {"python": platform.python_version(), "torch": torch.__version__,
                  "cuda_available": torch.cuda.is_available(),
                  "gpu_count": torch.cuda.device_count(),
                  "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    elif args.command == "ddp-train":
        from .distributed import ddp_train

        result = ddp_train(load_config(args.config))
    elif args.command == "external-evaluate":
        from pathlib import Path

        from .worlddynamics import WorldDynamicsDatasetAdapter, WorldDynamicsModelAdapter

        model = WorldDynamicsModelAdapter(args.repository, args.weights)
        dataset = WorldDynamicsDatasetAdapter(args.dataset)
        result = model.evaluate(dataset, args.split)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
    elif args.command == "rssm-evaluate":
        from pathlib import Path

        from .rssm_adapter import RSSMAdapter
        from .worlddynamics import WorldDynamicsDatasetAdapter

        model = RSSMAdapter(args.repository, args.weights, args.device)
        result = model.evaluate(WorldDynamicsDatasetAdapter(args.dataset))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
    elif args.command == "real-train":
        from pathlib import Path

        import yaml

        from .real_training import RealTrainingConfig, train_real

        config = RealTrainingConfig(**yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
        result = train_real(config, args.resume, args.max_steps)
    elif args.command == "rssm-benchmark":
        from .rssm_optimization import benchmark_rssm

        result = benchmark_rssm(args.repository, args.weights, args.dataset,
                                args.output_dir, args.runs, fastpath=args.fastpath)
    elif args.command == "rssm-eager-benchmark":
        from .rssm_optimization import benchmark_rssm_eager

        result = benchmark_rssm_eager(args.repository, args.weights, args.dataset,
                                      args.output_dir, args.runs)
    elif args.command == "rssm-serve":
        import uvicorn

        from .rssm_server import create_rssm_app
        uvicorn.run(create_rssm_app(args.repository, args.weights, args.device),
                    host="127.0.0.1", port=args.port)
        return
    elif args.command == "physical-train":
        from .physical import train_physical

        result = train_physical(args.repository, args.dataset, args.architecture_weights,
                                args.output_dir, args.epochs, args.seed)
    elif args.command == "benchmark-v11":
        from .benchmark_v11 import benchmark_v11

        result = benchmark_v11(load_config(args.config), args.checkpoint, args.output_dir,
                               args.runs, args.warmup)
    elif args.command == "profile-v11":
        from .profiling import profile_inference

        result = profile_inference(load_config(args.config), args.checkpoint, args.output_dir)
    elif args.command == "cuda-graph-v11":
        from .cuda_graph import compare_cuda_graph

        result = compare_cuda_graph(load_config(args.config), args.checkpoint,
                                    args.output_dir, args.runs)
    elif args.command == "checkpoint":
        result = verify_checkpoint(args.path)
    else:
        cfg = load_config(args.config)
        if args.command == "train":
            result = train(cfg, args.resume, max_steps=args.max_steps)
        elif args.command == "data":
            sample = SyntheticWorld(1, cfg.sequence_length, cfg.image_size,
                                    cfg.action_dim, cfg.seed)[0]
            validate_batch({k: v.unsqueeze(0) for k, v in sample.items()}, cfg.action_dim)
            result = {"status": "valid", "video_shape": list(sample["video"].shape)}
        elif args.command == "evaluate":
            result = evaluate(load_model(cfg, args.checkpoint), cfg)
        elif args.command == "infer":
            result = infer(cfg, args.checkpoint)
        elif args.command == "benchmark":
            result = benchmark(cfg, args.checkpoint)
        else:
            import uvicorn

            from .server import create_app
            uvicorn.run(create_app(cfg, args.checkpoint), host="127.0.0.1", port=8000)
            return
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
