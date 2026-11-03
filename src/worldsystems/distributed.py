"""CPU Gloo or CUDA NCCL DDP training contract for torchrun."""

import json
import os
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .checkpoint import save_checkpoint
from .config import Config
from .data import SyntheticWorld
from .engine import build_model


def ddp_train(cfg: Config) -> dict:
    if not torch.distributed.is_available():
        raise RuntimeError("PyTorch distributed unavailable")
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size < 2 or rank < 0 or local_rank < 0:
        raise RuntimeError("launch with torchrun and at least two processes")
    if cfg.device == "cuda":
        if torch.cuda.device_count() < world_size:
            raise RuntimeError("insufficient CUDA devices for DDP")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    shared_file = os.environ.get("WORLDSYSTEMS_DDP_FILE")
    init_method = f"file:///{Path(shared_file).as_posix()}" if shared_file else "env://"
    torch.distributed.init_process_group(backend=backend, init_method=init_method,
                                         rank=rank, world_size=world_size)
    try:
        torch.manual_seed(cfg.seed)
        base = build_model(cfg).to(device)
        model = DistributedDataParallel(base, device_ids=[local_rank] if cfg.device == "cuda" else None)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        dataset = SyntheticWorld(cfg.train_samples, cfg.sequence_length, cfg.image_size,
                                 cfg.action_dim, cfg.seed, namespace="train")
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=cfg.seed, drop_last=True)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler,
                            pin_memory=cfg.device == "cuda")
        if len(loader) == 0:
            raise ValueError("too few samples for DDP world size and batch size")
        metrics = []
        global_step = 0
        for epoch in range(cfg.epochs):
            sampler.set_epoch(epoch)
            model.train()
            loss_sum = torch.zeros(2, device=device)
            for batch in loader:
                video = batch["video"].to(device, non_blocking=True)
                actions = batch["actions"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                prediction, _ = model(video, actions)
                loss = torch.nn.functional.mse_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite distributed loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                global_step += 1
                loss_sum += torch.tensor([loss.item() * len(video), len(video)], device=device)
            scheduler.step()
            torch.distributed.all_reduce(loss_sum)
            metric = {"epoch": epoch, "global_step_per_rank": global_step,
                      "train_mse": (loss_sum[0] / loss_sum[1]).item(),
                      "world_size": world_size, "backend": backend}
            metrics.append(metric)
            if rank == 0:
                output = Path(cfg.checkpoint_dir)
                output.mkdir(parents=True, exist_ok=True)
                save_checkpoint(output / f"ddp-epoch-{epoch:04d}", model.module,
                                optimizer, scheduler, epoch, global_step,
                                {**cfg.__dict__, "backend": backend, "world_size": world_size})
                with (output / "ddp-metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(metric) + "\n")
            torch.distributed.barrier()
        return {"status": "completed", "rank": rank, "world_size": world_size,
                "backend": backend, "metrics": metrics}
    finally:
        torch.distributed.destroy_process_group()
