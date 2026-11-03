"""Windows-compatible two-rank Gloo contract runner; no multi-GPU performance claim."""

import argparse
import os
import socket
import uuid
from pathlib import Path

import torch.multiprocessing as mp

from worldsystems.config import load_config
from worldsystems.distributed import ddp_train


def worker(rank: int, port: int, config: str, store_file: str):
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
                       "WORLD_SIZE": "2", "RANK": str(rank), "LOCAL_RANK": str(rank),
                       "USE_LIBUV": "0", "WORLDSYSTEMS_DDP_FILE": store_file})
    result = ddp_train(load_config(config))
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/tiny_ddp_cpu_v11.yaml")
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    store = Path("artifacts") / f"ddp-store-{uuid.uuid4().hex}"
    store.parent.mkdir(exist_ok=True)
    mp.spawn(worker, args=(port, args.config, str(store.resolve())), nprocs=2, join=True)
