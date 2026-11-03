"""Refresh resumable execution state and evidence inventory from actual local reports."""

import json
import subprocess
from pathlib import Path

from worldsystems.checkpoint import sha256


def main():
    root = Path("results/v20")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    phases = [
        (0, "TESTED", "39 historical hashes verified; source drift and serving lifecycle fixed"),
        (1, "GPU_VERIFIED", "Full deterministic free rollout capture; partial graph and exhaustive timing pending"),
        (2, "BLOCKED", "Both real model Inductor executions fail: working Triton absent"),
        (3, "GPU_VERIFIED", "RSSM/ActionPredictor 200 steps each; formal VideoFlow training pending"),
        (4, "IMPLEMENTED", "Inference horizons 4/8/16/32 measured; video/training matrix pending"),
        (5, "GPU_VERIFIED", "Resident recurrent sessions tested; HTTP/video streaming integration pending"),
        (6, "IMPLEMENTED", "Existing service race/body-limit fixes tested; queued scheduler pending"),
        (7, "NOT_RUN", "Dynamic batching and bounded queue not implemented"),
        (8, "NOT_RUN", "V1.2 RSSM recovery inherited; V2 VideoFlow recovery pending"),
        (9, "NOT_RUN", "V1.1 CPU Gloo inherited; FSDP2/DeepSpeed implementation pending"),
        (10, "BLOCKED", "Only one local GPU; no compatible multi-GPU runner authorized/available"),
        (11, "IMPLEMENTED", "RSSM pinned source; remaining visual families pending"),
        (12, "IMPLEMENTED", "V1.2 ACT inherited; FetchPush/Diffusion integration pending"),
        (13, "NOT_RUN", "Registry not implemented"),
        (14, "NOT_RUN", "Research question identified; related work, novelty and ablations pending"),
        (15, "IMPLEMENTED", "Raw graph/training evidence and resume state; full benchmark suite pending"),
    ]
    state = {"schema": "worldsystems-execution-v20-1", "overall": "IN_PROGRESS",
             "complete": False, "git_commit": head,
             "validation": {"cuda_suite": "21 passed", "cpu_suite": "17 passed, 4 skipped",
                            "ruff": "passed", "warning": "upstream Starlette httpx deprecation"},
             "branch": "feat/v20-systems-research", "remote_pushed": False,
             "phases": [{"phase": p, "status": s, "detail": d} for p, s, d in phases],
             "next_command": ".\\.venv-cuda\\Scripts\\python.exe -m pytest tests/test_v20_graph.py tests/test_v20_streaming.py -q",
             "next_implementation": "Add bounded deadline-aware queue to RSSM HTTP service; integrate RSSMStreaming; run 1/2/4/8/16-client CPU/GPU benchmarks. Then formal VideoFlow training and cross-process recovery.",
             "constraints": ["write only WorldSystems", "never overwrite V1/V1.1/V1.2 evidence",
                             "no remote push authorization", "no driver/WSL/global environment changes",
                             "GPU timings exploratory: exclusive ownership not established"],
             "completed_commands": ["scripts/audit_v12_evidence.py",
                                    "scripts/audit_graph_v20.py --output results/v20/graph-audit-pinned-02 --runs 100",
                                    "scripts/compile_v20.py", "scripts/train_v20.py"],
             "rerun_rule": "Audit/compile/training scripts refuse existing output paths; choose a new versioned path before rerunning."}
    Path("execution").mkdir(exist_ok=True)
    Path("execution/v20-state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    files = [p for p in root.rglob("*") if p.is_file() and p.name != "evidence-manifest.json"]
    inventory = {"git_commit_at_generation": head,
                 "raw_files": [{"path": p.as_posix(), "sha256": sha256(p)} for p in files],
                 "source_files": [{"path": p.as_posix(), "sha256": sha256(p)}
                                  for p in Path("src/worldsystems").glob("*.py")]}
    (root / "evidence-manifest.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"saved resumable state; indexed {len(files)} V2 evidence files")


if __name__ == "__main__":
    main()
