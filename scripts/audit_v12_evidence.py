"""Verify historical evidence without modifying it."""

import json
from pathlib import Path

from worldsystems.checkpoint import sha256


def main():
    manifest = json.loads(Path("results/v12/evidence-manifest.json").read_text(encoding="utf-8"))
    verified, failed = [], []
    for row in manifest["raw_files"]:
        path = Path(row["path"])
        if path.is_file() and sha256(path) == row["sha256"]:
            verified.append(str(path))
        else:
            failed.append(str(path))
    output = Path("results/v20/v12-integrity.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"status": "TESTED" if not failed else "FAILED", "verified": verified,
                   "failed": failed}, stream, indent=2)
    print(f"verified={len(verified)} failed={len(failed)}")
    if failed:
        raise ValueError("historical evidence integrity failure")


if __name__ == "__main__":
    main()
