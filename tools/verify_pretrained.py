#!/usr/bin/env python3
"""Verify local StreamTalk pretrained checkpoints against the release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("checkpoints/pretrained"),
    )
    args = parser.parse_args()
    manifest_path = args.weights_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        path = args.weights_dir / artifact["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != artifact["size_bytes"]:
            raise RuntimeError(f"size mismatch: {path}")
        actual = sha256_file(path)
        if actual != artifact["sha256"]:
            raise RuntimeError(f"SHA256 mismatch: {path}")
        print(f"OK {artifact['sha256']}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
