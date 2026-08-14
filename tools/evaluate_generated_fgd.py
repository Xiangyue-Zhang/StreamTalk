#!/usr/bin/env python3
"""Score generated StreamTalk NPZ files with the official EMAGE AESK FGD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from streamtalk_fgd import PersistentOfficialFGD, load_persistent_aesk, sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground_truth", type=Path, required=True)
    parser.add_argument("--scope", choices=("speaker2", "all"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metric_batch_size", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser


def _poses(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        if "poses" not in payload:
            raise ValueError(f"missing poses in {path}")
        poses = np.asarray(payload["poses"], dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] < 165 or not np.isfinite(poses).all():
        raise ValueError(f"invalid SMPL-X poses in {path}: {poses.shape}")
    return poses[:, :165]


def main() -> int:
    args = build_parser().parse_args()
    if args.metric_batch_size <= 0:
        raise ValueError("--metric_batch_size must be positive")

    ids_path = REPO_ROOT / "tools/assets/official_beat2_test_ids.json"
    ids_payload = json.loads(ids_path.read_text(encoding="utf-8"))
    file_ids = ids_payload[args.scope]
    expected_count = 15 if args.scope == "speaker2" else 265
    if len(file_ids) != expected_count or len(set(file_ids)) != expected_count:
        raise RuntimeError(f"official {args.scope} id fixture is invalid")

    prediction_clips = []
    ground_truth_clips = []
    for file_id in file_ids:
        prediction = _poses(args.predictions / f"res_{file_id}.npz")
        ground_truth = _poses(args.ground_truth / f"{file_id}.npz")
        if len(ground_truth) < len(prediction):
            raise ValueError(f"prediction is longer than GT for {file_id}")
        prediction_clips.append(prediction)
        ground_truth_clips.append(ground_truth[: len(prediction)])

    import torch

    device = torch.device(args.device)
    _model, encoder = load_persistent_aesk(
        device, torch, REPO_ROOT / "Scripts/FM/EMAGE_VAE"
    )
    evaluator = PersistentOfficialFGD(
        encoder, ground_truth_clips, batch_size=args.metric_batch_size
    )
    metric = evaluator.evaluate(prediction_clips)
    result = {
        "format": "streamtalk-generated-npz-emage-fgd-v1",
        "scope": args.scope,
        "official_id_count": expected_count,
        "metric_batch_size": args.metric_batch_size,
        "ids_sha256": sha256_file(ids_path),
        "aesk_sha256": sha256_file(
            REPO_ROOT / "Scripts/FM/EMAGE_VAE/AESKConv_240_100.bin"
        ),
        **metric,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
