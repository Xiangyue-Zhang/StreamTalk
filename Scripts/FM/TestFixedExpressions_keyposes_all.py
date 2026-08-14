import argparse
import csv
from pathlib import Path
import subprocess
import sys


def parse_train_test_split_csv(csv_file):
    result = {split: [] for split in ("train", "test", "val", "additional")}
    with Path(csv_file).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "type"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"split CSV must contain columns {sorted(required)}")
        for row in reader:
            split = row["type"]
            if split in result:
                result[split].append(row["id"])
    return result


def _speaker_id(file_id):
    first = file_id.find("_")
    second = file_id.find("_", first + 1)
    if first < 0 or second < 0:
        raise ValueError(f"unexpected BEAT2 file id: {file_id}")
    return file_id[:second] + "_0"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run StreamTalk generation over the BEAT2 test split."
    )
    parser.add_argument("--data_folder", type=Path, required=True)
    parser.add_argument("--wav_folder", type=Path, required=True)
    parser.add_argument("--csv_file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_folder", type=Path, required=True)
    parser.add_argument("--wavlm_model", required=True)
    parser.add_argument("--smplx_model", type=Path, required=True)
    parser.add_argument(
        "--pid",
        help="optional speaker id such as 2_scott_0; omit for all speakers",
    )
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser


def Test_All(argv=None):
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.cfg_scale <= 7.0:
        raise ValueError("--cfg_scale must be between 0 and 7")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    file_ids = parse_train_test_split_csv(args.csv_file)["test"]
    if args.pid is not None:
        prefix = f"{args.pid}_"
        file_ids = [file_id for file_id in file_ids if file_id.startswith(prefix)]
    if args.limit is not None:
        file_ids = file_ids[: args.limit]
    if not file_ids:
        raise RuntimeError("no matching BEAT2 test files")

    test_script = Path(__file__).with_name("TestFixedExpressions_keyposes.py")
    args.output_folder.mkdir(parents=True, exist_ok=True)
    for file_id in file_ids:
        npz_file = args.data_folder / f"{file_id}.npz"
        wav_file = args.wav_folder / f"{file_id}.wav"
        if not npz_file.is_file() or not wav_file.is_file():
            raise FileNotFoundError(f"missing BEAT2 pair for {file_id}")
        command = [
            args.python,
            str(test_script),
            str(wav_file),
            str(args.checkpoint),
            "--example_motion_npz",
            str(npz_file),
            "--output_file",
            str(args.output_folder / f"res_{file_id}.npz"),
            "--pid",
            _speaker_id(file_id),
            "--wavlm_model",
            args.wavlm_model,
            "--smplx_model",
            str(args.smplx_model),
            "--retrieval_data_folder",
            str(args.data_folder),
            "--retrieval_csv_file",
            str(args.csv_file),
            "--cfg_scale",
            str(args.cfg_scale),
            "--seed",
            str(args.seed),
        ]
        if args.dry_run:
            print(subprocess.list2cmdline(command))
        else:
            subprocess.run(command, check=True)

    print(f"completed {len(file_ids)} test files in {args.output_folder}")


if __name__ == "__main__":
    Test_All()
