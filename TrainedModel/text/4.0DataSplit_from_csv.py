"""
4.0DataSplit_from_csv.py
Convert mbti_1.csv (columns: type, posts) into the folder layout required by the
text training scripts (train/val/test/<MBTI>/*.txt).

Example:
  python TrainedModel/text/4.0DataSplit_from_csv.py ^
      --csv .\\mbti_1.csv ^
      --out-root .\\data\\TextData ^
      --per-post --min-chars 30 --val 0.1 --test 0.1
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

MBTI_16: Sequence[str] = (
    "INTJ",
    "INTP",
    "ENTJ",
    "ENTP",
    "INFJ",
    "INFP",
    "ENFJ",
    "ENFP",
    "ISTJ",
    "ISFJ",
    "ESTJ",
    "ESFJ",
    "ISTP",
    "ISFP",
    "ESTP",
    "ESFP",
)

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def set_seed(seed: int) -> None:
    random.seed(seed)


def clean_text(raw: str) -> str:
    """Basic cleanup: drop URLs, collapse whitespace, trim markers."""
    text = raw.replace("|||", "\n")
    text = URL_RE.sub("", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_csv(
    csv_path: Path,
    per_post: bool,
    min_chars: int,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Returns:
        samples: list of (clean_text, mbti_type)
        classes: MBTI types present in the CSV (subset of MBTI_16)
    """
    samples: List[Tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mbti_type = (row.get("type") or "").strip().upper()
            posts = row.get("posts") or ""
            if mbti_type not in MBTI_16 or not posts:
                continue
            if per_post:
                entries = (clean_text(chunk) for chunk in posts.split("|||"))
                for entry in entries:
                    if len(entry) >= min_chars:
                        samples.append((entry, mbti_type))
            else:
                entry = clean_text(posts.replace("|||", " "))
                if len(entry) >= min_chars:
                    samples.append((entry, mbti_type))
    classes = sorted({cls for _, cls in samples})
    if not classes:
        raise RuntimeError("No valid MBTI samples found. Check CSV columns (type, posts).")
    return samples, classes


def stratified_split(
    samples: Sequence[Tuple[str, str]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    """Return nested dict split[label][split_name] -> list of samples."""
    rng = random.Random(seed)
    buckets: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for text, label in samples:
        buckets[label].append((text, label))

    splits: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
    for label, items in buckets.items():
        rng.shuffle(items)
        n_total = len(items)
        n_val = int(round(n_total * val_ratio))
        n_test = int(round(n_total * test_ratio))
        n_val = min(n_val, n_total)
        n_test = min(n_test, n_total - n_val)
        n_train = n_total - n_val - n_test
        splits[label] = {
            "train": items[:n_train],
            "val": items[n_train : n_train + n_val],
            "test": items[n_train + n_val :],
        }
    return splits


def ensure_output_root(root: Path, overwrite: bool) -> None:
    if root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory {root} already exists. "
                "Use --overwrite to rebuild it."
            )
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def write_split_tree(
    root: Path,
    splits: Dict[str, Dict[str, List[Tuple[str, str]]]],
) -> Dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for label, label_splits in splits.items():
        for split_name, items in label_splits.items():
            dest_dir = root / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            for idx, (text, _) in enumerate(items):
                file_path = dest_dir / f"{label}_{idx:05d}.txt"
                file_path.write_text(text, encoding="utf-8")
            counts[split_name] += len(items)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split mbti_1.csv into train/val/test text folders."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to mbti_1.csv")
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Destination folder (train/val/test directories will be created here).",
    )
    parser.add_argument(
        "--per-post",
        action="store_true",
        help="Treat each post segment (split by '|||') as an individual sample.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=30,
        help="Discard samples shorter than this many characters after cleaning.",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.1,
        help="Validation ratio (fraction of samples per class).",
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.1,
        help="Test ratio (fraction of samples per class).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for stratified splits."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the existing output directory before writing new data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if args.val < 0 or args.test < 0 or args.val + args.test >= 1:
        raise ValueError("val and test ratios must be between 0 and 1 and sum to < 1.")

    set_seed(args.seed)
    samples, classes = read_csv(args.csv, per_post=args.per_post, min_chars=args.min_chars)
    print(f"Detected classes ({len(classes)}): {classes}")

    splits = stratified_split(samples, val_ratio=args.val, test_ratio=args.test, seed=args.seed)

    ensure_output_root(args.out_root, overwrite=args.overwrite)
    counts = write_split_tree(args.out_root, splits)
    total = sum(counts.values())
    print(f"Wrote {total} samples → train:{counts['train']} val:{counts['val']} test:{counts['test']}")
    print(f"Dataset ready under: {args.out_root}")


if __name__ == "__main__":
    main()
