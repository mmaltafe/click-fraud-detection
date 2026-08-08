#!/usr/bin/env python3
"""
Binarize raw Attack_type labels for the prepared Kaggle subdatasets.

The downstream pipeline is binary: every request is either:
- legitimate
- attack

Any Attack_type/attack_type value different from legitimate/benign/normal-like
labels is mapped to attack. The script updates raw tables in place and writes a
summary JSON under data/raw.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils._env import VALID_DATASETS  # noqa: E402
from utils.target_utils import TARGET_COLUMNS, binary_attack_type  # noqa: E402


TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".parquet", ".feather"}


# Pipeline configuration constants. Edit these values to change this script.
CONFIG_RAW_ROOT = 'data/raw'
CONFIG_DATASET = 'all'


def parse_args() -> argparse.Namespace:
    """Return script configuration from global constants; command-line args are intentionally ignored."""
    return argparse.Namespace(
        raw_root=CONFIG_RAW_ROOT,
        dataset=CONFIG_DATASET,
    )

def normalize_name(value: str) -> str:
    return str(value).strip().lower()


def target_columns(columns: list[str]) -> list[str]:
    normalized_targets = {normalize_name(column) for column in TARGET_COLUMNS}
    return [column for column in columns if normalize_name(str(column)) in normalized_targets]


def is_table(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TABLE_SUFFIXES and not path.name.startswith(".")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    separator = "\t" if suffix == ".tsv" else None
    return pd.read_csv(path, sep=separator, engine="python")


def write_table(path: Path, table: pd.DataFrame) -> None:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        table.to_parquet(path, index=False)
        return
    if suffix == ".feather":
        table.to_feather(path)
        return
    separator = "\t" if suffix == ".tsv" else ","
    table.to_csv(path, index=False, sep=separator)


def dataset_paths(raw_root: Path, dataset: str) -> list[Path]:
    if dataset == "all":
        return [raw_root / name for name in sorted(VALID_DATASETS) if (raw_root / name).exists()]
    return [raw_root / dataset]


def process_table(path: Path) -> dict[str, Any]:
    table = read_table(path)
    columns = target_columns([str(column) for column in table.columns])
    if not columns:
        return {
            "path": str(path),
            "status": "skipped",
            "reason": "target_column_not_found",
            "n_rows": int(len(table)),
        }

    original_counts: dict[str, int] = {}
    binary_counts: dict[str, int] = {}
    for column in columns:
        original = table[column].fillna("__missing__").astype(str)
        original_counts.update({str(key): int(value) for key, value in original.value_counts(dropna=False).items()})
        table[column] = table[column].map(binary_attack_type)
        binary_counts.update({str(key): int(value) for key, value in table[column].value_counts(dropna=False).items()})

    write_table(path, table)
    return {
        "path": str(path),
        "status": "ok",
        "target_columns": columns,
        "n_rows": int(len(table)),
        "original_counts": original_counts,
        "binary_counts": binary_counts,
    }


def process_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Raw dataset folder not found: {dataset_path}")

    results: list[dict[str, Any]] = []
    for path in sorted(dataset_path.rglob("*")):
        if not is_table(path):
            continue
        try:
            results.append(process_table(path))
        except Exception as exc:
            results.append(
                {
                    "path": str(path),
                    "status": "error",
                    "error": str(exc),
                }
            )
            print(f"WARNING: failed to binarize labels in {path}: {exc}", file=sys.stderr)
    return results


def main() -> None:
    args = parse_args()
    raw_root = PROJECT_ROOT / args.raw_root
    all_results: dict[str, list[dict[str, Any]]] = {}
    for dataset_path in dataset_paths(raw_root, args.dataset):
        all_results[dataset_path.name] = process_dataset(dataset_path)

    summary_path = raw_root / "binary_attack_labels_metadata.json"
    summary_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    ok_tables = sum(1 for rows in all_results.values() for row in rows if row.get("status") == "ok")
    skipped_tables = sum(1 for rows in all_results.values() for row in rows if row.get("status") == "skipped")
    error_tables = sum(1 for rows in all_results.values() for row in rows if row.get("status") == "error")
    print(
        f"Binarized labels in {ok_tables} tables; "
        f"skipped={skipped_tables}; errors={error_tables}; metadata={summary_path}"
    )


if __name__ == "__main__":
    main()
