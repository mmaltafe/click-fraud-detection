#!/usr/bin/env python3
"""
Create SentenceTransformer embeddings for the subdataset selected in .env.

Each request row is converted into one raw text document by concatenating all
dataset columns as "column=value" tokens. Missing values are represented as
__missing__.

Input:
    data/raw/{DATASET}

Output:
    data/extracted_features/sentence_transformer/{DATASET}/embeddings.npy
    data/extracted_features/sentence_transformer/{DATASET}/target.parquet
    data/extracted_features/sentence_transformer/{DATASET}/metadata.json

Model cache:
    models/sentence_transformers
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils._columns import CAMPAIGN_COLUMNS  # noqa: E402
from utils._env import VALID_DATASETS, MISSING, read_env_value  # noqa: E402
from utils.target_utils import binary_target_series  # noqa: E402


MODEL_NAME = "all-MiniLM-L6-v2"
TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".parquet", ".feather"}
TARGET_COLUMNS = ("Attack_type", "attack_type")


# Pipeline configuration constants. Edit these values to change this script.
CONFIG_ENV_FILE = '.env'
CONFIG_RAW_ROOT = 'data/raw'
CONFIG_OUTPUT_ROOT = 'data/extracted_features/sentence_transformer'
CONFIG_MODEL_CACHE = 'models/sentence_transformers'
CONFIG_MODEL_NAME = 'all-MiniLM-L6-v2'
CONFIG_BATCH_SIZE = 128
CONFIG_NORMALIZE = False


def parse_args() -> argparse.Namespace:
    """Return script configuration from global constants; command-line args are intentionally ignored."""
    return argparse.Namespace(
        env_file=CONFIG_ENV_FILE,
        raw_root=CONFIG_RAW_ROOT,
        output_root=CONFIG_OUTPUT_ROOT,
        model_cache=CONFIG_MODEL_CACHE,
        model_name=CONFIG_MODEL_NAME,
        batch_size=CONFIG_BATCH_SIZE,
        normalize=CONFIG_NORMALIZE,
    )


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def campaign_column(columns: list[str]) -> str | None:
    normalized = {normalize_name(column): column for column in columns}
    for candidate in CAMPAIGN_COLUMNS:
        candidate_name = normalize_name(candidate)
        if candidate_name in normalized:
            return normalized[candidate_name]
    return None


def target_column(columns: list[str]) -> str | None:
    normalized = {normalize_name(column): column for column in columns}
    for candidate in TARGET_COLUMNS:
        candidate_name = normalize_name(candidate)
        if candidate_name in normalized:
            return normalized[candidate_name]
    return None


def is_table(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TABLE_SUFFIXES and not path.name.startswith(".")


def read_table(path: Path) -> pd.DataFrame | None:
    try:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".feather":
            return pd.read_feather(path)

        separator = "\t" if suffix == ".tsv" else None
        return pd.read_csv(path, sep=separator, engine="python")
    except Exception as exc:
        print(f"WARNING: skipping unreadable table {path}: {exc}", file=sys.stderr)
        return None


def raw_files_for_dataset(dataset: str, dataset_path: Path) -> list[tuple[Path, str, str]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Raw dataset folder not found: {dataset_path}")

    files: list[tuple[Path, str, str]] = []
    if dataset == "all50":
        for traffic_source in sorted(path for path in dataset_path.iterdir() if path.is_dir()):
            for path in sorted(traffic_source.rglob("*")):
                if is_table(path):
                    campaign = path.parent.name if path.parent != traffic_source else path.stem
                    files.append((path, traffic_source.name, campaign))
        return files

    for path in sorted(dataset_path.rglob("*")):
        if is_table(path):
            campaign = path.parent.name if path.parent != dataset_path else path.stem
            files.append((path, MISSING, campaign))
    return files


def stringify_value(value: object) -> str:
    if pd.isna(value):
        return MISSING
    text = str(value).strip()
    return text if text else MISSING


def split_target(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    column = target_column(list(table.columns))
    if column is None:
        target = binary_target_series([MISSING] * len(table), index=table.index)
        return table, target

    target = binary_target_series(table[column], index=table.index)
    return table.drop(columns=[column]), target


def table_to_documents(
    table: pd.DataFrame,
    traffic_source: str,
    campaign: str,
    source_path: Path,
) -> tuple[list[str], list[dict[str, str | int]], pd.DataFrame]:
    if table.empty:
        return [], [], pd.DataFrame({"attack_type": []})

    table = table.copy()
    table.columns = [str(column) for column in table.columns]
    table, target = split_target(table)
    existing_campaign_column = campaign_column(list(table.columns))

    documents: list[str] = []
    rows: list[dict[str, str | int]] = []
    for row_index, row in table.iterrows():
        row_campaign = stringify_value(row[existing_campaign_column]) if existing_campaign_column else campaign
        parts = [
            f"traffic_source={traffic_source}",
            f"campaign={row_campaign}",
        ]
        for column in sorted(table.columns, key=normalize_name):
            normalized_column = normalize_name(column)
            parts.append(f"{normalized_column}={stringify_value(row[column])}")

        documents.append(" ".join(parts))
        rows.append(
            {
                "source_path": str(source_path),
                "source_row_index": int(row_index),
                "traffic_source": traffic_source,
                "campaign": row_campaign,
            }
        )

    target_frame = pd.DataFrame({"attack_type": target.reset_index(drop=True)})
    return documents, rows, target_frame


def build_corpus(raw_files: list[tuple[Path, str, str]]) -> tuple[list[str], list[dict[str, str | int]], pd.DataFrame]:
    documents: list[str] = []
    row_index: list[dict[str, str | int]] = []
    target_parts: list[pd.DataFrame] = []

    for path, traffic_source, campaign in raw_files:
        table = read_table(path)
        if table is None or table.empty:
            continue
        file_documents, file_rows, file_target = table_to_documents(table, traffic_source, campaign, path)
        documents.extend(file_documents)
        row_index.extend(file_rows)
        target_parts.append(file_target)

    target = pd.concat(target_parts, ignore_index=True) if target_parts else pd.DataFrame({"attack_type": []})
    return documents, row_index, target


def save_embeddings(
    documents: list[str],
    row_index: list[dict[str, str | int]],
    target: pd.DataFrame,
    output_path: Path,
    model_cache: Path,
    model_name: str,
    batch_size: int,
    normalize: bool,
) -> None:
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model_cache.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(model_name, cache_folder=str(model_cache))
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    ).astype(np.float32, copy=False)

    np.save(output_path / "embeddings.npy", embeddings)
    target.to_parquet(output_path / "target.parquet", index=False)

    metadata = {
        "model": model_name,
        "model_cache": str(model_cache),
        "embeddings": "embeddings.npy",
        "target": "target.parquet",
        "n_rows": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "batch_size": batch_size,
        "normalize_embeddings": normalize,
        "missing_value": MISSING,
        "row_index": row_index,
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    dataset = read_env_value(project_root / args.env_file, "DATASET")

    if dataset not in VALID_DATASETS:
        valid = ", ".join(sorted(VALID_DATASETS))
        raise ValueError(f"Invalid DATASET={dataset!r}. Expected one of: {valid}")

    raw_files = raw_files_for_dataset(dataset, project_root / args.raw_root / dataset)
    if not raw_files:
        raise ValueError(f"No raw table files were found for DATASET={dataset}. Run 000_get_kaggle_data first.")

    documents, row_index, target = build_corpus(raw_files)
    if not documents:
        raise ValueError(f"No text documents were created for DATASET={dataset}.")

    output_path = project_root / args.output_root / dataset
    save_embeddings(
        documents=documents,
        row_index=row_index,
        target=target,
        output_path=output_path,
        model_cache=project_root / args.model_cache,
        model_name=args.model_name,
        batch_size=args.batch_size,
        normalize=args.normalize,
    )

    print(f"Saved SentenceTransformer embeddings with {len(documents)} rows to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
