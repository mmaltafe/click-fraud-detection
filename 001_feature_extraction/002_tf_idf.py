#!/usr/bin/env python3
"""
Create a TF-IDF representation for the subdataset selected in .env.

Each request row is converted into one raw text document by concatenating all
dataset columns as "column=value" tokens. Missing values are represented as
__missing__.

Input:
    data/raw/{DATASET}

Output:
    data/extracted_features/tf_idf/{DATASET}/tf_idf_matrix.npz
    data/extracted_features/tf_idf/{DATASET}/target.parquet
    data/extracted_features/tf_idf/{DATASET}/metadata.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils._columns import CAMPAIGN_COLUMNS, normalize_name, is_table, read_table, raw_files_for_dataset, stringify_value  # noqa: E402
from utils._env import VALID_DATASETS, MISSING, read_env_value  # noqa: E402
from utils.target_utils import binary_target_series, TARGET_COLUMNS  # noqa: E402


# Pipeline configuration constants. Edit these values to change this script.
CONFIG_ENV_FILE = '.env'
CONFIG_RAW_ROOT = 'data/raw'
CONFIG_OUTPUT_ROOT = 'data/extracted_features/tf_idf'
CONFIG_MAX_FEATURES = 50000
CONFIG_MIN_DF = 1
CONFIG_NGRAM_MAX = 2


def parse_args() -> argparse.Namespace:
    """Return script configuration from global constants; command-line args are intentionally ignored."""
    return argparse.Namespace(
        env_file=CONFIG_ENV_FILE,
        raw_root=CONFIG_RAW_ROOT,
        output_root=CONFIG_OUTPUT_ROOT,
        max_features=CONFIG_MAX_FEATURES,
        min_df=CONFIG_MIN_DF,
        ngram_max=CONFIG_NGRAM_MAX,
    )


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


def parse_min_df(value: str):
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError as exc:
        raise ValueError("--min-df must be an integer or float") from exc


def save_tfidf(
    documents: list[str],
    row_index: list[dict[str, str | int]],
    target: pd.DataFrame,
    output_path: Path,
    max_features: int,
    min_df,
    ngram_max: int,
) -> None:
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=max_features,
        min_df=min_df,
        ngram_range=(1, ngram_max),
        token_pattern=r"(?u)\b[^\s]+\b",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(documents)
    sparse.save_npz(output_path / "tf_idf_matrix.npz", matrix, compressed=True)
    target.to_parquet(output_path / "target.parquet", index=False)

    metadata = {
        "vectorizer": "sklearn.feature_extraction.text.TfidfVectorizer",
        "matrix": "tf_idf_matrix.npz",
        "target": "target.parquet",
        "n_rows": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "max_features": max_features,
        "min_df": min_df,
        "ngram_range": [1, ngram_max],
        "missing_value": MISSING,
        "feature_names": vectorizer.get_feature_names_out().tolist(),
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
    save_tfidf(
        documents=documents,
        row_index=row_index,
        target=target,
        output_path=output_path,
        max_features=args.max_features,
        min_df=parse_min_df(str(args.min_df)),
        ngram_max=args.ngram_max,
    )

    print(f"Saved TF-IDF features with {len(documents)} rows to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
