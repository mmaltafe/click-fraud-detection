#!/usr/bin/env python3
"""
Apply SelectKBest supervised feature selection to each extracted feature approach.

Inputs:
    data/extracted_features/label_encoder/{DATASET}/campaign_label_encoder.json
    data/extracted_features/semantic_headers/{DATASET}/semantic_headers.parquet
    data/extracted_features/tf_idf/{DATASET}/tf_idf_matrix.npz
    data/extracted_features/sentence_transformer/{DATASET}/embeddings.npy

Outputs:
    data/selected_features/selectKBest/{APPROACH}/{DATASET}/features.npy
    data/selected_features/selectKBest/{APPROACH}/{DATASET}/target.parquet
    data/selected_features/selectKBest/{APPROACH}/{DATASET}/metadata.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.target_utils import binary_target_frame, binary_target_series  # noqa: E402


MISSING = "__missing__"
VALID_DATASETS = {"dev5", "facebook50", "all50"}
TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".parquet", ".feather"}
CAMPAIGN_COLUMNS = (
    "campaign",
    "campaign_id",
    "campaignid",
    "campaignId",
    "Campaign",
    "CampaignID",
)
TARGET_COLUMNS = ("Attack_type", "attack_type")


# Pipeline configuration constants. Edit these values to change this script.
CONFIG_ENV_FILE = '.env'
CONFIG_RAW_ROOT = 'data/raw'
CONFIG_EXTRACTED_ROOT = 'data/extracted_features'
CONFIG_OUTPUT_ROOT = 'data/selected_features/selectKBest'
CONFIG_K = 50
CONFIG_APPROACHES = ['label_encoder', 'semantic_headers', 'tf_idf', 'sentence_transformer']


def parse_args() -> argparse.Namespace:
    """Return script configuration from global constants; command-line args are intentionally ignored."""
    return argparse.Namespace(
        env_file=CONFIG_ENV_FILE,
        raw_root=CONFIG_RAW_ROOT,
        extracted_root=CONFIG_EXTRACTED_ROOT,
        output_root=CONFIG_OUTPUT_ROOT,
        k=CONFIG_K,
        approaches=CONFIG_APPROACHES,
    )

def read_env_value(env_file: Path, key: str) -> str:
    if not env_file.exists():
        raise FileNotFoundError(f"Environment file not found: {env_file}")

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip('"').strip("'")

    raise KeyError(f"{key} was not found in {env_file}")


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {normalize_name(column): column for column in columns}
    for candidate in candidates:
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


def target_from_table(table: pd.DataFrame) -> pd.Series:
    column = find_column(list(table.columns), TARGET_COLUMNS)
    if column is None:
        return binary_target_series([MISSING] * len(table), index=table.index)
    return binary_target_series(table[column], index=table.index)


def load_raw_campaign_and_target(dataset: str, raw_root: Path) -> tuple[np.ndarray, pd.DataFrame]:
    campaigns: list[str] = []
    targets: list[str] = []

    for path, _traffic_source, campaign in raw_files_for_dataset(dataset, raw_root / dataset):
        table = read_table(path)
        if table is None or table.empty:
            continue

        campaign_column = find_column(list(table.columns), CAMPAIGN_COLUMNS)
        if campaign_column is None:
            campaigns.extend([campaign] * len(table))
        else:
            campaigns.extend(table[campaign_column].map(stringify_value).tolist())

        targets.extend(target_from_table(table).reset_index(drop=True).tolist())

    if not campaigns:
        raise ValueError(f"No raw rows found for DATASET={dataset}.")

    return np.asarray(campaigns, dtype=object), pd.DataFrame({"attack_type": targets})


def load_label_encoder_features(dataset: str, extracted_root: Path, raw_root: Path) -> tuple[np.ndarray, pd.DataFrame, dict]:
    artifact_path = extracted_root / "label_encoder" / dataset / "campaign_label_encoder.json"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing label encoder artifact: {artifact_path}")

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    mapping = {str(key): int(value) for key, value in artifact["mapping"].items()}
    campaigns, target = load_raw_campaign_and_target(dataset, raw_root)
    encoded = np.asarray([mapping.get(str(campaign), -1) for campaign in campaigns], dtype=np.float32).reshape(-1, 1)
    metadata = {
        "source_artifact": str(artifact_path),
        "input_type": "dense",
        "input_shape": list(encoded.shape),
        "unknown_campaign_value": -1,
    }
    return encoded, target, metadata


def load_semantic_headers_features(dataset: str, extracted_root: Path) -> tuple[np.ndarray, pd.DataFrame, dict]:
    approach_path = extracted_root / "semantic_headers" / dataset
    features_path = approach_path / "semantic_headers.parquet"
    target_path = approach_path / "target.parquet"
    if not features_path.exists() or not target_path.exists():
        raise FileNotFoundError(f"Missing semantic_headers artifacts under {approach_path}")

    features = pd.read_parquet(features_path)
    target = binary_target_frame(pd.read_parquet(target_path))
    matrix = features.to_numpy(dtype=np.float32)
    metadata = {
        "source_artifact": str(features_path),
        "input_type": "dense",
        "input_shape": list(matrix.shape),
    }
    return matrix, target, metadata


def load_tfidf_features(dataset: str, extracted_root: Path) -> tuple[sparse.spmatrix, pd.DataFrame, dict]:
    approach_path = extracted_root / "tf_idf" / dataset
    matrix_path = approach_path / "tf_idf_matrix.npz"
    target_path = approach_path / "target.parquet"
    if not matrix_path.exists() or not target_path.exists():
        raise FileNotFoundError(f"Missing tf_idf artifacts under {approach_path}")

    matrix = sparse.load_npz(matrix_path).astype(np.float32)
    target = binary_target_frame(pd.read_parquet(target_path))
    metadata = {
        "source_artifact": str(matrix_path),
        "input_type": "sparse",
        "input_shape": list(matrix.shape),
    }
    return matrix, target, metadata


def load_sentence_transformer_features(dataset: str, extracted_root: Path) -> tuple[np.ndarray, pd.DataFrame, dict]:
    approach_path = extracted_root / "sentence_transformer" / dataset
    embeddings_path = approach_path / "embeddings.npy"
    target_path = approach_path / "target.parquet"
    if not embeddings_path.exists() or not target_path.exists():
        raise FileNotFoundError(f"Missing sentence_transformer artifacts under {approach_path}")

    matrix = np.load(embeddings_path).astype(np.float32, copy=False)
    target = binary_target_frame(pd.read_parquet(target_path))
    metadata = {
        "source_artifact": str(embeddings_path),
        "input_type": "dense",
        "input_shape": list(matrix.shape),
    }
    return matrix, target, metadata


def effective_k(n_features: int, requested: int) -> int:
    return max(1, min(requested, n_features))


def remove_constant_features(matrix) -> tuple[Any, np.ndarray, dict]:
    selector = VarianceThreshold(threshold=0.0)
    filtered = selector.fit_transform(matrix)
    kept_indices = selector.get_support(indices=True).astype(int)
    if kept_indices.size == 0:
        raise ValueError("all input features are constant")
    return filtered, kept_indices, {
        "method": "sklearn.feature_selection.VarianceThreshold",
        "threshold": 0.0,
        "original_n_features": int(matrix.shape[1]),
        "kept_n_features": int(filtered.shape[1]),
        "removed_n_features": int(matrix.shape[1] - filtered.shape[1]),
        "kept_feature_indices": kept_indices.tolist(),
        "reason": "remove_constant_features_before_f_classif",
    }


def select_features(matrix, target: pd.DataFrame, requested_k: int) -> tuple[np.ndarray, SelectKBest | None, dict]:
    matrix = matrix.astype(np.float32) if sparse.issparse(matrix) else np.asarray(matrix, dtype=np.float32)
    matrix, kept_indices, preprocessing_metadata = remove_constant_features(matrix)
    k = effective_k(matrix.shape[1], requested_k)

    if matrix.shape[1] <= k:
        output = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
        return output.astype(np.float32, copy=False), None, {
            "method": "identity",
            "reason": "input_features_less_than_or_equal_to_requested_k",
            "k": int(matrix.shape[1]),
            "preprocessing": preprocessing_metadata,
        }

    labels = target["attack_type"].fillna(MISSING).astype(str)
    if labels.nunique() < 2:
        output = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
        return output.astype(np.float32, copy=False), None, {
            "method": "identity",
            "reason": "target_has_less_than_two_classes",
            "k": int(matrix.shape[1]),
            "preprocessing": preprocessing_metadata,
        }

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    selector = SelectKBest(score_func=f_classif, k=k)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.feature_selection")
        selected = selector.fit_transform(matrix, y)
    if sparse.issparse(selected):
        selected = selected.toarray()

    scores = selector.scores_
    pvalues = selector.pvalues_
    selected_indices = kept_indices[selector.get_support(indices=True)]

    return selected.astype(np.float32, copy=False), selector, {
        "method": "sklearn.feature_selection.SelectKBest",
        "score_func": "sklearn.feature_selection.f_classif",
        "k": int(k),
        "selected_feature_indices": selected_indices.astype(int).tolist(),
        "scores": np.nan_to_num(scores, nan=-1.0).astype(float).tolist(),
        "pvalues": np.nan_to_num(pvalues, nan=-1.0).astype(float).tolist(),
        "target_classes": label_encoder.classes_.tolist(),
        "preprocessing": preprocessing_metadata,
    }


def save_reduced_features(
    approach: str,
    dataset: str,
    features: np.ndarray,
    target: pd.DataFrame,
    output_root: Path,
    input_metadata: dict,
    reduction_metadata: dict,
) -> None:
    if len(features) != len(target):
        raise ValueError(
            f"Feature/target row mismatch for {approach}: features={len(features)}, target={len(target)}"
        )

    output_path = output_root / approach / dataset
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "features.npy", features.astype(np.float32, copy=False))
    target.to_parquet(output_path / "target.parquet", index=False)

    metadata = {
        "approach": approach,
        "dataset": dataset,
        "features": "features.npy",
        "target": "target.parquet",
        "output_shape": list(features.shape),
        "dtype": "float32",
        "input": input_metadata,
        "reduction": reduction_metadata,
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def load_approach(approach: str, dataset: str, extracted_root: Path, raw_root: Path):
    if approach == "label_encoder":
        return load_label_encoder_features(dataset, extracted_root, raw_root)
    if approach == "semantic_headers":
        return load_semantic_headers_features(dataset, extracted_root)
    if approach == "tf_idf":
        return load_tfidf_features(dataset, extracted_root)
    if approach == "sentence_transformer":
        return load_sentence_transformer_features(dataset, extracted_root)
    raise ValueError(f"Unknown approach: {approach}")


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    dataset = read_env_value(project_root / args.env_file, "DATASET")

    if dataset not in VALID_DATASETS:
        valid = ", ".join(sorted(VALID_DATASETS))
        raise ValueError(f"Invalid DATASET={dataset!r}. Expected one of: {valid}")

    extracted_root = project_root / args.extracted_root
    raw_root = project_root / args.raw_root
    output_root = project_root / args.output_root

    completed: list[str] = []
    skipped: list[str] = []

    for approach in args.approaches:
        try:
            matrix, target, input_metadata = load_approach(approach, dataset, extracted_root, raw_root)
            if sparse.issparse(matrix):
                reduced, _selector, reduction_metadata = select_features(matrix, target, args.k)
            else:
                reduced, _selector, reduction_metadata = select_features(np.asarray(matrix), target, args.k)

            save_reduced_features(
                approach=approach,
                dataset=dataset,
                features=reduced,
                target=target,
                output_root=output_root,
                input_metadata=input_metadata,
                reduction_metadata=reduction_metadata,
            )
            completed.append(approach)
            print(f"Saved SelectKBest features for {approach} to {output_root / approach / dataset}")
        except FileNotFoundError as exc:
            skipped.append(approach)
            print(f"WARNING: skipping {approach}: {exc}", file=sys.stderr)

    if not completed:
        skipped_text = ", ".join(skipped) if skipped else "none"
        raise ValueError(f"No SelectKBest outputs were created. Skipped approaches: {skipped_text}")

    if skipped:
        print(f"Completed {len(completed)} approach(es); skipped: {', '.join(skipped)}")
    else:
        print(f"Completed SelectKBest for all {len(completed)} approach(es).")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
