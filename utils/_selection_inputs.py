"""
Load one of the four extracted-feature approaches (label_encoder,
semantic_headers, tf_idf, sentence_transformer) as raw input for a feature
selection/reduction script, and save its reduced output. Shared by the
002_feature_selection/*.py scripts.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from utils._columns import CAMPAIGN_COLUMNS, find_column, read_table, raw_files_for_dataset, stringify_value
from utils._env import MISSING
from utils.target_utils import TARGET_COLUMNS, binary_target_frame, binary_target_series


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
