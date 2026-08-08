from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from utils._campaigns import ensure_campaign_row_index
from utils.target_utils import binary_target_frame


EXTRACTED_APPROACHES = ("semantic_headers", "tf_idf", "sentence_transformer")
SELECTED_METHODS = ("pca", "truncatedSVD", "chi2", "selectKBest")
SELECTED_APPROACHES = ("label_encoder", "semantic_headers", "tf_idf", "sentence_transformer")


@dataclass
class FeatureDataset:
    feature_stage: str
    feature_selection: str | None
    feature_approach: str
    path: Path
    X: Any
    target: pd.DataFrame
    row_index: pd.DataFrame | None


def read_metadata(path: Path) -> dict:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def metadata_row_index(path: Path) -> pd.DataFrame | None:
    row_index = read_metadata(path).get("row_index")
    if not row_index:
        return None
    return pd.DataFrame(row_index)


def load_extracted_dataset(root: Path, dataset: str, approach: str) -> FeatureDataset | None:
    path = root / approach / dataset
    if approach == "semantic_headers":
        features_path = path / "semantic_headers.parquet"
        target_path = path / "target.parquet"
        if not features_path.exists() or not target_path.exists():
            return None
        X = pd.read_parquet(features_path).to_numpy(dtype=np.float32)
        target = binary_target_frame(pd.read_parquet(target_path))
        return FeatureDataset("extracted_features", None, approach, path, X, target, None)

    if approach == "tf_idf":
        features_path = path / "tf_idf_matrix.npz"
        target_path = path / "target.parquet"
        if not features_path.exists() or not target_path.exists():
            return None
        X = sparse.load_npz(features_path).astype(np.float32)
        target = binary_target_frame(pd.read_parquet(target_path))
        return FeatureDataset("extracted_features", None, approach, path, X, target, metadata_row_index(path))

    if approach == "sentence_transformer":
        features_path = path / "embeddings.npy"
        target_path = path / "target.parquet"
        if not features_path.exists() or not target_path.exists():
            return None
        X = np.load(features_path).astype(np.float32, copy=False)
        target = binary_target_frame(pd.read_parquet(target_path))
        return FeatureDataset("extracted_features", None, approach, path, X, target, metadata_row_index(path))

    return None


def load_selected_dataset(root: Path, dataset: str, method: str, approach: str) -> FeatureDataset | None:
    path = root / method / approach / dataset
    features_path = path / "features.npy"
    target_path = path / "target.parquet"
    if not features_path.exists() or not target_path.exists():
        return None
    X = np.load(features_path).astype(np.float32, copy=False)
    target = binary_target_frame(pd.read_parquet(target_path))
    return FeatureDataset("selected_features", method, approach, path, X, target, None)


def discover_feature_datasets(extracted_root: Path, selected_root: Path, raw_root: Path, dataset: str) -> list[FeatureDataset]:
    datasets: list[FeatureDataset] = []
    for approach in EXTRACTED_APPROACHES:
        loaded = load_extracted_dataset(extracted_root, dataset, approach)
        if loaded is not None:
            datasets.append(ensure_campaign_row_index(loaded, dataset, raw_root))

    for method in SELECTED_METHODS:
        for approach in SELECTED_APPROACHES:
            loaded = load_selected_dataset(selected_root, dataset, method, approach)
            if loaded is not None:
                datasets.append(ensure_campaign_row_index(loaded, dataset, raw_root))
    return datasets
