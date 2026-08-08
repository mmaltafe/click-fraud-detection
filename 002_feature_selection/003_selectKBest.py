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
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils._env import VALID_DATASETS, MISSING, read_env_value  # noqa: E402
from utils._selection_inputs import load_approach, save_reduced_features  # noqa: E402


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
