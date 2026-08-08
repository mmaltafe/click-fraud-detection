#!/usr/bin/env python3
"""
Apply TruncatedSVD dimensionality reduction to each extracted feature approach.

Inputs:
    data/extracted_features/label_encoder/{DATASET}/campaign_label_encoder.json
    data/extracted_features/semantic_headers/{DATASET}/semantic_headers.parquet
    data/extracted_features/tf_idf/{DATASET}/tf_idf_matrix.npz
    data/extracted_features/sentence_transformer/{DATASET}/embeddings.npy

Outputs:
    data/selected_features/truncatedSVD/{APPROACH}/{DATASET}/features.npy
    data/selected_features/truncatedSVD/{APPROACH}/{DATASET}/target.parquet
    data/selected_features/truncatedSVD/{APPROACH}/{DATASET}/metadata.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils._env import VALID_DATASETS, read_env_value  # noqa: E402
from utils._selection_inputs import load_approach, save_reduced_features  # noqa: E402


# Pipeline configuration constants. Edit these values to change this script.
CONFIG_ENV_FILE = '.env'
CONFIG_RAW_ROOT = 'data/raw'
CONFIG_EXTRACTED_ROOT = 'data/extracted_features'
CONFIG_OUTPUT_ROOT = 'data/selected_features/truncatedSVD'
CONFIG_N_COMPONENTS = 50
CONFIG_APPROACHES = ['label_encoder', 'semantic_headers', 'tf_idf', 'sentence_transformer']


def parse_args() -> argparse.Namespace:
    """Return script configuration from global constants; command-line args are intentionally ignored."""
    return argparse.Namespace(
        env_file=CONFIG_ENV_FILE,
        raw_root=CONFIG_RAW_ROOT,
        extracted_root=CONFIG_EXTRACTED_ROOT,
        output_root=CONFIG_OUTPUT_ROOT,
        n_components=CONFIG_N_COMPONENTS,
        approaches=CONFIG_APPROACHES,
    )


def effective_components(matrix_shape: tuple[int, int], requested: int) -> int:
    n_samples, n_features = matrix_shape
    if n_features <= 1:
        return 1
    return max(1, min(requested, n_samples, n_features - 1))


def preprocessing_for_svd(matrix) -> tuple[Any, dict]:
    if sparse.issparse(matrix):
        # TruncatedSVD is commonly used for sparse TF-IDF/LSA without centering.
        # Scaling by standard deviation preserves sparsity and avoids densifying.
        scaler = StandardScaler(with_mean=False)
        scaled = scaler.fit_transform(matrix).astype(np.float32)
        return scaled, {
            "method": "sklearn.preprocessing.StandardScaler",
            "with_mean": False,
            "with_std": True,
            "reason": "preserve_sparse_input_for_truncated_svd",
            "n_features_in": int(getattr(scaler, "n_features_in_", 0)),
        }

    scaler = StandardScaler()
    scaled = scaler.fit_transform(np.asarray(matrix, dtype=np.float32)).astype(np.float32, copy=False)
    return scaled, {
        "method": "sklearn.preprocessing.StandardScaler",
        "with_mean": True,
        "with_std": True,
        "reason": "standardize_dense_features_before_svd",
        "n_features_in": int(getattr(scaler, "n_features_in_", 0)),
    }


def reduce_matrix(matrix, requested_components: int) -> tuple[np.ndarray, TruncatedSVD | None, dict]:
    matrix, preprocessing_metadata = preprocessing_for_svd(matrix)
    n_components = effective_components(matrix.shape, requested_components)
    if matrix.shape[1] <= n_components:
        output = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
        return output.astype(np.float32, copy=False), None, {
            "method": "identity",
            "reason": "input_features_less_than_or_equal_to_requested_components",
            "n_components": int(matrix.shape[1]),
            "preprocessing": preprocessing_metadata,
        }

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(matrix).astype(np.float32, copy=False)
    return reduced, svd, {
        "method": "sklearn.decomposition.TruncatedSVD",
        "n_components": int(n_components),
        "preprocessing": preprocessing_metadata,
        "explained_variance_ratio": svd.explained_variance_ratio_.astype(float).tolist(),
        "total_explained_variance_ratio": float(svd.explained_variance_ratio_.sum()),
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
                reduced, _svd, reduction_metadata = reduce_matrix(matrix, args.n_components)
            else:
                reduced, _svd, reduction_metadata = reduce_matrix(np.asarray(matrix), args.n_components)

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
            print(f"Saved TruncatedSVD features for {approach} to {output_root / approach / dataset}")
        except FileNotFoundError as exc:
            skipped.append(approach)
            print(f"WARNING: skipping {approach}: {exc}", file=sys.stderr)

    if not completed:
        skipped_text = ", ".join(skipped) if skipped else "none"
        raise ValueError(f"No TruncatedSVD outputs were created. Skipped approaches: {skipped_text}")

    if skipped:
        print(f"Completed {len(completed)} approach(es); skipped: {', '.join(skipped)}")
    else:
        print(f"Completed TruncatedSVD for all {len(completed)} approach(es).")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
