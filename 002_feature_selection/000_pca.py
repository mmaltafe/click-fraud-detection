#!/usr/bin/env python3
"""
Apply PCA dimensionality reduction to each extracted feature approach.

Inputs:
    data/extracted_features/label_encoder/{DATASET}/campaign_label_encoder.json
    data/extracted_features/semantic_headers/{DATASET}/semantic_headers.parquet
    data/extracted_features/tf_idf/{DATASET}/tf_idf_matrix.npz
    data/extracted_features/sentence_transformer/{DATASET}/embeddings.npy

Outputs:
    data/selected_features/pca/{APPROACH}/{DATASET}/features.npy
    data/selected_features/pca/{APPROACH}/{DATASET}/target.parquet
    data/selected_features/pca/{APPROACH}/{DATASET}/metadata.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import IncrementalPCA, PCA
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
CONFIG_OUTPUT_ROOT = 'data/selected_features/pca'
CONFIG_N_COMPONENTS = 50
CONFIG_BATCH_SIZE = 4096
CONFIG_APPROACHES = ['label_encoder', 'semantic_headers', 'tf_idf', 'sentence_transformer']


def parse_args() -> argparse.Namespace:
    """Return script configuration from global constants; command-line args are intentionally ignored."""
    return argparse.Namespace(
        env_file=CONFIG_ENV_FILE,
        raw_root=CONFIG_RAW_ROOT,
        extracted_root=CONFIG_EXTRACTED_ROOT,
        output_root=CONFIG_OUTPUT_ROOT,
        n_components=CONFIG_N_COMPONENTS,
        batch_size=CONFIG_BATCH_SIZE,
        approaches=CONFIG_APPROACHES,
    )


def effective_components(matrix_shape: tuple[int, int], requested: int) -> int:
    n_samples, n_features = matrix_shape
    return max(1, min(requested, n_samples, n_features))


def scaler_metadata(scaler: StandardScaler, with_mean: bool) -> dict:
    return {
        "method": "sklearn.preprocessing.StandardScaler",
        "with_mean": bool(with_mean),
        "with_std": True,
        "n_features_in": int(getattr(scaler, "n_features_in_", 0)),
    }


def reduce_dense(matrix: np.ndarray, requested_components: int) -> tuple[np.ndarray, PCA | None, dict]:
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(matrix).astype(np.float32, copy=False)
    n_components = effective_components(matrix.shape, requested_components)
    if matrix.shape[1] <= n_components:
        return scaled_matrix, None, {
            "method": "identity",
            "reason": "input_features_less_than_or_equal_to_requested_components",
            "n_components": int(matrix.shape[1]),
            "preprocessing": scaler_metadata(scaler, with_mean=True),
        }

    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(scaled_matrix).astype(np.float32, copy=False)
    return reduced, pca, {
        "method": "sklearn.decomposition.PCA",
        "n_components": int(n_components),
        "preprocessing": scaler_metadata(scaler, with_mean=True),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
        "total_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
    }


def sparse_batches(matrix: sparse.spmatrix, batch_size: int):
    for start in range(0, matrix.shape[0], batch_size):
        end = min(start + batch_size, matrix.shape[0])
        yield start, end, matrix[start:end].toarray().astype(np.float32, copy=False)


def reduce_sparse(matrix: sparse.spmatrix, requested_components: int, batch_size: int) -> tuple[np.ndarray, IncrementalPCA | None, dict]:
    scaler = StandardScaler(with_mean=False)
    scaled_matrix = scaler.fit_transform(matrix).astype(np.float32)
    n_components = effective_components(matrix.shape, requested_components)
    if matrix.shape[1] <= n_components:
        return scaled_matrix.toarray().astype(np.float32, copy=False), None, {
            "method": "identity",
            "reason": "input_features_less_than_or_equal_to_requested_components",
            "n_components": int(matrix.shape[1]),
            "preprocessing": scaler_metadata(scaler, with_mean=False),
        }

    fit_batch_size = max(batch_size, n_components)
    pca = IncrementalPCA(n_components=n_components, batch_size=fit_batch_size)

    for _start, _end, batch in sparse_batches(scaled_matrix, fit_batch_size):
        if len(batch) >= n_components:
            pca.partial_fit(batch)

    reduced_parts = []
    for _start, _end, batch in sparse_batches(scaled_matrix, fit_batch_size):
        reduced_parts.append(pca.transform(batch).astype(np.float32, copy=False))

    reduced = np.vstack(reduced_parts)
    return reduced, pca, {
        "method": "sklearn.decomposition.IncrementalPCA",
        "n_components": int(n_components),
        "batch_size": int(fit_batch_size),
        "preprocessing": scaler_metadata(scaler, with_mean=False),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
        "total_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
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
                reduced, _pca, reduction_metadata = reduce_sparse(matrix, args.n_components, args.batch_size)
            else:
                reduced, _pca, reduction_metadata = reduce_dense(np.asarray(matrix), args.n_components)

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
            print(f"Saved PCA features for {approach} to {output_root / approach / dataset}")
        except FileNotFoundError as exc:
            skipped.append(approach)
            print(f"WARNING: skipping {approach}: {exc}", file=sys.stderr)

    if not completed:
        skipped_text = ", ".join(skipped) if skipped else "none"
        raise ValueError(f"No PCA outputs were created. Skipped approaches: {skipped_text}")

    if skipped:
        print(f"Completed {len(completed)} approach(es); skipped: {', '.join(skipped)}")
    else:
        print(f"Completed PCA for all {len(completed)} approach(es).")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
