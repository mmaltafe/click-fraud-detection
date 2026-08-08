#!/usr/bin/env python3
"""
Evaluate all generated feature datasets with local TabPFN inference.

This script uses the local `tabpfn` package and avoids `tabpfn-client`/API
calls. On first use, TabPFN may download model weights and may require local
license authentication depending on the checkpoint/version installed.

Inputs:
    data/extracted_features/{approach}/{DATASET}
    data/selected_features/{selector}/{approach}/{DATASET}

Output:
    results/machine_learning_evaluation/tabpfn/{DATASET}/results.csv
    results/machine_learning_evaluation/tabpfn/{DATASET}/results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.target_utils import binary_target_frame  # noqa: E402

from utils._campaigns import (
    add_campaign_cv_args,
    aggregate_campaign_fold_results,
    campaign_indices,
    campaign_kfold_splits,
    cap_rows_by_campaign,
    ensure_campaign_row_index,
    subset_feature_dataset,
)
from utils._resume import add_resume_metadata, base_config, completed_keys, config_hash, load_existing_results, result_key, save_results


MISSING = "__missing__"
VALID_DATASETS = {"dev5", "facebook50", "all50"}
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


# Pipeline configuration constants. Edit these values to change this script.
CONFIG_ENV_FILE = '.env'
CONFIG_EXTRACTED_ROOT = 'data/extracted_features'
CONFIG_SELECTED_ROOT = 'data/selected_features'
CONFIG_OUTPUT_ROOT = 'results/machine_learning_evaluation/tabpfn'
CONFIG_RAW_ROOT = 'data/raw'
CONFIG_K_FOLDS = 5
CONFIG_MODEL_CACHE = 'models/tabpfn'
CONFIG_MODEL_PATH = None
CONFIG_ALLOW_BROWSER_LOGIN = False
CONFIG_DEVICE = 'auto'
CONFIG_IGNORE_PRETRAINING_LIMITS = False
CONFIG_ALLOW_CPU_LARGE_DATASET = False
CONFIG_MAX_CPU_TRAIN_ROWS = 1000
CONFIG_MAX_CPU_TEST_ROWS = 2000
CONFIG_N_ESTIMATORS = 1
CONFIG_TEST_SIZE = 0.3
CONFIG_RANDOM_STATE = 42
CONFIG_MAX_ROWS = 0
CONFIG_MAX_DENSE_CELLS = 10000000
CONFIG_RETRY_ERRORS = True


def parse_args() -> argparse.Namespace:
    """Return script configuration from global constants; command-line args are intentionally ignored."""
    return argparse.Namespace(
        env_file=CONFIG_ENV_FILE,
        extracted_root=CONFIG_EXTRACTED_ROOT,
        selected_root=CONFIG_SELECTED_ROOT,
        output_root=CONFIG_OUTPUT_ROOT,
        raw_root=CONFIG_RAW_ROOT,
        k_folds=CONFIG_K_FOLDS,
        model_cache=CONFIG_MODEL_CACHE,
        model_path=CONFIG_MODEL_PATH,
        allow_browser_login=CONFIG_ALLOW_BROWSER_LOGIN,
        device=CONFIG_DEVICE,
        ignore_pretraining_limits=CONFIG_IGNORE_PRETRAINING_LIMITS,
        allow_cpu_large_dataset=CONFIG_ALLOW_CPU_LARGE_DATASET,
        max_cpu_train_rows=CONFIG_MAX_CPU_TRAIN_ROWS,
        max_cpu_test_rows=CONFIG_MAX_CPU_TEST_ROWS,
        n_estimators=CONFIG_N_ESTIMATORS,
        test_size=CONFIG_TEST_SIZE,
        random_state=CONFIG_RANDOM_STATE,
        max_rows=CONFIG_MAX_ROWS,
        max_dense_cells=CONFIG_MAX_DENSE_CELLS,
        retry_errors=CONFIG_RETRY_ERRORS,
    )

def read_env_file(env_file: Path) -> dict[str, str]:
    if not env_file.exists():
        raise FileNotFoundError(f"Environment file not found: {env_file}")

    values: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def read_env_value(env_file: Path, key: str) -> str:
    values = read_env_file(env_file)
    if key in values:
        return values[key]
    raise KeyError(f"{key} was not found in {env_file}")


def configure_tabpfn_environment(project_root: Path, args: argparse.Namespace) -> None:
    model_cache = project_root / args.model_cache
    model_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TABPFN_MODEL_CACHE_DIR"] = str(model_cache)
    if args.allow_cpu_large_dataset:
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")
    else:
        os.environ.pop("TABPFN_ALLOW_CPU_LARGE_DATASET", None)

    env_values = read_env_file(project_root / args.env_file)
    if env_values.get("TABPFN_TOKEN"):
        os.environ.setdefault("TABPFN_TOKEN", env_values["TABPFN_TOKEN"])

    if args.allow_browser_login:
        os.environ.pop("TABPFN_NO_BROWSER", None)
    else:
        os.environ.setdefault("TABPFN_NO_BROWSER", "1")


def tabpfn_dependency_error() -> str | None:
    try:
        from tabpfn import TabPFNClassifier  # noqa: F401
    except ModuleNotFoundError:
        return (
            "tabpfn is not installed in the active Python environment. "
            "Install project requirements or run with the project virtualenv."
        )
    except Exception as exc:
        return f"TabPFN could not be imported: {exc}"
    return None


def tabpfn_device_error(args: argparse.Namespace) -> str | None:
    if args.device and args.device.lower().startswith("cuda") and not cuda_is_available():
        return (
            f"TabPFN was configured to use --device {args.device}, but torch.cuda.is_available() is False. "
            "Run on a machine with CUDA/GPU support, install a CUDA-enabled PyTorch build, or pass "
            "--device cpu --max-rows 1000 for a small local CPU test."
        )
    return None


def resolve_model_path(project_root: Path, args: argparse.Namespace) -> str:
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.is_absolute():
            model_path = project_root / model_path
        return str(model_path)

    model_cache = project_root / args.model_cache
    checkpoints = sorted(model_cache.glob("*.ckpt"))
    if checkpoints:
        return str(checkpoints[0])
    return "auto"


def classifier_factories(
    random_state: int,
    device: str | None,
    ignore_pretraining_limits: bool,
    model_path: str,
    n_estimators: int,
):
    return [
        (
            "TabPFN",
            lambda: make_tabpfn_classifier(
                random_state=random_state,
                device=device,
                ignore_pretraining_limits=ignore_pretraining_limits,
                model_path=model_path,
                n_estimators=n_estimators,
            ),
        )
    ]


def make_tabpfn_classifier(
    random_state: int,
    device: str | None,
    ignore_pretraining_limits: bool,
    model_path: str,
    n_estimators: int,
):
    from tabpfn import TabPFNClassifier

    # Local TabPFN usage following the official README:
    #   from tabpfn import TabPFNClassifier
    #   clf = TabPFNClassifier(...)
    #   clf.fit(X_train, y_train)
    # No tabpfn-client or external API call is used here.
    kwargs = {"random_state": random_state, "model_path": model_path, "n_estimators": n_estimators}
    if device:
        kwargs["device"] = device
    if ignore_pretraining_limits:
        kwargs["ignore_pretraining_limits"] = True

    try:
        return TabPFNClassifier(**kwargs)
    except TypeError:
        kwargs.pop("ignore_pretraining_limits", None)
        try:
            return TabPFNClassifier(**kwargs)
        except TypeError:
            kwargs.pop("random_state", None)
            return TabPFNClassifier(**kwargs)


def read_metadata(path: Path) -> dict:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def metadata_row_index(path: Path) -> pd.DataFrame | None:
    metadata = read_metadata(path)
    row_index = metadata.get("row_index")
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


def cap_rows(feature_dataset: FeatureDataset, max_rows: int, random_state: int) -> FeatureDataset:
    if max_rows <= 0 or len(feature_dataset.target) <= max_rows:
        return feature_dataset

    y = feature_dataset.target["attack_type"].astype(str)
    stratify = y if y.value_counts().min() >= 2 else None
    _, sample_idx = train_test_split(
        np.arange(len(y)),
        test_size=max_rows,
        random_state=random_state,
        stratify=stratify,
    )
    sample_idx = np.sort(sample_idx)
    X = feature_dataset.X[sample_idx]
    target = feature_dataset.target.iloc[sample_idx].reset_index(drop=True)
    row_index = (
        feature_dataset.row_index.iloc[sample_idx].reset_index(drop=True)
        if feature_dataset.row_index is not None and len(feature_dataset.row_index) == len(y)
        else feature_dataset.row_index
    )
    return FeatureDataset(
        feature_dataset.feature_stage,
        feature_dataset.feature_selection,
        feature_dataset.feature_approach,
        feature_dataset.path,
        X,
        target,
        row_index,
    )


def needs_dense(classifier_name: str) -> bool:
    return classifier_name in {"TabPFN"}


def maybe_dense(X, max_dense_cells: int = 10_000_000):
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float32, copy=False)
    if isinstance(X, pd.Series):
        return X.to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)
    if not sparse.issparse(X):
        return np.asarray(X, dtype=np.float32)
    cells = X.shape[0] * X.shape[1]
    if max_dense_cells > 0 and cells > max_dense_cells:
        raise MemoryError(f"Refusing to densify sparse matrix with {cells} cells")
    return X.toarray().astype(np.float32, copy=False)


def estimated_train_rows(n_samples: int, test_size: float) -> int:
    return int(math.ceil(n_samples * (1.0 - test_size)))


def cuda_is_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def effective_cpu_device(device: str | None) -> bool:
    if device:
        normalized = device.lower()
        if normalized == "cpu":
            return True
        if normalized == "auto":
            return not cuda_is_available()
        return False
    return not cuda_is_available()


def classifier_skip_reason(feature_dataset: FeatureDataset, classifier_name: str, args: argparse.Namespace) -> str | None:
    if classifier_name != "TabPFN":
        return None
    if sparse.issparse(feature_dataset.X):
        n_samples = len(feature_dataset.target)
        n_train = estimated_train_rows(n_samples, args.test_size)
        n_test = n_samples - n_train
        if effective_cpu_device(args.device) and not args.allow_cpu_large_dataset:
            if args.max_cpu_train_rows > 0:
                n_train = min(n_train, args.max_cpu_train_rows)
            if args.max_cpu_test_rows > 0:
                n_test = min(n_test, args.max_cpu_test_rows)
        cells = int(max(n_train, n_test) * feature_dataset.X.shape[1])
        if args.max_dense_cells > 0 and cells > args.max_dense_cells:
            return (
                "TabPFN requires dense inputs, but this sparse matrix would expand to "
                f"at least {cells} cells after sampling, exceeding --max-dense-cells={args.max_dense_cells}."
            )
    return None


def downsample_indices(indices: np.ndarray, y: np.ndarray, max_rows: int, random_state: int) -> np.ndarray:
    if max_rows <= 0 or len(indices) <= max_rows:
        return indices
    y_subset = y[indices]
    class_counts = pd.Series(y_subset).value_counts()
    stratify = y_subset if len(class_counts) > 1 and class_counts.min() >= 2 else None
    _, sampled_positions = train_test_split(
        np.arange(len(indices)),
        test_size=max_rows,
        random_state=random_state,
        stratify=stratify,
    )
    return np.sort(indices[sampled_positions])


def stratified_split_indices(y: np.ndarray, test_size: float, random_state: int):
    class_counts = pd.Series(y).value_counts()
    stratify = y if len(class_counts) > 1 and class_counts.min() >= 2 else None
    return train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def communication_cost(X) -> int:
    if sparse.issparse(X):
        return int((X.data.nbytes + X.indices.nbytes + X.indptr.nbytes))
    array = np.asarray(X)
    return int(array.nbytes)


def group_balanced_accuracy(y_true, y_pred, groups: pd.Series | None) -> dict[str, float] | None:
    if groups is None:
        return None
    values: dict[str, float] = {}
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": groups.astype(str).values})
    for group, part in frame.groupby("group"):
        if len(part["y_true"].unique()) < 2:
            values[str(group)] = None
            continue
        values[str(group)] = float(balanced_accuracy_score(part["y_true"], part["y_pred"]))
    return values


def metric_variance(values: dict[str, float] | None) -> float | None:
    if not values:
        return None
    numeric = [value for value in values.values() if value is not None and not math.isnan(value)]
    if len(numeric) < 2:
        return None
    return float(np.var(numeric))


def fairness_gap(values: dict[str, float] | None) -> float | None:
    if not values:
        return None
    numeric = [value for value in values.values() if value is not None and not math.isnan(value)]
    if len(numeric) < 2:
        return None
    return float(max(numeric) - min(numeric))


def extract_groups(feature_dataset: FeatureDataset, test_idx: np.ndarray):
    if feature_dataset.row_index is None or len(feature_dataset.row_index) != len(feature_dataset.target):
        return None, None
    row_index = feature_dataset.row_index.iloc[test_idx].reset_index(drop=True)
    campaign = row_index["campaign"] if "campaign" in row_index.columns else None
    traffic_source = row_index["traffic_source"] if "traffic_source" in row_index.columns else None
    return campaign, traffic_source


def evaluate_classifier(
    feature_dataset: FeatureDataset,
    classifier_name: str,
    classifier,
    args: argparse.Namespace,
) -> dict:
    y_text = feature_dataset.target["attack_type"].fillna(MISSING).astype(str).to_numpy()
    if len(np.unique(y_text)) < 2:
        raise ValueError("target has fewer than two classes")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    train_idx, test_idx = stratified_split_indices(y, args.test_size, args.random_state)
    original_train_rows = int(len(train_idx))
    original_test_rows = int(len(test_idx))
    cpu_limited_run = (
        classifier_name == "TabPFN"
        and effective_cpu_device(args.device)
        and not args.allow_cpu_large_dataset
    )
    if cpu_limited_run:
        train_idx = downsample_indices(train_idx, y, args.max_cpu_train_rows, args.random_state)
        test_idx = downsample_indices(test_idx, y, args.max_cpu_test_rows, args.random_state + 1)

    X_train = feature_dataset.X[train_idx]
    X_test = feature_dataset.X[test_idx]
    if needs_dense(classifier_name):
        X_train = maybe_dense(X_train, max_dense_cells=args.max_dense_cells)
        X_test = maybe_dense(X_test, max_dense_cells=args.max_dense_cells)

    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier.fit(X_train, y[train_idx])
        y_pred = classifier.predict(X_test)
    elapsed = time.perf_counter() - started
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)

    labels = np.arange(len(label_encoder.classes_))
    recall_values = recall_score(y[test_idx], y_pred, labels=labels, average=None, zero_division=0)
    recall_by_class = {
        str(class_name): float(value)
        for class_name, value in zip(label_encoder.classes_, recall_values)
    }

    campaign_groups, traffic_groups = extract_groups(feature_dataset, test_idx)
    campaign_performance = group_balanced_accuracy(y[test_idx], y_pred, campaign_groups)
    traffic_performance = group_balanced_accuracy(y[test_idx], y_pred, traffic_groups)

    return {
        "feature_stage": feature_dataset.feature_stage,
        "feature_selection": feature_dataset.feature_selection,
        "feature_approach": feature_dataset.feature_approach,
        "classifier": classifier_name,
        "n_samples": int(len(y)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_train_original": original_train_rows,
        "n_test_original": original_test_rows,
        "cpu_train_sampled": bool(len(train_idx) < original_train_rows),
        "cpu_test_sampled": bool(len(test_idx) < original_test_rows),
        "n_features": int(feature_dataset.X.shape[1]),
        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], y_pred)),
        "macro_f1": float(f1_score(y[test_idx], y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y[test_idx], y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y[test_idx], y_pred)),
        "recall_by_class": recall_by_class,
        "communication_cost": communication_cost(feature_dataset.X),
        "number_of_rounds": None,
        "client_variance": metric_variance(campaign_performance),
        "fairness_between_campaigns": fairness_gap(campaign_performance),
        "performance_by_traffic_source": traffic_performance,
        "fit_predict_seconds": float(elapsed),
        "status": "ok",
        "error": None,
    }


def evaluate_campaign_classifier(
    feature_dataset: FeatureDataset,
    campaign_id: str,
    classifier_name: str,
    factory,
    args: argparse.Namespace,
) -> dict:
    y_text = feature_dataset.target["attack_type"].fillna(MISSING).astype(str).to_numpy()
    if len(np.unique(y_text)) < 2:
        raise ValueError("campaign target has fewer than two classes")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    splits, split_strategy, effective_folds = campaign_kfold_splits(y, args.k_folds, args.random_state)
    fold_results = []
    cpu_limited_run = (
        classifier_name == "TabPFN"
        and effective_cpu_device(args.device)
        and not args.allow_cpu_large_dataset
    )

    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
        original_train_rows = int(len(train_idx))
        original_test_rows = int(len(test_idx))
        if cpu_limited_run:
            train_idx = downsample_indices(train_idx, y, args.max_cpu_train_rows, args.random_state + fold_number)
            test_idx = downsample_indices(test_idx, y, args.max_cpu_test_rows, args.random_state + 10_000 + fold_number)

        X_train = feature_dataset.X[train_idx]
        X_test = feature_dataset.X[test_idx]
        if needs_dense(classifier_name):
            X_train = maybe_dense(X_train, max_dense_cells=args.max_dense_cells)
            X_test = maybe_dense(X_test, max_dense_cells=args.max_dense_cells)

        classifier = factory()
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier.fit(X_train, y[train_idx])
            y_pred = np.asarray(classifier.predict(X_test)).reshape(-1).astype(int)
        elapsed = time.perf_counter() - started

        labels = np.arange(len(label_encoder.classes_))
        recall_values = recall_score(y[test_idx], y_pred, labels=labels, average=None, zero_division=0)
        fold_results.append(
            {
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_train_original": original_train_rows,
                "n_test_original": original_test_rows,
                "cpu_train_sampled": bool(len(train_idx) < original_train_rows),
                "cpu_test_sampled": bool(len(test_idx) < original_test_rows),
                "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], y_pred)),
                "macro_f1": float(f1_score(y[test_idx], y_pred, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y[test_idx], y_pred, average="weighted", zero_division=0)),
                "mcc": float(matthews_corrcoef(y[test_idx], y_pred)),
                "recall_by_class": {
                    str(class_name): float(value)
                    for class_name, value in zip(label_encoder.classes_, recall_values)
                },
                "communication_cost": communication_cost(feature_dataset.X),
                "number_of_rounds": None,
                "client_variance": None,
                "fairness_between_campaigns": None,
                "performance_by_traffic_source": None,
                "fit_predict_seconds": float(elapsed),
                "status": "ok",
                "error": None,
                "fold": fold_number,
            }
        )

    result = aggregate_campaign_fold_results(
        fold_results,
        feature_dataset,
        campaign_id,
        classifier_name,
    )
    result["cv_strategy"] = split_strategy
    result["k_folds"] = int(effective_folds)
    result["n_train_original"] = int(sum(item.get("n_train_original") or 0 for item in fold_results))
    result["n_test_original"] = int(sum(item.get("n_test_original") or 0 for item in fold_results))
    result["cpu_train_sampled"] = any(bool(item.get("cpu_train_sampled")) for item in fold_results)
    result["cpu_test_sampled"] = any(bool(item.get("cpu_test_sampled")) for item in fold_results)
    return result


def failed_result(feature_dataset: FeatureDataset, classifier_name: str, error: Exception) -> dict:
    return {
        "feature_stage": feature_dataset.feature_stage,
        "feature_selection": feature_dataset.feature_selection,
        "feature_approach": feature_dataset.feature_approach,
        "classifier": classifier_name,
        "evaluation_scope": "campaign",
        "campaign": feature_dataset.row_index["campaign"].iloc[0] if feature_dataset.row_index is not None and "campaign" in feature_dataset.row_index.columns and len(feature_dataset.row_index) else None,
        "n_samples": int(len(feature_dataset.target)),
        "n_train": None,
        "n_test": None,
        "n_train_original": None,
        "n_test_original": None,
        "cpu_train_sampled": None,
        "cpu_test_sampled": None,
        "n_features": int(feature_dataset.X.shape[1]),
        "balanced_accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
        "mcc": None,
        "recall_by_class": None,
        "communication_cost": communication_cost(feature_dataset.X),
        "number_of_rounds": None,
        "client_variance": None,
        "fairness_between_campaigns": None,
        "performance_by_traffic_source": None,
        "fit_predict_seconds": None,
        "status": "error",
        "error": str(error),
    }


def skipped_result(feature_dataset: FeatureDataset, classifier_name: str, reason: str) -> dict:
    return {
        "feature_stage": feature_dataset.feature_stage,
        "feature_selection": feature_dataset.feature_selection,
        "feature_approach": feature_dataset.feature_approach,
        "classifier": classifier_name,
        "evaluation_scope": "campaign",
        "campaign": feature_dataset.row_index["campaign"].iloc[0] if feature_dataset.row_index is not None and "campaign" in feature_dataset.row_index.columns and len(feature_dataset.row_index) else None,
        "n_samples": int(len(feature_dataset.target)),
        "n_train": None,
        "n_test": None,
        "n_train_original": None,
        "n_test_original": None,
        "cpu_train_sampled": None,
        "cpu_test_sampled": None,
        "n_features": int(feature_dataset.X.shape[1]),
        "balanced_accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
        "mcc": None,
        "recall_by_class": None,
        "communication_cost": communication_cost(feature_dataset.X),
        "number_of_rounds": None,
        "client_variance": None,
        "fairness_between_campaigns": None,
        "performance_by_traffic_source": None,
        "fit_predict_seconds": None,
        "status": "skipped",
        "error": reason,
    }


def remove_retryable_errors(results: list[dict], current_config_hash: str, retry_errors: bool) -> list[dict]:
    if not retry_errors:
        return results
    return [
        result
        for result in results
        if not (
            result.get("classifier") == "TabPFN"
            and result.get("status") in {"error", "failed", "skipped"}
        )
    ]


def normalize_for_csv(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = result.copy()
        for key in ("recall_by_class", "performance_by_traffic_source", "campaign_performance", "traffic_source_performance", "fold_statuses"):
            if row.get(key) is not None:
                row[key] = json.dumps(row[key], ensure_ascii=False, separators=(",", ":"))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    dataset = read_env_value(project_root / args.env_file, "DATASET")
    if dataset not in VALID_DATASETS:
        valid = ", ".join(sorted(VALID_DATASETS))
        raise ValueError(f"Invalid DATASET={dataset!r}. Expected one of: {valid}")

    feature_datasets = discover_feature_datasets(
        project_root / args.extracted_root,
        project_root / args.selected_root,
        project_root / args.raw_root,
        dataset,
    )
    if not feature_datasets:
        raise ValueError("No extracted or selected feature datasets were found.")

    output_path = project_root / args.output_root / dataset
    run_config = base_config(
        args,
        exclude={
            "allow_browser_login",
            "model_path",
            "retry_errors",
        },
    )
    run_config["dataset"] = dataset
    current_config_hash = config_hash(run_config)
    results: list[dict] = load_existing_results(output_path)
    results = remove_retryable_errors(results, current_config_hash, args.retry_errors)
    done = completed_keys(results, current_config_hash)
    configure_tabpfn_environment(project_root, args)
    model_path = resolve_model_path(project_root, args)
    dependency_error = tabpfn_dependency_error() or tabpfn_device_error(args)

    factories = classifier_factories(
        args.random_state,
        device=args.device,
        ignore_pretraining_limits=args.ignore_pretraining_limits,
        model_path=model_path,
        n_estimators=args.n_estimators,
    )
    for feature_dataset in feature_datasets:
        feature_dataset = cap_rows_by_campaign(feature_dataset, args.max_rows, args.random_state)
        campaigns = campaign_indices(feature_dataset)
        if not campaigns:
            print(f"{feature_dataset.feature_approach}/{feature_dataset.feature_selection or 'none'}: no campaign index; skipped")
            continue
        for classifier_name, factory in factories:
            for campaign_id, indices in campaigns:
                campaign_dataset = subset_feature_dataset(feature_dataset, indices, campaign_id)
                pending_key = {
                    "config_hash": current_config_hash,
                    "feature_stage": campaign_dataset.feature_stage,
                    "feature_selection": campaign_dataset.feature_selection,
                    "feature_approach": campaign_dataset.feature_approach,
                    "classifier": classifier_name,
                    "federated_algorithm": None,
                    "evaluation_scope": "campaign",
                    "campaign": campaign_id,
                }
                if result_key(pending_key) in done:
                    selector = campaign_dataset.feature_selection or "none"
                    print(f"{campaign_dataset.feature_approach}/{selector}/{campaign_id} + {classifier_name}: skipped")
                    continue
                skip_reason = classifier_skip_reason(campaign_dataset, classifier_name, args)
                if skip_reason:
                    result = skipped_result(campaign_dataset, classifier_name, skip_reason)
                    result["campaign"] = campaign_id
                    result["campaign_id"] = campaign_id
                    result["k_folds"] = args.k_folds
                elif dependency_error:
                    result = failed_result(campaign_dataset, classifier_name, RuntimeError(dependency_error))
                    result["campaign"] = campaign_id
                    result["campaign_id"] = campaign_id
                    result["k_folds"] = args.k_folds
                else:
                    try:
                        result = evaluate_campaign_classifier(
                            campaign_dataset,
                            campaign_id,
                            classifier_name,
                            factory,
                            args,
                        )
                    except Exception as exc:
                        result = failed_result(campaign_dataset, classifier_name, exc)
                        result["campaign"] = campaign_id
                        result["campaign_id"] = campaign_id
                        result["k_folds"] = args.k_folds
                result = add_resume_metadata(result, run_config, pending_key)
                results.append(result)
                done.add(result_key(result))
                save_results(output_path, results, normalize_for_csv)
                selector = campaign_dataset.feature_selection or "none"
                print(f"{campaign_dataset.feature_approach}/{selector}/{campaign_id} + {classifier_name}: {result['status']}")

    save_results(output_path, results, normalize_for_csv)
    print(f"Saved TabPFN evaluation results to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise
