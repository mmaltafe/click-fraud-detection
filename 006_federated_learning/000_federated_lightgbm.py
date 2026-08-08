#!/usr/bin/env python3
"""
Evaluate TabPFN global embeddings with a federated LightGBM meta-classifier.

Each campaign is treated as a federated client. For each campaign-aligned fold,
a single global TabPFN representation is built for all campaigns, using only the
fold training context. Clients then train local LightGBM models on those fixed
embeddings. The federated part is therefore only the LightGBM meta-classifier in
front of TabPFN embeddings; TabPFN weights are not federated.

The script first reads the best pipeline and LightGBM hyperparameters found by
`005_tabpfn/004_embeddings_lightgbm_grid_search.py`. It then uses that source
feature representation to produce global TabPFN embeddings and trains one local
LightGBM meta-classifier per campaign/client in each fold.

This is a practical Flower-style simulation inspired by FederBoost's horizontal
GBDT setting: GBDT training can be federated through lightweight aggregation of
intermediate tree statistics rather than centralized raw-data pooling. Here we
use a deployable approximation for this project: local LightGBM clients plus
server-side probability aggregation.

References used for the design:
    - FederBoost: Private Federated Learning for GBDT, arXiv:2011.02796v4.
    - IEEE Xplore document 11004389, provided as an additional reference for
      federated learning / LightGBM-style evaluation context.

Outputs:
    results/machine_learning_evaluation/federated_lightgbm/{DATASET}/results.csv
    results/machine_learning_evaluation/federated_lightgbm/{DATASET}/results.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ML_EVALUATION_DIR = PROJECT_ROOT / "003_machine_learning_evaluation"
if str(ML_EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(ML_EVALUATION_DIR))

from utils._env import VALID_DATASETS, MISSING, read_env_value  # noqa: E402
from utils._campaigns import (  # noqa: E402
    campaign_indices,
    campaign_kfold_splits,
    cap_rows_by_campaign,
    ensure_campaign_row_index,
    subset_feature_dataset,
    summarize_dicts,
    summarize_numeric,
    summarize_numeric_std,
)
from utils._fairness_metrics import fairness_gap, group_balanced_accuracy, metric_variance  # noqa: E402
from utils._feature_datasets import (  # noqa: E402
    EXTRACTED_APPROACHES,
    SELECTED_METHODS,
    SELECTED_APPROACHES,
    FeatureDataset,
)
from utils._resume import (  # noqa: E402
    add_resume_metadata,
    base_config,
    completed_keys,
    config_hash,
    load_existing_results,
    result_key,
    save_results,
)


CLASSIFIER_NAME = "Federated-LightGBM"
FEDERATED_ALGORITHM = "GlobalTabPFNEmbeddingsLocalLightGBMWeightedEnsemble"


# Pipeline configuration constants. Edit these values to change this script.
CONFIG_ENV_FILE = ".env"
CONFIG_EXTRACTED_ROOT = "data/extracted_features"
CONFIG_SELECTED_ROOT = "data/selected_features"
CONFIG_RAW_ROOT = "data/raw"
CONFIG_OUTPUT_ROOT = "results/machine_learning_evaluation/federated_lightgbm"
CONFIG_BAYESIAN_RESULTS_ROOT = "results/grid_search/tabpfn_embeddings_lightgbm"
CONFIG_USE_BEST_BAYESIAN_OPTIMIZATION = True
CONFIG_K_FOLDS = 5
CONFIG_RANDOM_STATE = 42
CONFIG_MAX_ROWS = 0
CONFIG_MIN_CLIENTS = 2
CONFIG_MAX_CLIENTS = 0
CONFIG_ROUNDS = 5
CONFIG_LOCAL_TREES_PER_ROUND = 0  # 0 derives max(1, best_n_estimators // rounds)
CONFIG_N_ESTIMATORS = 100
CONFIG_LEARNING_RATE = 0.1
CONFIG_NUM_LEAVES = 31
CONFIG_MAX_DEPTH = -1
CONFIG_MIN_CHILD_SAMPLES = 20
CONFIG_SUBSAMPLE = 1.0
CONFIG_COLSAMPLE_BYTREE = 1.0
CONFIG_REG_ALPHA = 0.0
CONFIG_REG_LAMBDA = 0.0
CONFIG_N_JOBS = 1
CONFIG_WEIGHTING = "examples"  # examples or uniform
CONFIG_PREDICTION_THRESHOLD = 0.5
CONFIG_MAX_DENSE_CELLS = 20_000_000
CONFIG_USE_TABPFN_GLOBAL_EMBEDDINGS = True
CONFIG_MODEL_CACHE = "models/tabpfn"
CONFIG_MODEL_PATH = None
CONFIG_DEVICE = "auto"
CONFIG_IGNORE_PRETRAINING_LIMITS = True
CONFIG_TABPFN_N_ESTIMATORS = 8
CONFIG_ALLOW_BROWSER_LOGIN = False
CONFIG_ALLOW_CPU_LARGE_DATASET = False
CONFIG_MAX_CPU_EMBEDDING_TRAIN_ROWS = 2000
CONFIG_TABPFN_EMBEDDING_CACHE = "data/tabpfn"
CONFIG_USE_TABPFN_EMBEDDING_CACHE = True
CONFIG_SAVE_CAMPAIGN_EMBEDDINGS = True


def parse_args() -> argparse.Namespace:
    """Return script configuration from global constants; command-line args are intentionally ignored."""
    return argparse.Namespace(
        env_file=CONFIG_ENV_FILE,
        extracted_root=CONFIG_EXTRACTED_ROOT,
        selected_root=CONFIG_SELECTED_ROOT,
        raw_root=CONFIG_RAW_ROOT,
        output_root=CONFIG_OUTPUT_ROOT,
        bayesian_results_root=CONFIG_BAYESIAN_RESULTS_ROOT,
        use_best_bayesian_optimization=CONFIG_USE_BEST_BAYESIAN_OPTIMIZATION,
        k_folds=CONFIG_K_FOLDS,
        random_state=CONFIG_RANDOM_STATE,
        max_rows=CONFIG_MAX_ROWS,
        min_clients=CONFIG_MIN_CLIENTS,
        max_clients=CONFIG_MAX_CLIENTS,
        rounds=CONFIG_ROUNDS,
        local_trees_per_round=CONFIG_LOCAL_TREES_PER_ROUND,
        n_estimators=CONFIG_N_ESTIMATORS,
        learning_rate=CONFIG_LEARNING_RATE,
        num_leaves=CONFIG_NUM_LEAVES,
        max_depth=CONFIG_MAX_DEPTH,
        min_child_samples=CONFIG_MIN_CHILD_SAMPLES,
        subsample=CONFIG_SUBSAMPLE,
        colsample_bytree=CONFIG_COLSAMPLE_BYTREE,
        reg_alpha=CONFIG_REG_ALPHA,
        reg_lambda=CONFIG_REG_LAMBDA,
        n_jobs=CONFIG_N_JOBS,
        weighting=CONFIG_WEIGHTING,
        prediction_threshold=CONFIG_PREDICTION_THRESHOLD,
        max_dense_cells=CONFIG_MAX_DENSE_CELLS,
        use_tabpfn_global_embeddings=CONFIG_USE_TABPFN_GLOBAL_EMBEDDINGS,
        model_cache=CONFIG_MODEL_CACHE,
        model_path=CONFIG_MODEL_PATH,
        device=CONFIG_DEVICE,
        ignore_pretraining_limits=CONFIG_IGNORE_PRETRAINING_LIMITS,
        tabpfn_n_estimators=CONFIG_TABPFN_N_ESTIMATORS,
        allow_browser_login=CONFIG_ALLOW_BROWSER_LOGIN,
        allow_cpu_large_dataset=CONFIG_ALLOW_CPU_LARGE_DATASET,
        max_cpu_embedding_train_rows=CONFIG_MAX_CPU_EMBEDDING_TRAIN_ROWS,
        tabpfn_embedding_cache=CONFIG_TABPFN_EMBEDDING_CACHE,
        use_tabpfn_embedding_cache=CONFIG_USE_TABPFN_EMBEDDING_CACHE,
        save_campaign_embeddings=CONFIG_SAVE_CAMPAIGN_EMBEDDINGS,
    )


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def discover_feature_datasets(extracted_root: Path, selected_root: Path, raw_root: Path, dataset: str) -> list[FeatureDataset]:
    boosting = load_module(PROJECT_ROOT / "003_machine_learning_evaluation" / "001_boosting_algorithms.py", "boosting_loaders_for_fed_lgbm")
    datasets: list[FeatureDataset] = []
    for approach in EXTRACTED_APPROACHES:
        loaded = boosting.load_extracted_dataset(extracted_root, dataset, approach)
        if loaded is not None:
            datasets.append(ensure_campaign_row_index(loaded, dataset, raw_root))

    for method in SELECTED_METHODS:
        for approach in SELECTED_APPROACHES:
            loaded = boosting.load_selected_dataset(selected_root, dataset, method, approach)
            if loaded is not None:
                datasets.append(ensure_campaign_row_index(loaded, dataset, raw_root))
    return datasets


def parse_json_value(value, default):
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def normalize_selection(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "<na>", "null"}:
        return None
    return text


def pipeline_from_result(row: dict) -> tuple[str, str | None, str]:
    pipeline = row.get("bayesian_feature_pipeline")
    if pipeline and str(pipeline).lower() not in {"none", "nan", "<na>"}:
        parts = str(pipeline).split("|")
        if len(parts) == 3:
            return parts[0], normalize_selection(parts[1]), parts[2]
    return (
        str(row.get("feature_stage")),
        normalize_selection(row.get("feature_selection")),
        str(row.get("feature_approach")),
    )


def load_best_bayesian_configuration(project_root: Path, args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    results_path = project_root / args.bayesian_results_root / dataset / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Bayesian optimization results not found: {results_path}. "
            "Run 005_tabpfn/005_embeddings_lightgbm_grid_search.py first."
        )
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    candidates = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        params = parse_json_value(row.get("lightgbm_params"), {})
        if not params:
            continue
        objective = row.get("trial_campaign_balanced_accuracy_mean")
        if objective is None or (isinstance(objective, float) and math.isnan(objective)):
            objective = row.get("balanced_accuracy")
        if objective is None:
            continue
        std = row.get("trial_campaign_balanced_accuracy_std")
        if std is None or (isinstance(std, float) and math.isnan(std)):
            std = float("inf")
        stage, selection, approach = pipeline_from_result(row)
        candidates.append(
            {
                "objective": float(objective),
                "std": float(std),
                "feature_stage": stage,
                "feature_selection": selection,
                "feature_approach": approach,
                "lightgbm_params": params,
                "tabpfn_params": parse_json_value(row.get("tabpfn_params"), {}),
                "meta_train_size": row.get("meta_train_size"),
                "bayesian_trial": row.get("bayesian_trial"),
                "bayesian_run_trial": row.get("bayesian_run_trial"),
                "bayesian_study_name": row.get("bayesian_study_name"),
                "bayesian_feature_pipeline": row.get("bayesian_feature_pipeline"),
                "bayesian_search_space_id": row.get("bayesian_search_space_id"),
                "source_results_path": str(results_path),
            }
        )
    if not candidates:
        raise ValueError(f"No successful Bayesian Optimization candidate with LightGBM params was found in {results_path}")
    candidates.sort(key=lambda item: (-item["objective"], item["std"]))
    return candidates[0]


def apply_lightgbm_params_from_bayesian(args: argparse.Namespace, best: dict[str, Any]) -> argparse.Namespace:
    params = best["lightgbm_params"]
    mapping = {
        "n_estimators": "n_estimators",
        "learning_rate": "learning_rate",
        "num_leaves": "num_leaves",
        "max_depth": "max_depth",
        "min_child_samples": "min_child_samples",
        "subsample": "subsample",
        "colsample_bytree": "colsample_bytree",
        "reg_alpha": "reg_alpha",
        "reg_lambda": "reg_lambda",
        "n_jobs": "n_jobs",
    }
    for source, target in mapping.items():
        if source in params and params[source] is not None:
            setattr(args, target, params[source])
    return args


def apply_tabpfn_params_from_bayesian(args: argparse.Namespace, best: dict[str, Any]) -> argparse.Namespace:
    params = best.get("tabpfn_params") or {}
    mapping = {
        "device": "device",
        "ignore_pretraining_limits": "ignore_pretraining_limits",
        "n_estimators": "tabpfn_n_estimators",
        "max_cpu_train_rows": "max_cpu_embedding_train_rows",
    }
    for source, target in mapping.items():
        if source in params and params[source] is not None:
            setattr(args, target, params[source])
    return args


def select_feature_dataset(feature_datasets: list[FeatureDataset], best: dict[str, Any]) -> FeatureDataset:
    for feature_dataset in feature_datasets:
        if (
            feature_dataset.feature_stage == best["feature_stage"]
            and normalize_selection(feature_dataset.feature_selection) == normalize_selection(best["feature_selection"])
            and feature_dataset.feature_approach == best["feature_approach"]
        ):
            return feature_dataset
    available = [
        f"{item.feature_stage}|{item.feature_selection or 'none'}|{item.feature_approach}"
        for item in feature_datasets
    ]
    desired = f"{best['feature_stage']}|{best['feature_selection'] or 'none'}|{best['feature_approach']}"
    raise ValueError(f"Best Bayesian pipeline {desired} was not found. Available: {available}")


def take_rows(X, indices: np.ndarray):
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[indices]
    return X[indices]


def maybe_dense(X, max_dense_cells: int):
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float32, copy=False)
    if isinstance(X, pd.Series):
        return X.to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)
    if not sparse.issparse(X):
        return np.asarray(X, dtype=np.float32)
    cells = X.shape[0] * X.shape[1]
    if cells > max_dense_cells:
        raise MemoryError(f"Refusing to densify sparse matrix with {cells} cells")
    return X.toarray().astype(np.float32, copy=False)


def ensure_2d_embeddings(embeddings: np.ndarray, expected_rows: int) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim > 1 and array.shape[0] != expected_rows:
        matching_axes = [axis for axis, size in enumerate(array.shape) if size == expected_rows]
        if matching_axes:
            array = np.moveaxis(array, matching_axes[0], 0)
        else:
            raise ValueError(
                "Could not align TabPFN embeddings with input rows: "
                f"embedding_shape={array.shape}, expected_rows={expected_rows}"
            )
    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    if array.shape[0] != expected_rows:
        raise ValueError(
            "TabPFN embeddings have incompatible row count after reshape: "
            f"embedding_shape={array.shape}, expected_rows={expected_rows}"
        )
    return array


def tabpfn_embeddings_or_proba(classifier, X: np.ndarray, expected_rows: int) -> tuple[np.ndarray, str]:
    try:
        embeddings = classifier.get_embeddings(X, data_source="test")
        return ensure_2d_embeddings(embeddings, expected_rows), "tabpfn_embeddings"
    except Exception:
        try:
            embeddings = classifier.get_embeddings(X)
            return ensure_2d_embeddings(embeddings, expected_rows), "tabpfn_embeddings_no_data_source"
        except Exception:
            proba = classifier.predict_proba(X)
            return ensure_2d_embeddings(proba, expected_rows), "predict_proba_fallback"


def safe_path_part(value: Any) -> str:
    text = str(value if value is not None else "none")
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in text)


def array_digest(array: np.ndarray) -> str:
    normalized = np.asarray(array, dtype=np.int64)
    return hashlib.sha256(normalized.tobytes()).hexdigest()[:16]


def cache_digest(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]


def tabpfn_embedding_cache_paths(
    source_dataset: FeatureDataset,
    dataset: str,
    fold_number: int,
    train_idx: np.ndarray,
    all_idx: np.ndarray,
    model_path: str,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    pipeline = (
        f"{safe_path_part(source_dataset.feature_stage)}__"
        f"{safe_path_part(source_dataset.feature_selection)}__"
        f"{safe_path_part(source_dataset.feature_approach)}"
    )
    payload = {
        "dataset": dataset,
        "feature_stage": source_dataset.feature_stage,
        "feature_selection": source_dataset.feature_selection,
        "feature_approach": source_dataset.feature_approach,
        "source_path": str(source_dataset.path),
        "source_shape": list(source_dataset.X.shape),
        "target_rows": int(len(source_dataset.target)),
        "fold": int(fold_number + 1),
        "train_idx_hash": array_digest(train_idx),
        "all_idx_hash": array_digest(all_idx),
        "random_state": int(args.random_state),
        "device": args.device,
        "ignore_pretraining_limits": bool(args.ignore_pretraining_limits),
        "tabpfn_n_estimators": int(args.tabpfn_n_estimators),
        "max_cpu_embedding_train_rows": int(args.max_cpu_embedding_train_rows),
        "model_path": model_path,
    }
    digest = cache_digest(payload)
    cache_dir = PROJECT_ROOT / args.tabpfn_embedding_cache / dataset / pipeline
    cache_file = cache_dir / f"fold_{fold_number + 1:02d}_{digest}.npz"
    return cache_file, payload


def load_tabpfn_embedding_cache(cache_file: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    if not cache_file.exists():
        return None
    metadata_file = cache_file.with_suffix(".json")
    metadata = {}
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    with np.load(cache_file, allow_pickle=False) as cached:
        all_idx = cached["all_idx"].astype(int)
        embeddings = cached["embeddings"].astype(np.float32, copy=False)
    return all_idx, embeddings, metadata


def save_tabpfn_embedding_cache(
    cache_file: Path,
    all_idx: np.ndarray,
    embeddings: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, all_idx=all_idx.astype(np.int64), embeddings=embeddings.astype(np.float32, copy=False))
    cache_file.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_campaign_embedding_cache(
    cache_file: Path,
    client_splits: list[tuple[str, np.ndarray, list[tuple[np.ndarray, np.ndarray]], str]],
    fold_number: int,
    all_idx: np.ndarray,
    embeddings: np.ndarray,
) -> None:
    position_by_index = {int(index): position for position, index in enumerate(all_idx)}
    campaign_dir = cache_file.parent / "campaigns" / cache_file.stem
    campaign_dir.mkdir(parents=True, exist_ok=True)
    for campaign_id, global_indices, splits, _split_strategy in client_splits:
        local_train_idx, local_test_idx = splits[fold_number]
        campaign_indices_for_fold = np.unique(
            np.concatenate([global_indices[local_train_idx], global_indices[local_test_idx]])
        ).astype(int)
        positions = np.array(
            [position_by_index[int(index)] for index in campaign_indices_for_fold if int(index) in position_by_index],
            dtype=int,
        )
        if len(positions) == 0:
            continue
        output_file = campaign_dir / f"{safe_path_part(campaign_id)}.npz"
        np.savez_compressed(
            output_file,
            row_indices=campaign_indices_for_fold.astype(np.int64),
            embeddings=embeddings[positions].astype(np.float32, copy=False),
        )


def build_global_tabpfn_embedding_dataset(
    source_dataset: FeatureDataset,
    dataset: str,
    y: np.ndarray,
    client_splits: list[tuple[str, np.ndarray, list[tuple[np.ndarray, np.ndarray]], str]],
    fold_number: int,
    args: argparse.Namespace,
) -> tuple[FeatureDataset, dict[str, Any]]:
    tabpfn_module = load_module(
        PROJECT_ROOT / "003_machine_learning_evaluation" / "002_tabpfn.py",
        f"tabpfn_embedding_for_federated_lightgbm_fold_{fold_number}",
    )
    tabpfn_module.configure_tabpfn_environment(PROJECT_ROOT, args)
    model_path = tabpfn_module.resolve_model_path(PROJECT_ROOT, args)

    train_parts = []
    test_parts = []
    for _campaign_id, global_indices, splits, _split_strategy in client_splits:
        local_train_idx, local_test_idx = splits[fold_number]
        train_parts.append(global_indices[local_train_idx])
        test_parts.append(global_indices[local_test_idx])

    train_idx = np.unique(np.concatenate(train_parts)).astype(int)
    test_idx = np.unique(np.concatenate(test_parts)).astype(int)
    all_idx = np.unique(np.concatenate([train_idx, test_idx])).astype(int)
    cache_file, cache_payload = tabpfn_embedding_cache_paths(
        source_dataset,
        dataset,
        fold_number,
        train_idx,
        all_idx,
        model_path,
        args,
    )
    if args.use_tabpfn_embedding_cache:
        cached = load_tabpfn_embedding_cache(cache_file)
        if cached is not None:
            cached_all_idx, cached_embeddings, cached_metadata = cached
            X_embedded = np.zeros((len(source_dataset.target), cached_embeddings.shape[1]), dtype=np.float32)
            X_embedded[cached_all_idx] = cached_embeddings
            embedded_dataset = FeatureDataset(
                feature_stage=source_dataset.feature_stage,
                feature_selection=source_dataset.feature_selection,
                feature_approach=source_dataset.feature_approach,
                path=source_dataset.path,
                X=X_embedded,
                target=source_dataset.target,
                row_index=source_dataset.row_index,
            )
            metadata = {
                "feature_representation": "tabpfn_global_embeddings",
                "tabpfn_embedding_scope": "global_train_fold_context",
                "tabpfn_embedding_cache_hit": True,
                "tabpfn_embedding_cache_file": str(cache_file),
                "tabpfn_embedding_train_rows": cached_metadata.get("tabpfn_embedding_train_rows"),
                "tabpfn_embedding_total_rows": int(len(cached_all_idx)),
                "tabpfn_embedding_features": int(cached_embeddings.shape[1]),
                "tabpfn_embedding_model_path": model_path,
                "tabpfn_embedding_device": args.device,
                "tabpfn_embedding_n_estimators": int(args.tabpfn_n_estimators),
                "tabpfn_embedding_source": cached_metadata.get("tabpfn_embedding_source", "cache"),
            }
            return embedded_dataset, metadata

    fit_idx = train_idx
    if tabpfn_module.effective_cpu_device(args.device) and not args.allow_cpu_large_dataset:
        fit_idx = tabpfn_module.downsample_indices(
            train_idx,
            y,
            int(args.max_cpu_embedding_train_rows),
            int(args.random_state) + 97 * (fold_number + 1),
        )

    X_fit = maybe_dense(take_rows(source_dataset.X, fit_idx), int(args.max_dense_cells))
    X_all = maybe_dense(take_rows(source_dataset.X, all_idx), int(args.max_dense_cells))
    classifier = tabpfn_module.make_tabpfn_classifier(
        random_state=int(args.random_state) + 31 * (fold_number + 1),
        device=args.device,
        ignore_pretraining_limits=bool(args.ignore_pretraining_limits),
        model_path=model_path,
        n_estimators=int(args.tabpfn_n_estimators),
    )
    classifier.fit(X_fit, y[fit_idx])
    embeddings, embedding_source = tabpfn_embeddings_or_proba(classifier, X_all, len(all_idx))

    position_by_index = {int(index): position for position, index in enumerate(all_idx)}
    train_positions = np.array([position_by_index[int(index)] for index in train_idx], dtype=int)
    scaler = StandardScaler()
    scaler.fit(embeddings[train_positions])
    embeddings = scaler.transform(embeddings).astype(np.float32, copy=False)
    cache_metadata = {
        **cache_payload,
        "feature_representation": "tabpfn_global_embeddings",
        "tabpfn_embedding_scope": "global_train_fold_context",
        "tabpfn_embedding_source": embedding_source,
        "tabpfn_embedding_train_rows": int(len(fit_idx)),
        "tabpfn_embedding_total_rows": int(len(all_idx)),
        "tabpfn_embedding_features": int(embeddings.shape[1]),
        "tabpfn_embedding_model_path": model_path,
        "tabpfn_embedding_device": args.device,
        "tabpfn_embedding_n_estimators": int(args.tabpfn_n_estimators),
    }
    if args.use_tabpfn_embedding_cache:
        save_tabpfn_embedding_cache(cache_file, all_idx, embeddings, cache_metadata)
        if args.save_campaign_embeddings:
            save_campaign_embedding_cache(cache_file, client_splits, fold_number, all_idx, embeddings)

    X_embedded = np.zeros((len(source_dataset.target), embeddings.shape[1]), dtype=np.float32)
    X_embedded[all_idx] = embeddings
    embedded_dataset = FeatureDataset(
        feature_stage=source_dataset.feature_stage,
        feature_selection=source_dataset.feature_selection,
        feature_approach=source_dataset.feature_approach,
        path=source_dataset.path,
        X=X_embedded,
        target=source_dataset.target,
        row_index=source_dataset.row_index,
    )
    metadata = {
        "feature_representation": "tabpfn_global_embeddings",
        "tabpfn_embedding_scope": "global_train_fold_context",
        "tabpfn_embedding_source": embedding_source,
        "tabpfn_embedding_cache_hit": False,
        "tabpfn_embedding_cache_file": str(cache_file) if args.use_tabpfn_embedding_cache else None,
        "tabpfn_embedding_train_rows": int(len(fit_idx)),
        "tabpfn_embedding_total_rows": int(len(all_idx)),
        "tabpfn_embedding_features": int(embeddings.shape[1]),
        "tabpfn_embedding_model_path": model_path,
        "tabpfn_embedding_device": args.device,
        "tabpfn_embedding_n_estimators": int(args.tabpfn_n_estimators),
    }
    return embedded_dataset, metadata


def communication_cost_matrix(X) -> int:
    if sparse.issparse(X):
        return int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)
    return int(np.asarray(X).nbytes)


def model_communication_cost(model) -> int:
    try:
        booster = model.booster_
        return int(len(booster.model_to_string().encode("utf-8")))
    except Exception:
        return 0


def make_lightgbm(args: argparse.Namespace, random_state: int):
    try:
        from lightgbm import LGBMClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("lightgbm is required for Federated LightGBM") from exc

    return LGBMClassifier(
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        max_depth=int(args.max_depth),
        min_child_samples=int(args.min_child_samples),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        reg_alpha=float(args.reg_alpha),
        reg_lambda=float(args.reg_lambda),
        class_weight="balanced",
        random_state=random_state,
        n_jobs=int(args.n_jobs),
        force_col_wise=True,
        verbosity=-1,
    )


def make_round_lightgbm(args: argparse.Namespace, random_state: int):
    round_args = argparse.Namespace(**vars(args))
    round_trees = int(args.local_trees_per_round) if int(args.local_trees_per_round) > 0 else max(1, int(args.n_estimators) // max(1, int(args.rounds)))
    round_args.n_estimators = round_trees
    return make_lightgbm(round_args, random_state)


def client_weight(n_train: int, weighting: str) -> float:
    if weighting == "examples":
        return float(n_train)
    if weighting == "sqrt_examples":
        return float(np.sqrt(max(1, n_train)))
    if weighting == "log_examples":
        return float(np.log1p(max(1, n_train)))
    if weighting in {"uniform", "balanced_by_campaign"}:
        return 1.0
    raise ValueError(f"unknown weighting strategy: {weighting}")


def sigmoid(raw_score: np.ndarray) -> np.ndarray:
    raw_score = np.clip(np.asarray(raw_score, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-raw_score))


def model_raw_contribution(model, X) -> np.ndarray:
    raw = np.asarray(model.predict(X, raw_score=True), dtype=np.float64)
    if raw.ndim > 1:
        raw = raw[:, -1]
    return raw.reshape(-1)


def fit_client_model(
    feature_dataset: FeatureDataset,
    y: np.ndarray,
    train_idx: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    init_score: np.ndarray | None = None,
):
    X_train = take_rows(feature_dataset.X, train_idx)
    X_train = maybe_dense(X_train, args.max_dense_cells)
    scaler = None
    if not sparse.issparse(X_train):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    model = make_round_lightgbm(args, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_kwargs = {"init_score": init_score} if init_score is not None else {}
        model.fit(X_train, y[train_idx], **fit_kwargs)
    return model, scaler


def transform_with_scaler(X, scaler, args: argparse.Namespace):
    X_transformed = maybe_dense(X, args.max_dense_cells)
    if scaler is not None:
        X_transformed = scaler.transform(X_transformed)
    return X_transformed


def global_raw_score(global_rounds: list[list[tuple[Any, Any, float]]], X, args: argparse.Namespace) -> np.ndarray:
    if not global_rounds:
        return np.zeros(X.shape[0], dtype=np.float64)
    raw = np.zeros(X.shape[0], dtype=np.float64)
    for round_models in global_rounds:
        contribution = np.zeros(X.shape[0], dtype=np.float64)
        weight_sum = 0.0
        for model, scaler, weight in round_models:
            X_for_model = transform_with_scaler(X, scaler, args)
            contribution += model_raw_contribution(model, X_for_model) * float(weight)
            weight_sum += float(weight)
        if weight_sum > 0:
            raw += contribution / weight_sum
    return raw


def evaluate_feature_dataset(feature_dataset: FeatureDataset, dataset: str, args: argparse.Namespace) -> dict:
    y_text = feature_dataset.target["attack_type"].fillna(MISSING).astype(str).to_numpy()
    if len(np.unique(y_text)) < 2:
        raise ValueError("target has fewer than two classes")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    n_classes = len(label_encoder.classes_)
    campaigns = campaign_indices(feature_dataset)
    if args.max_clients > 0:
        campaigns = campaigns[: args.max_clients]
    if len(campaigns) < args.min_clients:
        raise ValueError(f"fewer than min_clients={args.min_clients}: {len(campaigns)}")

    client_splits = []
    effective_folds = []
    for campaign_id, indices in campaigns:
        local_y = y[indices]
        if len(np.unique(local_y)) < 2:
            continue
        splits, split_strategy, n_folds = campaign_kfold_splits(local_y, args.k_folds, args.random_state)
        client_splits.append((campaign_id, indices, splits, split_strategy))
        effective_folds.append(n_folds)
    if len(client_splits) < args.min_clients:
        raise ValueError(f"fewer usable clients after class filtering: {len(client_splits)}")

    n_global_folds = int(min(effective_folds))
    fold_results = []
    total_started = time.perf_counter()
    for fold_number in range(n_global_folds):
        started = time.perf_counter()
        client_train_sizes = {}
        model_bytes = 0
        raw_feature_bytes_read_locally = 0
        embedding_feature_bytes_read_locally = 0
        global_rounds: list[list[tuple[Any, Any, float]]] = []
        fold_feature_dataset = feature_dataset
        fold_embedding_metadata = {
            "feature_representation": "source_features",
            "tabpfn_embedding_scope": None,
            "tabpfn_embedding_source": None,
            "tabpfn_embedding_train_rows": None,
            "tabpfn_embedding_total_rows": None,
            "tabpfn_embedding_features": None,
        }
        if args.use_tabpfn_global_embeddings:
            fold_feature_dataset, fold_embedding_metadata = build_global_tabpfn_embedding_dataset(
                feature_dataset,
                dataset,
                y,
                client_splits,
                fold_number,
                args,
            )

        for round_number in range(int(args.rounds)):
            round_models = []
            for client_position, (campaign_id, global_indices, splits, _split_strategy) in enumerate(client_splits):
                local_train_idx, _local_test_idx = splits[fold_number]
                train_idx = global_indices[local_train_idx]
                if len(np.unique(y[train_idx])) < 2:
                    continue
                init_score = global_raw_score(global_rounds, take_rows(fold_feature_dataset.X, train_idx), args) if global_rounds else None
                model, scaler = fit_client_model(
                    fold_feature_dataset,
                    y,
                    train_idx,
                    args,
                    args.random_state + 10_000 * (fold_number + 1) + 1_000 * (round_number + 1) + client_position,
                    init_score=init_score,
                )
                n_train = int(len(train_idx))
                weight = client_weight(n_train, args.weighting)
                round_models.append((model, scaler, weight))
                client_train_sizes[str(campaign_id)] = n_train
                model_bytes += model_communication_cost(model)
                raw_feature_bytes_read_locally += communication_cost_matrix(take_rows(feature_dataset.X, train_idx))
                embedding_feature_bytes_read_locally += communication_cost_matrix(take_rows(fold_feature_dataset.X, train_idx))

            if len(round_models) < args.min_clients:
                raise ValueError(f"fold {fold_number + 1}, round {round_number + 1} has fewer usable trained clients: {len(round_models)}")
            global_rounds.append(round_models)

        y_true_parts = []
        y_pred_parts = []
        campaign_groups: list[str] = []
        traffic_groups: list[str] = []
        local_test_rows = 0

        for campaign_id, global_indices, splits, _split_strategy in client_splits:
            _local_train_idx, local_test_idx = splits[fold_number]
            test_idx = global_indices[local_test_idx]
            if len(test_idx) == 0:
                continue

            raw_score = global_raw_score(global_rounds, take_rows(fold_feature_dataset.X, test_idx), args)
            positive_proba = sigmoid(raw_score)
            y_pred = (positive_proba >= float(args.prediction_threshold)).astype(int)
            y_true_parts.append(y[test_idx])
            y_pred_parts.append(y_pred)
            campaign_groups.extend([str(campaign_id)] * len(test_idx))
            if feature_dataset.row_index is not None and "traffic_source" in feature_dataset.row_index.columns:
                traffic_groups.extend(feature_dataset.row_index.iloc[test_idx]["traffic_source"].fillna(MISSING).astype(str).tolist())
            else:
                traffic_groups.extend([MISSING] * len(test_idx))
            local_test_rows += int(len(test_idx))

        if not y_true_parts:
            raise ValueError(f"fold {fold_number + 1} has no test rows")

        y_true = np.concatenate(y_true_parts)
        y_pred = np.concatenate(y_pred_parts)
        labels = np.arange(n_classes)
        recall_values = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        campaign_performance = group_balanced_accuracy(y_true, y_pred, campaign_groups)
        traffic_source_performance = group_balanced_accuracy(y_true, y_pred, traffic_groups)
        elapsed = time.perf_counter() - started
        fold_results.append(
            {
                "fold": int(fold_number + 1),
                "n_train": int(sum(client_train_sizes.values())),
                "n_test": int(local_test_rows),
                "n_clients": int(len(client_splits)),
                "client_train_sizes": client_train_sizes,
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
                "recall_by_class": {
                    str(class_name): float(value)
                    for class_name, value in zip(label_encoder.classes_, recall_values)
                },
                "communication_cost": int(model_bytes),
                "raw_feature_bytes_read_locally": int(raw_feature_bytes_read_locally),
                "embedding_feature_bytes_read_locally": int(embedding_feature_bytes_read_locally),
                "server_received_raw_rows": 0,
                "server_received_raw_columns": 0,
                "federated_component": "lightgbm_meta_classifier",
                "meta_classifier_federated": "LightGBM",
                "tabpfn_is_federated": False,
                **fold_embedding_metadata,
                "number_of_rounds": int(args.rounds),
                "local_trees_per_round": int(args.local_trees_per_round) if int(args.local_trees_per_round) > 0 else max(1, int(args.n_estimators) // max(1, int(args.rounds))),
                "prediction_threshold": float(args.prediction_threshold),
                "weighting": args.weighting,
                "campaign_performance": campaign_performance,
                "campaign_metric_variance": metric_variance(campaign_performance),
                "campaign_fairness_gap": fairness_gap(campaign_performance),
                "client_variance": metric_variance(campaign_performance),
                "fairness_between_campaigns": fairness_gap(campaign_performance),
                "traffic_source_performance": traffic_source_performance,
                "performance_by_traffic_source": traffic_source_performance,
                "traffic_source_metric_variance": metric_variance(traffic_source_performance),
                "traffic_source_fairness_gap": fairness_gap(traffic_source_performance),
                "fit_predict_seconds": float(elapsed),
                "elapsed_seconds": float(elapsed),
                "status": "ok",
                "error": None,
            }
        )

    result = summarize_fold_results(fold_results, feature_dataset, dataset)
    result["fit_predict_seconds"] = float(time.perf_counter() - total_started)
    result["elapsed_seconds"] = result["fit_predict_seconds"]
    return result


def summarize_fold_results(fold_results: list[dict], feature_dataset: FeatureDataset, dataset: str) -> dict:
    ok_results = [result for result in fold_results if result.get("status") == "ok"]
    if not ok_results:
        raise ValueError("no successful federated folds")

    result = {
        "feature_stage": feature_dataset.feature_stage,
        "feature_selection": feature_dataset.feature_selection,
        "feature_approach": feature_dataset.feature_approach,
        "classifier": CLASSIFIER_NAME,
        "model": CLASSIFIER_NAME,
        "federated_algorithm": FEDERATED_ALGORITHM,
        "dataset": dataset,
        "evaluation_scope": "federated_clients",
        "campaign": None,
        "campaign_id": None,
        "n_samples": int(len(feature_dataset.target)),
        "n_train": int(sum(fold.get("n_train") or 0 for fold in ok_results)),
        "n_test": int(sum(fold.get("n_test") or 0 for fold in ok_results)),
        "source_n_features": int(feature_dataset.X.shape[1]),
        "n_features": int(ok_results[0].get("tabpfn_embedding_features") or feature_dataset.X.shape[1]),
        "n_clients": int(round(summarize_numeric([fold.get("n_clients") for fold in ok_results]) or 0)),
        "k_folds": int(len(ok_results)),
        "cv_strategy": "campaign_aligned_kfold",
        "status": "ok",
        "error": None,
        "references": [
            "FederBoost: Private Federated Learning for GBDT, arXiv:2011.02796v4",
            "IEEE Xplore document 11004389",
        ],
        "federated_lightgbm_strategy": "roundwise_residual_boosting_with_weighted_client_tree_aggregation",
        "feature_representation": ok_results[0].get("feature_representation") or "source_features",
        "federated_component": ok_results[0].get("federated_component") or "lightgbm_meta_classifier",
        "meta_classifier_federated": ok_results[0].get("meta_classifier_federated") or "LightGBM",
        "tabpfn_is_federated": bool(ok_results[0].get("tabpfn_is_federated", False)),
        "tabpfn_embedding_scope": ok_results[0].get("tabpfn_embedding_scope"),
        "tabpfn_embedding_source": ok_results[0].get("tabpfn_embedding_source"),
        "tabpfn_embedding_cache_file": ok_results[0].get("tabpfn_embedding_cache_file"),
        "tabpfn_embedding_device": ok_results[0].get("tabpfn_embedding_device"),
        "tabpfn_embedding_n_estimators": ok_results[0].get("tabpfn_embedding_n_estimators"),
        "weighting": ok_results[0].get("weighting") or CONFIG_WEIGHTING,
        "prediction_threshold": float(ok_results[0].get("prediction_threshold") or CONFIG_PREDICTION_THRESHOLD),
        "rounds": int(ok_results[0].get("number_of_rounds") or CONFIG_ROUNDS),
        "local_trees_per_round": int(ok_results[0].get("local_trees_per_round") or CONFIG_LOCAL_TREES_PER_ROUND),
        "server_received_raw_rows": 0,
        "server_received_raw_columns": 0,
    }
    for key in (
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "communication_cost",
        "number_of_rounds",
        "campaign_metric_variance",
        "campaign_fairness_gap",
        "client_variance",
        "fairness_between_campaigns",
        "traffic_source_metric_variance",
        "traffic_source_fairness_gap",
        "server_received_raw_rows",
        "server_received_raw_columns",
        "raw_feature_bytes_read_locally",
        "embedding_feature_bytes_read_locally",
        "tabpfn_embedding_cache_hit",
        "tabpfn_embedding_train_rows",
        "tabpfn_embedding_total_rows",
        "tabpfn_embedding_features",
    ):
        values = [fold.get(key) for fold in ok_results]
        result[key] = summarize_numeric(values)
        result[f"{key}_fold_std"] = summarize_numeric_std(values)

    result["recall_by_class"] = summarize_dicts([fold.get("recall_by_class") for fold in ok_results])
    result["campaign_performance"] = summarize_dicts([fold.get("campaign_performance") for fold in ok_results])
    result["traffic_source_performance"] = summarize_dicts([fold.get("traffic_source_performance") for fold in ok_results])
    result["performance_by_traffic_source"] = result["traffic_source_performance"]
    result["fold_results"] = fold_results
    return result


def failed_result(feature_dataset: FeatureDataset, error: Exception, dataset: str) -> dict:
    return {
        "feature_stage": feature_dataset.feature_stage,
        "feature_selection": feature_dataset.feature_selection,
        "feature_approach": feature_dataset.feature_approach,
        "classifier": CLASSIFIER_NAME,
        "model": CLASSIFIER_NAME,
        "federated_algorithm": FEDERATED_ALGORITHM,
        "dataset": dataset,
        "evaluation_scope": "federated_clients",
        "campaign": None,
        "campaign_id": None,
        "n_samples": int(len(feature_dataset.target)),
        "n_train": None,
        "n_test": None,
        "n_features": int(feature_dataset.X.shape[1]),
        "source_n_features": int(feature_dataset.X.shape[1]),
        "feature_representation": "tabpfn_global_embeddings" if CONFIG_USE_TABPFN_GLOBAL_EMBEDDINGS else "source_features",
        "federated_component": "lightgbm_meta_classifier",
        "meta_classifier_federated": "LightGBM",
        "tabpfn_is_federated": False,
        "balanced_accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
        "mcc": None,
        "recall_by_class": None,
        "communication_cost": None,
        "number_of_rounds": 1,
        "client_variance": None,
        "fairness_between_campaigns": None,
        "performance_by_traffic_source": None,
        "status": "error",
        "error": str(error),
    }


def normalize_for_csv(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = result.copy()
        for key, value in list(row.items()):
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    dataset = read_env_value(project_root / args.env_file, "DATASET")
    if dataset not in VALID_DATASETS:
        raise ValueError(f"Invalid DATASET={dataset!r}. Expected one of: {sorted(VALID_DATASETS)}")

    feature_datasets = discover_feature_datasets(
        project_root / args.extracted_root,
        project_root / args.selected_root,
        project_root / args.raw_root,
        dataset,
    )
    if not feature_datasets:
        raise ValueError("No extracted or selected feature datasets were found.")

    best_bayesian = None
    if args.use_best_bayesian_optimization:
        best_bayesian = load_best_bayesian_configuration(project_root, args, dataset)
        args = apply_lightgbm_params_from_bayesian(args, best_bayesian)
        args = apply_tabpfn_params_from_bayesian(args, best_bayesian)
        feature_datasets = [select_feature_dataset(feature_datasets, best_bayesian)]
        print(
            "Using best Bayesian Optimization pipeline: "
            f"{best_bayesian['feature_stage']}/{best_bayesian['feature_selection'] or 'none'}/"
            f"{best_bayesian['feature_approach']} "
            f"objective={best_bayesian['objective']} std={best_bayesian['std']} "
            f"lightgbm_params={best_bayesian['lightgbm_params']} "
            f"tabpfn_params={best_bayesian.get('tabpfn_params')}",
            flush=True,
        )

    output_path = project_root / args.output_root / dataset
    run_config = base_config(args)
    run_config["dataset"] = dataset
    if best_bayesian is not None:
        run_config["source_best_bayesian_optimization"] = best_bayesian
    current_config_hash = config_hash(run_config)
    results: list[dict] = load_existing_results(output_path)
    done = completed_keys(results, current_config_hash)

    for feature_dataset in feature_datasets:
        feature_dataset = cap_rows_by_campaign(feature_dataset, args.max_rows, args.random_state)
        pending_key = {
            "config_hash": current_config_hash,
            "feature_stage": feature_dataset.feature_stage,
            "feature_selection": feature_dataset.feature_selection,
            "feature_approach": feature_dataset.feature_approach,
            "classifier": CLASSIFIER_NAME,
            "federated_algorithm": FEDERATED_ALGORITHM,
            "evaluation_scope": "federated_clients",
            "campaign": None,
        }
        if result_key(pending_key) in done:
            selector = feature_dataset.feature_selection or "none"
            print(f"{feature_dataset.feature_approach}/{selector} + {CLASSIFIER_NAME}: skipped")
            continue
        try:
            result = evaluate_feature_dataset(feature_dataset, dataset, args)
        except Exception as exc:
            result = failed_result(feature_dataset, exc, dataset)
        if best_bayesian is not None:
            result["source_best_bayesian_optimization"] = best_bayesian
            result["source_best_bayesian_objective"] = best_bayesian["objective"]
            result["source_best_bayesian_objective_std"] = best_bayesian["std"]
            result["source_best_bayesian_trial"] = best_bayesian.get("bayesian_trial")
            result["source_best_bayesian_study_name"] = best_bayesian.get("bayesian_study_name")
            result["source_best_bayesian_search_space_id"] = best_bayesian.get("bayesian_search_space_id")
            result["lightgbm_params_from_bayesian_optimization"] = best_bayesian["lightgbm_params"]
            result["tabpfn_params_from_bayesian_optimization"] = best_bayesian.get("tabpfn_params")
        result = add_resume_metadata(result, run_config, pending_key)
        results.append(result)
        done.add(result_key(result))
        save_results(output_path, results, normalize_for_csv)
        selector = feature_dataset.feature_selection or "none"
        print(f"{feature_dataset.feature_approach}/{selector} + {CLASSIFIER_NAME}: {result['status']} ba={result.get('balanced_accuracy')}")

    save_results(output_path, results, normalize_for_csv)
    print(f"Saved Federated LightGBM results to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise
