#!/usr/bin/env python3
"""
Bayesian optimization for TabPFN embeddings followed by a LightGBM classifier.

This experiment evaluates every available feature representation from
data/extracted_features and data/selected_features. For each campaign and fold,
TabPFN is fitted on a private context split, frozen embeddings are extracted for
a meta-training split and the test fold, and LightGBM is trained on those
embeddings. Optuna/TPE searches both TabPFN and LightGBM hyperparameters.

Outputs:
    results/grid_search/tabpfn_embeddings_lightgbm/{DATASET}/results.csv
    results/grid_search/tabpfn_embeddings_lightgbm/{DATASET}/results.json
    results/grid_search/tabpfn_embeddings_lightgbm/{DATASET}/summary.csv
    results/grid_search/tabpfn_embeddings_lightgbm/{DATASET}/optuna_studies/*.db
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import re
import sys
import time
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils._env import VALID_DATASETS, MISSING, PRIMARY_METRIC  # noqa: E402

BASE_SCRIPT = PROJECT_ROOT / "005_tabpfn" / "000_fine_tune_head.py"
STACKING_SCRIPT = PROJECT_ROOT / "005_tabpfn" / "003_stacking_meta_classifier.py"

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

CLASSIFIER_NAME = "TabPFN-Embeddings-LightGBM-BayesianOpt"


# Pipeline configuration constants. Edit these values to change this script.
CONFIG_ENV_FILE = ".env"
CONFIG_EXTRACTED_ROOT = "data/extracted_features"
CONFIG_SELECTED_ROOT = "data/selected_features"
CONFIG_RAW_ROOT = "data/raw"
CONFIG_OUTPUT_ROOT = "results/grid_search/tabpfn_embeddings_lightgbm"
CONFIG_MODEL_CACHE = "models/tabpfn"
CONFIG_MODEL_PATH = None
CONFIG_ALLOW_BROWSER_LOGIN = False
CONFIG_DATASET = None
CONFIG_RANDOM_STATE = 42
CONFIG_K_FOLDS = 3
CONFIG_MAX_ROWS = 0
CONFIG_MAX_CAMPAIGNS = 0
CONFIG_MAX_DENSE_CELLS = 10_000_000
CONFIG_ALLOW_CPU_LARGE_DATASET = False
CONFIG_FEATURE_STAGE = "auto"  # auto, extracted_features, selected_features
CONFIG_FEATURE_SELECTION = "none,pca,truncatedSVD"  # auto, none, pca, truncatedSVD, chi2, selectKBest, or comma-separated values
CONFIG_FEATURE_APPROACH = "sentence_transformer"  # auto, label_encoder, semantic_headers, tf_idf, sentence_transformer, or comma-separated values
CONFIG_MAX_FEATURE_DATASETS = 0
CONFIG_N_TRIALS = 50  # New Optuna trials to run per execution, not the total study budget.
CONFIG_N_STARTUP_TRIALS = 8
CONFIG_TIMEOUT_SECONDS = None
CONFIG_META_TRAIN_SIZE_CHOICES = "0.20,0.30,0.40,0.50"
CONFIG_DEVICE_CHOICES = "auto"
CONFIG_IGNORE_PRETRAINING_LIMITS_CHOICES = "true"
CONFIG_TABPFN_N_ESTIMATORS_CHOICES = "4,8,16"
CONFIG_MAX_CPU_TRAIN_ROWS_CHOICES = "1000,2000"
CONFIG_MAX_CPU_TEST_ROWS_CHOICES = "1000"
CONFIG_LIGHTGBM_N_ESTIMATORS_MIN = 80
CONFIG_LIGHTGBM_N_ESTIMATORS_MAX = 500
CONFIG_LIGHTGBM_LEARNING_RATE_MIN = 0.01
CONFIG_LIGHTGBM_LEARNING_RATE_MAX = 0.20
CONFIG_LIGHTGBM_NUM_LEAVES_MIN = 7
CONFIG_LIGHTGBM_NUM_LEAVES_MAX = 63
CONFIG_LIGHTGBM_MAX_DEPTH_CHOICES = "-1,3,5,7,9"
CONFIG_LIGHTGBM_MIN_CHILD_SAMPLES_MIN = 5
CONFIG_LIGHTGBM_MIN_CHILD_SAMPLES_MAX = 60
CONFIG_LIGHTGBM_SUBSAMPLE_MIN = 0.60
CONFIG_LIGHTGBM_SUBSAMPLE_MAX = 1.00
CONFIG_LIGHTGBM_COLSAMPLE_BYTREE_MIN = 0.60
CONFIG_LIGHTGBM_COLSAMPLE_BYTREE_MAX = 1.00
CONFIG_LIGHTGBM_REG_ALPHA_MIN = 1e-8
CONFIG_LIGHTGBM_REG_ALPHA_MAX = 10.0
CONFIG_LIGHTGBM_REG_LAMBDA_MIN = 1e-8
CONFIG_LIGHTGBM_REG_LAMBDA_MAX = 10.0
CONFIG_LIGHTGBM_N_JOBS = 1
CONFIG_PLAN_ONLY = False
CONFIG_RETRY_ERRORS = True


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> Namespace:
    """Return script configuration from global constants; command-line args are intentionally ignored."""
    return Namespace(
        env_file=CONFIG_ENV_FILE,
        extracted_root=CONFIG_EXTRACTED_ROOT,
        selected_root=CONFIG_SELECTED_ROOT,
        raw_root=CONFIG_RAW_ROOT,
        output_root=CONFIG_OUTPUT_ROOT,
        model_cache=CONFIG_MODEL_CACHE,
        model_path=CONFIG_MODEL_PATH,
        allow_browser_login=CONFIG_ALLOW_BROWSER_LOGIN,
        dataset=CONFIG_DATASET,
        random_state=CONFIG_RANDOM_STATE,
        k_folds=CONFIG_K_FOLDS,
        max_rows=CONFIG_MAX_ROWS,
        max_campaigns=CONFIG_MAX_CAMPAIGNS,
        max_dense_cells=CONFIG_MAX_DENSE_CELLS,
        allow_cpu_large_dataset=CONFIG_ALLOW_CPU_LARGE_DATASET,
        feature_stage=CONFIG_FEATURE_STAGE,
        feature_selection=CONFIG_FEATURE_SELECTION,
        feature_approach=CONFIG_FEATURE_APPROACH,
        max_feature_datasets=CONFIG_MAX_FEATURE_DATASETS,
        n_trials=CONFIG_N_TRIALS,
        n_startup_trials=CONFIG_N_STARTUP_TRIALS,
        timeout_seconds=CONFIG_TIMEOUT_SECONDS,
        meta_train_size_choices=CONFIG_META_TRAIN_SIZE_CHOICES,
        device_choices=CONFIG_DEVICE_CHOICES,
        ignore_pretraining_limits_choices=CONFIG_IGNORE_PRETRAINING_LIMITS_CHOICES,
        tabpfn_n_estimators_choices=CONFIG_TABPFN_N_ESTIMATORS_CHOICES,
        max_cpu_train_rows_choices=CONFIG_MAX_CPU_TRAIN_ROWS_CHOICES,
        max_cpu_test_rows_choices=CONFIG_MAX_CPU_TEST_ROWS_CHOICES,
        lightgbm_n_estimators_min=CONFIG_LIGHTGBM_N_ESTIMATORS_MIN,
        lightgbm_n_estimators_max=CONFIG_LIGHTGBM_N_ESTIMATORS_MAX,
        lightgbm_learning_rate_min=CONFIG_LIGHTGBM_LEARNING_RATE_MIN,
        lightgbm_learning_rate_max=CONFIG_LIGHTGBM_LEARNING_RATE_MAX,
        lightgbm_num_leaves_min=CONFIG_LIGHTGBM_NUM_LEAVES_MIN,
        lightgbm_num_leaves_max=CONFIG_LIGHTGBM_NUM_LEAVES_MAX,
        lightgbm_max_depth_choices=CONFIG_LIGHTGBM_MAX_DEPTH_CHOICES,
        lightgbm_min_child_samples_min=CONFIG_LIGHTGBM_MIN_CHILD_SAMPLES_MIN,
        lightgbm_min_child_samples_max=CONFIG_LIGHTGBM_MIN_CHILD_SAMPLES_MAX,
        lightgbm_subsample_min=CONFIG_LIGHTGBM_SUBSAMPLE_MIN,
        lightgbm_subsample_max=CONFIG_LIGHTGBM_SUBSAMPLE_MAX,
        lightgbm_colsample_bytree_min=CONFIG_LIGHTGBM_COLSAMPLE_BYTREE_MIN,
        lightgbm_colsample_bytree_max=CONFIG_LIGHTGBM_COLSAMPLE_BYTREE_MAX,
        lightgbm_reg_alpha_min=CONFIG_LIGHTGBM_REG_ALPHA_MIN,
        lightgbm_reg_alpha_max=CONFIG_LIGHTGBM_REG_ALPHA_MAX,
        lightgbm_reg_lambda_min=CONFIG_LIGHTGBM_REG_LAMBDA_MIN,
        lightgbm_reg_lambda_max=CONFIG_LIGHTGBM_REG_LAMBDA_MAX,
        lightgbm_n_jobs=CONFIG_LIGHTGBM_N_JOBS,
        plan_only=CONFIG_PLAN_ONLY,
        retry_errors=CONFIG_RETRY_ERRORS,
    )


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_int_grid(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def parse_float_grid(value: str) -> list[float]:
    return [float(item) for item in parse_csv(value)]


def parse_bool_grid(value: str) -> list[bool]:
    values = []
    for item in parse_csv(value):
        normalized = item.lower()
        if normalized in {"1", "true", "yes", "y"}:
            values.append(True)
        elif normalized in {"0", "false", "no", "n"}:
            values.append(False)
        else:
            raise ValueError(f"Invalid boolean value: {item}")
    return values


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def load_optuna():
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "optuna is required for Bayesian optimization. Add it to requirements.txt and install the project dependencies."
        ) from exc
    return optuna


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")[:180]


def feature_dataset_key(feature_dataset) -> str:
    return "|".join(
        [
            str(feature_dataset.feature_stage),
            str(feature_dataset.feature_selection or "none"),
            str(feature_dataset.feature_approach),
        ]
    )


def pipeline_search_space_id(pipeline_keys: list[str]) -> str:
    encoded = stable_json(sorted(pipeline_keys)).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def study_name(dataset: str, search_space_id: str) -> str:
    return sanitize_name(f"{dataset}_pipeline_feature_search_{search_space_id}_campaign_mean_objective")


def suggest_trial_params(trial, args: Namespace) -> tuple[float, dict[str, Any], dict[str, Any]]:
    meta_train_size = float(trial.suggest_categorical("meta_train_size", parse_float_grid(args.meta_train_size_choices)))
    tabpfn_params = {
        "device": trial.suggest_categorical("tabpfn_device", parse_csv(args.device_choices)),
        "ignore_pretraining_limits": bool(
            trial.suggest_categorical("tabpfn_ignore_pretraining_limits", parse_bool_grid(args.ignore_pretraining_limits_choices))
        ),
        "n_estimators": int(trial.suggest_categorical("tabpfn_n_estimators", parse_int_grid(args.tabpfn_n_estimators_choices))),
        "max_cpu_train_rows": int(trial.suggest_categorical("tabpfn_max_cpu_train_rows", parse_int_grid(args.max_cpu_train_rows_choices))),
        "max_cpu_test_rows": int(trial.suggest_categorical("tabpfn_max_cpu_test_rows", parse_int_grid(args.max_cpu_test_rows_choices))),
    }
    lightgbm_params = {
        "n_estimators": int(trial.suggest_int("lightgbm_n_estimators", int(args.lightgbm_n_estimators_min), int(args.lightgbm_n_estimators_max), step=20)),
        "learning_rate": float(
            trial.suggest_float("lightgbm_learning_rate", float(args.lightgbm_learning_rate_min), float(args.lightgbm_learning_rate_max), log=True)
        ),
        "num_leaves": int(trial.suggest_int("lightgbm_num_leaves", int(args.lightgbm_num_leaves_min), int(args.lightgbm_num_leaves_max))),
        "max_depth": int(trial.suggest_categorical("lightgbm_max_depth", parse_int_grid(args.lightgbm_max_depth_choices))),
        "min_child_samples": int(
            trial.suggest_int("lightgbm_min_child_samples", int(args.lightgbm_min_child_samples_min), int(args.lightgbm_min_child_samples_max))
        ),
        "subsample": float(trial.suggest_float("lightgbm_subsample", float(args.lightgbm_subsample_min), float(args.lightgbm_subsample_max))),
        "colsample_bytree": float(
            trial.suggest_float("lightgbm_colsample_bytree", float(args.lightgbm_colsample_bytree_min), float(args.lightgbm_colsample_bytree_max))
        ),
        "reg_alpha": float(trial.suggest_float("lightgbm_reg_alpha", float(args.lightgbm_reg_alpha_min), float(args.lightgbm_reg_alpha_max), log=True)),
        "reg_lambda": float(trial.suggest_float("lightgbm_reg_lambda", float(args.lightgbm_reg_lambda_min), float(args.lightgbm_reg_lambda_max), log=True)),
        "n_jobs": int(args.lightgbm_n_jobs),
    }
    return meta_train_size, tabpfn_params, lightgbm_params


def filter_feature_datasets(feature_datasets: list[Any], args: Namespace) -> list[Any]:
    output = []
    allowed_selections = None
    if args.feature_selection != "auto":
        allowed_selections = set(parse_csv(args.feature_selection))
    allowed_approaches = None
    if args.feature_approach != "auto":
        allowed_approaches = set(parse_csv(args.feature_approach))
    for dataset in feature_datasets:
        if args.feature_stage != "auto" and dataset.feature_stage != args.feature_stage:
            continue
        selection = dataset.feature_selection or "none"
        if allowed_selections is not None and selection not in allowed_selections:
            continue
        if allowed_approaches is not None and dataset.feature_approach not in allowed_approaches:
            continue
        output.append(dataset)
    if args.max_feature_datasets > 0:
        output = output[: args.max_feature_datasets]
    return output


def grid_key(
    dataset: str,
    feature_dataset,
    campaign_id: str,
    meta_train_size: float,
    tabpfn_params: dict,
    lightgbm_params: dict,
    search_space_id: str,
) -> str:
    return stable_json(
        {
            "dataset": dataset,
            "feature_stage": feature_dataset.feature_stage,
            "feature_selection": feature_dataset.feature_selection,
            "feature_approach": feature_dataset.feature_approach,
            "campaign": campaign_id,
            "objective_scope": "campaign_mean",
            "pipeline_search": True,
            "pipeline_search_space_id": search_space_id,
            "meta_train_size": meta_train_size,
            "tabpfn_params": tabpfn_params,
            "lightgbm_params": lightgbm_params,
        }
    )


def load_existing_results(output_path: Path) -> list[dict]:
    path = output_path / "results.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def completed_keys(results: list[dict], retry_errors: bool) -> set[str]:
    done = set()
    for result in results:
        if retry_errors and result.get("status") not in {"ok", "skipped"}:
            continue
        key = result.get("grid_key")
        if key:
            done.add(str(key))
    return done


def normalize_for_csv(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = result.copy()
        for key, value in list(row.items()):
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_results(results: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame([row for row in results if row.get("status") == "ok"])
    if frame.empty:
        return pd.DataFrame()
    metrics = [column for column in ("balanced_accuracy", "macro_f1", "weighted_f1", "mcc", "fit_predict_seconds") if column in frame.columns]
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    groups = [
        "dataset",
        "feature_stage",
        "feature_selection",
        "feature_approach",
        "meta_train_size",
        "tabpfn_params_key",
        "lightgbm_params_key",
    ]
    summary = frame.groupby(groups, dropna=False)[metrics].agg(["mean", "std", "max", "count"]).reset_index()
    summary.columns = ["_".join(str(part) for part in column if part) if isinstance(column, tuple) else str(column) for column in summary.columns]
    return summary.sort_values(["balanced_accuracy_mean", "macro_f1_mean"], ascending=[False, False])


def save_results(output_path: Path, results: list[dict]) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    normalize_for_csv(results).to_csv(output_path / "results.csv", index=False)
    summary = summarize_results(results)
    if not summary.empty:
        summary.to_csv(output_path / "summary.csv", index=False)


def remove_retryable_errors(results: list[dict], retry_errors: bool) -> list[dict]:
    if not retry_errors:
        return results
    return [
        result
        for result in results
        if not (
            result.get("classifier") == CLASSIFIER_NAME
            and result.get("status") in {"error", "failed", "skipped"}
        )
    ]


def make_lightgbm(params: dict, random_state: int):
    try:
        from lightgbm import LGBMClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("lightgbm is required for this Bayesian optimization") from exc
    return LGBMClassifier(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        min_child_samples=int(params["min_child_samples"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        class_weight="balanced",
        random_state=random_state,
        n_jobs=int(params["n_jobs"]),
        force_col_wise=True,
        verbosity=-1,
    )


def split_context_and_meta(stack_module, train_idx: np.ndarray, y: np.ndarray, meta_train_size: float, random_state: int):
    return stack_module.split_context_and_meta(train_idx, y, meta_train_size, random_state)


def prepare_embedding_folds(
    base,
    stack_module,
    feature_dataset,
    tabpfn_factory,
    tabpfn_params: dict,
    meta_train_size: float,
    args: Namespace,
):
    y_text = feature_dataset.target["attack_type"].fillna(MISSING).astype(str).to_numpy()
    if len(np.unique(y_text)) < 2:
        raise ValueError("campaign target has fewer than two classes")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    class_names = [str(item) for item in label_encoder.classes_.tolist()]
    if len(class_names) != 2:
        raise ValueError(f"Bayesian optimization expects binary target, got {class_names}")

    splits, split_strategy, effective_folds = base.campaign_kfold_splits(y, args.k_folds, args.random_state)
    cpu_limited_run = base.effective_cpu_device(tabpfn_params.get("device")) and not args.allow_cpu_large_dataset
    folds = []

    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
        original_train_rows = int(len(train_idx))
        original_test_rows = int(len(test_idx))
        if cpu_limited_run:
            train_idx = base.downsample_indices(train_idx, y, int(tabpfn_params.get("max_cpu_train_rows") or 0), args.random_state + fold_number)
            test_idx = base.downsample_indices(test_idx, y, int(tabpfn_params.get("max_cpu_test_rows") or 0), args.random_state + 10_000 + fold_number)

        context_idx, meta_idx, split_type = split_context_and_meta(
            stack_module,
            train_idx,
            y,
            meta_train_size,
            args.random_state + 100_000 + fold_number,
        )
        X_context = base.maybe_dense(feature_dataset.X[context_idx], args.max_dense_cells)
        X_meta_raw = base.maybe_dense(feature_dataset.X[meta_idx], args.max_dense_cells)
        X_test_raw = base.maybe_dense(feature_dataset.X[test_idx], args.max_dense_cells)

        classifier = tabpfn_factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier.fit(X_context, y[context_idx])
            X_meta, embedding_source = stack_module.tabpfn_embeddings_or_proba(base, classifier, X_meta_raw, len(meta_idx), "test")
            X_test, _ = stack_module.tabpfn_embeddings_or_proba(base, classifier, X_test_raw, len(test_idx), "test")

        scaler = StandardScaler()
        X_meta = scaler.fit_transform(X_meta).astype(np.float32, copy=False)
        X_test = scaler.transform(X_test).astype(np.float32, copy=False)
        folds.append(
            {
                "fold_number": int(fold_number),
                "X_meta": X_meta,
                "X_test": X_test,
                "y_meta": y[meta_idx],
                "y_test": y[test_idx],
                "class_names": class_names,
                "labels": np.arange(len(class_names)),
                "n_train": int(len(context_idx)),
                "n_meta_train": int(len(meta_idx)),
                "n_test": int(len(test_idx)),
                "n_train_original": original_train_rows,
                "n_test_original": original_test_rows,
                "cpu_train_sampled": bool(len(train_idx) < original_train_rows),
                "cpu_test_sampled": bool(len(test_idx) < original_test_rows),
                "context_meta_split": split_type,
                "embedding_source": embedding_source,
            }
        )
    return folds, split_strategy, effective_folds


def fold_metrics(y_test: np.ndarray, y_pred: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    recall_values = recall_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "recall_by_class": {
            str(class_name): float(value)
            for class_name, value in zip(class_names, recall_values)
        },
    }


def evaluate_lightgbm_from_folds(base, feature_dataset, campaign_id: str, folds: list[dict], split_strategy: str, effective_folds: int, lightgbm_params: dict, tabpfn_params: dict, meta_train_size: float, dataset: str) -> dict:
    fold_results = []
    for fold in folds:
        started = time.perf_counter()
        model = make_lightgbm(lightgbm_params, random_state=42 + int(fold["fold_number"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(fold["X_meta"], fold["y_meta"])
            y_pred = np.asarray(model.predict(fold["X_test"])).reshape(-1).astype(int)
        elapsed = time.perf_counter() - started
        fold_results.append(
            {
                "n_train": int(fold["n_train"]),
                "n_meta_train": int(fold["n_meta_train"]),
                "n_test": int(fold["n_test"]),
                "n_train_original": int(fold["n_train_original"]),
                "n_test_original": int(fold["n_test_original"]),
                "cpu_train_sampled": bool(fold["cpu_train_sampled"]),
                "cpu_test_sampled": bool(fold["cpu_test_sampled"]),
                "context_meta_split": fold["context_meta_split"],
                "communication_cost": base.communication_cost(feature_dataset.X),
                "number_of_rounds": None,
                "client_variance": None,
                "fairness_between_campaigns": None,
                "performance_by_traffic_source": None,
                "fit_predict_seconds": float(elapsed),
                "elapsed_seconds": float(elapsed),
                "status": "ok",
                "error": None,
                "fold": int(fold["fold_number"]),
                "meta_approach": "tabpfn_embeddings_lightgbm",
                "meta_feature_source": fold["embedding_source"],
                **fold_metrics(fold["y_test"], y_pred, fold["labels"], fold["class_names"]),
            }
        )

    result = base.aggregate_campaign_fold_results(
        fold_results,
        feature_dataset,
        campaign_id,
        CLASSIFIER_NAME,
        dataset=dataset,
    )
    result["cv_strategy"] = split_strategy
    result["k_folds"] = int(effective_folds)
    result["meta_approach"] = "tabpfn_embeddings_lightgbm"
    result["meta_feature_source"] = fold_results[0].get("meta_feature_source") if fold_results else None
    result["meta_train_size"] = float(meta_train_size)
    result["n_meta_train"] = int(sum(fold.get("n_meta_train") or 0 for fold in fold_results))
    result["tabpfn_frozen"] = True
    result["tabpfn_params"] = tabpfn_params
    result["lightgbm_params"] = lightgbm_params
    result["tabpfn_params_key"] = stable_json(tabpfn_params)
    result["lightgbm_params_key"] = stable_json(lightgbm_params)
    result["grid_search_method"] = "tabpfn_embeddings_lightgbm_bayesian_optimization"
    result["search_method"] = "bayesian_optimization"
    result["grid_model"] = CLASSIFIER_NAME
    return result


def failed_result(feature_dataset, campaign_id: str, error: Exception, dataset: str, meta_train_size: float, tabpfn_params: dict, lightgbm_params: dict) -> dict:
    return {
        "feature_stage": feature_dataset.feature_stage,
        "feature_selection": feature_dataset.feature_selection,
        "feature_approach": feature_dataset.feature_approach,
        "classifier": CLASSIFIER_NAME,
        "dataset": dataset,
        "evaluation_scope": "campaign",
        "campaign": campaign_id,
        "campaign_id": campaign_id,
        "n_samples": int(len(feature_dataset.target)),
        "n_features": int(feature_dataset.X.shape[1]),
        "balanced_accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
        "mcc": None,
        "status": "error",
        "error": str(error),
        "meta_train_size": float(meta_train_size),
        "tabpfn_params": tabpfn_params,
        "lightgbm_params": lightgbm_params,
        "tabpfn_params_key": stable_json(tabpfn_params),
        "lightgbm_params_key": stable_json(lightgbm_params),
        "grid_search_method": "tabpfn_embeddings_lightgbm_bayesian_optimization",
        "search_method": "bayesian_optimization",
        "grid_model": CLASSIFIER_NAME,
    }


def main() -> None:
    args = parse_args()
    base = load_module(BASE_SCRIPT, "tabpfn_embedding_lgbm_base")
    stack_module = load_module(STACKING_SCRIPT, "tabpfn_embedding_lgbm_stack")
    tabpfn_module = base.load_tabpfn_module()
    project_root = PROJECT_ROOT
    dataset = args.dataset or base.read_env_value(project_root / args.env_file, "DATASET")
    if dataset not in VALID_DATASETS:
        raise ValueError(f"Invalid DATASET={dataset!r}. Expected one of: {sorted(VALID_DATASETS)}")

    feature_datasets = tabpfn_module.discover_feature_datasets(
        project_root / args.extracted_root,
        project_root / args.selected_root,
        project_root / args.raw_root,
        dataset,
    )
    feature_datasets = filter_feature_datasets(feature_datasets, args)
    if not feature_datasets:
        raise ValueError("No feature datasets were found for this Bayesian optimization.")

    base.configure_tabpfn_environment(project_root, args)
    model_path = base.resolve_model_path(project_root, args)
    output_path = project_root / args.output_root / dataset
    results = remove_retryable_errors(load_existing_results(output_path), args.retry_errors)
    done = completed_keys(results, args.retry_errors)
    studies_path = output_path / "optuna_studies"

    pipeline_candidates = []
    for feature_dataset in feature_datasets:
        feature_dataset = base.cap_rows_by_campaign(feature_dataset, args.max_rows, args.random_state)
        campaigns = base.campaign_indices(feature_dataset)
        if args.max_campaigns > 0:
            campaigns = campaigns[: args.max_campaigns]
        pipeline_candidates.append(
            {
                "key": feature_dataset_key(feature_dataset),
                "feature_dataset": feature_dataset,
                "campaigns": campaigns,
                "feature_stage": feature_dataset.feature_stage,
                "feature_selection": feature_dataset.feature_selection,
                "feature_approach": feature_dataset.feature_approach,
            }
        )
    if not pipeline_candidates:
        raise ValueError("No pipeline candidates were found for this Bayesian optimization.")

    search_space_id = pipeline_search_space_id([item["key"] for item in pipeline_candidates])
    plan = [
        {
            "pipeline_key": item["key"],
            "pipeline_search_space_id": search_space_id,
            "feature_stage": item["feature_stage"],
            "feature_selection": item["feature_selection"],
            "feature_approach": item["feature_approach"],
            "campaigns": len(item["campaigns"]),
            "objective": "mean_balanced_accuracy_across_campaigns",
            "search_space": "feature_pipeline + tabpfn + lightgbm",
            "new_trials_this_run": int(args.n_trials),
            "sampler": "TPESampler",
        }
        for item in pipeline_candidates
    ]

    if args.plan_only:
        output_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(plan).to_csv(output_path / "plan.csv", index=False)
        print(f"Saved Bayesian optimization plan with {len(plan)} pipeline candidates to {output_path / 'plan.csv'}")
        return

    optuna = load_optuna()
    pipeline_by_key = {item["key"]: item for item in pipeline_candidates}
    print(
        f"TabPFN embeddings + LightGBM Bayesian optimization: pipeline_candidates={len(pipeline_candidates)}, "
        f"new_trials_this_run={args.n_trials}, objective=campaign_mean, search=pipeline+tabpfn+lightgbm, "
        f"search_space_id={search_space_id}",
        flush=True,
    )

    current_study_name = study_name(dataset, search_space_id)
    studies_path.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{studies_path / (current_study_name + '.db')}"
    sampler = optuna.samplers.TPESampler(
        seed=int(args.random_state),
        n_startup_trials=int(args.n_startup_trials),
        multivariate=True,
        warn_independent_sampling=False,
    )
    study = optuna.create_study(
        study_name=current_study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True,
    )
    folds_cache: dict[str, tuple[list[dict], str, int]] = {}
    completed_before = len(
        [
            trial
            for trial in study.trials
            if trial.state in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL}
        ]
    )
    new_trials_this_run = int(args.n_trials)
    run_trial_counter = {"count": 0}

    def objective(trial) -> float:
        pipeline_key = trial.suggest_categorical("feature_pipeline", list(pipeline_by_key.keys()))
        selected_pipeline = pipeline_by_key[pipeline_key]
        feature_dataset = selected_pipeline["feature_dataset"]
        campaign_datasets = [
            (campaign_id, base.subset_feature_dataset(feature_dataset, indices, campaign_id))
            for campaign_id, indices in selected_pipeline["campaigns"]
        ]
        meta_train_size, tabpfn_params, lightgbm_params = suggest_trial_params(trial, args)
        run_trial_counter["count"] += 1
        run_trial_number = int(run_trial_counter["count"])
        trial_results = []
        campaign_metrics = []

        for campaign_id, campaign_dataset in campaign_datasets:
            key = grid_key(
                dataset,
                campaign_dataset,
                campaign_id,
                meta_train_size,
                tabpfn_params,
                lightgbm_params,
                search_space_id,
            )
            if key in done:
                matching = [
                    row
                    for row in results
                    if row.get("grid_key") == key and row.get("status") == "ok" and row.get(PRIMARY_METRIC) is not None
                ]
                if matching:
                    campaign_metrics.append(float(matching[-1][PRIMARY_METRIC]))
                    continue

            def tabpfn_factory(params=tabpfn_params):
                return tabpfn_module.make_tabpfn_classifier(
                    random_state=args.random_state,
                    device=params.get("device", "auto"),
                    ignore_pretraining_limits=bool(params.get("ignore_pretraining_limits", False)),
                    model_path=model_path,
                    n_estimators=int(params.get("n_estimators", 8)),
                )

            cache_key = stable_json(
                {
                    "pipeline": pipeline_key,
                    "campaign": campaign_id,
                    "meta_train_size": meta_train_size,
                    "tabpfn_params": tabpfn_params,
                }
            )
            try:
                if cache_key not in folds_cache:
                    folds_cache[cache_key] = prepare_embedding_folds(
                        base,
                        stack_module,
                        campaign_dataset,
                        tabpfn_factory,
                        tabpfn_params,
                        meta_train_size,
                        args,
                    )
                folds, split_strategy, effective_folds = folds_cache[cache_key]
                result = evaluate_lightgbm_from_folds(
                    base,
                    campaign_dataset,
                    campaign_id,
                    folds,
                    split_strategy,
                    effective_folds,
                    lightgbm_params,
                    tabpfn_params,
                    meta_train_size,
                    dataset,
                )
            except Exception as exc:
                result = failed_result(campaign_dataset, campaign_id, exc, dataset, meta_train_size, tabpfn_params, lightgbm_params)
                result["grid_key"] = key
                result["bayesian_study_name"] = current_study_name
                result["bayesian_trial"] = int(trial.number)
                result["bayesian_run_trial"] = run_trial_number
                result["bayesian_trial_params"] = dict(trial.params)
                result["bayesian_sampler"] = "TPESampler"
                result["bayesian_objective_scope"] = "campaign_mean"
                result["bayesian_pipeline_search"] = True
                result["bayesian_feature_pipeline"] = pipeline_key
                result["bayesian_search_space_id"] = search_space_id
                result["new_trials_requested_this_run"] = new_trials_this_run
                result["completed_trials_before_run"] = int(completed_before)
                trial_results.append(result)
                print(
                    f"[run_trial {run_trial_number}/{new_trials_this_run} optuna_trial {trial.number}] {campaign_id} "
                    f"{campaign_dataset.feature_approach}/{campaign_dataset.feature_selection or 'none'}: "
                    f"{result['status']} ba={result.get(PRIMARY_METRIC)}",
                    flush=True,
                )
                continue

            result["grid_key"] = key
            result["bayesian_study_name"] = current_study_name
            result["bayesian_trial"] = int(trial.number)
            result["bayesian_run_trial"] = run_trial_number
            result["bayesian_trial_params"] = dict(trial.params)
            result["bayesian_sampler"] = "TPESampler"
            result["bayesian_objective_scope"] = "campaign_mean"
            result["bayesian_pipeline_search"] = True
            result["bayesian_feature_pipeline"] = pipeline_key
            result["bayesian_search_space_id"] = search_space_id
            result["new_trials_requested_this_run"] = new_trials_this_run
            result["completed_trials_before_run"] = int(completed_before)
            trial_results.append(result)
            metric = result.get(PRIMARY_METRIC)
            print(
                f"[run_trial {run_trial_number}/{new_trials_this_run} optuna_trial {trial.number}] {campaign_id} "
                f"{campaign_dataset.feature_approach}/{campaign_dataset.feature_selection or 'none'}: "
                f"{result['status']} ba={metric}",
                flush=True,
            )
            if metric is not None:
                campaign_metrics.append(float(metric))

        if len(campaign_metrics) < len(campaign_datasets):
            for result in trial_results:
                results.append(result)
                done.add(str(result.get("grid_key")))
            save_results(output_path, results)
            raise optuna.TrialPruned(
                f"trial evaluated {len(campaign_metrics)}/{len(campaign_datasets)} campaigns successfully"
            )

        campaign_mean = float(np.mean(campaign_metrics))
        campaign_std = float(np.std(campaign_metrics, ddof=1)) if len(campaign_metrics) > 1 else 0.0
        for result in trial_results:
            result["trial_campaign_balanced_accuracy_mean"] = campaign_mean
            result["trial_campaign_balanced_accuracy_std"] = campaign_std
            result["trial_campaign_count"] = int(len(campaign_metrics))
            results.append(result)
            done.add(str(result.get("grid_key")))
        save_results(output_path, results)
        print(
            f"[run_trial {run_trial_number}/{new_trials_this_run} optuna_trial {trial.number}] "
            f"{feature_dataset.feature_approach}/{feature_dataset.feature_selection or 'none'}: "
            f"campaign_mean_ba={campaign_mean} campaign_std_ba={campaign_std}",
            flush=True,
        )
        return campaign_mean

    try:
        study.optimize(
            objective,
            n_trials=new_trials_this_run,
            timeout=args.timeout_seconds,
            gc_after_trial=True,
            show_progress_bar=False,
            catch=(Exception,),
        )
    except Exception as exc:
        print(f"Study {current_study_name} stopped with error: {exc}", flush=True)

    best_value = None
    best_params = None
    try:
        best_value = float(study.best_value)
        best_params = dict(study.best_params)
    except Exception:
        pass
    print(
        f"[study 1/1] {current_study_name}: "
        f"completed_before={completed_before} new_trials_this_run={run_trial_counter['count']} "
        f"total_trials={len(study.trials)} best={best_value} params={best_params}",
        flush=True,
    )

    save_results(output_path, results)
    print(f"Saved TabPFN embeddings + LightGBM Bayesian optimization results to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise
