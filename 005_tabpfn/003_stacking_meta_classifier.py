#!/usr/bin/env python3
"""
Evaluate TabPFN stacking/meta-classifier alternatives.

The TabPFN checkpoint is kept frozen. For each campaign and fold, the training
split is divided into a TabPFN context split and a meta-classifier split. TabPFN
is fitted only on the context split, generates probabilities or embeddings for
the meta split and the test fold, and a classical model is trained on those
derived signals.

Inputs:
    results/grid_search/tabpfn/{DATASET}/results.csv
    data/extracted_features/{approach}/{DATASET}
    data/selected_features/{selector}/{approach}/{DATASET}

Outputs:
    results/tabpfn_stacking_meta_classifier/{DATASET}/results.csv
    results/tabpfn_stacking_meta_classifier/{DATASET}/results.json
    results/tabpfn_stacking_meta_classifier/{DATASET}/summary.csv
    results/tabpfn_stacking_meta_classifier/{DATASET}/best_tabpfn_config.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils._env import VALID_DATASETS, MISSING, PRIMARY_METRIC  # noqa: E402

BASE_SCRIPT = PROJECT_ROOT / "005_tabpfn" / "000_fine_tune_head.py"

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


# Pipeline configuration constants. Edit these values to change this script.
CONFIG_ENV_FILE = ".env"
CONFIG_GRID_RESULTS_ROOT = "results/grid_search/tabpfn"
CONFIG_EXTRACTED_ROOT = "data/extracted_features"
CONFIG_SELECTED_ROOT = "data/selected_features"
CONFIG_RAW_ROOT = "data/raw"
CONFIG_OUTPUT_ROOT = "results/tabpfn_stacking_meta_classifier"
CONFIG_MODEL_CACHE = "models/tabpfn"
CONFIG_MODEL_PATH = None
CONFIG_ALLOW_BROWSER_LOGIN = False
CONFIG_DATASET = None
CONFIG_RANDOM_STATE = 42
CONFIG_K_FOLDS = 5
CONFIG_MAX_ROWS = 0
CONFIG_MAX_CAMPAIGNS = 0
CONFIG_MAX_DENSE_CELLS = 10_000_000
CONFIG_ALLOW_CPU_LARGE_DATASET = False
CONFIG_META_TRAIN_SIZE = 0.35
CONFIG_META_APPROACHES = (
    "tabpfn_proba_lightgbm,"
    "tabpfn_proba_catboost,"
    "tabpfn_proba_logistic,"
    "tabpfn_embeddings_lightgbm,"
    "original_plus_tabpfn_proba_lightgbm"
)
CONFIG_LIGHTGBM_N_ESTIMATORS = 300
CONFIG_LIGHTGBM_LEARNING_RATE = 0.03
CONFIG_LIGHTGBM_NUM_LEAVES = 31
CONFIG_LIGHTGBM_N_JOBS = 1
CONFIG_CATBOOST_ITERATIONS = 300
CONFIG_CATBOOST_LEARNING_RATE = 0.03
CONFIG_CATBOOST_DEPTH = 6
CONFIG_LOGISTIC_C = 1.0
CONFIG_RETRY_ERRORS = True


META_CLASSIFIERS = {
    "tabpfn_proba_lightgbm": "TabPFN-Proba-LightGBM",
    "tabpfn_proba_catboost": "TabPFN-Proba-CatBoost",
    "tabpfn_proba_logistic": "TabPFN-Proba-LogisticRegression",
    "tabpfn_embeddings_lightgbm": "TabPFN-Embeddings-LightGBM",
    "original_plus_tabpfn_proba_lightgbm": "OriginalPlus-TabPFN-Proba-LightGBM",
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("tabpfn_stacking_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> Namespace:
    """Return script configuration from global constants; command-line args are intentionally ignored."""
    return Namespace(
        env_file=CONFIG_ENV_FILE,
        grid_results_root=CONFIG_GRID_RESULTS_ROOT,
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
        meta_train_size=CONFIG_META_TRAIN_SIZE,
        meta_approaches=CONFIG_META_APPROACHES,
        lightgbm_n_estimators=CONFIG_LIGHTGBM_N_ESTIMATORS,
        lightgbm_learning_rate=CONFIG_LIGHTGBM_LEARNING_RATE,
        lightgbm_num_leaves=CONFIG_LIGHTGBM_NUM_LEAVES,
        lightgbm_n_jobs=CONFIG_LIGHTGBM_N_JOBS,
        catboost_iterations=CONFIG_CATBOOST_ITERATIONS,
        catboost_learning_rate=CONFIG_CATBOOST_LEARNING_RATE,
        catboost_depth=CONFIG_CATBOOST_DEPTH,
        logistic_c=CONFIG_LOGISTIC_C,
        retry_errors=CONFIG_RETRY_ERRORS,
    )


def parse_csv_values(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def best_config_to_dict(best) -> dict[str, Any]:
    return {
        "feature_stage": best.feature_stage,
        "feature_selection": best.feature_selection,
        "feature_approach": best.feature_approach,
        "grid_params": best.grid_params,
        "source_balanced_accuracy_mean": best.source_balanced_accuracy_mean,
        "source_balanced_accuracy_std": best.source_balanced_accuracy_std,
        "source_macro_f1_mean": best.source_macro_f1_mean,
        "source_runs": best.source_runs,
    }


def stacking_key(dataset: str, campaign_id: str, best, meta_approach: str) -> str:
    return json.dumps(
        {
            "dataset": dataset,
            "campaign": campaign_id,
            "best_tabpfn_config": best_config_to_dict(best),
            "meta_approach": meta_approach,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def completed_stacking_keys(results: list[dict], retry_errors: bool) -> set[str]:
    done = set()
    for result in results:
        if retry_errors and result.get("status") not in {"ok", "skipped"}:
            continue
        key = result.get("stacking_key")
        if key:
            done.add(str(key))
    return done


def split_context_and_meta(train_idx: np.ndarray, y: np.ndarray, meta_train_size: float, random_state: int):
    if meta_train_size <= 0 or len(train_idx) < 4:
        return train_idx, train_idx, "shared_train_context"
    y_train = y[train_idx]
    class_counts = pd.Series(y_train).value_counts()
    stratify = y_train if len(class_counts) == 2 and class_counts.min() >= 2 else None
    try:
        context_pos, meta_pos = train_test_split(
            np.arange(len(train_idx)),
            test_size=float(meta_train_size),
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        return train_idx, train_idx, "shared_train_context"
    context_idx = np.sort(train_idx[context_pos])
    meta_idx = np.sort(train_idx[meta_pos])
    if len(np.unique(y[context_idx])) < 2 or len(np.unique(y[meta_idx])) < 2:
        return train_idx, train_idx, "shared_train_context"
    return context_idx, meta_idx, "context_meta_split"


def positive_proba_column(classifier, proba: np.ndarray, positive_label: int) -> int:
    classes = getattr(classifier, "classes_", None)
    if classes is not None:
        classes = np.asarray(classes).reshape(-1)
        matches = np.flatnonzero(classes.astype(int) == int(positive_label))
        if len(matches):
            return int(matches[0])
    if positive_label < proba.shape[1]:
        return int(positive_label)
    raise ValueError(f"Could not locate positive label {positive_label} in proba with shape {proba.shape}")


def tabpfn_probabilities(classifier, X) -> np.ndarray:
    return np.asarray(classifier.predict_proba(X), dtype=np.float32)


def tabpfn_embeddings_or_proba(base, classifier, X, expected_rows: int, data_source: str) -> tuple[np.ndarray, str]:
    try:
        embeddings = classifier.get_embeddings(X, data_source=data_source)
        return base.ensure_2d_embeddings(embeddings, expected_rows), "tabpfn_embeddings"
    except Exception:
        proba = classifier.predict_proba(X)
        return base.ensure_2d_embeddings(proba, expected_rows), "predict_proba_fallback"


def original_plus_proba(X_original, proba: np.ndarray):
    proba = np.asarray(proba, dtype=np.float32)
    if sparse.issparse(X_original):
        return sparse.hstack([X_original, sparse.csr_matrix(proba)], format="csr")
    return np.hstack([np.asarray(X_original, dtype=np.float32), proba]).astype(np.float32, copy=False)


def make_meta_classifier(meta_approach: str, args: Namespace, random_state: int):
    if meta_approach.endswith("lightgbm"):
        try:
            from lightgbm import LGBMClassifier
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("lightgbm is required for this stacking approach") from exc
        return LGBMClassifier(
            n_estimators=int(args.lightgbm_n_estimators),
            learning_rate=float(args.lightgbm_learning_rate),
            num_leaves=int(args.lightgbm_num_leaves),
            class_weight="balanced",
            random_state=random_state,
            n_jobs=int(args.lightgbm_n_jobs),
            force_col_wise=True,
            verbosity=-1,
        )
    if meta_approach.endswith("catboost"):
        try:
            from catboost import CatBoostClassifier
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("catboost is required for this stacking approach") from exc
        return CatBoostClassifier(
            iterations=int(args.catboost_iterations),
            learning_rate=float(args.catboost_learning_rate),
            depth=int(args.catboost_depth),
            loss_function="Logloss",
            auto_class_weights="Balanced",
            random_seed=random_state,
            verbose=False,
        )
    if meta_approach.endswith("logistic"):
        return LogisticRegression(
            C=float(args.logistic_c),
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        )
    raise ValueError(f"Unknown meta approach: {meta_approach}")


def build_meta_features(
    base,
    classifier,
    feature_dataset,
    meta_approach: str,
    meta_idx: np.ndarray,
    test_idx: np.ndarray,
    max_dense_cells: int,
):
    X_meta_original = feature_dataset.X[meta_idx]
    X_test_original = feature_dataset.X[test_idx]

    if meta_approach in {
        "tabpfn_proba_lightgbm",
        "tabpfn_proba_catboost",
        "tabpfn_proba_logistic",
    }:
        return (
            tabpfn_probabilities(classifier, base.maybe_dense(X_meta_original, max_dense_cells)),
            tabpfn_probabilities(classifier, base.maybe_dense(X_test_original, max_dense_cells)),
            "tabpfn_predict_proba",
        )

    if meta_approach == "tabpfn_embeddings_lightgbm":
        X_meta_dense = base.maybe_dense(X_meta_original, max_dense_cells)
        X_test_dense = base.maybe_dense(X_test_original, max_dense_cells)
        X_meta, source = tabpfn_embeddings_or_proba(base, classifier, X_meta_dense, len(meta_idx), "test")
        X_test, _source_test = tabpfn_embeddings_or_proba(base, classifier, X_test_dense, len(test_idx), "test")
        scaler = StandardScaler()
        return (
            scaler.fit_transform(X_meta).astype(np.float32, copy=False),
            scaler.transform(X_test).astype(np.float32, copy=False),
            source,
        )

    if meta_approach == "original_plus_tabpfn_proba_lightgbm":
        X_meta_dense = base.maybe_dense(X_meta_original, max_dense_cells)
        X_test_dense = base.maybe_dense(X_test_original, max_dense_cells)
        meta_proba = tabpfn_probabilities(classifier, X_meta_dense)
        test_proba = tabpfn_probabilities(classifier, X_test_dense)
        return (
            original_plus_proba(X_meta_original, meta_proba),
            original_plus_proba(X_test_original, test_proba),
            "original_features_plus_tabpfn_predict_proba",
        )

    raise ValueError(f"Unknown meta approach: {meta_approach}")


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


def evaluate_campaign_stacking(base, feature_dataset, campaign_id: str, factory, args: Namespace, best, meta_approach: str) -> dict:
    y_text = feature_dataset.target["attack_type"].fillna(MISSING).astype(str).to_numpy()
    if len(np.unique(y_text)) < 2:
        raise ValueError("campaign target has fewer than two classes")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    class_names = [str(item) for item in label_encoder.classes_.tolist()]
    if len(class_names) != 2:
        raise ValueError(f"stacking expects binary target, got {class_names}")

    splits, split_strategy, effective_folds = base.campaign_kfold_splits(y, args.k_folds, args.random_state)
    params = best.grid_params
    cpu_limited_run = base.effective_cpu_device(params.get("device")) and not args.allow_cpu_large_dataset
    labels = np.arange(len(class_names))
    fold_results = []

    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
        started = time.perf_counter()
        original_train_rows = int(len(train_idx))
        original_test_rows = int(len(test_idx))
        if cpu_limited_run:
            train_idx = base.downsample_indices(train_idx, y, int(params.get("max_cpu_train_rows") or 0), args.random_state + fold_number)
            test_idx = base.downsample_indices(test_idx, y, int(params.get("max_cpu_test_rows") or 0), args.random_state + 10_000 + fold_number)

        context_idx, meta_idx, split_type = split_context_and_meta(
            train_idx,
            y,
            args.meta_train_size,
            args.random_state + 100_000 + fold_number,
        )
        X_context = base.maybe_dense(feature_dataset.X[context_idx], args.max_dense_cells)
        y_context = y[context_idx]
        y_meta = y[meta_idx]
        y_test = y[test_idx]

        classifier = factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier.fit(X_context, y_context)
            X_meta, X_test, meta_feature_source = build_meta_features(
                base,
                classifier,
                feature_dataset,
                meta_approach,
                meta_idx,
                test_idx,
                args.max_dense_cells,
            )
            meta_classifier = make_meta_classifier(meta_approach, args, args.random_state + fold_number)
            meta_classifier.fit(X_meta, y_meta)
            y_pred = np.asarray(meta_classifier.predict(X_test)).reshape(-1).astype(int)

        elapsed = time.perf_counter() - started
        fold_results.append(
            {
                "n_train": int(len(context_idx)),
                "n_meta_train": int(len(meta_idx)),
                "n_test": int(len(test_idx)),
                "n_train_original": original_train_rows,
                "n_test_original": original_test_rows,
                "cpu_train_sampled": bool(len(train_idx) < original_train_rows),
                "cpu_test_sampled": bool(len(test_idx) < original_test_rows),
                "context_meta_split": split_type,
                "communication_cost": base.communication_cost(feature_dataset.X),
                "number_of_rounds": None,
                "client_variance": None,
                "fairness_between_campaigns": None,
                "performance_by_traffic_source": None,
                "fit_predict_seconds": float(elapsed),
                "elapsed_seconds": float(elapsed),
                "status": "ok",
                "error": None,
                "fold": fold_number,
                "meta_approach": meta_approach,
                "meta_feature_source": meta_feature_source,
                **fold_metrics(y_test, y_pred, labels, class_names),
            }
        )

    classifier_name = META_CLASSIFIERS[meta_approach]
    result = base.aggregate_campaign_fold_results(
        fold_results,
        feature_dataset,
        campaign_id,
        classifier_name,
    )
    result["cv_strategy"] = split_strategy
    result["k_folds"] = int(effective_folds)
    result["meta_approach"] = meta_approach
    result["meta_feature_source"] = fold_results[0].get("meta_feature_source") if fold_results else None
    result["meta_train_size"] = float(args.meta_train_size)
    result["n_meta_train"] = int(sum(fold.get("n_meta_train") or 0 for fold in fold_results))
    result["tabpfn_frozen"] = True
    result["fine_tuned_component"] = None
    result["source_grid_params"] = best.grid_params
    result["lightgbm_params"] = {
        "n_estimators": int(args.lightgbm_n_estimators),
        "learning_rate": float(args.lightgbm_learning_rate),
        "num_leaves": int(args.lightgbm_num_leaves),
        "n_jobs": int(args.lightgbm_n_jobs),
    } if meta_approach.endswith("lightgbm") else None
    result["catboost_params"] = {
        "iterations": int(args.catboost_iterations),
        "learning_rate": float(args.catboost_learning_rate),
        "depth": int(args.catboost_depth),
    } if meta_approach.endswith("catboost") else None
    result["logistic_params"] = {
        "C": float(args.logistic_c),
        "class_weight": "balanced",
    } if meta_approach.endswith("logistic") else None
    return result


def failed_result(feature_dataset, campaign_id: str, classifier_name: str, error: Exception, best, meta_approach: str) -> dict:
    return {
        "feature_stage": feature_dataset.feature_stage,
        "feature_selection": feature_dataset.feature_selection,
        "feature_approach": feature_dataset.feature_approach,
        "classifier": classifier_name,
        "dataset": None,
        "evaluation_scope": "campaign",
        "campaign": campaign_id,
        "campaign_id": campaign_id,
        "n_samples": int(len(feature_dataset.target)),
        "n_features": int(feature_dataset.X.shape[1]),
        "balanced_accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
        "mcc": None,
        "recall_by_class": None,
        "status": "error",
        "error": str(error),
        "meta_approach": meta_approach,
        "tabpfn_frozen": True,
        "fine_tuned_component": None,
        "source_grid_params": best.grid_params,
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


def summarize_results(results: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame([row for row in results if row.get("status") == "ok"])
    if frame.empty:
        return pd.DataFrame()
    metrics = [
        column
        for column in (
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "mcc",
            "fit_predict_seconds",
        )
        if column in frame.columns
    ]
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    groups = ["classifier", "meta_approach", "meta_feature_source"]
    summary = frame.groupby(groups, dropna=False)[metrics].agg(["mean", "std", "max", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part) if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    return summary.sort_values(["balanced_accuracy_mean", "macro_f1_mean"], ascending=[False, False])


def save_stacking_results(base, output_path: Path, results: list[dict]) -> None:
    base.save_results(output_path, results, normalize_for_csv)
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
            str(result.get("classifier", "")).startswith(("TabPFN", "OriginalPlus"))
            and result.get("status") in {"error", "failed", "skipped"}
        )
    ]


def main() -> None:
    args = parse_args()
    base = load_base_module()
    project_root = PROJECT_ROOT
    dataset = args.dataset or base.read_env_value(project_root / args.env_file, "DATASET")
    if dataset not in VALID_DATASETS:
        raise ValueError(f"Invalid DATASET={dataset!r}. Expected one of: {sorted(VALID_DATASETS)}")

    meta_approaches = parse_csv_values(args.meta_approaches)
    invalid = [approach for approach in meta_approaches if approach not in META_CLASSIFIERS]
    if invalid:
        raise ValueError(f"Invalid meta approach(es): {invalid}. Expected: {sorted(META_CLASSIFIERS)}")

    module = base.load_tabpfn_module()
    best = base.load_best_tabpfn_config(project_root / args.grid_results_root, dataset)
    feature_dataset = base.load_best_feature_dataset(module, project_root, args, dataset, best)
    feature_dataset = base.cap_rows_by_campaign(feature_dataset, args.max_rows, args.random_state)

    base.configure_tabpfn_environment(project_root, args)
    model_path = base.resolve_model_path(project_root, args)
    params = best.grid_params

    output_path = project_root / args.output_root / dataset
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "best_tabpfn_config.json").write_text(
        json.dumps(best_config_to_dict(best), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    run_config = base.base_config(args, exclude={"retry_errors", "allow_browser_login", "model_path"})
    run_config["dataset"] = dataset
    run_config["best_tabpfn_config"] = best_config_to_dict(best)
    current_config_hash = base.config_hash(run_config)
    results = remove_retryable_errors(base.load_existing_results(output_path), args.retry_errors)
    done = completed_stacking_keys(results, args.retry_errors)

    campaigns = base.campaign_indices(feature_dataset)
    if args.max_campaigns > 0:
        campaigns = campaigns[: args.max_campaigns]
    if not campaigns:
        raise ValueError("No campaign index was found for the selected feature dataset.")

    jobs = [(campaign_id, approach) for campaign_id, _indices in campaigns for approach in meta_approaches]
    print(
        "TabPFN stacking meta-classifier using "
        f"{best.feature_stage}/{best.feature_selection or 'none'}/{best.feature_approach} "
        f"and params={best.grid_params}; jobs={len(jobs)}",
        flush=True,
    )

    def factory():
        return module.make_tabpfn_classifier(
            random_state=args.random_state,
            device=params.get("device", "auto"),
            ignore_pretraining_limits=bool(params.get("ignore_pretraining_limits", False)),
            model_path=model_path,
            n_estimators=int(params.get("n_estimators", 8)),
        )

    completed_now = 0
    for campaign_id, indices in campaigns:
        campaign_dataset = base.subset_feature_dataset(feature_dataset, indices, campaign_id)
        for meta_approach in meta_approaches:
            key = stacking_key(dataset, campaign_id, best, meta_approach)
            classifier_name = META_CLASSIFIERS[meta_approach]
            if key in done:
                completed_now += 1
                print(f"[{completed_now}/{len(jobs)}] {campaign_id} + {classifier_name}: skipped", flush=True)
                continue
            pending_key = {
                "config_hash": current_config_hash,
                "feature_stage": campaign_dataset.feature_stage,
                "feature_selection": campaign_dataset.feature_selection,
                "feature_approach": campaign_dataset.feature_approach,
                "classifier": classifier_name,
                "federated_algorithm": None,
                "evaluation_scope": "campaign",
                "campaign": campaign_id,
                "meta_approach": meta_approach,
            }
            try:
                result = evaluate_campaign_stacking(
                    base,
                    campaign_dataset,
                    campaign_id,
                    factory,
                    args,
                    best,
                    meta_approach,
                )
                result["dataset"] = dataset
            except Exception as exc:
                result = failed_result(campaign_dataset, campaign_id, classifier_name, exc, best, meta_approach)
                result["dataset"] = dataset
            result["stacking_key"] = key
            result["stacking_method"] = "tabpfn_meta_classifier"
            result["grid_model"] = classifier_name
            result = base.add_resume_metadata(result, run_config, pending_key)
            results.append(result)
            done.add(key)
            completed_now += 1
            save_stacking_results(base, output_path, results)
            print(
                f"[{completed_now}/{len(jobs)}] {campaign_id} + {classifier_name}: "
                f"{result['status']} ba={result.get(PRIMARY_METRIC)}",
                flush=True,
            )

    save_stacking_results(base, output_path, results)
    print(f"Saved TabPFN stacking results to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise
