#!/usr/bin/env python3
"""
Calibrate the TabPFN decision threshold using the best TabPFN grid-search setup.

This experiment keeps the TabPFN checkpoint unchanged. For each campaign and
fold, it fits TabPFN on the private training split, selects a binary decision
threshold on an internal calibration split, and evaluates on the held-out fold.
It also compares a light logistic head trained on frozen TabPFN embeddings.

Inputs:
    results/grid_search/tabpfn/{DATASET}/results.csv
    data/extracted_features/{approach}/{DATASET}
    data/selected_features/{selector}/{approach}/{DATASET}

Outputs:
    results/tabpfn_threshold_calibration/{DATASET}/results.csv
    results/tabpfn_threshold_calibration/{DATASET}/results.json
    results/tabpfn_threshold_calibration/{DATASET}/summary.csv
    results/tabpfn_threshold_calibration/{DATASET}/best_tabpfn_config.json
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils._env import VALID_DATASETS, MISSING, PRIMARY_METRIC  # noqa: E402

BASE_SCRIPT = PROJECT_ROOT / "005_tabpfn" / "000_fine_tune_head.py"


# Pipeline configuration constants. Edit these values to change this script.
CONFIG_ENV_FILE = ".env"
CONFIG_GRID_RESULTS_ROOT = "results/grid_search/tabpfn"
CONFIG_EXTRACTED_ROOT = "data/extracted_features"
CONFIG_SELECTED_ROOT = "data/selected_features"
CONFIG_RAW_ROOT = "data/raw"
CONFIG_OUTPUT_ROOT = "results/tabpfn_threshold_calibration"
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
CONFIG_CALIBRATION_SIZE = 0.25
CONFIG_THRESHOLD_START = 0.05
CONFIG_THRESHOLD_END = 0.95
CONFIG_THRESHOLD_STEP = 0.01
CONFIG_SCORE_SOURCES = "tabpfn_proba_default,tabpfn_proba,tabpfn_embedding_logistic"
CONFIG_LOGISTIC_C_GRID = "0.1,1.0,10.0"
CONFIG_RETRY_ERRORS = True


CLASSIFIER_NAMES = {
    "tabpfn_proba_default": "TabPFN-Threshold-Default",
    "tabpfn_proba": "TabPFN-Threshold-Calibrated",
    "tabpfn_embedding_logistic": "TabPFN-Embedding-Logistic-Threshold",
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("tabpfn_threshold_base", BASE_SCRIPT)
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
        calibration_size=CONFIG_CALIBRATION_SIZE,
        threshold_start=CONFIG_THRESHOLD_START,
        threshold_end=CONFIG_THRESHOLD_END,
        threshold_step=CONFIG_THRESHOLD_STEP,
        score_sources=CONFIG_SCORE_SOURCES,
        logistic_c_grid=CONFIG_LOGISTIC_C_GRID,
        retry_errors=CONFIG_RETRY_ERRORS,
    )


def parse_csv_values(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_float_grid(value: str) -> list[float]:
    parsed = [float(item) for item in parse_csv_values(value)]
    if not parsed:
        raise ValueError("float grid cannot be empty")
    return parsed


def threshold_grid(args: Namespace) -> np.ndarray:
    if args.threshold_step <= 0:
        raise ValueError("threshold_step must be positive")
    values = np.arange(
        float(args.threshold_start),
        float(args.threshold_end) + (float(args.threshold_step) / 2.0),
        float(args.threshold_step),
    )
    values = np.clip(values, 0.0, 1.0)
    return np.unique(np.round(values, 6))


def threshold_key(dataset: str, campaign_id: str, best, score_source: str, logistic_c: float | None) -> str:
    return json.dumps(
        {
            "dataset": dataset,
            "campaign": campaign_id,
            "best_tabpfn_config": best_config_to_dict(best),
            "score_source": score_source,
            "logistic_c": logistic_c,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def completed_threshold_keys(results: list[dict], retry_errors: bool) -> set[str]:
    done = set()
    for result in results:
        if retry_errors and result.get("status") not in {"ok", "skipped"}:
            continue
        key = result.get("threshold_calibration_key")
        if key:
            done.add(str(key))
    return done


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


def binary_labels(label_classes: list[str], positive_class_name: str = "attack") -> tuple[int, int]:
    normalized = [str(item).lower() for item in label_classes]
    if positive_class_name in normalized:
        positive_label = normalized.index(positive_class_name)
    elif len(label_classes) == 2:
        positive_label = 1
    else:
        raise ValueError(f"Could not infer positive class from labels: {label_classes}")
    negative = [idx for idx in range(len(label_classes)) if idx != positive_label]
    if len(negative) != 1:
        raise ValueError(f"Threshold calibration expects binary labels, got: {label_classes}")
    return int(positive_label), int(negative[0])


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


def labels_from_scores(scores: np.ndarray, threshold: float, positive_label: int, negative_label: int) -> np.ndarray:
    return np.where(np.asarray(scores) >= float(threshold), positive_label, negative_label).astype(int)


def choose_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
    positive_label: int,
    negative_label: int,
) -> tuple[float, float, float]:
    best_threshold = 0.5
    best_ba = -np.inf
    best_macro_f1 = -np.inf
    for threshold in thresholds:
        y_pred = labels_from_scores(scores, float(threshold), positive_label, negative_label)
        ba = float(balanced_accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        better = ba > best_ba
        tied_better = np.isclose(ba, best_ba) and (
            macro_f1 > best_macro_f1 or abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
        )
        if better or tied_better:
            best_threshold = float(threshold)
            best_ba = ba
            best_macro_f1 = macro_f1
    return best_threshold, best_ba, best_macro_f1


def calibration_split(train_idx: np.ndarray, y: np.ndarray, calibration_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    if calibration_size <= 0 or len(train_idx) < 4:
        return train_idx, train_idx
    y_train = y[train_idx]
    class_counts = pd.Series(y_train).value_counts()
    stratify = y_train if len(class_counts) == 2 and class_counts.min() >= 2 else None
    try:
        fit_pos, cal_pos = train_test_split(
            np.arange(len(train_idx)),
            test_size=calibration_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        return train_idx, train_idx
    fit_idx = np.sort(train_idx[fit_pos])
    cal_idx = np.sort(train_idx[cal_pos])
    if len(np.unique(y[fit_idx])) < 2 or len(np.unique(y[cal_idx])) < 2:
        return train_idx, train_idx
    return fit_idx, cal_idx


def maybe_get_embeddings(base, classifier, X_fit, X_other, expected_rows: int, data_source: str) -> np.ndarray:
    try:
        embeddings = classifier.get_embeddings(X_other, data_source=data_source)
        return base.ensure_2d_embeddings(embeddings, expected_rows)
    except Exception:
        proba = classifier.predict_proba(X_other)
        return base.ensure_2d_embeddings(proba, expected_rows)


def best_embedding_logistic(
    X_fit_embeddings: np.ndarray,
    y_fit: np.ndarray,
    X_cal_embeddings: np.ndarray,
    y_cal: np.ndarray,
    c_grid: list[float],
    random_state: int,
    positive_label: int,
) -> tuple[LogisticRegression, float, np.ndarray]:
    best_model = None
    best_c = c_grid[0]
    best_score = -np.inf
    best_scores = None
    for c_value in c_grid:
        model = LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        )
        model.fit(X_fit_embeddings, y_fit)
        proba = model.predict_proba(X_cal_embeddings)
        column = positive_proba_column(model, proba, positive_label)
        scores = proba[:, column]
        y_pred = labels_from_scores(scores, 0.5, positive_label, 1 - positive_label)
        score = float(balanced_accuracy_score(y_cal, y_pred))
        if score > best_score:
            best_model = model
            best_c = float(c_value)
            best_score = score
            best_scores = scores
    if best_model is None or best_scores is None:
        raise ValueError("Could not train embedding logistic calibration model")
    return best_model, best_c, best_scores


def fold_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
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


def evaluate_campaign_threshold(
    base,
    feature_dataset,
    campaign_id: str,
    factory,
    args: Namespace,
    best,
    score_source: str,
    logistic_c_override: float | None,
) -> dict:
    y_text = feature_dataset.target["attack_type"].fillna(MISSING).astype(str).to_numpy()
    if len(np.unique(y_text)) < 2:
        raise ValueError("campaign target has fewer than two classes")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    class_names = [str(item) for item in label_encoder.classes_.tolist()]
    if len(class_names) != 2:
        raise ValueError(f"threshold calibration expects binary target, got {class_names}")
    positive_label, negative_label = binary_labels(class_names)

    splits, split_strategy, effective_folds = base.campaign_kfold_splits(y, args.k_folds, args.random_state)
    thresholds = threshold_grid(args)
    params = best.grid_params
    cpu_limited_run = base.effective_cpu_device(params.get("device")) and not args.allow_cpu_large_dataset
    c_grid = [float(logistic_c_override)] if logistic_c_override is not None else parse_float_grid(args.logistic_c_grid)
    labels = np.arange(len(class_names))
    fold_results = []

    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
        started = time.perf_counter()
        original_train_rows = int(len(train_idx))
        original_test_rows = int(len(test_idx))
        if cpu_limited_run:
            train_idx = base.downsample_indices(train_idx, y, int(params.get("max_cpu_train_rows") or 0), args.random_state + fold_number)
            test_idx = base.downsample_indices(test_idx, y, int(params.get("max_cpu_test_rows") or 0), args.random_state + 10_000 + fold_number)

        fit_idx, cal_idx = calibration_split(
            train_idx,
            y,
            float(args.calibration_size),
            args.random_state + 100_000 + fold_number,
        )

        X_fit = base.maybe_dense(feature_dataset.X[fit_idx], args.max_dense_cells)
        X_cal = base.maybe_dense(feature_dataset.X[cal_idx], args.max_dense_cells)
        X_test = base.maybe_dense(feature_dataset.X[test_idx], args.max_dense_cells)
        y_fit = y[fit_idx]
        y_cal = y[cal_idx]
        y_test = y[test_idx]

        classifier = factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier.fit(X_fit, y_fit)

            if score_source in {"tabpfn_proba_default", "tabpfn_proba"}:
                cal_proba = classifier.predict_proba(X_cal)
                test_proba = classifier.predict_proba(X_test)
                column = positive_proba_column(classifier, cal_proba, positive_label)
                cal_scores = cal_proba[:, column]
                test_scores = test_proba[:, positive_proba_column(classifier, test_proba, positive_label)]
                embedding_source = None
                logistic_c = None
            elif score_source == "tabpfn_embedding_logistic":
                fit_embeddings = maybe_get_embeddings(base, classifier, X_fit, X_fit, len(X_fit), "train")
                cal_embeddings = maybe_get_embeddings(base, classifier, X_fit, X_cal, len(X_cal), "test")
                test_embeddings = maybe_get_embeddings(base, classifier, X_fit, X_test, len(X_test), "test")
                scaler = StandardScaler()
                fit_embeddings = scaler.fit_transform(fit_embeddings).astype(np.float32, copy=False)
                cal_embeddings = scaler.transform(cal_embeddings).astype(np.float32, copy=False)
                test_embeddings = scaler.transform(test_embeddings).astype(np.float32, copy=False)
                logistic, logistic_c, cal_scores = best_embedding_logistic(
                    fit_embeddings,
                    y_fit,
                    cal_embeddings,
                    y_cal,
                    c_grid,
                    args.random_state + fold_number,
                    positive_label,
                )
                test_proba = logistic.predict_proba(test_embeddings)
                test_scores = test_proba[:, positive_proba_column(logistic, test_proba, positive_label)]
                embedding_source = "tabpfn_embeddings_or_proba_fallback"
            else:
                raise ValueError(f"Unknown score_source: {score_source}")

        if score_source == "tabpfn_proba_default":
            threshold = 0.5
            y_cal_pred = labels_from_scores(cal_scores, threshold, positive_label, negative_label)
            cal_ba = float(balanced_accuracy_score(y_cal, y_cal_pred))
            cal_macro_f1 = float(f1_score(y_cal, y_cal_pred, average="macro", zero_division=0))
        else:
            threshold, cal_ba, cal_macro_f1 = choose_threshold(
                y_cal,
                cal_scores,
                thresholds,
                positive_label,
                negative_label,
            )
        y_pred = labels_from_scores(test_scores, threshold, positive_label, negative_label)
        metrics = fold_metrics(y_test, y_pred, labels, class_names)
        elapsed = time.perf_counter() - started
        fold_results.append(
            {
                "n_train": int(len(fit_idx)),
                "n_calibration": int(len(cal_idx)),
                "n_test": int(len(test_idx)),
                "n_train_original": original_train_rows,
                "n_test_original": original_test_rows,
                "cpu_train_sampled": bool(len(train_idx) < original_train_rows),
                "cpu_test_sampled": bool(len(test_idx) < original_test_rows),
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
                "threshold": float(threshold),
                "calibration_balanced_accuracy": float(cal_ba),
                "calibration_macro_f1": float(cal_macro_f1),
                "score_source": score_source,
                "positive_class": class_names[positive_label],
                "negative_class": class_names[negative_label],
                "logistic_c": logistic_c,
                "embedding_source": embedding_source,
                **metrics,
            }
        )

    classifier_name = CLASSIFIER_NAMES.get(score_source, f"TabPFN-{score_source}")
    result = base.aggregate_campaign_fold_results(
        fold_results,
        feature_dataset,
        campaign_id,
        classifier_name,
    )
    ok_folds = [fold for fold in fold_results if fold.get("status") == "ok"]
    result["cv_strategy"] = split_strategy
    result["k_folds"] = int(effective_folds)
    result["score_source"] = score_source
    result["threshold_strategy"] = (
        "fixed_0_5"
        if score_source == "tabpfn_proba_default"
        else "train_calibration_balanced_accuracy"
    )
    result["threshold_mean"] = float(np.mean([fold["threshold"] for fold in ok_folds])) if ok_folds else None
    result["threshold_fold_std"] = float(np.std([fold["threshold"] for fold in ok_folds], ddof=1)) if len(ok_folds) > 1 else None
    result["calibration_balanced_accuracy"] = float(np.mean([fold["calibration_balanced_accuracy"] for fold in ok_folds])) if ok_folds else None
    result["calibration_macro_f1"] = float(np.mean([fold["calibration_macro_f1"] for fold in ok_folds])) if ok_folds else None
    result["n_calibration"] = int(sum(fold.get("n_calibration") or 0 for fold in fold_results))
    result["positive_class"] = class_names[positive_label]
    result["negative_class"] = class_names[negative_label]
    result["tabpfn_frozen"] = True
    result["fine_tuned_component"] = None
    result["source_grid_params"] = best.grid_params
    if score_source == "tabpfn_embedding_logistic":
        logistic_values = [fold.get("logistic_c") for fold in ok_folds if fold.get("logistic_c") is not None]
        result["logistic_c_values"] = logistic_values
        result["logistic_c"] = logistic_c_override
    return result


def failed_result(feature_dataset, campaign_id: str, classifier_name: str, error: Exception, best, score_source: str) -> dict:
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
        "score_source": score_source,
        "threshold_strategy": "train_calibration_balanced_accuracy",
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
            "threshold_mean",
            "calibration_balanced_accuracy",
            "fit_predict_seconds",
        )
        if column in frame.columns
    ]
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    groups = ["classifier", "score_source", "logistic_c"]
    existing_groups = [column for column in groups if column in frame.columns]
    summary = frame.groupby(existing_groups, dropna=False)[metrics].agg(["mean", "std", "max", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part) if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    return summary.sort_values(["balanced_accuracy_mean", "macro_f1_mean"], ascending=[False, False])


def save_threshold_results(base, output_path: Path, results: list[dict]) -> None:
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
            str(result.get("classifier", "")).startswith("TabPFN")
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

    score_sources = parse_csv_values(args.score_sources)
    invalid_sources = [source for source in score_sources if source not in CLASSIFIER_NAMES]
    if invalid_sources:
        raise ValueError(f"Invalid score source(s): {invalid_sources}. Expected: {sorted(CLASSIFIER_NAMES)}")

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
    done = completed_threshold_keys(results, args.retry_errors)

    campaigns = base.campaign_indices(feature_dataset)
    if args.max_campaigns > 0:
        campaigns = campaigns[: args.max_campaigns]
    if not campaigns:
        raise ValueError("No campaign index was found for the selected feature dataset.")

    logistic_grid = parse_float_grid(args.logistic_c_grid)
    jobs = []
    for campaign_id, _indices in campaigns:
        for score_source in score_sources:
            if score_source == "tabpfn_embedding_logistic":
                for c_value in logistic_grid:
                    jobs.append((campaign_id, score_source, float(c_value)))
            else:
                jobs.append((campaign_id, score_source, None))

    print(
        "TabPFN threshold calibration using "
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
        for score_source in score_sources:
            logistic_values = logistic_grid if score_source == "tabpfn_embedding_logistic" else [None]
            for logistic_c in logistic_values:
                key = threshold_key(dataset, campaign_id, best, score_source, logistic_c)
                classifier_name = CLASSIFIER_NAMES[score_source]
                if key in done:
                    completed_now += 1
                    print(f"[{completed_now}/{len(jobs)}] {campaign_id} + {classifier_name} c={logistic_c}: skipped", flush=True)
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
                    "score_source": score_source,
                    "logistic_c": logistic_c,
                }
                try:
                    result = evaluate_campaign_threshold(
                        base,
                        campaign_dataset,
                        campaign_id,
                        factory,
                        args,
                        best,
                        score_source,
                        logistic_c,
                    )
                    result["dataset"] = dataset
                except Exception as exc:
                    result = failed_result(campaign_dataset, campaign_id, classifier_name, exc, best, score_source)
                    result["dataset"] = dataset
                    result["logistic_c"] = logistic_c
                result["threshold_calibration_key"] = key
                result["threshold_calibration_method"] = "tabpfn_calibrated_binary_threshold"
                result["grid_model"] = classifier_name
                result = base.add_resume_metadata(result, run_config, pending_key)
                results.append(result)
                done.add(key)
                completed_now += 1
                save_threshold_results(base, output_path, results)
                print(
                    f"[{completed_now}/{len(jobs)}] {campaign_id} + {classifier_name} "
                    f"c={logistic_c}: {result['status']} ba={result.get(PRIMARY_METRIC)}",
                    flush=True,
                )

    save_threshold_results(base, output_path, results)
    print(f"Saved TabPFN threshold calibration results to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise
