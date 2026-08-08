#!/usr/bin/env python3
"""
Train a centralized LightGBM server model on frozen global TabPFN embeddings.

This script implements the first alternative after the federated experiments:
keep TabPFN as a frozen embedding extractor, make the embeddings available to
the server, and train one LightGBM model on the union of campaign/client train
folds. The evaluation uses the same campaign-aligned folds and the same best
Bayesian-Optimization pipeline used by the federated LightGBM scripts.

The purpose is diagnostic: if this centralized server model clearly beats the
federated LightGBM on the same embeddings, the main bottleneck is the federated
aggregation protocol. If it does not, the bottleneck is more likely the frozen
representation or the meta-classifier configuration.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from utils._fairness_metrics import fairness_gap, group_balanced_accuracy, metric_variance  # noqa: E402
from utils.federated_lightgbm_runner import (  # noqa: E402
    PROJECT_ROOT,
    load_base_module,
    prepare_base_experiment,
    save_results,
)


CONFIG_OUTPUT_ROOT = "results/federated_learning/lightgbm_centralized_tabpfn_embeddings"
CONFIG_EXPERIMENT_NAME = "centralized_tabpfn_embeddings_lightgbm"
CONFIG_PREDICTION_THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60)
CONFIG_FORCE_RERUN = False

CLASSIFIER_NAME = "Centralized-TabPFN-Embeddings-LightGBM"


def load_existing(output_path: Path) -> list[dict]:
    path = output_path / "results.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def summarize_numeric(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None and not pd.isna(value)]
    return float(np.mean(numeric)) if numeric else None


def summarize_numeric_std(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None and not pd.isna(value)]
    return float(np.std(numeric, ddof=1)) if len(numeric) > 1 else None


def summarize_dicts(dicts: list[dict | None]) -> dict | None:
    keys: set[str] = set()
    for value in dicts:
        if value:
            keys.update(str(key) for key in value.keys())
    if not keys:
        return None
    return {
        key: summarize_numeric([value.get(key) for value in dicts if value is not None and key in value])
        for key in sorted(keys)
    }


def build_client_splits(base, feature_dataset, y: np.ndarray, args):
    campaigns = base.campaign_indices(feature_dataset)
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
        splits, split_strategy, n_folds = base.campaign_kfold_splits(local_y, args.k_folds, args.random_state)
        client_splits.append((campaign_id, indices, splits, split_strategy))
        effective_folds.append(n_folds)
    if len(client_splits) < args.min_clients:
        raise ValueError(f"fewer usable clients after class filtering: {len(client_splits)}")
    return client_splits, int(min(effective_folds))


def fit_server_lightgbm(base, active_dataset, y: np.ndarray, train_idx: np.ndarray, args, seed: int):
    X_train = base.maybe_dense(base.take_rows(active_dataset.X, train_idx), args.max_dense_cells)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = base.make_lightgbm(args, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y[train_idx])
    return model, scaler


def predict_server_lightgbm(base, model, scaler, active_dataset, test_idx: np.ndarray, args) -> np.ndarray:
    X_test = base.maybe_dense(base.take_rows(active_dataset.X, test_idx), args.max_dense_cells)
    X_test = scaler.transform(X_test)
    proba = np.asarray(model.predict_proba(X_test), dtype=np.float64)
    if proba.ndim == 1:
        return proba.reshape(-1)
    return proba[:, -1]


def evaluate_threshold(base, feature_dataset, dataset: str, args, threshold: float) -> dict:
    started_total = time.perf_counter()
    y_text = feature_dataset.target["attack_type"].fillna(base.MISSING).astype(str).to_numpy()
    if len(np.unique(y_text)) < 2:
        raise ValueError("target has fewer than two classes")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    labels = np.arange(len(label_encoder.classes_))
    client_splits, n_global_folds = build_client_splits(base, feature_dataset, y, args)

    fold_rows = []
    for fold_number in range(n_global_folds):
        started = time.perf_counter()
        active_dataset = feature_dataset
        embedding_metadata = {"feature_representation": "source_features"}
        if getattr(args, "use_tabpfn_global_embeddings", False):
            active_dataset, embedding_metadata = base.build_global_tabpfn_embedding_dataset(
                feature_dataset,
                dataset,
                y,
                client_splits,
                fold_number,
                args,
            )

        train_idx = np.concatenate([
            global_indices[splits[fold_number][0]]
            for _campaign_id, global_indices, splits, _split_strategy in client_splits
        ])
        test_idx = np.concatenate([
            global_indices[splits[fold_number][1]]
            for _campaign_id, global_indices, splits, _split_strategy in client_splits
        ])

        model, scaler = fit_server_lightgbm(
            base,
            active_dataset,
            y,
            train_idx,
            args,
            int(args.random_state) + 20_000 + fold_number,
        )
        positive_proba = predict_server_lightgbm(base, model, scaler, active_dataset, test_idx, args)
        y_pred = (positive_proba >= float(threshold)).astype(int)

        campaign_groups: list[str] = []
        traffic_groups: list[str] = []
        for campaign_id, global_indices, splits, _split_strategy in client_splits:
            local_test_idx = splits[fold_number][1]
            campaign_test_idx = global_indices[local_test_idx]
            campaign_groups.extend([str(campaign_id)] * len(campaign_test_idx))
            if feature_dataset.row_index is not None and "traffic_source" in feature_dataset.row_index.columns:
                traffic_groups.extend(
                    feature_dataset.row_index.iloc[campaign_test_idx]["traffic_source"]
                    .fillna(base.MISSING)
                    .astype(str)
                    .tolist()
                )
            else:
                traffic_groups.extend([base.MISSING] * len(campaign_test_idx))

        recall_values = recall_score(y[test_idx], y_pred, labels=labels, average=None, zero_division=0)
        campaign_performance = group_balanced_accuracy(y[test_idx], y_pred, campaign_groups)
        traffic_source_performance = group_balanced_accuracy(y[test_idx], y_pred, traffic_groups)
        elapsed = time.perf_counter() - started
        fold_rows.append(
            {
                "fold": int(fold_number + 1),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_clients": int(len(client_splits)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], y_pred)),
                "macro_f1": float(f1_score(y[test_idx], y_pred, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y[test_idx], y_pred, average="weighted", zero_division=0)),
                "mcc": float(matthews_corrcoef(y[test_idx], y_pred)),
                "recall_by_class": {
                    str(class_name): float(value)
                    for class_name, value in zip(label_encoder.classes_, recall_values)
                },
                "communication_cost": 0,
                "server_received_raw_rows": 0,
                "server_received_raw_columns": 0,
                "server_received_embedding_rows": int(len(train_idx)),
                "server_received_embedding_columns": int(active_dataset.X.shape[1]),
                "prediction_threshold": float(threshold),
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
                **embedding_metadata,
            }
        )

    result = {
        "feature_stage": feature_dataset.feature_stage,
        "feature_selection": feature_dataset.feature_selection,
        "feature_approach": feature_dataset.feature_approach,
        "classifier": CLASSIFIER_NAME,
        "model": CLASSIFIER_NAME,
        "dataset": dataset,
        "evaluation_scope": "centralized_server_embeddings",
        "campaign": None,
        "campaign_id": None,
        "n_samples": int(len(feature_dataset.target)),
        "n_train": int(sum(row.get("n_train") or 0 for row in fold_rows)),
        "n_test": int(sum(row.get("n_test") or 0 for row in fold_rows)),
        "source_n_features": int(feature_dataset.X.shape[1]),
        "n_features": int(fold_rows[0].get("tabpfn_embedding_features") or feature_dataset.X.shape[1]),
        "n_clients": int(round(summarize_numeric([row.get("n_clients") for row in fold_rows]) or 0)),
        "k_folds": int(len(fold_rows)),
        "cv_strategy": "campaign_aligned_kfold",
        "status": "ok",
        "error": None,
        "feature_representation": fold_rows[0].get("feature_representation") or "source_features",
        "tabpfn_is_federated": False,
        "lightgbm_training_scope": "centralized_server",
        "prediction_threshold": float(threshold),
        "server_received_raw_rows": 0,
        "server_received_raw_columns": 0,
        "fit_predict_seconds": float(time.perf_counter() - started_total),
        "elapsed_seconds": float(time.perf_counter() - started_total),
        "fold_results": fold_rows,
    }
    for key in (
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "communication_cost",
        "campaign_metric_variance",
        "campaign_fairness_gap",
        "client_variance",
        "fairness_between_campaigns",
        "traffic_source_metric_variance",
        "traffic_source_fairness_gap",
        "server_received_raw_rows",
        "server_received_raw_columns",
        "server_received_embedding_rows",
        "server_received_embedding_columns",
        "tabpfn_embedding_cache_hit",
    ):
        result[key] = summarize_numeric([row.get(key) for row in fold_rows])
        result[f"{key}_fold_std"] = summarize_numeric_std([row.get(key) for row in fold_rows])

    result["recall_by_class"] = summarize_dicts([row.get("recall_by_class") for row in fold_rows])
    result["campaign_performance"] = summarize_dicts([row.get("campaign_performance") for row in fold_rows])
    result["traffic_source_performance"] = summarize_dicts([row.get("traffic_source_performance") for row in fold_rows])
    result["performance_by_traffic_source"] = result["traffic_source_performance"]
    return result


def failed_result(feature_dataset, dataset: str, error: Exception, threshold: float) -> dict:
    return {
        "feature_stage": feature_dataset.feature_stage,
        "feature_selection": feature_dataset.feature_selection,
        "feature_approach": feature_dataset.feature_approach,
        "classifier": CLASSIFIER_NAME,
        "model": CLASSIFIER_NAME,
        "dataset": dataset,
        "evaluation_scope": "centralized_server_embeddings",
        "prediction_threshold": float(threshold),
        "n_samples": int(len(feature_dataset.target)),
        "n_features": int(feature_dataset.X.shape[1]),
        "status": "error",
        "error": str(error),
    }


def main() -> None:
    base = load_base_module()
    args, dataset, feature_dataset, best_bayesian = prepare_base_experiment(base, CONFIG_OUTPUT_ROOT)
    output_path = PROJECT_ROOT / CONFIG_OUTPUT_ROOT / dataset
    results = [] if CONFIG_FORCE_RERUN else load_existing(output_path)
    done = {
        float(result.get("prediction_threshold"))
        for result in results
        if result.get("status") in {"ok", "error", "failed", "skipped"}
        and result.get("prediction_threshold") is not None
    }

    print(
        f"{CONFIG_EXPERIMENT_NAME}: {feature_dataset.feature_stage}/"
        f"{feature_dataset.feature_selection or 'none'}/{feature_dataset.feature_approach}",
        flush=True,
    )
    for position, threshold in enumerate(CONFIG_PREDICTION_THRESHOLDS, start=1):
        if float(threshold) in done:
            print(f"[{position}/{len(CONFIG_PREDICTION_THRESHOLDS)}] skipped threshold={threshold}", flush=True)
            continue
        try:
            result = evaluate_threshold(base, feature_dataset, dataset, args, float(threshold))
        except Exception as exc:
            result = failed_result(feature_dataset, dataset, exc, float(threshold))
        result["experiment_name"] = CONFIG_EXPERIMENT_NAME
        result["experiment_config"] = {"prediction_threshold": float(threshold)}
        result["grid_search_method"] = CONFIG_EXPERIMENT_NAME
        result["grid_model"] = CLASSIFIER_NAME
        result["grid_params"] = result["experiment_config"]
        if best_bayesian is not None:
            result["source_best_bayesian_optimization"] = best_bayesian
            result["source_best_bayesian_objective"] = best_bayesian["objective"]
            result["source_best_bayesian_objective_std"] = best_bayesian["std"]
            result["lightgbm_params_from_bayesian_optimization"] = best_bayesian["lightgbm_params"]
            result["tabpfn_params_from_bayesian_optimization"] = best_bayesian.get("tabpfn_params")
        results.append(result)
        done.add(float(threshold))
        save_results(output_path, results)
        print(
            f"[{position}/{len(CONFIG_PREDICTION_THRESHOLDS)}] "
            f"{result['status']} ba={result.get('balanced_accuracy')} threshold={threshold}",
            flush=True,
        )

    save_results(output_path, results)
    print(f"Saved centralized TabPFN-embeddings LightGBM results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
