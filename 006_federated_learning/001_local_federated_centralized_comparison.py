#!/usr/bin/env python3
"""
Compare Local, Federated and Centralized LightGBM on the same campaign folds.

This script answers whether the federated protocol transfers useful knowledge
between campaigns. It evaluates:

- Local: one LightGBM trained and tested inside each campaign.
- Federated: roundwise residual LightGBM meta-classifier from `000_federated_lightgbm.py`,
  using global TabPFN embeddings when enabled in the base script.
- Centralized: one LightGBM trained on the union of client train folds and tested
  on the union of client test folds.
"""

from __future__ import annotations

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


PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from utils.federated_lightgbm_runner import (
    PROJECT_ROOT,
    args_with_overrides,
    load_base_module,
    load_existing_results,
    prepare_base_experiment,
    save_results,
)


CONFIG_OUTPUT_ROOT = "results/federated_learning/lightgbm_local_federated_centralized_comparison"
CONFIG_EXPERIMENT_NAME = "lightgbm_local_federated_centralized_comparison"
CONFIG_ROUNDS = 10
CONFIG_LOCAL_TREES_PER_ROUND = 20
CONFIG_WEIGHTING = "uniform"
CONFIG_PREDICTION_THRESHOLD = 0.5


def metric_row(y_true, y_pred, labels, class_names) -> dict[str, Any]:
    recall_values = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "recall_by_class": {
            str(class_name): float(value)
            for class_name, value in zip(class_names, recall_values)
        },
    }


def summarize_rows(rows: list[dict], base_fields: dict) -> dict:
    ok = [row for row in rows if row.get("status") == "ok"]
    result = dict(base_fields)
    result["fold_results"] = rows
    result["status"] = "ok" if ok else "error"
    result["error"] = None if ok else "; ".join(str(row.get("error")) for row in rows)
    for key in ("balanced_accuracy", "macro_f1", "weighted_f1", "mcc", "fit_predict_seconds", "communication_cost"):
        values = [row.get(key) for row in ok if row.get(key) is not None and not pd.isna(row.get(key))]
        result[key] = float(np.mean(values)) if values else None
        result[f"{key}_fold_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    result["n_train"] = int(sum(row.get("n_train") or 0 for row in ok))
    result["n_test"] = int(sum(row.get("n_test") or 0 for row in ok))
    result["k_folds"] = int(len(ok))
    return result


def model_predict_proba(base, model, scaler, X, args):
    X_eval = base.maybe_dense(X, args.max_dense_cells)
    if scaler is not None:
        X_eval = scaler.transform(X_eval)
    proba = np.asarray(model.predict_proba(X_eval), dtype=np.float64)
    if proba.ndim == 1:
        return proba
    return proba[:, -1]


def fit_lightgbm(base, X, y, args, seed):
    X_train = base.maybe_dense(X, args.max_dense_cells)
    scaler = None
    if not sparse.issparse(X_train):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
    model = base.make_lightgbm(args, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y)
    return model, scaler


def usable_client_splits(base, feature_dataset, y, campaigns, args):
    client_splits = []
    effective_folds = []
    for campaign_id, indices in campaigns:
        local_y = y[indices]
        if len(np.unique(local_y)) < 2:
            continue
        splits, split_strategy, n_folds = base.campaign_kfold_splits(local_y, args.k_folds, args.random_state)
        client_splits.append((campaign_id, indices, splits, split_strategy))
        effective_folds.append(n_folds)
    if not client_splits:
        raise ValueError("no usable campaign/client splits")
    return client_splits, int(min(effective_folds))


def fold_feature_dataset(base, feature_dataset, dataset, y, client_splits, fold_number, args):
    if not getattr(args, "use_tabpfn_global_embeddings", False):
        return feature_dataset, {"feature_representation": "source_features"}
    return base.build_global_tabpfn_embedding_dataset(
        feature_dataset,
        dataset,
        y,
        client_splits,
        fold_number,
        args,
    )


def evaluate_local(base, feature_dataset, y, labels, class_names, campaigns, args, dataset: str) -> list[dict]:
    client_splits, n_global_folds = usable_client_splits(base, feature_dataset, y, campaigns, args)
    campaign_rows: dict[str, list[dict]] = {str(campaign_id): [] for campaign_id, *_rest in client_splits}
    campaign_sizes: dict[str, int] = {str(campaign_id): int(len(indices)) for campaign_id, indices, *_rest in client_splits}

    for fold_index in range(n_global_folds):
        active_dataset, embedding_metadata = fold_feature_dataset(
            base,
            feature_dataset,
            dataset,
            y,
            client_splits,
            fold_index,
            args,
        )
        for campaign_id, indices, splits, _split_strategy in client_splits:
            started = time.perf_counter()
            local_train_idx, local_test_idx = splits[fold_index]
            train_idx = indices[local_train_idx]
            test_idx = indices[local_test_idx]
            try:
                model, scaler = fit_lightgbm(
                    base,
                    base.take_rows(active_dataset.X, train_idx),
                    y[train_idx],
                    args,
                    args.random_state + fold_index + 1,
                )
                proba = model_predict_proba(base, model, scaler, base.take_rows(active_dataset.X, test_idx), args)
                y_pred = (proba >= float(args.prediction_threshold)).astype(int)
                row = {
                    "fold": int(fold_index + 1),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "fit_predict_seconds": float(time.perf_counter() - started),
                    "communication_cost": 0,
                    "status": "ok",
                    "error": None,
                    **embedding_metadata,
                    **metric_row(y[test_idx], y_pred, labels, class_names),
                }
            except Exception as exc:
                row = {"fold": int(fold_index + 1), "status": "error", "error": str(exc)}
            campaign_rows[str(campaign_id)].append(row)

    results = []
    for campaign_id, fold_rows in campaign_rows.items():
        results.append(
            summarize_rows(
                fold_rows,
                {
                    "feature_stage": feature_dataset.feature_stage,
                    "feature_selection": feature_dataset.feature_selection,
                    "feature_approach": feature_dataset.feature_approach,
                    "classifier": "Local-LightGBM",
                    "model": "Local-LightGBM",
                    "dataset": dataset,
                    "evaluation_scope": "campaign",
                    "campaign": str(campaign_id),
                    "campaign_id": str(campaign_id),
                    "n_samples": int(campaign_sizes[str(campaign_id)]),
                    "source_n_features": int(feature_dataset.X.shape[1]),
                    "n_features": int(
                        next(
                            (
                                row.get("tabpfn_embedding_features")
                                for row in fold_rows
                                if row.get("tabpfn_embedding_features") is not None
                            ),
                            feature_dataset.X.shape[1],
                        )
                    ),
                    "experiment_name": CONFIG_EXPERIMENT_NAME,
                    "comparison_scope": "local",
                    "feature_representation": (
                        "tabpfn_global_embeddings"
                        if getattr(args, "use_tabpfn_global_embeddings", False)
                        else "source_features"
                    ),
                },
            )
        )
    return results


def evaluate_centralized(base, feature_dataset, y, labels, class_names, campaigns, args, dataset: str) -> dict:
    client_splits, n_global_folds = usable_client_splits(base, feature_dataset, y, campaigns, args)
    fold_rows = []
    for fold_number in range(n_global_folds):
        started = time.perf_counter()
        active_dataset, embedding_metadata = fold_feature_dataset(
            base,
            feature_dataset,
            dataset,
            y,
            client_splits,
            fold_number,
            args,
        )
        train_idx = np.concatenate([indices[splits[fold_number][0]] for _campaign, indices, splits, _strategy in client_splits])
        test_idx = np.concatenate([indices[splits[fold_number][1]] for _campaign, indices, splits, _strategy in client_splits])
        try:
            model, scaler = fit_lightgbm(
                base,
                base.take_rows(active_dataset.X, train_idx),
                y[train_idx],
                args,
                args.random_state + 1000 + fold_number,
            )
            proba = model_predict_proba(base, model, scaler, base.take_rows(active_dataset.X, test_idx), args)
            y_pred = (proba >= float(args.prediction_threshold)).astype(int)
            row = {
                "fold": int(fold_number + 1),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "fit_predict_seconds": float(time.perf_counter() - started),
                "communication_cost": 0,
                "status": "ok",
                "error": None,
                **embedding_metadata,
                **metric_row(y[test_idx], y_pred, labels, class_names),
            }
        except Exception as exc:
            row = {"fold": int(fold_number + 1), "status": "error", "error": str(exc)}
        fold_rows.append(row)

    return summarize_rows(
        fold_rows,
        {
            "feature_stage": feature_dataset.feature_stage,
            "feature_selection": feature_dataset.feature_selection,
            "feature_approach": feature_dataset.feature_approach,
            "classifier": "Centralized-LightGBM",
            "model": "Centralized-LightGBM",
            "dataset": dataset,
            "evaluation_scope": "centralized_clients_union",
            "campaign": None,
            "campaign_id": None,
            "n_samples": int(len(feature_dataset.target)),
            "source_n_features": int(feature_dataset.X.shape[1]),
            "n_features": int(
                next(
                    (
                        row.get("tabpfn_embedding_features")
                        for row in fold_rows
                        if row.get("tabpfn_embedding_features") is not None
                    ),
                    feature_dataset.X.shape[1],
                )
            ),
            "experiment_name": CONFIG_EXPERIMENT_NAME,
            "comparison_scope": "centralized",
            "feature_representation": (
                "tabpfn_global_embeddings"
                if getattr(args, "use_tabpfn_global_embeddings", False)
                else "source_features"
            ),
        },
    )


def main() -> None:
    base = load_base_module()
    args, dataset, feature_dataset, best_bayesian = prepare_base_experiment(base, CONFIG_OUTPUT_ROOT)
    args = args_with_overrides(
        args,
        {
            "rounds": CONFIG_ROUNDS,
            "local_trees_per_round": CONFIG_LOCAL_TREES_PER_ROUND,
            "weighting": CONFIG_WEIGHTING,
            "prediction_threshold": CONFIG_PREDICTION_THRESHOLD,
        },
    )
    output_path = PROJECT_ROOT / CONFIG_OUTPUT_ROOT / dataset
    results = load_existing_results(output_path)
    if results:
        print(f"{CONFIG_EXPERIMENT_NAME}: existing results found in {output_path}; skipping")
        return

    y_text = feature_dataset.target["attack_type"].fillna(base.MISSING).astype(str).to_numpy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    labels = np.arange(len(label_encoder.classes_))
    class_names = [str(item) for item in label_encoder.classes_.tolist()]
    campaigns = base.campaign_indices(feature_dataset)
    if args.max_clients > 0:
        campaigns = campaigns[: args.max_clients]
    if len(campaigns) < args.min_clients:
        raise ValueError(f"fewer than min_clients={args.min_clients}: {len(campaigns)}")

    print(f"{CONFIG_EXPERIMENT_NAME}: running local baselines", flush=True)
    results.extend(evaluate_local(base, feature_dataset, y, labels, class_names, campaigns, args, dataset))

    print(f"{CONFIG_EXPERIMENT_NAME}: running centralized baseline", flush=True)
    results.append(evaluate_centralized(base, feature_dataset, y, labels, class_names, campaigns, args, dataset))

    print(f"{CONFIG_EXPERIMENT_NAME}: running federated baseline", flush=True)
    federated = base.evaluate_feature_dataset(feature_dataset, dataset, args)
    federated["experiment_name"] = CONFIG_EXPERIMENT_NAME
    federated["comparison_scope"] = "federated"
    if best_bayesian is not None:
        federated["source_best_bayesian_optimization"] = best_bayesian
        federated["lightgbm_params_from_bayesian_optimization"] = best_bayesian["lightgbm_params"]
    results.append(federated)

    save_results(output_path, results)
    print(f"Saved comparison results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
