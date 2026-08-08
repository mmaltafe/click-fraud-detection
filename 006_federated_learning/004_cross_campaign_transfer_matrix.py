#!/usr/bin/env python3
"""
Measure cross-campaign transfer for TabPFN embeddings + LightGBM.

For each campaign-aligned fold, this script trains one local LightGBM on a
source campaign and evaluates it on every target campaign. The resulting
train-campaign x test-campaign matrix is the practical basis for a future
similarity-weighted federated ensemble.
"""

from __future__ import annotations

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

from utils._campaigns import build_client_splits, summarize_numeric, summarize_numeric_std  # noqa: E402
from utils.federated_lightgbm_runner import (  # noqa: E402
    PROJECT_ROOT,
    load_base_module,
    load_existing_results,
    prepare_base_experiment,
    save_results,
)


CONFIG_OUTPUT_ROOT = "results/federated_learning/lightgbm_cross_campaign_transfer"
CONFIG_EXPERIMENT_NAME = "cross_campaign_transfer_matrix"
CONFIG_PREDICTION_THRESHOLD = 0.50
CONFIG_FORCE_RERUN = False

CLASSIFIER_NAME = "Cross-Campaign-LightGBM"


def fit_source_model(base, active_dataset, y: np.ndarray, train_idx: np.ndarray, args, seed: int):
    X_train = base.maybe_dense(base.take_rows(active_dataset.X, train_idx), args.max_dense_cells)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = base.make_lightgbm(args, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y[train_idx])
    return model, scaler


def predict_positive_proba(base, model, scaler, active_dataset, test_idx: np.ndarray, args) -> np.ndarray:
    X_test = base.maybe_dense(base.take_rows(active_dataset.X, test_idx), args.max_dense_cells)
    X_test = scaler.transform(X_test)
    proba = np.asarray(model.predict_proba(X_test), dtype=np.float64)
    if proba.ndim == 1:
        return proba.reshape(-1)
    return proba[:, -1]


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict[str, Any]:
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


def summarize_pair(
    fold_rows: list[dict],
    *,
    feature_dataset,
    dataset: str,
    source_campaign: str,
    target_campaign: str,
    class_names: list[str],
) -> dict:
    ok_rows = [row for row in fold_rows if row.get("status") == "ok"]
    result = {
        "feature_stage": feature_dataset.feature_stage,
        "feature_selection": feature_dataset.feature_selection,
        "feature_approach": feature_dataset.feature_approach,
        "classifier": CLASSIFIER_NAME,
        "model": CLASSIFIER_NAME,
        "dataset": dataset,
        "evaluation_scope": "cross_campaign_transfer",
        "source_campaign": source_campaign,
        "target_campaign": target_campaign,
        "transfer_type": "self" if source_campaign == target_campaign else "cross",
        "campaign": target_campaign,
        "campaign_id": target_campaign,
        "n_samples": int(len(feature_dataset.target)),
        "n_train": int(sum(row.get("n_train") or 0 for row in ok_rows)),
        "n_test": int(sum(row.get("n_test") or 0 for row in ok_rows)),
        "source_n_features": int(feature_dataset.X.shape[1]),
        "n_features": int((ok_rows[0].get("tabpfn_embedding_features") if ok_rows else None) or feature_dataset.X.shape[1]),
        "k_folds": int(len(ok_rows)),
        "prediction_threshold": float(CONFIG_PREDICTION_THRESHOLD),
        "status": "ok" if ok_rows else "error",
        "error": None if ok_rows else "; ".join(str(row.get("error")) for row in fold_rows),
        "feature_representation": (ok_rows[0].get("feature_representation") if ok_rows else None) or "source_features",
        "fold_results": fold_rows,
    }
    for key in (
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "communication_cost",
        "fit_predict_seconds",
        "tabpfn_embedding_cache_hit",
    ):
        result[key] = summarize_numeric([row.get(key) for row in ok_rows])
        result[f"{key}_fold_std"] = summarize_numeric_std([row.get(key) for row in ok_rows])

    for class_name in class_names:
        key = f"recall_{class_name}"
        result[key] = summarize_numeric([
            (row.get("recall_by_class") or {}).get(class_name)
            for row in ok_rows
        ])
        result[f"{key}_fold_std"] = summarize_numeric_std([
            (row.get("recall_by_class") or {}).get(class_name)
            for row in ok_rows
        ])
    return result


def add_transfer_baselines(results: list[dict]) -> list[dict]:
    self_scores = {
        row["target_campaign"]: row.get("balanced_accuracy")
        for row in results
        if row.get("transfer_type") == "self" and row.get("balanced_accuracy") is not None
    }
    source_cross_means = {}
    target_cross_means = {}
    for row in results:
        if row.get("transfer_type") != "cross" or row.get("balanced_accuracy") is None:
            continue
        source_cross_means.setdefault(row["source_campaign"], []).append(row["balanced_accuracy"])
        target_cross_means.setdefault(row["target_campaign"], []).append(row["balanced_accuracy"])

    for row in results:
        target_self = self_scores.get(row["target_campaign"])
        row["target_self_balanced_accuracy"] = target_self
        if target_self is not None and row.get("balanced_accuracy") is not None:
            row["delta_vs_target_self"] = float(row["balanced_accuracy"] - target_self)
        else:
            row["delta_vs_target_self"] = None
        row["source_cross_campaign_mean"] = summarize_numeric(source_cross_means.get(row["source_campaign"], []))
        row["target_received_cross_campaign_mean"] = summarize_numeric(target_cross_means.get(row["target_campaign"], []))
    return results


def save_transfer_matrices(output_path: Path, results: list[dict]) -> None:
    frame = pd.DataFrame(results)
    if frame.empty or "balanced_accuracy" not in frame.columns:
        return
    matrix = frame.pivot_table(
        index="source_campaign",
        columns="target_campaign",
        values="balanced_accuracy",
        aggfunc="mean",
    )
    matrix.to_csv(output_path / "transfer_matrix_balanced_accuracy.csv")
    delta_matrix = frame.pivot_table(
        index="source_campaign",
        columns="target_campaign",
        values="delta_vs_target_self",
        aggfunc="mean",
    )
    delta_matrix.to_csv(output_path / "transfer_matrix_delta_vs_target_self.csv")


def evaluate_transfer_matrix(base, feature_dataset, dataset: str, args) -> list[dict]:
    y_text = feature_dataset.target["attack_type"].fillna(base.MISSING).astype(str).to_numpy()
    if len(np.unique(y_text)) < 2:
        raise ValueError("target has fewer than two classes")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    labels = np.arange(len(label_encoder.classes_))
    class_names = [str(item) for item in label_encoder.classes_.tolist()]
    client_splits, n_global_folds = build_client_splits(
        feature_dataset, y, args.k_folds, args.random_state, args.min_clients, args.max_clients
    )
    pair_fold_rows: dict[tuple[str, str], list[dict]] = {
        (source_campaign, target_campaign): []
        for source_campaign, *_source_rest in client_splits
        for target_campaign, *_target_rest in client_splits
    }

    for fold_number in range(n_global_folds):
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

        source_models = {}
        for source_position, (source_campaign, source_indices, source_splits, _source_strategy) in enumerate(client_splits):
            source_train_idx = source_indices[source_splits[fold_number][0]]
            started = time.perf_counter()
            try:
                if len(np.unique(y[source_train_idx])) < 2:
                    raise ValueError("source train fold has fewer than two classes")
                model, scaler = fit_source_model(
                    base,
                    active_dataset,
                    y,
                    source_train_idx,
                    args,
                    int(args.random_state) + 30_000 * (fold_number + 1) + source_position,
                )
                source_models[source_campaign] = {
                    "model": model,
                    "scaler": scaler,
                    "train_idx": source_train_idx,
                    "fit_seconds": float(time.perf_counter() - started),
                    "model_bytes": base.model_communication_cost(model),
                }
            except Exception as exc:
                for target_campaign, *_target_rest in client_splits:
                    pair_fold_rows[(source_campaign, target_campaign)].append(
                        {
                            "fold": int(fold_number + 1),
                            "status": "error",
                            "error": str(exc),
                            **embedding_metadata,
                        }
                    )

        for source_campaign, source_payload in source_models.items():
            for target_campaign, target_indices, target_splits, _target_strategy in client_splits:
                started = time.perf_counter()
                target_test_idx = target_indices[target_splits[fold_number][1]]
                try:
                    positive_proba = predict_positive_proba(
                        base,
                        source_payload["model"],
                        source_payload["scaler"],
                        active_dataset,
                        target_test_idx,
                        args,
                    )
                    y_pred = (positive_proba >= float(CONFIG_PREDICTION_THRESHOLD)).astype(int)
                    row = {
                        "fold": int(fold_number + 1),
                        "n_train": int(len(source_payload["train_idx"])),
                        "n_test": int(len(target_test_idx)),
                        "communication_cost": int(source_payload["model_bytes"]),
                        "fit_predict_seconds": float(source_payload["fit_seconds"] + time.perf_counter() - started),
                        "status": "ok",
                        "error": None,
                        **embedding_metadata,
                        **metric_row(y[target_test_idx], y_pred, labels, class_names),
                    }
                except Exception as exc:
                    row = {
                        "fold": int(fold_number + 1),
                        "status": "error",
                        "error": str(exc),
                        **embedding_metadata,
                    }
                pair_fold_rows[(source_campaign, target_campaign)].append(row)

    results = [
        summarize_pair(
            rows,
            feature_dataset=feature_dataset,
            dataset=dataset,
            source_campaign=source_campaign,
            target_campaign=target_campaign,
            class_names=class_names,
        )
        for (source_campaign, target_campaign), rows in sorted(pair_fold_rows.items())
    ]
    return add_transfer_baselines(results)


def main() -> None:
    base = load_base_module()
    args, dataset, feature_dataset, best_bayesian = prepare_base_experiment(base, CONFIG_OUTPUT_ROOT)
    output_path = PROJECT_ROOT / CONFIG_OUTPUT_ROOT / dataset
    if output_path.exists() and not CONFIG_FORCE_RERUN:
        existing = load_existing_results(output_path)
        if existing:
            print(f"{CONFIG_EXPERIMENT_NAME}: existing results found in {output_path}; skipping")
            return

    print(
        f"{CONFIG_EXPERIMENT_NAME}: {feature_dataset.feature_stage}/"
        f"{feature_dataset.feature_selection or 'none'}/{feature_dataset.feature_approach}",
        flush=True,
    )
    results = evaluate_transfer_matrix(base, feature_dataset, dataset, args)
    for result in results:
        result["experiment_name"] = CONFIG_EXPERIMENT_NAME
        result["experiment_config"] = {"prediction_threshold": CONFIG_PREDICTION_THRESHOLD}
        result["grid_search_method"] = CONFIG_EXPERIMENT_NAME
        result["grid_model"] = CLASSIFIER_NAME
        result["grid_params"] = result["experiment_config"]
        if best_bayesian is not None:
            result["source_best_bayesian_optimization"] = best_bayesian
            result["source_best_bayesian_objective"] = best_bayesian["objective"]
            result["source_best_bayesian_objective_std"] = best_bayesian["std"]
            result["lightgbm_params_from_bayesian_optimization"] = best_bayesian["lightgbm_params"]
            result["tabpfn_params_from_bayesian_optimization"] = best_bayesian.get("tabpfn_params")

    save_results(output_path, results)
    save_transfer_matrices(output_path, results)
    best_cross = max(
        (row for row in results if row.get("transfer_type") == "cross" and row.get("balanced_accuracy") is not None),
        key=lambda row: row["balanced_accuracy"],
        default=None,
    )
    if best_cross:
        print(
            "Best cross transfer: "
            f"{best_cross['source_campaign']} -> {best_cross['target_campaign']} "
            f"ba={best_cross['balanced_accuracy']}",
            flush=True,
        )
    print(f"Saved cross-campaign transfer results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
