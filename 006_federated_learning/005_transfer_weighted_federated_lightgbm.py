#!/usr/bin/env python3
"""
Evaluate transfer-weighted federated LightGBM on frozen TabPFN embeddings.

This script uses the cross-campaign transfer matrix produced by
`004_cross_campaign_transfer_matrix.py` to personalize the server aggregation.
Instead of one global client weight vector, each target campaign receives a
different weighted ensemble of local LightGBM clients:

    s_target(x) = sum_source w[source,target] * s_source(x)

The raw data still stay local. The server uses only local LightGBM models,
their scores, and the previously measured transfer matrix.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, recall_score
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from utils.federated_lightgbm_runner import (  # noqa: E402
    PROJECT_ROOT,
    load_base_module,
    prepare_base_experiment,
    save_results,
    stable_key,
)


CONFIG_OUTPUT_ROOT = "results/federated_learning/lightgbm_transfer_weighted"
CONFIG_TRANSFER_ROOT = "results/federated_learning/lightgbm_cross_campaign_transfer"
CONFIG_EXPERIMENT_NAME = "transfer_weighted_federated_lightgbm"
CONFIG_FORCE_RERUN = False

CONFIG_GLOBAL_ROUNDS = 5
CONFIG_GLOBAL_LOCAL_TREES_PER_ROUND = 20
CONFIG_PREDICTION_THRESHOLD = 0.50

CONFIG_WEIGHT_CONFIGS = [
    {
        "transfer_weight_mode": "direct",
        "temperature": 1.0,
        "top_k": 0,
        "self_weight_floor": 0.0,
        "description": "normalize raw transfer balanced accuracy by target campaign",
    },
    {
        "transfer_weight_mode": "softmax",
        "temperature": 0.05,
        "top_k": 0,
        "self_weight_floor": 0.0,
        "description": "softmax over transfer balanced accuracy by target campaign",
    },
    {
        "transfer_weight_mode": "softmax",
        "temperature": 0.10,
        "top_k": 3,
        "self_weight_floor": 0.0,
        "description": "top-3 softmax over transfer balanced accuracy",
    },
    {
        "transfer_weight_mode": "positive_delta",
        "temperature": 0.05,
        "top_k": 3,
        "self_weight_floor": 0.50,
        "description": "top-3 softmax over non-negative delta versus target self model",
    },
]

CLASSIFIER_NAME = "Transfer-Weighted-Federated-LightGBM"
FEDERATED_ALGORITHM = "GlobalTabPFNEmbeddingsLightGBMTransferWeightedAggregation"


def clone_args(args: Namespace, **overrides) -> Namespace:
    payload = vars(args).copy()
    payload.update(overrides)
    return Namespace(**payload)


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


def group_balanced_accuracy(y_true, y_pred, groups: list[str]) -> dict[str, float | None]:
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": groups})
    values: dict[str, float | None] = {}
    for group, part in frame.groupby("group"):
        if part["y_true"].nunique() < 2:
            values[str(group)] = None
        else:
            values[str(group)] = float(balanced_accuracy_score(part["y_true"], part["y_pred"]))
    return values


def metric_variance(values: dict[str, float | None] | None) -> float | None:
    if not values:
        return None
    numeric = [float(value) for value in values.values() if value is not None and not pd.isna(value)]
    return float(np.var(numeric)) if len(numeric) > 1 else None


def fairness_gap(values: dict[str, float | None] | None) -> float | None:
    if not values:
        return None
    numeric = [float(value) for value in values.values() if value is not None and not pd.isna(value)]
    return float(max(numeric) - min(numeric)) if len(numeric) > 1 else None


def build_client_splits(base, feature_dataset, y: np.ndarray, args: Namespace):
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
        client_splits.append((str(campaign_id), indices, splits, split_strategy))
        effective_folds.append(n_folds)
    if len(client_splits) < args.min_clients:
        raise ValueError(f"fewer usable clients after class filtering: {len(client_splits)}")
    return client_splits, int(min(effective_folds))


def load_transfer_matrix(dataset: str) -> pd.DataFrame:
    matrix_path = PROJECT_ROOT / CONFIG_TRANSFER_ROOT / dataset / "transfer_matrix_balanced_accuracy.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Transfer matrix not found: {matrix_path}. "
            "Run 006_federated_learning/004_cross_campaign_transfer_matrix.py first."
        )
    matrix = pd.read_csv(matrix_path, index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    return matrix.astype(float)


def normalize_vector(values: pd.Series, mode: str, temperature: float, top_k: int, self_campaign: str, self_weight_floor: float) -> dict[str, float]:
    vector = values.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if top_k > 0 and top_k < len(vector):
        keep = set(vector.nlargest(top_k).index.astype(str))
        if self_campaign in vector.index:
            keep.add(self_campaign)
        vector = vector.where(vector.index.isin(keep), 0.0)

    if mode == "direct":
        scores = vector.clip(lower=0.0)
    elif mode == "softmax":
        tau = max(float(temperature), 1e-6)
        centered = vector - float(vector.max())
        scores = pd.Series(np.exp(centered / tau), index=vector.index)
        scores = scores.where(vector > 0.0, 0.0)
    elif mode == "positive_delta":
        self_score = float(vector.get(self_campaign, vector.max()))
        delta = (vector - self_score).clip(lower=0.0)
        if float(self_weight_floor) > 0.0 and self_campaign in delta.index:
            delta.loc[self_campaign] = max(float(delta.loc[self_campaign]), float(self_weight_floor))
        tau = max(float(temperature), 1e-6)
        centered = delta - float(delta.max())
        scores = pd.Series(np.exp(centered / tau), index=delta.index)
        scores = scores.where(delta > 0.0, 0.0)
    else:
        raise ValueError(f"Unknown transfer weight mode: {mode}")

    if float(self_weight_floor) > 0.0 and self_campaign in scores.index:
        scores.loc[self_campaign] = max(float(scores.loc[self_campaign]), float(self_weight_floor))

    total = float(scores.sum())
    if total <= 0.0:
        if self_campaign in scores.index:
            scores = pd.Series(0.0, index=scores.index)
            scores.loc[self_campaign] = 1.0
        else:
            scores = pd.Series(1.0 / len(scores), index=scores.index)
    else:
        scores = scores / total
    return {str(index): float(value) for index, value in scores.items() if float(value) > 0.0}


def build_transfer_weights(matrix: pd.DataFrame, campaigns: list[str], config: dict[str, Any]) -> dict[str, dict[str, float]]:
    available = matrix.reindex(index=campaigns, columns=campaigns)
    weights: dict[str, dict[str, float]] = {}
    for target_campaign in campaigns:
        target_column = available[target_campaign]
        weights[target_campaign] = normalize_vector(
            target_column,
            mode=str(config["transfer_weight_mode"]),
            temperature=float(config["temperature"]),
            top_k=int(config["top_k"]),
            self_campaign=target_campaign,
            self_weight_floor=float(config["self_weight_floor"]),
        )
    return weights


def transfer_weighted_raw_score(
    base,
    global_rounds: list[list[tuple[str, Any, Any]]],
    X,
    args: Namespace,
    target_campaign: str,
    transfer_weights: dict[str, dict[str, float]],
) -> np.ndarray:
    if not global_rounds:
        return np.zeros(X.shape[0], dtype=np.float64)
    raw = np.zeros(X.shape[0], dtype=np.float64)
    target_weights = transfer_weights.get(str(target_campaign), {})
    for round_models in global_rounds:
        contribution = np.zeros(X.shape[0], dtype=np.float64)
        weight_sum = 0.0
        for source_campaign, model, scaler in round_models:
            weight = float(target_weights.get(str(source_campaign), 0.0))
            if weight <= 0.0:
                continue
            X_for_model = base.transform_with_scaler(X, scaler, args)
            contribution += base.model_raw_contribution(model, X_for_model) * weight
            weight_sum += weight
        if weight_sum > 0.0:
            raw += contribution / weight_sum
    return raw


def train_transfer_weighted_rounds(base, active_dataset, y: np.ndarray, client_splits, fold_number: int, args: Namespace, transfer_weights):
    global_rounds: list[list[tuple[str, Any, Any]]] = []
    model_bytes = 0
    client_train_sizes: dict[str, int] = {}

    for round_number in range(int(args.rounds)):
        round_models = []
        for client_position, (campaign_id, global_indices, splits, _split_strategy) in enumerate(client_splits):
            local_train_idx, _local_test_idx = splits[fold_number]
            train_idx = global_indices[local_train_idx]
            if len(np.unique(y[train_idx])) < 2:
                continue
            init_score = (
                transfer_weighted_raw_score(
                    base,
                    global_rounds,
                    base.take_rows(active_dataset.X, train_idx),
                    args,
                    str(campaign_id),
                    transfer_weights,
                )
                if global_rounds
                else None
            )
            model, scaler = base.fit_client_model(
                active_dataset,
                y,
                train_idx,
                args,
                int(args.random_state) + 70_000 * (fold_number + 1) + 1_000 * (round_number + 1) + client_position,
                init_score=init_score,
            )
            round_models.append((str(campaign_id), model, scaler))
            client_train_sizes[str(campaign_id)] = int(len(train_idx))
            model_bytes += base.model_communication_cost(model)

        if len(round_models) < args.min_clients:
            raise ValueError(f"fold {fold_number + 1}, round {round_number + 1} has fewer usable clients")
        global_rounds.append(round_models)
    return global_rounds, model_bytes, client_train_sizes


def evaluate_transfer_weighted_config(base, feature_dataset, dataset: str, args: Namespace, config: dict[str, Any], transfer_matrix: pd.DataFrame) -> dict:
    started_total = time.perf_counter()
    y_text = feature_dataset.target["attack_type"].fillna(base.MISSING).astype(str).to_numpy()
    if len(np.unique(y_text)) < 2:
        raise ValueError("target has fewer than two classes")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    labels = np.arange(len(label_encoder.classes_))
    client_splits, n_global_folds = build_client_splits(base, feature_dataset, y, args)
    campaign_ids = [campaign_id for campaign_id, *_ in client_splits]
    transfer_weights = build_transfer_weights(transfer_matrix, campaign_ids, config)

    run_args = clone_args(
        args,
        rounds=int(config["rounds"]),
        local_trees_per_round=int(config["local_trees_per_round"]),
        prediction_threshold=float(config["prediction_threshold"]),
    )

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

        global_rounds, model_bytes, client_train_sizes = train_transfer_weighted_rounds(
            base,
            active_dataset,
            y,
            client_splits,
            fold_number,
            run_args,
            transfer_weights,
        )

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
            test_X = base.take_rows(active_dataset.X, test_idx)
            raw_score = transfer_weighted_raw_score(
                base,
                global_rounds,
                test_X,
                run_args,
                str(campaign_id),
                transfer_weights,
            )
            positive_proba = base.sigmoid(raw_score)
            y_pred = (positive_proba >= float(run_args.prediction_threshold)).astype(int)

            y_true_parts.append(y[test_idx])
            y_pred_parts.append(y_pred)
            campaign_groups.extend([str(campaign_id)] * len(test_idx))
            if feature_dataset.row_index is not None and "traffic_source" in feature_dataset.row_index.columns:
                traffic_groups.extend(
                    feature_dataset.row_index.iloc[test_idx]["traffic_source"].fillna(base.MISSING).astype(str).tolist()
                )
            else:
                traffic_groups.extend([base.MISSING] * len(test_idx))
            local_test_rows += int(len(test_idx))

        if not y_true_parts:
            raise ValueError(f"fold {fold_number + 1} has no test rows")

        y_true = np.concatenate(y_true_parts)
        y_pred = np.concatenate(y_pred_parts)
        recall_values = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        campaign_performance = group_balanced_accuracy(y_true, y_pred, campaign_groups)
        traffic_source_performance = group_balanced_accuracy(y_true, y_pred, traffic_groups)
        elapsed = time.perf_counter() - started
        fold_rows.append(
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
                "server_received_raw_rows": 0,
                "server_received_raw_columns": 0,
                "number_of_rounds": int(run_args.rounds),
                "local_trees_per_round": int(run_args.local_trees_per_round),
                "prediction_threshold": float(run_args.prediction_threshold),
                "transfer_weight_mode": str(config["transfer_weight_mode"]),
                "transfer_weight_temperature": float(config["temperature"]),
                "transfer_weight_top_k": int(config["top_k"]),
                "transfer_self_weight_floor": float(config["self_weight_floor"]),
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
        "federated_algorithm": FEDERATED_ALGORITHM,
        "dataset": dataset,
        "evaluation_scope": "federated_clients_transfer_weighted",
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
        "federated_component": "lightgbm_meta_classifier",
        "tabpfn_is_federated": False,
        "transfer_matrix_source": str(PROJECT_ROOT / CONFIG_TRANSFER_ROOT / dataset / "transfer_matrix_balanced_accuracy.csv"),
        "transfer_weights": transfer_weights,
        "rounds": int(run_args.rounds),
        "local_trees_per_round": int(run_args.local_trees_per_round),
        "prediction_threshold": float(run_args.prediction_threshold),
        "transfer_weight_mode": str(config["transfer_weight_mode"]),
        "transfer_weight_temperature": float(config["temperature"]),
        "transfer_weight_top_k": int(config["top_k"]),
        "transfer_self_weight_floor": float(config["self_weight_floor"]),
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
        "tabpfn_embedding_cache_hit",
    ):
        result[key] = summarize_numeric([row.get(key) for row in fold_rows])
        result[f"{key}_fold_std"] = summarize_numeric_std([row.get(key) for row in fold_rows])

    result["recall_by_class"] = summarize_dicts([row.get("recall_by_class") for row in fold_rows])
    result["campaign_performance"] = summarize_dicts([row.get("campaign_performance") for row in fold_rows])
    result["traffic_source_performance"] = summarize_dicts([row.get("traffic_source_performance") for row in fold_rows])
    result["performance_by_traffic_source"] = result["traffic_source_performance"]
    return result


def failed_result(feature_dataset, dataset: str, error: Exception, config: dict[str, Any]) -> dict:
    return {
        "feature_stage": feature_dataset.feature_stage,
        "feature_selection": feature_dataset.feature_selection,
        "feature_approach": feature_dataset.feature_approach,
        "classifier": CLASSIFIER_NAME,
        "model": CLASSIFIER_NAME,
        "federated_algorithm": FEDERATED_ALGORITHM,
        "dataset": dataset,
        "evaluation_scope": "federated_clients_transfer_weighted",
        "n_samples": int(len(feature_dataset.target)),
        "n_features": int(feature_dataset.X.shape[1]),
        "status": "error",
        "error": str(error),
        "experiment_config": config,
    }


def main() -> None:
    base = load_base_module()
    args, dataset, feature_dataset, best_bayesian = prepare_base_experiment(base, CONFIG_OUTPUT_ROOT)
    output_path = PROJECT_ROOT / CONFIG_OUTPUT_ROOT / dataset
    transfer_matrix = load_transfer_matrix(dataset)
    results = load_existing(output_path)
    done = {
        stable_key(result.get("experiment_key", {}))
        for result in results
        if result.get("status") in {"ok", "error", "failed", "skipped"} and not CONFIG_FORCE_RERUN
    }

    print(
        f"{CONFIG_EXPERIMENT_NAME}: {feature_dataset.feature_stage}/"
        f"{feature_dataset.feature_selection or 'none'}/{feature_dataset.feature_approach}",
        flush=True,
    )

    configs = [
        {
            "rounds": CONFIG_GLOBAL_ROUNDS,
            "local_trees_per_round": CONFIG_GLOBAL_LOCAL_TREES_PER_ROUND,
            "prediction_threshold": CONFIG_PREDICTION_THRESHOLD,
            **weight_config,
        }
        for weight_config in CONFIG_WEIGHT_CONFIGS
    ]

    for position, config in enumerate(configs, start=1):
        experiment_key = {
            "experiment_name": CONFIG_EXPERIMENT_NAME,
            "dataset": dataset,
            "feature_stage": feature_dataset.feature_stage,
            "feature_selection": feature_dataset.feature_selection,
            "feature_approach": feature_dataset.feature_approach,
            "feature_representation": "tabpfn_global_embeddings"
            if getattr(args, "use_tabpfn_global_embeddings", False)
            else "source_features",
            "federated_component": "lightgbm_meta_classifier",
            "aggregation": "target_campaign_transfer_weighted",
            "tabpfn_n_estimators": getattr(args, "tabpfn_n_estimators", None),
            "tabpfn_embedding_cache": getattr(args, "tabpfn_embedding_cache", None),
            "transfer_matrix_root": CONFIG_TRANSFER_ROOT,
            "config": config,
        }
        key = stable_key(experiment_key)
        if key in done:
            print(f"[{position}/{len(configs)}] skipped {config}", flush=True)
            continue

        try:
            result = evaluate_transfer_weighted_config(base, feature_dataset, dataset, args, config, transfer_matrix)
        except Exception as exc:
            result = failed_result(feature_dataset, dataset, exc, config)

        result["experiment_name"] = CONFIG_EXPERIMENT_NAME
        result["experiment_key"] = experiment_key
        result["experiment_config"] = config
        result["grid_search_method"] = CONFIG_EXPERIMENT_NAME
        result["grid_model"] = CLASSIFIER_NAME
        result["grid_params"] = config
        if best_bayesian is not None:
            result["source_best_bayesian_optimization"] = best_bayesian
            result["source_best_bayesian_objective"] = best_bayesian["objective"]
            result["source_best_bayesian_objective_std"] = best_bayesian["std"]
            result["lightgbm_params_from_bayesian_optimization"] = best_bayesian["lightgbm_params"]
            result["tabpfn_params_from_bayesian_optimization"] = best_bayesian.get("tabpfn_params")

        results.append(result)
        done.add(key)
        save_results(output_path, results)
        print(
            f"[{position}/{len(configs)}] "
            f"{result['status']} ba={result.get('balanced_accuracy')} config={config}",
            flush=True,
        )

    save_results(output_path, results)
    print(f"Saved transfer-weighted federated LightGBM results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
