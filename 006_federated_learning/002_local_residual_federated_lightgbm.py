#!/usr/bin/env python3
"""
Evaluate personalized residual LightGBM on top of the federated TabPFN-embeddings model.

The global model follows `000_federated_lightgbm.py`: TabPFN produces a shared
embedding space for all campaigns in each fold, and LightGBM is trained in a
Flower-style federated loop. After the global rounds, each campaign trains a
small local residual LightGBM using the global raw score as `init_score`.

The goal is to keep the federated model as the shared base while recovering
campaign-specific behavior that is lost by global aggregation.
"""

from __future__ import annotations

import sys
import time
import warnings
from argparse import Namespace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, recall_score
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from utils._campaigns import build_client_splits, summarize_dicts, summarize_numeric, summarize_numeric_std  # noqa: E402
from utils._fairness_metrics import fairness_gap, group_balanced_accuracy, metric_variance  # noqa: E402
from utils.federated_lightgbm_runner import (  # noqa: E402
    PROJECT_ROOT,
    args_with_overrides,
    load_base_module,
    load_existing_results,
    prepare_base_experiment,
    save_results,
    stable_key,
)


CONFIG_OUTPUT_ROOT = "results/federated_learning/lightgbm_personalized_residual"
CONFIG_EXPERIMENT_NAME = "federated_lightgbm_personalized_residual"

CONFIG_GLOBAL_ROUNDS = 10
CONFIG_GLOBAL_LOCAL_TREES_PER_ROUND = 20
CONFIG_GLOBAL_WEIGHTING = "uniform"
CONFIG_GLOBAL_PREDICTION_THRESHOLD = 0.5

CONFIG_PERSONALIZATION_CONFIGS = [
    {
        "global_rounds": CONFIG_GLOBAL_ROUNDS,
        "global_local_trees_per_round": CONFIG_GLOBAL_LOCAL_TREES_PER_ROUND,
        "global_weighting": CONFIG_GLOBAL_WEIGHTING,
        "prediction_threshold": threshold,
        "personalization_trees": trees,
        "personalization_learning_rate": learning_rate,
        "personalization_num_leaves": leaves,
        "personalization_max_depth": max_depth,
        "personalization_min_child_samples": min_child_samples,
        "personalization_alpha": alpha,
    }
    for trees, learning_rate, leaves, max_depth, min_child_samples, alpha, threshold in product(
        (5, 10, 20),
        (0.02, 0.05),
        (7, 15),
        (3, 5),
        (10, 25),
        (0.50, 1.00),
        (0.45, 0.50),
    )
]


CLASSIFIER_NAME = "Personalized-Federated-LightGBM"
FEDERATED_ALGORITHM = "GlobalTabPFNEmbeddingsFederatedLightGBMWithLocalResidual"


def make_personalization_args(args: Namespace, config: dict[str, Any]) -> Namespace:
    return args_with_overrides(
        args,
        {
            "n_estimators": int(config["personalization_trees"]),
            "learning_rate": float(config["personalization_learning_rate"]),
            "num_leaves": int(config["personalization_num_leaves"]),
            "max_depth": int(config["personalization_max_depth"]),
            "min_child_samples": int(config["personalization_min_child_samples"]),
        },
    )


def train_global_rounds(base, active_dataset, y: np.ndarray, client_splits, fold_number: int, args: Namespace):
    global_rounds: list[list[tuple[Any, Any, float]]] = []
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
                base.global_raw_score(global_rounds, base.take_rows(active_dataset.X, train_idx), args)
                if global_rounds
                else None
            )
            model, scaler = base.fit_client_model(
                active_dataset,
                y,
                train_idx,
                args,
                int(args.random_state) + 10_000 * (fold_number + 1) + 1_000 * (round_number + 1) + client_position,
                init_score=init_score,
            )
            weight = base.client_weight(len(train_idx), args.weighting)
            round_models.append((model, scaler, weight))
            client_train_sizes[str(campaign_id)] = int(len(train_idx))
            model_bytes += base.model_communication_cost(model)

        if len(round_models) < args.min_clients:
            raise ValueError(f"fold {fold_number + 1}, round {round_number + 1} has fewer usable clients")
        global_rounds.append(round_models)
    return global_rounds, model_bytes, client_train_sizes


def fit_personalized_residual(base, active_dataset, y: np.ndarray, train_idx: np.ndarray, init_score: np.ndarray, args: Namespace, seed: int):
    X_train = base.maybe_dense(base.take_rows(active_dataset.X, train_idx), args.max_dense_cells)
    scaler = None
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = base.make_lightgbm(args, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y[train_idx], init_score=init_score)
    return model, scaler


def evaluate_personalized_config(base, feature_dataset, dataset: str, args: Namespace, config: dict[str, Any]) -> dict:
    started_total = time.perf_counter()
    y_text = feature_dataset.target["attack_type"].fillna(base.MISSING).astype(str).to_numpy()
    if len(np.unique(y_text)) < 2:
        raise ValueError("target has fewer than two classes")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    labels = np.arange(len(label_encoder.classes_))
    client_splits, n_global_folds = build_client_splits(
        feature_dataset, y, args.k_folds, args.random_state, args.min_clients, args.max_clients
    )

    global_args = args_with_overrides(
        args,
        {
            "rounds": int(config["global_rounds"]),
            "local_trees_per_round": int(config["global_local_trees_per_round"]),
            "weighting": str(config["global_weighting"]),
            "prediction_threshold": float(config["prediction_threshold"]),
        },
    )
    personal_args = make_personalization_args(args, config)
    alpha = float(config["personalization_alpha"])
    threshold = float(config["prediction_threshold"])

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

        global_rounds, global_model_bytes, client_train_sizes = train_global_rounds(
            base,
            active_dataset,
            y,
            client_splits,
            fold_number,
            global_args,
        )

        y_true_parts = []
        y_pred_parts = []
        campaign_groups: list[str] = []
        traffic_groups: list[str] = []
        local_model_bytes = 0
        local_test_rows = 0

        for client_position, (campaign_id, global_indices, splits, _split_strategy) in enumerate(client_splits):
            local_train_idx, local_test_idx = splits[fold_number]
            train_idx = global_indices[local_train_idx]
            test_idx = global_indices[local_test_idx]
            if len(test_idx) == 0 or len(np.unique(y[train_idx])) < 2:
                continue

            train_global_raw = base.global_raw_score(global_rounds, base.take_rows(active_dataset.X, train_idx), global_args)
            local_model, local_scaler = fit_personalized_residual(
                base,
                active_dataset,
                y,
                train_idx,
                train_global_raw,
                personal_args,
                int(args.random_state) + 50_000 * (fold_number + 1) + client_position,
            )
            test_X = base.take_rows(active_dataset.X, test_idx)
            test_global_raw = base.global_raw_score(global_rounds, test_X, global_args)
            local_raw = base.model_raw_contribution(
                local_model,
                base.transform_with_scaler(test_X, local_scaler, personal_args),
            )
            positive_proba = base.sigmoid(test_global_raw + alpha * local_raw)
            y_pred = (positive_proba >= threshold).astype(int)

            y_true_parts.append(y[test_idx])
            y_pred_parts.append(y_pred)
            campaign_groups.extend([str(campaign_id)] * len(test_idx))
            if feature_dataset.row_index is not None and "traffic_source" in feature_dataset.row_index.columns:
                traffic_groups.extend(
                    feature_dataset.row_index.iloc[test_idx]["traffic_source"].fillna(base.MISSING).astype(str).tolist()
                )
            else:
                traffic_groups.extend([base.MISSING] * len(test_idx))
            local_model_bytes += base.model_communication_cost(local_model)
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
                "communication_cost": int(global_model_bytes + local_model_bytes),
                "global_communication_cost": int(global_model_bytes),
                "personalization_model_bytes": int(local_model_bytes),
                "server_received_raw_rows": 0,
                "server_received_raw_columns": 0,
                "number_of_rounds": int(global_args.rounds),
                "local_trees_per_round": int(global_args.local_trees_per_round),
                "personalization_trees": int(config["personalization_trees"]),
                "personalization_alpha": alpha,
                "prediction_threshold": threshold,
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
        "evaluation_scope": "federated_clients_personalized",
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
        "personalized_component": "local_lightgbm_residual",
        "tabpfn_is_federated": False,
        "rounds": int(global_args.rounds),
        "local_trees_per_round": int(global_args.local_trees_per_round),
        "weighting": str(global_args.weighting),
        "prediction_threshold": threshold,
        "personalization_trees": int(config["personalization_trees"]),
        "personalization_learning_rate": float(config["personalization_learning_rate"]),
        "personalization_num_leaves": int(config["personalization_num_leaves"]),
        "personalization_max_depth": int(config["personalization_max_depth"]),
        "personalization_min_child_samples": int(config["personalization_min_child_samples"]),
        "personalization_alpha": alpha,
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
        "global_communication_cost",
        "personalization_model_bytes",
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
        "evaluation_scope": "federated_clients_personalized",
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
    results = load_existing_results(output_path)
    done = {
        stable_key(result.get("experiment_key", {}))
        for result in results
        if result.get("status") in {"ok", "error", "failed", "skipped"}
    }

    print(
        f"{CONFIG_EXPERIMENT_NAME}: {feature_dataset.feature_stage}/"
        f"{feature_dataset.feature_selection or 'none'}/{feature_dataset.feature_approach}",
        flush=True,
    )

    for position, config in enumerate(CONFIG_PERSONALIZATION_CONFIGS, start=1):
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
            "personalized_component": "local_lightgbm_residual",
            "tabpfn_n_estimators": getattr(args, "tabpfn_n_estimators", None),
            "tabpfn_embedding_cache": getattr(args, "tabpfn_embedding_cache", None),
            "config": config,
        }
        key = stable_key(experiment_key)
        if key in done:
            print(f"[{position}/{len(CONFIG_PERSONALIZATION_CONFIGS)}] skipped {config}", flush=True)
            continue

        try:
            result = evaluate_personalized_config(base, feature_dataset, dataset, args, config)
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
            f"[{position}/{len(CONFIG_PERSONALIZATION_CONFIGS)}] "
            f"{result['status']} ba={result.get('balanced_accuracy')} config={config}",
            flush=True,
        )

    save_results(output_path, results)
    print(f"Saved personalized federated LightGBM results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
