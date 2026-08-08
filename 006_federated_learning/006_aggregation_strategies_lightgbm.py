#!/usr/bin/env python3
"""
Compare four federated aggregation strategies for TabPFN embeddings + LightGBM.

This script extends the transfer-weighted federated LightGBM experiment with
four aggregation families motivated by federated aggregation surveys:

1. adaptive weighting;
2. similarity-based aggregation;
3. robust aggregation;
4. personalized aggregation.

References:
    - Qi et al. (2024), Model aggregation techniques in federated learning:
      A comprehensive survey.
    - Sah & Singh (2022), Aggregation Techniques in Federated Learning:
      Comprehensive Survey, Challenges and Opportunities.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.federated_lightgbm_runner import (  # noqa: E402
    prepare_base_experiment,
    save_results,
    stable_key,
)


TRANSFER_SCRIPT = PROJECT_ROOT / "006_federated_learning" / "005_transfer_weighted_federated_lightgbm.py"
CONFIG_OUTPUT_ROOT = "results/federated_learning/lightgbm_aggregation_strategies"
CONFIG_EXPERIMENT_NAME = "federated_lightgbm_aggregation_strategies"
CONFIG_FORCE_RERUN = False

CONFIG_GLOBAL_ROUNDS = 5
CONFIG_GLOBAL_LOCAL_TREES_PER_ROUND = 20
CONFIG_PREDICTION_THRESHOLD = 0.50

CONFIG_STRATEGIES = [
    {
        "aggregation_strategy": "adaptive_weighting",
        "transfer_weight_mode": "adaptive",
        "temperature": 0.10,
        "top_k": 0,
        "self_weight_floor": 0.10,
        "size_power": 0.25,
        "description": "weights combine transfer performance and source-client size prior",
    },
    {
        "aggregation_strategy": "similarity_based",
        "transfer_weight_mode": "similarity",
        "temperature": 0.08,
        "top_k": 3,
        "self_weight_floor": 0.20,
        "description": "weights use mutual campaign similarity derived from the transfer matrix",
    },
    {
        "aggregation_strategy": "robust_trimmed_mean",
        "transfer_weight_mode": "robust_trimmed_mean",
        "temperature": 1.00,
        "top_k": 0,
        "self_weight_floor": 0.00,
        "trim_fraction": 0.20,
        "description": "server aggregates local raw scores with per-sample trimmed mean",
    },
    {
        "aggregation_strategy": "personalized_transfer",
        "transfer_weight_mode": "personalized_softmax",
        "temperature": 0.10,
        "top_k": 3,
        "self_weight_floor": 0.35,
        "description": "each target campaign receives a different transfer-weighted ensemble",
    },
]


SOURCE_SIZE_PRIORS: dict[str, float] = {}


def load_transfer_module():
    spec = importlib.util.spec_from_file_location("transfer_weighted_federated_lightgbm_base", TRANSFER_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {TRANSFER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_existing(output_path: Path) -> list[dict]:
    path = output_path / "results.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def normalize_scores(scores: pd.Series, self_campaign: str, self_floor: float) -> dict[str, float]:
    scores = scores.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    if self_floor > 0.0 and self_campaign in scores.index:
        scores.loc[self_campaign] = max(float(scores.loc[self_campaign]), float(self_floor))
    total = float(scores.sum())
    if total <= 0.0:
        scores = pd.Series(0.0, index=scores.index)
        if self_campaign in scores.index:
            scores.loc[self_campaign] = 1.0
        else:
            scores[:] = 1.0 / max(1, len(scores))
    else:
        scores = scores / total
    return {str(key): float(value) for key, value in scores.items() if float(value) > 0.0}


def softmax_scores(scores: pd.Series, temperature: float) -> pd.Series:
    tau = max(float(temperature), 1e-6)
    scores = scores.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    centered = scores - float(scores.max())
    values = np.exp(centered / tau)
    values = np.where(scores.to_numpy() > 0.0, values, 0.0)
    return pd.Series(values, index=scores.index)


def keep_top_k(scores: pd.Series, top_k: int, self_campaign: str) -> pd.Series:
    if top_k <= 0 or top_k >= len(scores):
        return scores
    keep = set(scores.nlargest(top_k).index.astype(str))
    if self_campaign in scores.index:
        keep.add(self_campaign)
    return scores.where(scores.index.isin(keep), 0.0)


def adaptive_weights(matrix: pd.DataFrame, campaigns: list[str], config: dict[str, Any]) -> dict[str, dict[str, float]]:
    available = matrix.reindex(index=campaigns, columns=campaigns)
    size_power = float(config.get("size_power", 0.25))
    size_prior = pd.Series({campaign: SOURCE_SIZE_PRIORS.get(campaign, 1.0) for campaign in campaigns}, dtype=float)
    size_prior = (size_prior / max(float(size_prior.mean()), 1e-9)).pow(size_power)
    weights = {}
    for target_campaign in campaigns:
        scores = available[target_campaign].clip(lower=0.0).fillna(0.0) * size_prior
        scores = keep_top_k(scores, int(config.get("top_k", 0)), target_campaign)
        weights[target_campaign] = normalize_scores(scores, target_campaign, float(config.get("self_weight_floor", 0.0)))
    return weights


def similarity_weights(matrix: pd.DataFrame, campaigns: list[str], config: dict[str, Any]) -> dict[str, dict[str, float]]:
    available = matrix.reindex(index=campaigns, columns=campaigns)
    symmetric = (available + available.T) / 2.0
    weights = {}
    for target_campaign in campaigns:
        scores = symmetric[target_campaign].clip(lower=0.0).fillna(0.0)
        scores = keep_top_k(scores, int(config.get("top_k", 3)), target_campaign)
        scores = softmax_scores(scores, float(config.get("temperature", 0.1)))
        weights[target_campaign] = normalize_scores(scores, target_campaign, float(config.get("self_weight_floor", 0.0)))
    return weights


def personalized_weights(matrix: pd.DataFrame, campaigns: list[str], config: dict[str, Any]) -> dict[str, dict[str, float]]:
    available = matrix.reindex(index=campaigns, columns=campaigns)
    weights = {}
    for target_campaign in campaigns:
        scores = available[target_campaign].clip(lower=0.0).fillna(0.0)
        scores = keep_top_k(scores, int(config.get("top_k", 3)), target_campaign)
        scores = softmax_scores(scores, float(config.get("temperature", 0.1)))
        weights[target_campaign] = normalize_scores(scores, target_campaign, float(config.get("self_weight_floor", 0.0)))
    return weights


def uniform_weights(_matrix: pd.DataFrame, campaigns: list[str], _config: dict[str, Any]) -> dict[str, dict[str, float]]:
    value = 1.0 / max(1, len(campaigns))
    return {target: {source: value for source in campaigns} for target in campaigns}


def build_strategy_weights(matrix: pd.DataFrame, campaigns: list[str], config: dict[str, Any]) -> dict[str, dict[str, float]]:
    strategy = str(config.get("aggregation_strategy"))
    if strategy == "adaptive_weighting":
        return adaptive_weights(matrix, campaigns, config)
    if strategy == "similarity_based":
        return similarity_weights(matrix, campaigns, config)
    if strategy == "personalized_transfer":
        return personalized_weights(matrix, campaigns, config)
    if strategy == "robust_trimmed_mean":
        return uniform_weights(matrix, campaigns, config)
    raise ValueError(f"Unknown aggregation_strategy={strategy!r}")


def strategy_raw_score(original_raw_score, base, global_rounds, X, args, target_campaign: str, transfer_weights):
    strategy = str(getattr(args, "aggregation_strategy", ""))
    if strategy != "robust_trimmed_mean":
        return original_raw_score(base, global_rounds, X, args, target_campaign, transfer_weights)

    if not global_rounds:
        return np.zeros(X.shape[0], dtype=np.float64)
    raw = np.zeros(X.shape[0], dtype=np.float64)
    trim_fraction = float(getattr(args, "trim_fraction", 0.20))
    for round_models in global_rounds:
        contributions = []
        for _source_campaign, model, scaler in round_models:
            X_for_model = base.transform_with_scaler(X, scaler, args)
            contributions.append(base.model_raw_contribution(model, X_for_model))
        if not contributions:
            continue
        stacked = np.vstack(contributions)
        trim_count = int(np.floor(stacked.shape[0] * trim_fraction))
        if trim_count > 0 and stacked.shape[0] > 2 * trim_count:
            stacked = np.sort(stacked, axis=0)[trim_count:-trim_count]
        raw += np.mean(stacked, axis=0)
    return raw


def campaign_size_priors(base, feature_dataset) -> dict[str, float]:
    priors = {}
    for campaign_id, indices in base.campaign_indices(feature_dataset):
        priors[str(campaign_id)] = float(len(indices))
    return priors


def main() -> None:
    transfer_module = load_transfer_module()
    base = transfer_module.load_base_module()
    args, dataset, feature_dataset, best_bayesian = prepare_base_experiment(base, CONFIG_OUTPUT_ROOT)
    output_path = PROJECT_ROOT / CONFIG_OUTPUT_ROOT / dataset
    transfer_matrix = transfer_module.load_transfer_matrix(dataset)
    results = load_existing(output_path)
    done = {
        stable_key(result.get("experiment_key", {}))
        for result in results
        if result.get("status") in {"ok", "error", "failed", "skipped"} and not CONFIG_FORCE_RERUN
    }

    global SOURCE_SIZE_PRIORS
    SOURCE_SIZE_PRIORS = campaign_size_priors(base, feature_dataset)

    original_build_weights = transfer_module.build_transfer_weights
    original_raw_score = transfer_module.transfer_weighted_raw_score

    def patched_build_weights(matrix, campaigns, config):
        return build_strategy_weights(matrix, campaigns, config)

    def patched_raw_score(base_arg, global_rounds, X, args_arg, target_campaign, transfer_weights):
        return strategy_raw_score(original_raw_score, base_arg, global_rounds, X, args_arg, target_campaign, transfer_weights)

    transfer_module.build_transfer_weights = patched_build_weights
    transfer_module.transfer_weighted_raw_score = patched_raw_score

    print(
        f"{CONFIG_EXPERIMENT_NAME}: {feature_dataset.feature_stage}/"
        f"{feature_dataset.feature_selection or 'none'}/{feature_dataset.feature_approach}",
        flush=True,
    )

    try:
        configs = [
            {
                "rounds": CONFIG_GLOBAL_ROUNDS,
                "local_trees_per_round": CONFIG_GLOBAL_LOCAL_TREES_PER_ROUND,
                "prediction_threshold": CONFIG_PREDICTION_THRESHOLD,
                **strategy_config,
            }
            for strategy_config in CONFIG_STRATEGIES
        ]

        for position, config in enumerate(configs, start=1):
            experiment_key = {
                "experiment_name": CONFIG_EXPERIMENT_NAME,
                "dataset": dataset,
                "feature_stage": feature_dataset.feature_stage,
                "feature_selection": feature_dataset.feature_selection,
                "feature_approach": feature_dataset.feature_approach,
                "aggregation_strategy": config["aggregation_strategy"],
                "feature_representation": "tabpfn_global_embeddings"
                if getattr(args, "use_tabpfn_global_embeddings", False)
                else "source_features",
                "config": config,
            }
            key = stable_key(experiment_key)
            if key in done:
                print(f"[{position}/{len(configs)}] skipped {config}", flush=True)
                continue

            run_args = transfer_module.args_with_overrides(
                args,
                {
                    "aggregation_strategy": config["aggregation_strategy"],
                    "trim_fraction": float(config.get("trim_fraction", 0.0)),
                },
            )
            try:
                result = transfer_module.evaluate_transfer_weighted_config(
                    base,
                    feature_dataset,
                    dataset,
                    run_args,
                    config,
                    transfer_matrix,
                )
            except Exception as exc:
                result = transfer_module.failed_result(feature_dataset, dataset, exc, config)

            result["experiment_name"] = CONFIG_EXPERIMENT_NAME
            result["experiment_key"] = experiment_key
            result["experiment_config"] = config
            result["grid_search_method"] = CONFIG_EXPERIMENT_NAME
            result["grid_model"] = "Aggregation-Strategy-Federated-LightGBM"
            result["grid_params"] = config
            result["classifier"] = "Aggregation-Strategy-Federated-LightGBM"
            result["model"] = "Aggregation-Strategy-Federated-LightGBM"
            result["federated_algorithm"] = "GlobalTabPFNEmbeddingsLightGBMAggregationStrategyComparison"
            result["aggregation_strategy"] = config["aggregation_strategy"]
            result["aggregation_strategy_description"] = config["description"]
            result["aggregation_strategy_references"] = [
                "Qi et al. (2024), Model aggregation techniques in federated learning: A comprehensive survey.",
                "Sah & Singh (2022), Aggregation Techniques in Federated Learning: Comprehensive Survey, Challenges and Opportunities.",
            ]
            result["source_size_priors"] = SOURCE_SIZE_PRIORS
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
                f"[{position}/{len(configs)}] {result['status']} "
                f"ba={result.get('balanced_accuracy')} strategy={config['aggregation_strategy']}",
                flush=True,
            )
    finally:
        transfer_module.build_transfer_weights = original_build_weights
        transfer_module.transfer_weighted_raw_score = original_raw_score

    save_results(output_path, results)
    print(f"Saved aggregation strategy results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
