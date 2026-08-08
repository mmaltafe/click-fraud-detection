#!/usr/bin/env python3
"""
Train four LightGBM variants on dev_new_5 campaigns with personalized transfer aggregation.

The `dev_new_5` subset contains five Facebook campaigns outside the original
`dev5` set. Each campaign is evaluated as a low-data target with a reverse
5-fold protocol: one fold is used for training and the remaining four folds are
used for testing. The other campaigns in `dev_new_5` are used as support
clients for the combined/federated variants.

Compared variants:
    - Bayes local: target-only TabPFN context/meta split matching the Bayesian run.
    - Combined: one server LightGBM trained on target train + support campaign train rows.
    - Federated personalized: local LightGBMs aggregated with target-specific
      transfer weights, following the best aggregation strategy from
      `006_aggregation_strategies_lightgbm.py`.
    - Residual local personalized: personalized federated score plus a
      target-campaign local residual model.

Shared loaders, TabPFN-embedding helpers, LightGBM utilities, and the
reverse-5-fold experiment loop live in `_dev5new_transfer_shared.py`. This
script only defines the aggregation strategy: personalized transfer weighting.
See `007_dev5new_low_data_transfer_models.py` for the uniform-weighting
variant of the same experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import _dev5new_transfer_shared as shared  # noqa: E402


CONFIG_OUTPUT_ROOT = "results/federated_learning/dev_new_5_personalized_transfer_models"
CONFIG_FEDERATED_WEIGHTING = "personalized_transfer"
CONFIG_PERSONALIZED_TRANSFER_TEMPERATURE = 0.10
CONFIG_PERSONALIZED_TRANSFER_TOP_K = 3
CONFIG_PERSONALIZED_TRANSFER_SELF_WEIGHT_FLOOR = 0.35
CONFIG_FORCE_RERUN = True


def softmax_positive_scores(scores: dict[str, float], temperature: float) -> dict[str, float]:
    if not scores:
        return {}
    keys = list(scores)
    values = np.asarray([max(float(scores[key]), 0.0) for key in keys], dtype=np.float64)
    positive = values > 0.0
    if not positive.any():
        return {key: 1.0 / len(keys) for key in keys}
    centered = values - float(values.max())
    weights = np.exp(centered / max(float(temperature), 1e-6))
    weights = np.where(positive, weights, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return {key: 1.0 / len(keys) for key in keys}
    return {key: float(weight / total) for key, weight in zip(keys, weights)}


def personalized_transfer_weights(
    models: list[dict[str, Any]],
    X_target_train: np.ndarray,
    y_target_train: np.ndarray,
    target_campaign: str,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for payload in models:
        campaign = str(payload["campaign"])
        try:
            predictions = shared.predict_from_raw(shared.raw_score(payload["model"], X_target_train))
            scores[campaign] = float(balanced_accuracy_score(y_target_train, predictions))
        except Exception:
            scores[campaign] = 0.0

    if CONFIG_PERSONALIZED_TRANSFER_TOP_K > 0 and CONFIG_PERSONALIZED_TRANSFER_TOP_K < len(scores):
        keep = {
            campaign
            for campaign, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[
                : CONFIG_PERSONALIZED_TRANSFER_TOP_K
            ]
        }
        keep.add(target_campaign)
        scores = {campaign: score if campaign in keep else 0.0 for campaign, score in scores.items()}

    weights = softmax_positive_scores(scores, CONFIG_PERSONALIZED_TRANSFER_TEMPERATURE)
    if target_campaign in weights:
        weights[target_campaign] = max(
            float(weights[target_campaign]),
            float(CONFIG_PERSONALIZED_TRANSFER_SELF_WEIGHT_FLOOR),
        )
        total = float(sum(weights.values()))
        weights = {campaign: float(weight / total) for campaign, weight in weights.items()}
    return {campaign: weight for campaign, weight in weights.items() if weight > 0.0}


def aggregate_personalized_transfer_raw(
    models: list[dict[str, Any]],
    X: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    raw = np.zeros(X.shape[0], dtype=np.float64)
    weight_sum = 0.0
    for payload in models:
        campaign = str(payload["campaign"])
        weight = float(weights.get(campaign, 0.0))
        if weight <= 0.0:
            continue
        raw += shared.raw_score(payload["model"], X) * weight
        weight_sum += weight
    if weight_sum <= 0.0:
        return shared.aggregate_federated_raw(models, X, "uniform")
    return raw / weight_sum


def aggregate_test_raw(
    federated_models: list[dict[str, Any]],
    X_target_train: np.ndarray,
    y_target_train: np.ndarray,
    X_target_test: np.ndarray,
    target_campaign: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    transfer_weights = personalized_transfer_weights(federated_models, X_target_train, y_target_train, target_campaign)
    federated_raw = aggregate_personalized_transfer_raw(federated_models, X_target_test, transfer_weights)
    extra = {
        "aggregation_strategy": "personalized_transfer",
        "federated_weighting": CONFIG_FEDERATED_WEIGHTING,
        "personalized_transfer_weights": transfer_weights,
        "personalized_transfer_temperature": CONFIG_PERSONALIZED_TRANSFER_TEMPERATURE,
        "personalized_transfer_top_k": CONFIG_PERSONALIZED_TRANSFER_TOP_K,
        "personalized_transfer_self_weight_floor": CONFIG_PERSONALIZED_TRANSFER_SELF_WEIGHT_FLOOR,
    }
    return federated_raw, extra


def aggregate_train_raw(
    federated_models: list[dict[str, Any]],
    X_target_train: np.ndarray,
    y_target_train: np.ndarray,
    target_campaign: str,
) -> np.ndarray:
    transfer_weights = personalized_transfer_weights(federated_models, X_target_train, y_target_train, target_campaign)
    return aggregate_personalized_transfer_raw(federated_models, X_target_train, transfer_weights)


def main() -> None:
    shared.run_transfer_experiment(
        output_root=CONFIG_OUTPUT_ROOT,
        experiment_name="dev_new_5_personalized_transfer_models",
        force_rerun=CONFIG_FORCE_RERUN,
        federated_weighting=CONFIG_FEDERATED_WEIGHTING,
        federated_variant_name="Federated personalized",
        residual_variant_name="Residual local personalized",
        aggregate_test_fn=aggregate_test_raw,
        aggregate_train_fn=aggregate_train_raw,
        extra_run_metadata={
            "aggregation_strategy": "personalized_transfer",
            "aggregation_strategy_source": "best result from 006_federated_learning/006_aggregation_strategies_lightgbm.py",
            "personalized_transfer_temperature": CONFIG_PERSONALIZED_TRANSFER_TEMPERATURE,
            "personalized_transfer_top_k": CONFIG_PERSONALIZED_TRANSFER_TOP_K,
            "personalized_transfer_self_weight_floor": CONFIG_PERSONALIZED_TRANSFER_SELF_WEIGHT_FLOOR,
        },
    )


if __name__ == "__main__":
    main()
