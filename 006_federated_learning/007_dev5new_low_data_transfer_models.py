#!/usr/bin/env python3
"""
Train four LightGBM variants on dev_new_5 campaigns with reverse 5-fold.

The `dev_new_5` subset contains five Facebook campaigns outside the original
`dev5` set. Each campaign is evaluated as a low-data target with a reverse
5-fold protocol: one fold is used for training and the remaining four folds are
used for testing. The other campaigns in `dev_new_5` are used as support
clients for the combined/federated variants.

Compared variants:
    - Bayes local: target-only TabPFN context/meta split matching the Bayesian run.
    - Combined: one server LightGBM trained on target train + support campaign train rows.
    - Federated: local LightGBMs per campaign with uniform score aggregation.
    - Residual local: federated score plus a target-campaign local residual model.

Shared loaders, TabPFN-embedding helpers, LightGBM utilities, and the
reverse-5-fold experiment loop live in `_dev5new_transfer_shared.py`. This
script only defines the aggregation strategy: uniform federated averaging.
See `008_dev5new_personalized_transfer_models.py` for the personalized
transfer-weighted variant of the same experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import _dev5new_transfer_shared as shared  # noqa: E402


CONFIG_OUTPUT_ROOT = "results/federated_learning/dev_new_5_low_data_transfer_models"
CONFIG_FEDERATED_WEIGHTING = "uniform"  # examples or uniform
CONFIG_FORCE_RERUN = True


def aggregate_test_raw(
    federated_models: list[dict[str, Any]],
    X_target_train: np.ndarray,
    y_target_train: np.ndarray,
    X_target_test: np.ndarray,
    target_campaign: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    federated_raw = shared.aggregate_federated_raw(federated_models, X_target_test, CONFIG_FEDERATED_WEIGHTING)
    return federated_raw, {"federated_weighting": CONFIG_FEDERATED_WEIGHTING}


def aggregate_train_raw(
    federated_models: list[dict[str, Any]],
    X_target_train: np.ndarray,
    y_target_train: np.ndarray,
    target_campaign: str,
) -> np.ndarray:
    return shared.aggregate_federated_raw(federated_models, X_target_train, CONFIG_FEDERATED_WEIGHTING)


def main() -> None:
    shared.run_transfer_experiment(
        output_root=CONFIG_OUTPUT_ROOT,
        experiment_name="dev_new_5_low_data_transfer_models",
        force_rerun=CONFIG_FORCE_RERUN,
        federated_weighting=CONFIG_FEDERATED_WEIGHTING,
        federated_variant_name="Federated",
        residual_variant_name="Residual local",
        aggregate_test_fn=aggregate_test_raw,
        aggregate_train_fn=aggregate_train_raw,
    )


if __name__ == "__main__":
    main()
