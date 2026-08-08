from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, recall_score
from sklearn.model_selection import train_test_split

from utils._feature_datasets import FeatureDataset


def fairness_gap(values: dict[str, float | None] | None) -> float | None:
    if not values:
        return None
    numeric = [float(value) for value in values.values() if value is not None and not pd.isna(value)]
    if len(numeric) < 2:
        return None
    return float(max(numeric) - min(numeric))


def metric_variance(values: dict[str, float | None] | None) -> float | None:
    if not values:
        return None
    numeric = [float(value) for value in values.values() if value is not None and not pd.isna(value)]
    if len(numeric) < 2:
        return None
    return float(np.var(numeric))


def group_balanced_accuracy(y_true, y_pred, groups) -> dict[str, float | None] | None:
    if groups is None:
        return None
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": pd.Series(groups).astype(str).values})
    values: dict[str, float | None] = {}
    for group, part in frame.groupby("group"):
        if part["y_true"].nunique() < 2:
            values[str(group)] = None
        else:
            values[str(group)] = float(balanced_accuracy_score(part["y_true"], part["y_pred"]))
    return values


def extract_groups(feature_dataset: FeatureDataset, test_idx: np.ndarray):
    if feature_dataset.row_index is None or len(feature_dataset.row_index) != len(feature_dataset.target):
        return None, None
    row_index = feature_dataset.row_index.iloc[test_idx].reset_index(drop=True)
    campaign = row_index["campaign"] if "campaign" in row_index.columns else None
    traffic_source = row_index["traffic_source"] if "traffic_source" in row_index.columns else None
    return campaign, traffic_source


def communication_cost(X) -> int:
    if sparse.issparse(X):
        return int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)
    return int(np.asarray(X).nbytes)


def stratified_split_indices(y: np.ndarray, test_size: float, random_state: int):
    class_counts = pd.Series(y).value_counts()
    stratify = y if len(class_counts) > 1 and class_counts.min() >= 2 else None
    return train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


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
