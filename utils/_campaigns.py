from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


MISSING = "__missing__"


def count_table_rows(path: Path) -> int:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return len(pd.read_parquet(path))
    return len(pd.read_csv(path))


def raw_row_index(dataset: str, raw_root: Path, expected_rows: int) -> pd.DataFrame | None:
    dataset_path = raw_root / dataset
    if not dataset_path.exists():
        return None

    rows: list[dict[str, Any]] = []
    if dataset == "all50":
        files = [
            (path, path.parent.name, path.stem)
            for path in sorted(dataset_path.glob("TS_*/*"))
            if path.is_file() and path.suffix.lower() in {".csv", ".parquet", ".pq"}
        ]
    else:
        files = [
            (path, MISSING, path.stem)
            for path in sorted(dataset_path.glob("*"))
            if path.is_file() and path.suffix.lower() in {".csv", ".parquet", ".pq"}
        ]

    for path, traffic_source, campaign in files:
        for source_row_index in range(count_table_rows(path)):
            rows.append(
                {
                    "traffic_source": traffic_source,
                    "campaign": campaign,
                    "source_path": str(path),
                    "source_row_index": source_row_index,
                }
            )

    if len(rows) != expected_rows:
        return None
    return pd.DataFrame(rows)


def ensure_campaign_row_index(feature_dataset, dataset: str, raw_root: Path):
    row_index = getattr(feature_dataset, "row_index", None)
    target = getattr(feature_dataset, "target")
    if (
        row_index is not None
        and len(row_index) == len(target)
        and "campaign" in row_index.columns
    ):
        return feature_dataset

    reconstructed = raw_row_index(dataset, raw_root, len(target))
    if reconstructed is None:
        return feature_dataset

    return type(feature_dataset)(
        feature_dataset.feature_stage,
        feature_dataset.feature_selection,
        feature_dataset.feature_approach,
        feature_dataset.path,
        feature_dataset.X,
        feature_dataset.target,
        reconstructed,
    )


def campaign_indices(feature_dataset) -> list[tuple[str, np.ndarray]]:
    row_index = getattr(feature_dataset, "row_index", None)
    target = getattr(feature_dataset, "target")
    if row_index is None or len(row_index) != len(target) or "campaign" not in row_index.columns:
        return []

    campaigns = row_index["campaign"].fillna(MISSING).astype(str)
    groups: list[tuple[str, np.ndarray]] = []
    for campaign in sorted(campaigns.unique()):
        indices = np.flatnonzero(campaigns.to_numpy() == campaign)
        if len(indices) > 0:
            groups.append((str(campaign), indices))
    return groups


def subset_feature_dataset(feature_dataset, indices: np.ndarray, campaign_id: str | None = None):
    X = feature_dataset.X[indices]
    target = feature_dataset.target.iloc[indices].reset_index(drop=True)
    row_index = None
    if feature_dataset.row_index is not None and len(feature_dataset.row_index) == len(feature_dataset.target):
        row_index = feature_dataset.row_index.iloc[indices].reset_index(drop=True)
        if campaign_id is not None and "campaign" not in row_index.columns:
            row_index["campaign"] = campaign_id
    return type(feature_dataset)(
        feature_dataset.feature_stage,
        feature_dataset.feature_selection,
        feature_dataset.feature_approach,
        feature_dataset.path,
        X,
        target,
        row_index,
    )


def campaign_kfold_splits(y: np.ndarray, n_splits: int, random_state: int):
    if n_splits < 2:
        raise ValueError("--k-folds must be at least 2")
    if len(y) < 2:
        raise ValueError("campaign has fewer than two rows")

    effective_splits = min(n_splits, len(y))
    class_counts = pd.Series(y).value_counts()
    if len(class_counts) > 1 and class_counts.min() >= effective_splits:
        splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
        return list(splitter.split(np.arange(len(y)), y)), "stratified_kfold", effective_splits

    splitter = KFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(np.arange(len(y)))), "kfold", effective_splits


def build_client_splits(
    feature_dataset,
    y: np.ndarray,
    k_folds: int,
    random_state: int,
    min_clients: int,
    max_clients: int,
) -> tuple[list[tuple[str, np.ndarray, list, str]], int]:
    campaigns = campaign_indices(feature_dataset)
    if max_clients > 0:
        campaigns = campaigns[:max_clients]
    if len(campaigns) < min_clients:
        raise ValueError(f"fewer than min_clients={min_clients}: {len(campaigns)}")

    client_splits = []
    effective_folds = []
    for campaign_id, indices in campaigns:
        local_y = y[indices]
        if len(np.unique(local_y)) < 2:
            continue
        splits, split_strategy, n_folds = campaign_kfold_splits(local_y, k_folds, random_state)
        client_splits.append((campaign_id, indices, splits, split_strategy))
        effective_folds.append(n_folds)

    if len(client_splits) < min_clients:
        raise ValueError(f"fewer usable clients after class filtering: {len(client_splits)}")
    return client_splits, int(min(effective_folds))


def summarize_numeric(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None and not pd.isna(value)]
    if not numeric:
        return None
    return float(np.mean(numeric))


def summarize_numeric_std(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None and not pd.isna(value)]
    if len(numeric) < 2:
        return None
    return float(np.std(numeric, ddof=1))


def summarize_dicts(dicts: list[dict | None]) -> dict | None:
    keys: set[str] = set()
    for value in dicts:
        if value:
            keys.update(str(key) for key in value.keys())
    if not keys:
        return None
    summary = {}
    for key in sorted(keys):
        summary[key] = summarize_numeric([
            value.get(key)
            for value in dicts
            if value is not None and key in value
        ])
    return summary


def aggregate_campaign_fold_results(
    fold_results: list[dict],
    feature_dataset,
    campaign_id: str,
    classifier_name: str,
    dataset: str | None = None,
    federated_algorithm: str | None = None,
) -> dict:
    if not fold_results:
        raise ValueError("no fold results to aggregate")

    first = fold_results[0]
    status_values = [result.get("status") for result in fold_results]
    ok_results = [result for result in fold_results if result.get("status") == "ok"]
    base = {
        "feature_stage": feature_dataset.feature_stage,
        "feature_selection": feature_dataset.feature_selection,
        "feature_approach": feature_dataset.feature_approach,
        "classifier": classifier_name,
        "federated_algorithm": federated_algorithm,
        "dataset": dataset,
        "evaluation_scope": "campaign",
        "campaign": campaign_id,
        "campaign_id": campaign_id,
        "n_samples": int(len(feature_dataset.target)),
        "n_train": int(sum(result.get("n_train") or 0 for result in fold_results)),
        "n_test": int(sum(result.get("n_test") or 0 for result in fold_results)),
        "n_features": int(feature_dataset.X.shape[1]),
        "k_folds": int(len(fold_results)),
        "fold_statuses": status_values,
        "status": "ok" if ok_results else "error",
        "error": None if ok_results else "; ".join(str(result.get("error")) for result in fold_results),
    }

    for key in (
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "fit_predict_seconds",
        "elapsed_seconds",
        "communication_cost",
        "number_of_rounds",
        "client_variance",
        "fairness_between_campaigns",
        "campaign_metric_variance",
        "campaign_fairness_gap",
        "traffic_source_metric_variance",
        "traffic_source_fairness_gap",
    ):
        values = [result.get(key) for result in ok_results]
        base[key] = summarize_numeric(values)
        base[f"{key}_fold_std"] = summarize_numeric_std(values)

    base["ok_folds"] = int(len(ok_results))

    base["recall_by_class"] = summarize_dicts([result.get("recall_by_class") for result in ok_results])
    base["performance_by_traffic_source"] = summarize_dicts([
        result.get("performance_by_traffic_source") or result.get("traffic_source_performance")
        for result in ok_results
    ])
    base["campaign_performance"] = {campaign_id: base["balanced_accuracy"]} if base["balanced_accuracy"] is not None else None
    base["traffic_source_performance"] = base["performance_by_traffic_source"]

    for key, value in first.items():
        if key not in base and key not in {"config", "resume_key", "config_hash"}:
            base[key] = value
    return base


def cap_rows_by_campaign(feature_dataset, max_rows: int, random_state: int):
    if max_rows <= 0 or len(feature_dataset.target) <= max_rows:
        return feature_dataset
    row_index = getattr(feature_dataset, "row_index", None)
    if row_index is None or len(row_index) != len(feature_dataset.target) or "campaign" not in row_index.columns:
        return feature_dataset

    campaigns = campaign_indices(feature_dataset)
    if not campaigns:
        return feature_dataset

    rng = np.random.default_rng(random_state)
    sampled: list[int] = []
    per_campaign = max(1, max_rows // len(campaigns))
    for _campaign, indices in campaigns:
        if len(indices) <= per_campaign:
            sampled.extend(indices.tolist())
            continue
        sampled.extend(rng.choice(indices, size=per_campaign, replace=False).tolist())

    if len(sampled) < max_rows:
        remaining = np.setdiff1d(np.arange(len(feature_dataset.target)), np.asarray(sampled), assume_unique=False)
        extra = min(len(remaining), max_rows - len(sampled))
        if extra > 0:
            sampled.extend(rng.choice(remaining, size=extra, replace=False).tolist())

    sampled = sorted(sampled[:max_rows])
    return subset_feature_dataset(feature_dataset, np.asarray(sampled, dtype=int))
