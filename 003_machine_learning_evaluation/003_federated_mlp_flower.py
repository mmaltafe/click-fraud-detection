#!/usr/bin/env python3
"""
Evaluate generated feature datasets with local federated MLP training.

The implementation runs a local Flower-compatible simulation: model parameters
are serialized with `flwr.common.ndarrays_to_parameters`, while the training
loop remains in-process to avoid external services and API calls. It evaluates
FedAvg and FedProx over client partitions built from campaigns when available.

Output:
    results/machine_learning_evaluation/federated_mlp_flower/{DATASET}/results.csv
    results/machine_learning_evaluation/federated_mlp_flower/{DATASET}/results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from flwr.common import ndarrays_to_parameters
from scipy import sparse
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils._env import VALID_DATASETS, MISSING, read_env_value  # noqa: E402
from utils.target_utils import binary_target_frame  # noqa: E402

from utils._campaigns import (
    aggregate_campaign_fold_results,
    campaign_indices,
    campaign_kfold_splits,
    cap_rows_by_campaign,
    subset_feature_dataset,
)
from utils._resume import add_resume_metadata, base_config, completed_keys, config_hash, load_existing_results, result_key, save_results


EXTRACTED_APPROACHES = ("semantic_headers", "tf_idf", "sentence_transformer")
SELECTED_METHODS = ("pca", "truncatedSVD", "chi2", "selectKBest")
SELECTED_APPROACHES = ("label_encoder", "semantic_headers", "tf_idf", "sentence_transformer")
FEDERATED_ALGORITHMS = ("FedAvg", "FedProx")


@dataclass
class FeatureDataset:
    feature_stage: str
    feature_selection: str | None
    feature_approach: str
    path: Path
    X: Any
    target: pd.DataFrame
    row_index: pd.DataFrame | None


class MLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x):
        return self.network(x)


# Pipeline configuration constants. Edit these values to change this script.
CONFIG_ENV_FILE = '.env'
CONFIG_EXTRACTED_ROOT = 'data/extracted_features'
CONFIG_SELECTED_ROOT = 'data/selected_features'
CONFIG_RAW_ROOT = 'data/raw'
CONFIG_OUTPUT_ROOT = 'results/machine_learning_evaluation/federated_mlp_flower'
CONFIG_K_FOLDS = 5
CONFIG_TEST_SIZE = 0.3
CONFIG_RANDOM_STATE = 42
CONFIG_MAX_ROWS = 0
CONFIG_ROUNDS = 5
CONFIG_LOCAL_EPOCHS = 1
CONFIG_BATCH_SIZE = 128
CONFIG_LEARNING_RATE = 0.001
CONFIG_HIDDEN_SIZE = 64
CONFIG_MIN_CLIENTS = 2
CONFIG_MAX_CLIENTS = 0
CONFIG_FEDPROX_MU = 0.01


def parse_args() -> argparse.Namespace:
    """Return script configuration from global constants; command-line args are intentionally ignored."""
    return argparse.Namespace(
        env_file=CONFIG_ENV_FILE,
        extracted_root=CONFIG_EXTRACTED_ROOT,
        selected_root=CONFIG_SELECTED_ROOT,
        raw_root=CONFIG_RAW_ROOT,
        output_root=CONFIG_OUTPUT_ROOT,
        k_folds=CONFIG_K_FOLDS,
        test_size=CONFIG_TEST_SIZE,
        random_state=CONFIG_RANDOM_STATE,
        max_rows=CONFIG_MAX_ROWS,
        rounds=CONFIG_ROUNDS,
        local_epochs=CONFIG_LOCAL_EPOCHS,
        batch_size=CONFIG_BATCH_SIZE,
        learning_rate=CONFIG_LEARNING_RATE,
        hidden_size=CONFIG_HIDDEN_SIZE,
        min_clients=CONFIG_MIN_CLIENTS,
        max_clients=CONFIG_MAX_CLIENTS,
        fedprox_mu=CONFIG_FEDPROX_MU,
    )


def read_metadata(path: Path) -> dict:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def metadata_row_index(path: Path) -> pd.DataFrame | None:
    row_index = read_metadata(path).get("row_index")
    if not row_index:
        return None
    return pd.DataFrame(row_index)


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


def ensure_campaign_row_index(feature_dataset: FeatureDataset, dataset: str, raw_root: Path) -> FeatureDataset:
    if (
        feature_dataset.row_index is not None
        and len(feature_dataset.row_index) == len(feature_dataset.target)
        and "campaign" in feature_dataset.row_index.columns
    ):
        return feature_dataset
    row_index = raw_row_index(dataset, raw_root, len(feature_dataset.target))
    if row_index is None:
        return feature_dataset
    return FeatureDataset(
        feature_dataset.feature_stage,
        feature_dataset.feature_selection,
        feature_dataset.feature_approach,
        feature_dataset.path,
        feature_dataset.X,
        feature_dataset.target,
        row_index,
    )


def load_extracted_dataset(root: Path, dataset: str, approach: str) -> FeatureDataset | None:
    path = root / approach / dataset
    if approach == "semantic_headers":
        features_path = path / "semantic_headers.parquet"
        target_path = path / "target.parquet"
        if not features_path.exists() or not target_path.exists():
            return None
        X = pd.read_parquet(features_path).to_numpy(dtype=np.float32)
        target = binary_target_frame(pd.read_parquet(target_path))
        return FeatureDataset("extracted_features", None, approach, path, X, target, None)

    if approach == "tf_idf":
        features_path = path / "tf_idf_matrix.npz"
        target_path = path / "target.parquet"
        if not features_path.exists() or not target_path.exists():
            return None
        X = sparse.load_npz(features_path).astype(np.float32)
        target = binary_target_frame(pd.read_parquet(target_path))
        return FeatureDataset("extracted_features", None, approach, path, X, target, metadata_row_index(path))

    if approach == "sentence_transformer":
        features_path = path / "embeddings.npy"
        target_path = path / "target.parquet"
        if not features_path.exists() or not target_path.exists():
            return None
        X = np.load(features_path).astype(np.float32, copy=False)
        target = binary_target_frame(pd.read_parquet(target_path))
        return FeatureDataset("extracted_features", None, approach, path, X, target, metadata_row_index(path))

    return None


def load_selected_dataset(root: Path, dataset: str, method: str, approach: str) -> FeatureDataset | None:
    path = root / method / approach / dataset
    features_path = path / "features.npy"
    target_path = path / "target.parquet"
    if not features_path.exists() or not target_path.exists():
        return None
    X = np.load(features_path).astype(np.float32, copy=False)
    target = binary_target_frame(pd.read_parquet(target_path))
    return FeatureDataset("selected_features", method, approach, path, X, target, None)


def discover_feature_datasets(extracted_root: Path, selected_root: Path, raw_root: Path, dataset: str) -> list[FeatureDataset]:
    datasets: list[FeatureDataset] = []
    for approach in EXTRACTED_APPROACHES:
        loaded = load_extracted_dataset(extracted_root, dataset, approach)
        if loaded is not None:
            datasets.append(ensure_campaign_row_index(loaded, dataset, raw_root))

    for method in SELECTED_METHODS:
        for approach in SELECTED_APPROACHES:
            loaded = load_selected_dataset(selected_root, dataset, method, approach)
            if loaded is not None:
                datasets.append(ensure_campaign_row_index(loaded, dataset, raw_root))
    return datasets


def dense_array(X, max_dense_cells: int = 20_000_000) -> np.ndarray:
    if sparse.issparse(X):
        cells = X.shape[0] * X.shape[1]
        if cells > max_dense_cells:
            raise MemoryError(f"Refusing to densify sparse matrix with {cells} cells")
        return X.toarray().astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def stratified_split_indices(y: np.ndarray, test_size: float, random_state: int):
    class_counts = pd.Series(y).value_counts()
    stratify = y if len(class_counts) > 1 and class_counts.min() >= 2 else None
    return train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def model_parameters(model: nn.Module) -> list[np.ndarray]:
    return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]


def set_model_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    state_dict = model.state_dict()
    new_state = {
        key: torch.tensor(value, dtype=state_dict[key].dtype)
        for key, value in zip(state_dict.keys(), parameters)
    }
    model.load_state_dict(new_state, strict=True)


def parameter_bytes(parameters: list[np.ndarray]) -> int:
    # Flower serialization is used here to keep this estimate tied to Flower's
    # parameter representation while avoiding a networked simulation.
    flower_parameters = ndarrays_to_parameters(parameters)
    tensor_bytes = sum(len(tensor) for tensor in flower_parameters.tensors)
    return int(tensor_bytes)


def train_local_model(
    global_parameters: list[np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    hidden_size: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    algorithm: str,
    fedprox_mu: float,
) -> list[np.ndarray]:
    model = MLP(X.shape[1], n_classes, hidden_size)
    set_model_parameters(model, global_parameters)
    global_tensors = [torch.tensor(parameter, dtype=torch.float32) for parameter in global_parameters]

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _epoch in range(local_epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_X), batch_y)
            if algorithm == "FedProx":
                proximal = torch.tensor(0.0)
                for parameter, global_parameter in zip(model.parameters(), global_tensors):
                    proximal = proximal + torch.sum((parameter - global_parameter) ** 2)
                loss = loss + (fedprox_mu / 2.0) * proximal
            loss.backward()
            optimizer.step()

    return model_parameters(model)


def weighted_average(updates: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    total_examples = sum(num_examples for _parameters, num_examples in updates)
    averaged = []
    for layer_idx in range(len(updates[0][0])):
        layer = sum(parameters[layer_idx] * (num_examples / total_examples) for parameters, num_examples in updates)
        averaged.append(layer.astype(np.float32, copy=False))
    return averaged


def client_partitions(
    y: np.ndarray,
    row_index: pd.DataFrame | None,
    train_idx: np.ndarray,
    min_clients: int,
    max_clients: int,
    random_state: int,
) -> tuple[list[np.ndarray], pd.Series | None]:
    rng = np.random.default_rng(random_state)
    if row_index is not None and len(row_index) == len(y) and "campaign" in row_index.columns:
        campaigns = row_index.iloc[train_idx]["campaign"].fillna(MISSING).astype(str)
        partitions = [train_idx[campaigns.values == campaign] for campaign in sorted(campaigns.unique())]
        partitions = [partition for partition in partitions if len(partition) > 0]
        if len(partitions) >= min_clients:
            return (partitions[:max_clients] if max_clients and max_clients > 0 else partitions), campaigns

    shuffled = train_idx.copy()
    rng.shuffle(shuffled)
    fallback_max_clients = max_clients if max_clients and max_clients > 0 else min(5, len(shuffled))
    n_clients = min(fallback_max_clients, max(min_clients, min(5, len(shuffled))))
    partitions = [partition for partition in np.array_split(shuffled, n_clients) if len(partition) > 0]
    return partitions, None


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.argmax(logits, dim=1).cpu().numpy()


def group_balanced_accuracy(y_true, y_pred, groups: pd.Series | None) -> dict[str, float] | None:
    if groups is None:
        return None
    values: dict[str, float] = {}
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": groups.astype(str).values})
    for group, part in frame.groupby("group"):
        if len(part["y_true"].unique()) < 2:
            values[str(group)] = None
            continue
        values[str(group)] = float(balanced_accuracy_score(part["y_true"], part["y_pred"]))
    return values


def metric_variance(values: dict[str, float] | None) -> float | None:
    if not values:
        return None
    numeric = [value for value in values.values() if value is not None and not math.isnan(value)]
    if len(numeric) < 2:
        return None
    return float(np.var(numeric))


def fairness_gap(values: dict[str, float] | None) -> float | None:
    if not values:
        return None
    numeric = [value for value in values.values() if value is not None and not math.isnan(value)]
    if len(numeric) < 2:
        return None
    return float(max(numeric) - min(numeric))


def extract_groups(feature_dataset: FeatureDataset, test_idx: np.ndarray):
    if feature_dataset.row_index is None or len(feature_dataset.row_index) != len(feature_dataset.target):
        return None, None
    row_index = feature_dataset.row_index.iloc[test_idx].reset_index(drop=True)
    campaign = row_index["campaign"] if "campaign" in row_index.columns else None
    traffic_source = row_index["traffic_source"] if "traffic_source" in row_index.columns else None
    return campaign, traffic_source


def evaluate_federated(
    feature_dataset: FeatureDataset,
    algorithm: str,
    test_size: float,
    random_state: int,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    min_clients: int,
    max_clients: int,
    fedprox_mu: float,
    train_idx_override: np.ndarray | None = None,
    test_idx_override: np.ndarray | None = None,
) -> dict:
    y_text = feature_dataset.target["attack_type"].fillna(MISSING).astype(str).to_numpy()
    if len(np.unique(y_text)) < 2:
        raise ValueError("target has fewer than two classes")

    X = dense_array(feature_dataset.X)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    if train_idx_override is None or test_idx_override is None:
        train_idx, test_idx = stratified_split_indices(y, test_size, random_state)
    else:
        train_idx = np.asarray(train_idx_override, dtype=int)
        test_idx = np.asarray(test_idx_override, dtype=int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    X_scaled = X.copy()
    X_scaled[train_idx] = X_train
    X_scaled[test_idx] = X_test

    partitions, _campaign_groups_train = client_partitions(
        y,
        feature_dataset.row_index,
        train_idx,
        min_clients,
        max_clients,
        random_state,
    )
    if len(partitions) < 1:
        raise ValueError("no client partitions were created")

    model = MLP(X.shape[1], len(label_encoder.classes_), hidden_size)
    global_parameters = model_parameters(model)
    per_client_final_scores: dict[str, float] = {}
    communication = 0
    started = time.perf_counter()

    for _round in range(rounds):
        round_updates = []
        serialized_global_bytes = parameter_bytes(global_parameters)
        communication += serialized_global_bytes * len(partitions)
        for client_id, partition in enumerate(partitions):
            local_parameters = train_local_model(
                global_parameters,
                X_scaled[partition],
                y[partition],
                len(label_encoder.classes_),
                hidden_size,
                local_epochs,
                batch_size,
                learning_rate,
                algorithm,
                fedprox_mu,
            )
            communication += parameter_bytes(local_parameters)
            round_updates.append((local_parameters, len(partition)))

            local_model = MLP(X.shape[1], len(label_encoder.classes_), hidden_size)
            set_model_parameters(local_model, local_parameters)
            local_pred = predict(local_model, X_scaled[partition])
            per_client_final_scores[f"client_{client_id}"] = float(
                balanced_accuracy_score(y[partition], local_pred)
            )
        global_parameters = weighted_average(round_updates)

    set_model_parameters(model, global_parameters)
    y_pred = predict(model, X_test)
    elapsed = time.perf_counter() - started

    labels = np.arange(len(label_encoder.classes_))
    recall_values = recall_score(y[test_idx], y_pred, labels=labels, average=None, zero_division=0)
    recall_by_class = {
        str(class_name): float(value)
        for class_name, value in zip(label_encoder.classes_, recall_values)
    }

    campaign_groups, traffic_groups = extract_groups(feature_dataset, test_idx)
    campaign_performance = group_balanced_accuracy(y[test_idx], y_pred, campaign_groups)
    traffic_performance = group_balanced_accuracy(y[test_idx], y_pred, traffic_groups)

    return {
        "feature_stage": feature_dataset.feature_stage,
        "feature_selection": feature_dataset.feature_selection,
        "feature_approach": feature_dataset.feature_approach,
        "classifier": f"Flower MLP {algorithm}",
        "federated_algorithm": algorithm,
        "n_samples": int(len(y)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_features": int(feature_dataset.X.shape[1]),
        "n_clients": int(len(partitions)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], y_pred)),
        "macro_f1": float(f1_score(y[test_idx], y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y[test_idx], y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y[test_idx], y_pred)),
        "recall_by_class": recall_by_class,
        "communication_cost": int(communication),
        "number_of_rounds": int(rounds),
        "client_variance": metric_variance(per_client_final_scores),
        "fairness_between_campaigns": fairness_gap(campaign_performance),
        "performance_by_traffic_source": traffic_performance,
        "fit_predict_seconds": float(elapsed),
        "status": "ok",
        "error": None,
    }


def evaluate_campaign_federated(
    feature_dataset: FeatureDataset,
    campaign_id: str,
    algorithm: str,
    random_state: int,
    k_folds: int,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    fedprox_mu: float,
) -> dict:
    y_text = feature_dataset.target["attack_type"].fillna(MISSING).astype(str).to_numpy()
    if len(np.unique(y_text)) < 2:
        raise ValueError("campaign target has fewer than two classes")

    y = LabelEncoder().fit_transform(y_text)
    splits, split_strategy, effective_folds = campaign_kfold_splits(y, k_folds, random_state)
    fold_results = []
    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
        result = evaluate_federated(
            feature_dataset,
            algorithm,
            0.0,
            random_state + fold_number,
            rounds,
            local_epochs,
            batch_size,
            learning_rate,
            hidden_size,
            1,
            1,
            fedprox_mu,
            train_idx_override=train_idx,
            test_idx_override=test_idx,
        )
        result["fold"] = fold_number
        fold_results.append(result)

    result = aggregate_campaign_fold_results(
        fold_results,
        feature_dataset,
        campaign_id,
        f"Flower MLP {algorithm}",
        federated_algorithm=algorithm,
    )
    result["cv_strategy"] = split_strategy
    result["k_folds"] = int(effective_folds)
    result["n_clients"] = 1
    return result


def failed_result(feature_dataset: FeatureDataset, algorithm: str, error: Exception) -> dict:
    return {
        "feature_stage": feature_dataset.feature_stage,
        "feature_selection": feature_dataset.feature_selection,
        "feature_approach": feature_dataset.feature_approach,
        "classifier": f"Flower MLP {algorithm}",
        "federated_algorithm": algorithm,
        "evaluation_scope": "campaign",
        "campaign": feature_dataset.row_index["campaign"].iloc[0] if feature_dataset.row_index is not None and "campaign" in feature_dataset.row_index.columns and len(feature_dataset.row_index) else None,
        "n_samples": int(len(feature_dataset.target)),
        "n_train": None,
        "n_test": None,
        "n_features": int(feature_dataset.X.shape[1]),
        "n_clients": None,
        "balanced_accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
        "mcc": None,
        "recall_by_class": None,
        "communication_cost": None,
        "number_of_rounds": None,
        "client_variance": None,
        "fairness_between_campaigns": None,
        "performance_by_traffic_source": None,
        "fit_predict_seconds": None,
        "status": "error",
        "error": str(error),
    }


def normalize_for_csv(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = result.copy()
        for key in ("recall_by_class", "performance_by_traffic_source", "campaign_performance", "traffic_source_performance", "fold_statuses"):
            if row.get(key) is not None:
                row[key] = json.dumps(row[key], ensure_ascii=False, separators=(",", ":"))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    dataset = read_env_value(project_root / args.env_file, "DATASET")
    if dataset not in VALID_DATASETS:
        valid = ", ".join(sorted(VALID_DATASETS))
        raise ValueError(f"Invalid DATASET={dataset!r}. Expected one of: {valid}")

    feature_datasets = discover_feature_datasets(
        project_root / args.extracted_root,
        project_root / args.selected_root,
        project_root / args.raw_root,
        dataset,
    )
    if not feature_datasets:
        raise ValueError("No extracted or selected feature datasets were found.")

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    output_path = project_root / args.output_root / dataset
    run_config = base_config(args)
    run_config["dataset"] = dataset
    current_config_hash = config_hash(run_config)
    results: list[dict] = load_existing_results(output_path)
    done = completed_keys(results, current_config_hash)
    for feature_dataset in feature_datasets:
        feature_dataset = cap_rows_by_campaign(feature_dataset, args.max_rows, args.random_state)
        campaigns = campaign_indices(feature_dataset)
        if not campaigns:
            print(f"{feature_dataset.feature_approach}/{feature_dataset.feature_selection or 'none'}: no campaign index; skipped")
            continue
        for algorithm in FEDERATED_ALGORITHMS:
            for campaign_id, indices in campaigns:
                campaign_dataset = subset_feature_dataset(feature_dataset, indices, campaign_id)
                pending_key = {
                    "config_hash": current_config_hash,
                    "feature_stage": campaign_dataset.feature_stage,
                    "feature_selection": campaign_dataset.feature_selection,
                    "feature_approach": campaign_dataset.feature_approach,
                    "classifier": f"Flower MLP {algorithm}",
                    "federated_algorithm": algorithm,
                    "evaluation_scope": "campaign",
                    "campaign": campaign_id,
                }
                if result_key(pending_key) in done:
                    selector = campaign_dataset.feature_selection or "none"
                    print(f"{campaign_dataset.feature_approach}/{selector}/{campaign_id} + {algorithm}: skipped")
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = evaluate_campaign_federated(
                            campaign_dataset,
                            campaign_id,
                            algorithm,
                            args.random_state,
                            args.k_folds,
                            args.rounds,
                            args.local_epochs,
                            args.batch_size,
                            args.learning_rate,
                            args.hidden_size,
                            args.fedprox_mu,
                        )
                except Exception as exc:
                    result = failed_result(campaign_dataset, algorithm, exc)
                    result["campaign"] = campaign_id
                    result["campaign_id"] = campaign_id
                    result["k_folds"] = args.k_folds
                result = add_resume_metadata(result, run_config, pending_key)
                results.append(result)
                done.add(result_key(result))
                save_results(output_path, results, normalize_for_csv)
                selector = campaign_dataset.feature_selection or "none"
                print(f"{campaign_dataset.feature_approach}/{selector}/{campaign_id} + {algorithm}: {result['status']}")

    save_results(output_path, results, normalize_for_csv)
    print(f"Saved federated MLP evaluation results to {output_path}")


if __name__ == "__main__":
    main()
