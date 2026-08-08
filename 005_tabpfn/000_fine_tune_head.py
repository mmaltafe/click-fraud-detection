#!/usr/bin/env python3
"""
Fine-tune only a final trainable head on top of frozen TabPFN embeddings.

This experiment uses the best standard TabPFN configuration found by:

    004_grid_search/000_tabpfn_grid_search.py

The TabPFN checkpoint is not updated. For each campaign and k-fold split, the
script fits the local TabPFN context, extracts frozen embeddings, and trains a
small PyTorch head with cross-entropy loss.

Inputs:
    results/grid_search/tabpfn/{DATASET}/results.csv
    data/extracted_features/{approach}/{DATASET}
    data/selected_features/{selector}/{approach}/{DATASET}

Outputs:
    results/tabpfn_fine_tuning/{DATASET}/results.csv
    results/tabpfn_fine_tuning/{DATASET}/results.json
    results/tabpfn_fine_tuning/{DATASET}/best_tabpfn_config.json
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import warnings
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_EVALUATION_DIR = PROJECT_ROOT / "003_machine_learning_evaluation"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(ML_EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(ML_EVALUATION_DIR))

from utils._env import VALID_DATASETS, MISSING, PRIMARY_METRIC, read_env_value  # noqa: E402
from utils._campaigns import (  # noqa: E402
    aggregate_campaign_fold_results,
    campaign_indices,
    campaign_kfold_splits,
    cap_rows_by_campaign,
    subset_feature_dataset,
)
from utils._resume import add_resume_metadata, base_config, config_hash, load_existing_results, save_results  # noqa: E402


CLASSIFIER_NAME = "TabPFN-Frozen-Head"


# Pipeline configuration constants. Edit these values to change this script.
CONFIG_ENV_FILE = ".env"
CONFIG_GRID_RESULTS_ROOT = "results/grid_search/tabpfn"
CONFIG_EXTRACTED_ROOT = "data/extracted_features"
CONFIG_SELECTED_ROOT = "data/selected_features"
CONFIG_RAW_ROOT = "data/raw"
CONFIG_OUTPUT_ROOT = "results/tabpfn_fine_tuning"
CONFIG_MODEL_CACHE = "models/tabpfn"
CONFIG_MODEL_PATH = None
CONFIG_ALLOW_BROWSER_LOGIN = False
CONFIG_DATASET = None
CONFIG_RANDOM_STATE = 42
CONFIG_K_FOLDS = 5
CONFIG_MAX_ROWS = 0
CONFIG_MAX_CAMPAIGNS = 0
CONFIG_MAX_DENSE_CELLS = 10_000_000
CONFIG_ALLOW_CPU_LARGE_DATASET = False
CONFIG_HEAD_EPOCHS_GRID = "30,80"
CONFIG_HEAD_BATCH_SIZE_GRID = "256"
CONFIG_HEAD_LEARNING_RATE_GRID = "0.001,0.0003"
CONFIG_HEAD_WEIGHT_DECAY_GRID = "0.0001"
CONFIG_HEAD_HIDDEN_SIZE_GRID = "0,256"
CONFIG_HEAD_DROPOUT_GRID = "0.0,0.15"
CONFIG_USE_CLASS_WEIGHTS_GRID = "true,false"
CONFIG_MAX_CONFIGS = 0
CONFIG_PLAN_ONLY = False
CONFIG_RETRY_ERRORS = True


@dataclass
class BestTabPFNConfig:
    feature_stage: str
    feature_selection: str | None
    feature_approach: str
    grid_params: dict[str, Any]
    source_balanced_accuracy_mean: float
    source_balanced_accuracy_std: float | None
    source_macro_f1_mean: float | None
    source_runs: int


def parse_args() -> Namespace:
    """Return script configuration from global constants; command-line args are intentionally ignored."""
    return Namespace(
        env_file=CONFIG_ENV_FILE,
        grid_results_root=CONFIG_GRID_RESULTS_ROOT,
        extracted_root=CONFIG_EXTRACTED_ROOT,
        selected_root=CONFIG_SELECTED_ROOT,
        raw_root=CONFIG_RAW_ROOT,
        output_root=CONFIG_OUTPUT_ROOT,
        model_cache=CONFIG_MODEL_CACHE,
        model_path=CONFIG_MODEL_PATH,
        allow_browser_login=CONFIG_ALLOW_BROWSER_LOGIN,
        dataset=CONFIG_DATASET,
        random_state=CONFIG_RANDOM_STATE,
        k_folds=CONFIG_K_FOLDS,
        max_rows=CONFIG_MAX_ROWS,
        max_campaigns=CONFIG_MAX_CAMPAIGNS,
        max_dense_cells=CONFIG_MAX_DENSE_CELLS,
        allow_cpu_large_dataset=CONFIG_ALLOW_CPU_LARGE_DATASET,
        head_epochs_grid=CONFIG_HEAD_EPOCHS_GRID,
        head_batch_size_grid=CONFIG_HEAD_BATCH_SIZE_GRID,
        head_learning_rate_grid=CONFIG_HEAD_LEARNING_RATE_GRID,
        head_weight_decay_grid=CONFIG_HEAD_WEIGHT_DECAY_GRID,
        head_hidden_size_grid=CONFIG_HEAD_HIDDEN_SIZE_GRID,
        head_dropout_grid=CONFIG_HEAD_DROPOUT_GRID,
        use_class_weights_grid=CONFIG_USE_CLASS_WEIGHTS_GRID,
        max_configs=CONFIG_MAX_CONFIGS,
        plan_only=CONFIG_PLAN_ONLY,
        retry_errors=CONFIG_RETRY_ERRORS,
    )


def load_tabpfn_module():
    path = ML_EVALUATION_DIR / "002_tabpfn.py"
    spec = importlib.util.spec_from_file_location("ml_eval_tabpfn_fine_tune", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def parse_int_grid(value: str) -> list[int]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("integer grid cannot be empty")
    return [int(item) for item in values]


def parse_float_grid(value: str) -> list[float]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("float grid cannot be empty")
    return [float(item) for item in values]


def parse_bool_grid(value: str) -> list[bool]:
    output = []
    for item in [part.strip().lower() for part in value.split(",") if part.strip()]:
        if item in {"1", "true", "yes", "y"}:
            output.append(True)
        elif item in {"0", "false", "no", "n"}:
            output.append(False)
        else:
            raise ValueError(f"Invalid boolean grid value: {item}")
    if not output:
        raise ValueError("boolean grid cannot be empty")
    return output


def head_grid_configs(args: Namespace) -> list[dict[str, Any]]:
    from itertools import product

    configs = []
    for epochs, batch_size, learning_rate, weight_decay, hidden_size, dropout, use_weights in product(
        parse_int_grid(args.head_epochs_grid),
        parse_int_grid(args.head_batch_size_grid),
        parse_float_grid(args.head_learning_rate_grid),
        parse_float_grid(args.head_weight_decay_grid),
        parse_int_grid(args.head_hidden_size_grid),
        parse_float_grid(args.head_dropout_grid),
        parse_bool_grid(args.use_class_weights_grid),
    ):
        if hidden_size <= 0 and dropout > 0:
            continue
        configs.append(
            {
                "head_epochs": epochs,
                "head_batch_size": batch_size,
                "head_learning_rate": learning_rate,
                "head_weight_decay": weight_decay,
                "head_hidden_size": hidden_size,
                "head_dropout": dropout,
                "use_class_weights": use_weights,
            }
        )
    if args.max_configs > 0 and len(configs) > args.max_configs:
        rng = np.random.default_rng(args.random_state)
        chosen = np.sort(rng.choice(np.arange(len(configs)), size=args.max_configs, replace=False))
        configs = [configs[int(index)] for index in chosen]
    return configs


def args_with_head_config(args: Namespace, head_config: dict[str, Any]) -> Namespace:
    payload = vars(args).copy()
    payload.update(head_config)
    return Namespace(**payload)


def fine_tune_grid_key(dataset: str, campaign_id: str, best: BestTabPFNConfig, head_config: dict[str, Any]) -> str:
    return stable_json(
        {
            "dataset": dataset,
            "campaign": campaign_id,
            "classifier": CLASSIFIER_NAME,
            "best_tabpfn_config": best_config_to_dict(best),
            "head_config": head_config,
        }
    )


def completed_fine_tune_keys(results: list[dict], retry_errors: bool) -> set[str]:
    done = set()
    for result in results:
        if retry_errors and result.get("status") not in {"ok", "skipped"}:
            continue
        key = result.get("fine_tune_grid_key")
        if key:
            done.add(str(key))
    return done


def normalize_metric(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_grid_params(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or pd.isna(value):
        return {}
    return json.loads(str(value))


def load_best_tabpfn_config(grid_results_root: Path, dataset: str) -> BestTabPFNConfig:
    path = grid_results_root / dataset / "results.csv"
    if not path.exists():
        raise FileNotFoundError(f"TabPFN grid-search results not found: {path}")

    rows = pd.read_csv(path)
    rows = rows[rows["status"].astype(str).eq("ok")].copy()
    if rows.empty:
        raise ValueError(f"No successful TabPFN grid-search rows were found in {path}")

    rows[PRIMARY_METRIC] = pd.to_numeric(rows[PRIMARY_METRIC], errors="coerce")
    rows["macro_f1"] = pd.to_numeric(rows.get("macro_f1"), errors="coerce")
    rows = rows[rows[PRIMARY_METRIC].notna()].copy()
    if rows.empty:
        raise ValueError(f"No valid {PRIMARY_METRIC} values were found in {path}")

    rows["grid_params_dict"] = rows["grid_params"].apply(parse_grid_params)
    rows["grid_params_key"] = rows["grid_params_dict"].apply(stable_json)
    group_columns = ["feature_stage", "feature_selection", "feature_approach", "grid_params_key"]
    grouped = (
        rows.groupby(group_columns, dropna=False)
        .agg(
            balanced_accuracy_mean=(PRIMARY_METRIC, "mean"),
            balanced_accuracy_std=(PRIMARY_METRIC, "std"),
            macro_f1_mean=("macro_f1", "mean"),
            source_runs=(PRIMARY_METRIC, "count"),
        )
        .reset_index()
        .sort_values(["balanced_accuracy_mean", "macro_f1_mean", "source_runs"], ascending=[False, False, False])
    )
    best = grouped.iloc[0]
    sample = rows[rows["grid_params_key"].eq(best["grid_params_key"])].iloc[0]
    feature_selection = best["feature_selection"]
    if feature_selection in {None, "none"} or pd.isna(feature_selection):
        feature_selection = None
    return BestTabPFNConfig(
        feature_stage=str(best["feature_stage"]),
        feature_selection=None if feature_selection is None else str(feature_selection),
        feature_approach=str(best["feature_approach"]),
        grid_params=dict(sample["grid_params_dict"]),
        source_balanced_accuracy_mean=float(best["balanced_accuracy_mean"]),
        source_balanced_accuracy_std=normalize_metric(best["balanced_accuracy_std"]),
        source_macro_f1_mean=normalize_metric(best["macro_f1_mean"]),
        source_runs=int(best["source_runs"]),
    )


def best_config_to_dict(best: BestTabPFNConfig) -> dict[str, Any]:
    return {
        "feature_stage": best.feature_stage,
        "feature_selection": best.feature_selection,
        "feature_approach": best.feature_approach,
        "grid_params": best.grid_params,
        "source_balanced_accuracy_mean": best.source_balanced_accuracy_mean,
        "source_balanced_accuracy_std": best.source_balanced_accuracy_std,
        "source_macro_f1_mean": best.source_macro_f1_mean,
        "source_runs": best.source_runs,
    }


def load_best_feature_dataset(module, project_root: Path, args: Namespace, dataset: str, best: BestTabPFNConfig):
    if best.feature_stage == "extracted_features":
        loaded = module.load_extracted_dataset(project_root / args.extracted_root, dataset, best.feature_approach)
    elif best.feature_stage == "selected_features":
        if not best.feature_selection:
            raise ValueError("selected_features best config is missing feature_selection")
        loaded = module.load_selected_dataset(project_root / args.selected_root, dataset, best.feature_selection, best.feature_approach)
    else:
        raise ValueError(f"Unknown feature_stage in best config: {best.feature_stage}")

    if loaded is None:
        raise FileNotFoundError(f"Could not load best feature dataset: {best_config_to_dict(best)}")
    return module.ensure_campaign_row_index(loaded, dataset, project_root / args.raw_root)


def configure_tabpfn_environment(project_root: Path, args: Namespace) -> None:
    model_cache = project_root / args.model_cache
    model_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TABPFN_MODEL_CACHE_DIR"] = str(model_cache)
    if args.allow_browser_login:
        os.environ.pop("TABPFN_NO_BROWSER", None)
    else:
        os.environ.setdefault("TABPFN_NO_BROWSER", "1")

    env_file = project_root / args.env_file
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("TABPFN_TOKEN="):
                os.environ.setdefault("TABPFN_TOKEN", line.split("=", 1)[1].strip().strip('"').strip("'"))


def resolve_model_path(project_root: Path, args: Namespace) -> str:
    if args.model_path:
        model_path = Path(args.model_path)
        return str(model_path if model_path.is_absolute() else project_root / model_path)
    checkpoints = sorted((project_root / args.model_cache).glob("*.ckpt"))
    return str(checkpoints[0]) if checkpoints else "auto"


def effective_cpu_device(device: str | None) -> bool:
    if device and str(device).lower() == "cpu":
        return True
    if device and str(device).lower() not in {"auto", ""}:
        return False
    try:
        import torch

        return not bool(torch.cuda.is_available())
    except Exception:
        return True


def downsample_indices(indices: np.ndarray, y: np.ndarray, max_rows: int, random_state: int) -> np.ndarray:
    if max_rows <= 0 or len(indices) <= max_rows:
        return indices
    from sklearn.model_selection import train_test_split

    y_subset = y[indices]
    class_counts = pd.Series(y_subset).value_counts()
    stratify = y_subset if len(class_counts) > 1 and class_counts.min() >= 2 else None
    _, sampled_positions = train_test_split(
        np.arange(len(indices)),
        test_size=max_rows,
        random_state=random_state,
        stratify=stratify,
    )
    return np.sort(indices[sampled_positions])


def maybe_dense(X, max_dense_cells: int):
    if not sparse.issparse(X):
        return np.asarray(X, dtype=np.float32)
    cells = X.shape[0] * X.shape[1]
    if max_dense_cells > 0 and cells > max_dense_cells:
        raise MemoryError(f"Refusing to densify sparse matrix with {cells} cells")
    return X.toarray().astype(np.float32, copy=False)


def ensure_2d_embeddings(embeddings: np.ndarray, expected_rows: int) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim > 1 and array.shape[0] != expected_rows:
        matching_axes = [axis for axis, size in enumerate(array.shape) if size == expected_rows]
        if matching_axes:
            array = np.moveaxis(array, matching_axes[0], 0)
        else:
            raise ValueError(
                "Could not align TabPFN embeddings with input rows: "
                f"embedding_shape={array.shape}, expected_rows={expected_rows}"
            )
    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    if array.shape[0] != expected_rows:
        raise ValueError(
            "TabPFN embeddings have incompatible row count after reshape: "
            f"embedding_shape={array.shape}, expected_rows={expected_rows}"
        )
    return array


def class_weight_tensor(y: np.ndarray, n_classes: int, device):
    import torch

    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    args: Namespace,
    random_state: int,
) -> tuple[Any, float]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = int(X_train.shape[1])
    if args.head_hidden_size and args.head_hidden_size > 0:
        model = nn.Sequential(
            nn.Linear(input_dim, int(args.head_hidden_size)),
            nn.GELU(),
            nn.Dropout(float(args.head_dropout)),
            nn.Linear(int(args.head_hidden_size), n_classes),
        )
    else:
        model = nn.Linear(input_dim, n_classes)
    model.to(device)

    features = torch.tensor(X_train, dtype=torch.float32)
    labels = torch.tensor(y_train, dtype=torch.long)
    generator = torch.Generator().manual_seed(random_state)
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=max(1, int(args.head_batch_size)),
        shuffle=True,
        generator=generator,
    )
    weights = class_weight_tensor(y_train, n_classes, device) if args.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.head_learning_rate),
        weight_decay=float(args.head_weight_decay),
    )

    final_loss = None
    for _epoch in range(int(args.head_epochs)):
        model.train()
        losses = []
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        final_loss = float(np.mean(losses)) if losses else None
    return model, final_loss


def predict_head(model, X_test: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32, device=device))
        return torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)


def extract_tabpfn_embeddings(classifier, X_train, y_train, X_test) -> tuple[np.ndarray, np.ndarray, str]:
    classifier.fit(X_train, y_train)
    try:
        train_embeddings = classifier.get_embeddings(X_train, data_source="train")
        test_embeddings = classifier.get_embeddings(X_test, data_source="test")
        return (
            ensure_2d_embeddings(train_embeddings, len(X_train)),
            ensure_2d_embeddings(test_embeddings, len(X_test)),
            "tabpfn_embeddings",
        )
    except Exception:
        train_proba = classifier.predict_proba(X_train)
        test_proba = classifier.predict_proba(X_test)
        return (
            ensure_2d_embeddings(train_proba, len(X_train)),
            ensure_2d_embeddings(test_proba, len(X_test)),
            "predict_proba_fallback",
        )


def communication_cost(X) -> int:
    if sparse.issparse(X):
        return int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)
    return int(np.asarray(X).nbytes)


def prepare_campaign_embedding_folds(feature_dataset, campaign_id: str, factory, args: Namespace, best: BestTabPFNConfig):
    y_text = feature_dataset.target["attack_type"].fillna(MISSING).astype(str).to_numpy()
    if len(np.unique(y_text)) < 2:
        raise ValueError("campaign target has fewer than two classes")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    splits, split_strategy, effective_folds = campaign_kfold_splits(y, args.k_folds, args.random_state)
    params = best.grid_params
    cpu_limited_run = effective_cpu_device(params.get("device")) and not args.allow_cpu_large_dataset
    n_classes = int(len(label_encoder.classes_))
    embedding_folds = []

    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
        started = time.perf_counter()
        original_train_rows = int(len(train_idx))
        original_test_rows = int(len(test_idx))
        if cpu_limited_run:
            train_idx = downsample_indices(train_idx, y, int(params.get("max_cpu_train_rows") or 0), args.random_state + fold_number)
            test_idx = downsample_indices(test_idx, y, int(params.get("max_cpu_test_rows") or 0), args.random_state + 10_000 + fold_number)

        X_train = maybe_dense(feature_dataset.X[train_idx], args.max_dense_cells)
        X_test = maybe_dense(feature_dataset.X[test_idx], args.max_dense_cells)
        y_train = y[train_idx]
        y_test = y[test_idx]

        classifier = factory()
        fit_started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_embeddings, test_embeddings, embedding_source = extract_tabpfn_embeddings(classifier, X_train, y_train, X_test)

        scaler = StandardScaler()
        train_embeddings = scaler.fit_transform(train_embeddings).astype(np.float32, copy=False)
        test_embeddings = scaler.transform(test_embeddings).astype(np.float32, copy=False)
        embedding_elapsed = time.perf_counter() - started
        embedding_folds.append(
            {
                "fold_number": fold_number,
                "train_embeddings": train_embeddings,
                "test_embeddings": test_embeddings,
                "y_train": y_train,
                "y_test": y_test,
                "label_classes": label_encoder.classes_.tolist(),
                "original_train_rows": original_train_rows,
                "original_test_rows": original_test_rows,
                "cpu_train_sampled": bool(len(train_idx) < original_train_rows),
                "cpu_test_sampled": bool(len(test_idx) < original_test_rows),
                "embedding_source": embedding_source,
                "embedding_dim": int(train_embeddings.shape[1]),
                "embedding_seconds": float(embedding_elapsed),
                "n_classes": n_classes,
            }
        )
    return embedding_folds, split_strategy, effective_folds


def evaluate_campaign_head_from_embeddings(
    feature_dataset,
    campaign_id: str,
    embedding_folds: list[dict],
    split_strategy: str,
    effective_folds: int,
    args: Namespace,
    best: BestTabPFNConfig,
) -> dict:
    fold_results = []
    for fold_data in embedding_folds:
        fit_started = time.perf_counter()
        fold_number = int(fold_data["fold_number"])
        train_embeddings = fold_data["train_embeddings"]
        test_embeddings = fold_data["test_embeddings"]
        y_train = fold_data["y_train"]
        y_test = fold_data["y_test"]
        n_classes = int(fold_data["n_classes"])
        head, final_loss = train_head(train_embeddings, y_train, n_classes, args, args.random_state + fold_number)
        y_pred = predict_head(head, test_embeddings)
        fit_elapsed = time.perf_counter() - fit_started
        elapsed = fit_elapsed + float(fold_data.get("embedding_seconds") or 0.0)

        labels = np.arange(n_classes)
        recall_values = recall_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
        fold_results.append(
            {
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "n_train_original": int(fold_data["original_train_rows"]),
                "n_test_original": int(fold_data["original_test_rows"]),
                "cpu_train_sampled": bool(fold_data["cpu_train_sampled"]),
                "cpu_test_sampled": bool(fold_data["cpu_test_sampled"]),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
                "mcc": float(matthews_corrcoef(y_test, y_pred)),
                "recall_by_class": {
                    str(class_name): float(value)
                    for class_name, value in zip(fold_data["label_classes"], recall_values)
                },
                "communication_cost": communication_cost(feature_dataset.X),
                "number_of_rounds": None,
                "client_variance": None,
                "fairness_between_campaigns": None,
                "performance_by_traffic_source": None,
                "fit_predict_seconds": float(fit_elapsed),
                "elapsed_seconds": float(elapsed),
                "status": "ok",
                "error": None,
                "fold": fold_number,
                "embedding_source": fold_data["embedding_source"],
                "embedding_dim": int(fold_data["embedding_dim"]),
                "embedding_seconds": float(fold_data.get("embedding_seconds") or 0.0),
                "head_final_loss": final_loss,
            }
        )

    result = aggregate_campaign_fold_results(
        fold_results,
        feature_dataset,
        campaign_id,
        CLASSIFIER_NAME,
    )
    result["cv_strategy"] = split_strategy
    result["k_folds"] = int(effective_folds)
    result["tabpfn_frozen"] = True
    result["fine_tuned_component"] = "final_head_only"
    result["head_epochs"] = int(args.head_epochs)
    result["head_batch_size"] = int(args.head_batch_size)
    result["head_learning_rate"] = float(args.head_learning_rate)
    result["head_weight_decay"] = float(args.head_weight_decay)
    result["head_hidden_size"] = int(args.head_hidden_size)
    result["head_dropout"] = float(args.head_dropout)
    result["use_class_weights"] = bool(args.use_class_weights)
    result["head_config"] = {
        "head_epochs": int(args.head_epochs),
        "head_batch_size": int(args.head_batch_size),
        "head_learning_rate": float(args.head_learning_rate),
        "head_weight_decay": float(args.head_weight_decay),
        "head_hidden_size": int(args.head_hidden_size),
        "head_dropout": float(args.head_dropout),
        "use_class_weights": bool(args.use_class_weights),
    }
    result["source_grid_params"] = best.grid_params
    return result


def failed_result(feature_dataset, campaign_id: str, error: Exception, best: BestTabPFNConfig, head_config: dict[str, Any]) -> dict:
    return {
        "feature_stage": feature_dataset.feature_stage,
        "feature_selection": feature_dataset.feature_selection,
        "feature_approach": feature_dataset.feature_approach,
        "classifier": CLASSIFIER_NAME,
        "dataset": None,
        "evaluation_scope": "campaign",
        "campaign": campaign_id,
        "campaign_id": campaign_id,
        "n_samples": int(len(feature_dataset.target)),
        "n_features": int(feature_dataset.X.shape[1]),
        "balanced_accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
        "mcc": None,
        "status": "error",
        "error": str(error),
        "tabpfn_frozen": True,
        "fine_tuned_component": "final_head_only",
        "head_config": head_config,
        **head_config,
        "source_grid_params": best.grid_params,
    }


def normalize_for_csv(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = result.copy()
        for key, value in list(row.items()):
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_results(results: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame([row for row in results if row.get("status") == "ok"])
    if frame.empty or PRIMARY_METRIC not in frame.columns:
        return pd.DataFrame()
    metrics = [column for column in (PRIMARY_METRIC, "macro_f1", "weighted_f1", "mcc", "fit_predict_seconds") if column in frame.columns]
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    groups = [
        "head_epochs",
        "head_batch_size",
        "head_learning_rate",
        "head_weight_decay",
        "head_hidden_size",
        "head_dropout",
        "use_class_weights",
    ]
    summary = frame.groupby(groups, dropna=False)[metrics].agg(["mean", "std", "max", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part) if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    return summary.sort_values([f"{PRIMARY_METRIC}_mean", "macro_f1_mean"], ascending=[False, False])


def save_fine_tune_results(output_path: Path, results: list[dict]) -> None:
    save_results(output_path, results, normalize_for_csv)
    summary = summarize_results(results)
    if not summary.empty:
        summary.to_csv(output_path / "summary.csv", index=False)


def remove_retryable_errors(results: list[dict], retry_errors: bool) -> list[dict]:
    if not retry_errors:
        return results
    return [
        result
        for result in results
        if not (
            result.get("classifier") == CLASSIFIER_NAME
            and result.get("status") in {"error", "failed", "skipped"}
        )
    ]


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    dataset = args.dataset or read_env_value(project_root / args.env_file, "DATASET")
    if dataset not in VALID_DATASETS:
        raise ValueError(f"Invalid DATASET={dataset!r}. Expected one of: {sorted(VALID_DATASETS)}")

    module = load_tabpfn_module()
    best = load_best_tabpfn_config(project_root / args.grid_results_root, dataset)
    feature_dataset = load_best_feature_dataset(module, project_root, args, dataset, best)
    feature_dataset = cap_rows_by_campaign(feature_dataset, args.max_rows, args.random_state)

    configure_tabpfn_environment(project_root, args)
    model_path = resolve_model_path(project_root, args)
    params = best.grid_params

    output_path = project_root / args.output_root / dataset
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "best_tabpfn_config.json").write_text(
        json.dumps(best_config_to_dict(best), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    run_config = base_config(args, exclude={"retry_errors", "allow_browser_login", "model_path"})
    run_config["dataset"] = dataset
    run_config["best_tabpfn_config"] = best_config_to_dict(best)
    current_config_hash = config_hash(run_config)
    results = remove_retryable_errors(load_existing_results(output_path), args.retry_errors)
    done = completed_fine_tune_keys(results, args.retry_errors)

    campaigns = campaign_indices(feature_dataset)
    if args.max_campaigns > 0:
        campaigns = campaigns[: args.max_campaigns]
    if not campaigns:
        raise ValueError("No campaign index was found for the selected feature dataset.")

    head_configs = head_grid_configs(args)
    plan = [
        {
            "dataset": dataset,
            "campaign": campaign_id,
            "feature_stage": best.feature_stage,
            "feature_selection": best.feature_selection,
            "feature_approach": best.feature_approach,
            "best_tabpfn_params": stable_json(best.grid_params),
            "head_config": stable_json(head_config),
        }
        for campaign_id, _indices in campaigns
        for head_config in head_configs
    ]
    if args.plan_only:
        pd.DataFrame(plan).to_csv(output_path / "plan.csv", index=False)
        print(f"Saved TabPFN fine-tuning grid plan with {len(plan)} jobs to {output_path / 'plan.csv'}")
        return

    print(
        "TabPFN frozen-head fine-tuning grid search using "
        f"{best.feature_stage}/{best.feature_selection or 'none'}/{best.feature_approach} "
        f"and params={best.grid_params}; head_configs={len(head_configs)}; jobs={len(plan)}",
        flush=True,
    )

    def factory():
        return module.make_tabpfn_classifier(
            random_state=args.random_state,
            device=params.get("device", "auto"),
            ignore_pretraining_limits=bool(params.get("ignore_pretraining_limits", False)),
            model_path=model_path,
            n_estimators=int(params.get("n_estimators", 8)),
        )

    completed_now = 0
    total_jobs = len(plan)
    for campaign_id, indices in campaigns:
        campaign_dataset = subset_feature_dataset(feature_dataset, indices, campaign_id)
        embedding_folds = None
        split_strategy = None
        effective_folds = None
        for head_config in head_configs:
            key = fine_tune_grid_key(dataset, campaign_id, best, head_config)
            if key in done:
                completed_now += 1
                print(f"[{completed_now}/{total_jobs}] {campaign_id} + {CLASSIFIER_NAME} {head_config}: skipped", flush=True)
                continue
            eval_args = args_with_head_config(args, head_config)
            pending_key = {
                "config_hash": current_config_hash,
                "feature_stage": campaign_dataset.feature_stage,
                "feature_selection": campaign_dataset.feature_selection,
                "feature_approach": campaign_dataset.feature_approach,
                "classifier": CLASSIFIER_NAME,
                "federated_algorithm": None,
                "evaluation_scope": "campaign",
                "campaign": campaign_id,
                "fine_tune_grid_key": key,
            }
            try:
                if embedding_folds is None:
                    embedding_folds, split_strategy, effective_folds = prepare_campaign_embedding_folds(
                        campaign_dataset,
                        campaign_id,
                        factory,
                        eval_args,
                        best,
                    )
                result = evaluate_campaign_head_from_embeddings(
                    campaign_dataset,
                    campaign_id,
                    embedding_folds,
                    split_strategy,
                    effective_folds,
                    eval_args,
                    best,
                )
                result["dataset"] = dataset
            except Exception as exc:
                result = failed_result(campaign_dataset, campaign_id, exc, best, head_config)
                result["dataset"] = dataset
            result["fine_tune_grid_key"] = key
            result["fine_tune_grid_method"] = "tabpfn_frozen_head"
            result["grid_model"] = CLASSIFIER_NAME
            result = add_resume_metadata(result, run_config, pending_key)
            results.append(result)
            done.add(key)
            completed_now += 1
            save_fine_tune_results(output_path, results)
            print(
                f"[{completed_now}/{total_jobs}] {campaign_id} + {CLASSIFIER_NAME} "
                f"{head_config}: {result['status']} ba={result.get(PRIMARY_METRIC)}",
                flush=True,
            )

    save_fine_tune_results(output_path, results)
    print(f"Saved TabPFN fine-tuning results to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise
