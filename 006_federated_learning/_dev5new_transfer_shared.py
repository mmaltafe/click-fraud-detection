"""
Shared loaders, TabPFN-embedding helpers, LightGBM training/scoring utilities,
and the reverse-5-fold experiment loop used by both
`007_dev5new_low_data_transfer_models.py` (uniform federated aggregation) and
`008_dev5new_personalized_transfer_models.py` (personalized transfer
aggregation). Not a runnable pipeline step on its own.
"""

from __future__ import annotations

import json
import random
import sys
import time
import importlib.util
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils._env import MISSING  # noqa: E402
from utils._tabpfn_embeddings import tabpfn_embeddings_or_proba  # noqa: E402
from utils.target_utils import TARGET_COLUMN, binary_attack_type  # noqa: E402


CONFIG_RAW_DEV_NEW_ROOT = "data/raw/dev_new_5"
CONFIG_BAYESIAN_RESULTS = "results/grid_search/tabpfn_embeddings_lightgbm/dev5/results.json"
CONFIG_MODEL_CACHE = "models/tabpfn"
CONFIG_MODEL_PATH = None
CONFIG_ALLOW_BROWSER_LOGIN = False
CONFIG_DEVICE = "auto"
CONFIG_IGNORE_PRETRAINING_LIMITS = True
CONFIG_TABPFN_N_ESTIMATORS = 8
CONFIG_ALLOW_CPU_LARGE_DATASET = False
CONFIG_MAX_CPU_EMBEDDING_TRAIN_ROWS = 2000
CONFIG_RANDOM_STATE = 42
CONFIG_K_FOLDS = 5
CONFIG_TRAIN_SIZE = 1.0 / CONFIG_K_FOLDS
CONFIG_META_TRAIN_SIZE = 0.40
CONFIG_SUPPORT_TRAIN_SIZE = 0.10
CONFIG_MAX_SUPPORT_CAMPAIGNS = 0  # 0 starts from every other usable Facebook campaign before compatibility filtering.
CONFIG_MAX_COMPATIBLE_SUPPORT_CAMPAIGNS = 5
CONFIG_RESIDUAL_ALPHA = 0.50
CONFIG_PREDICTION_THRESHOLD = 0.50

CLASS_NAMES = ["attack", "legitimate"]
SEMANTIC_HEADERS_SCRIPT = PROJECT_ROOT / "001_feature_extraction" / "001_semantic_headers.py"
TABPFN_SCRIPT = PROJECT_ROOT / "003_machine_learning_evaluation" / "002_tabpfn.py"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="The copy keyword is deprecated.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


def normalize_for_csv(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = result.copy()
        for key, value in list(row.items()):
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        rows.append(row)
    return pd.DataFrame(rows)


def save_results(output_path: Path, results: list[dict]) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    normalize_for_csv(results).to_csv(output_path / "results.csv", index=False)


def load_semantic_headers_module():
    spec = importlib.util.spec_from_file_location("semantic_headers_for_10_90", SEMANTIC_HEADERS_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {SEMANTIC_HEADERS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_tabpfn_module():
    spec = importlib.util.spec_from_file_location("tabpfn_for_10_90", TABPFN_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {TABPFN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tabpfn_args() -> Any:
    from argparse import Namespace

    return Namespace(
        env_file=".env",
        model_cache=CONFIG_MODEL_CACHE,
        model_path=CONFIG_MODEL_PATH,
        allow_browser_login=CONFIG_ALLOW_BROWSER_LOGIN,
        device=CONFIG_DEVICE,
        ignore_pretraining_limits=CONFIG_IGNORE_PRETRAINING_LIMITS,
        allow_cpu_large_dataset=CONFIG_ALLOW_CPU_LARGE_DATASET,
        max_cpu_embedding_train_rows=CONFIG_MAX_CPU_EMBEDDING_TRAIN_ROWS,
    )


def campaign_files() -> list[Path]:
    root = PROJECT_ROOT / CONFIG_RAW_DEV_NEW_ROOT
    files = sorted(root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No campaign CSV files found in {root}")
    return files


def count_rows(path: Path) -> int:
    return int(sum(1 for _ in path.open("rb")) - 1)


def ordered_facebook_campaigns() -> list[Path]:
    return sorted(campaign_files(), key=lambda path: (count_rows(path), path.stem))


def target_column(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        if str(column).lower() == TARGET_COLUMN:
            return column
    raise ValueError(f"target column {TARGET_COLUMN!r} not found")


def load_campaign(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    target = target_column(frame)
    frame[TARGET_COLUMN] = frame[target].map(binary_attack_type)
    if target != TARGET_COLUMN:
        frame = frame.drop(columns=[target])
    return frame


def usable_for_10_90(path: Path) -> bool:
    try:
        frame = load_campaign(path)
    except Exception:
        return False
    counts = frame[TARGET_COLUMN].value_counts()
    return len(counts) >= 2 and counts.min() >= 2 and len(frame) >= 20


def support_campaigns(target_path: Path) -> list[Path]:
    candidates = [path for path in ordered_facebook_campaigns() if path != target_path and usable_for_10_90(path)]
    if CONFIG_MAX_SUPPORT_CAMPAIGNS > 0:
        rng = random.Random(CONFIG_RANDOM_STATE + 1)
        candidates = rng.sample(candidates, k=min(CONFIG_MAX_SUPPORT_CAMPAIGNS, len(candidates)))
        candidates = sorted(candidates, key=lambda path: path.stem)
    return candidates


def semantic_features(module, frame: pd.DataFrame, campaign: str) -> pd.DataFrame:
    features, _target = module.extract_features_from_table(
        frame.copy(),
        campaign=campaign,
        traffic_source="TS_1",
    )
    return features


def encode_semantic_features(train_features: pd.DataFrame, feature_frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    train_encoded = pd.get_dummies(train_features.fillna(MISSING), dummy_na=False)
    columns = train_encoded.columns
    encoded_frames = []
    for frame in feature_frames:
        encoded = pd.get_dummies(frame.fillna(MISSING), dummy_na=False)
        encoded = encoded.reindex(columns=columns, fill_value=0)
        encoded_frames.append(encoded.astype(np.float32, copy=False))
    return train_encoded.astype(np.float32, copy=False), encoded_frames


def build_tabpfn_embedding_blocks(
    encoded_train_reference: pd.DataFrame,
    encoded_frames: list[pd.DataFrame],
    train_block_count: int,
    y_train_context: np.ndarray,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    module = load_tabpfn_module()
    args = tabpfn_args()
    module.configure_tabpfn_environment(PROJECT_ROOT, args)
    model_path = module.resolve_model_path(PROJECT_ROOT, args)

    train_matrix = encoded_train_reference.to_numpy(dtype=np.float32, copy=False)
    all_matrix = np.vstack([frame.to_numpy(dtype=np.float32, copy=False) for frame in encoded_frames])
    fit_indices = np.arange(len(train_matrix))
    if module.effective_cpu_device(CONFIG_DEVICE) and not CONFIG_ALLOW_CPU_LARGE_DATASET:
        fit_indices = module.downsample_indices(
            fit_indices,
            y_train_context,
            int(CONFIG_MAX_CPU_EMBEDDING_TRAIN_ROWS),
            int(CONFIG_RANDOM_STATE) + 97,
        )

    classifier = module.make_tabpfn_classifier(
        random_state=int(CONFIG_RANDOM_STATE),
        device=CONFIG_DEVICE,
        ignore_pretraining_limits=bool(CONFIG_IGNORE_PRETRAINING_LIMITS),
        model_path=model_path,
        n_estimators=int(CONFIG_TABPFN_N_ESTIMATORS),
    )
    classifier.fit(train_matrix[fit_indices], y_train_context[fit_indices])
    embeddings, embedding_source = tabpfn_embeddings_or_proba(classifier, all_matrix, len(all_matrix))

    train_positions = np.arange(len(train_matrix))
    scaler = StandardScaler()
    scaler.fit(embeddings[train_positions])
    embeddings = scaler.transform(embeddings).astype(np.float32, copy=False)

    blocks = []
    start = 0
    for frame in encoded_frames:
        end = start + len(frame)
        blocks.append(embeddings[start:end])
        start = end

    metadata = {
        "feature_representation": "tabpfn_global_embeddings",
        "tabpfn_embedding_scope": "global_train_context_10_90",
        "tabpfn_embedding_source": embedding_source,
        "tabpfn_embedding_train_rows": int(len(fit_indices)),
        "tabpfn_embedding_total_rows": int(len(all_matrix)),
        "tabpfn_embedding_features": int(embeddings.shape[1]),
        "tabpfn_embedding_model_path": model_path,
        "tabpfn_embedding_device": CONFIG_DEVICE,
        "tabpfn_embedding_n_estimators": int(CONFIG_TABPFN_N_ESTIMATORS),
        "tabpfn_context_train_blocks": int(train_block_count),
    }
    return blocks, metadata


def split_campaign(frame: pd.DataFrame, train_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = frame[TARGET_COLUMN]
    train, test = train_test_split(
        frame,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def reverse_campaign_kfold_splits(frame: pd.DataFrame) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    y = frame[TARGET_COLUMN].astype(str).to_numpy()
    counts = pd.Series(y).value_counts()
    n_splits = min(CONFIG_K_FOLDS, int(counts.min()), len(frame))
    if n_splits < 2:
        raise ValueError("Target campaign does not have enough rows per class for reverse k-fold.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CONFIG_RANDOM_STATE)
    folds = []
    row_indices = np.arange(len(frame))
    for fold_number, (large_train_idx, small_fold_idx) in enumerate(splitter.split(row_indices, y), start=1):
        train = frame.iloc[small_fold_idx].reset_index(drop=True)
        test = frame.iloc[large_train_idx].reset_index(drop=True)
        if train[TARGET_COLUMN].nunique() < 2 or test[TARGET_COLUMN].nunique() < 2:
            continue
        folds.append((fold_number, train, test))
    if not folds:
        raise ValueError("No usable reverse folds were created.")
    return folds


def split_context_and_meta(frame: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if len(frame) < 4 or frame[TARGET_COLUMN].nunique() < 2:
        return frame.reset_index(drop=True), frame.reset_index(drop=True), "fallback_same_context_and_meta"
    counts = frame[TARGET_COLUMN].value_counts()
    if counts.min() < 2:
        return frame.reset_index(drop=True), frame.reset_index(drop=True), "fallback_same_context_and_meta"
    try:
        context, meta = train_test_split(
            frame,
            test_size=CONFIG_META_TRAIN_SIZE,
            random_state=random_state,
            stratify=frame[TARGET_COLUMN],
        )
    except ValueError:
        return frame.reset_index(drop=True), frame.reset_index(drop=True), "fallback_same_context_and_meta"
    if context[TARGET_COLUMN].nunique() < 2 or meta[TARGET_COLUMN].nunique() < 2:
        return frame.reset_index(drop=True), frame.reset_index(drop=True), "fallback_same_context_and_meta"
    return context.reset_index(drop=True), meta.reset_index(drop=True), "stratified_context_meta"


def sample_support_train(frame: pd.DataFrame, train_size: float, random_state: int) -> pd.DataFrame | None:
    if frame[TARGET_COLUMN].nunique() < 2:
        return None
    counts = frame[TARGET_COLUMN].value_counts()
    if counts.min() < 2:
        return None
    train, _test = split_campaign(frame, train_size, random_state)
    if train[TARGET_COLUMN].nunique() < 2:
        return None
    return train


def select_compatible_support_payloads(
    semantic_module,
    target_train: pd.DataFrame,
    support_payloads: list[dict[str, Any]],
    params: dict[str, Any],
    label_encoder: LabelEncoder,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if CONFIG_MAX_COMPATIBLE_SUPPORT_CAMPAIGNS <= 0 or len(support_payloads) <= CONFIG_MAX_COMPATIBLE_SUPPORT_CAMPAIGNS:
        scores = [{"campaign": payload["campaign"], "compatibility_score": None} for payload in support_payloads]
        return support_payloads, scores

    target_y = label_encoder.transform(target_train[TARGET_COLUMN])
    scored_payloads = []
    target_features = semantic_features(semantic_module, target_train, "target_validation")
    for position, payload in enumerate(support_payloads, start=1):
        try:
            support_y = label_encoder.transform(payload["train"][TARGET_COLUMN])
            if len(np.unique(support_y)) < 2:
                continue
            support_features = semantic_features(semantic_module, payload["train"], payload["campaign"])
            train_encoded, encoded_frames = encode_semantic_features(
                support_features,
                [support_features, target_features],
            )
            support_X, target_X = encoded_frames
            model = fit_model(support_X, support_y, params, CONFIG_RANDOM_STATE + 20_000 + position)
            score = balanced_accuracy_score(target_y, predict_from_raw(raw_score(model, target_X)))
            scored_payloads.append((float(score), payload))
        except Exception:
            continue

    if not scored_payloads:
        return support_payloads[:CONFIG_MAX_COMPATIBLE_SUPPORT_CAMPAIGNS], [
            {"campaign": payload["campaign"], "compatibility_score": None}
            for payload in support_payloads[:CONFIG_MAX_COMPATIBLE_SUPPORT_CAMPAIGNS]
        ]

    scored_payloads.sort(key=lambda item: (-item[0], item[1]["campaign"]))
    selected = [payload for _score, payload in scored_payloads[:CONFIG_MAX_COMPATIBLE_SUPPORT_CAMPAIGNS]]
    scores = [
        {"campaign": payload["campaign"], "compatibility_score": score}
        for score, payload in scored_payloads
    ]
    return selected, scores


def load_best_lightgbm_params() -> dict[str, Any]:
    path = PROJECT_ROOT / CONFIG_BAYESIAN_RESULTS
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    candidates = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        params = row.get("lightgbm_params")
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception:
                params = {}
        if not isinstance(params, dict) or not params:
            continue
        score = row.get("trial_campaign_balanced_accuracy_mean", row.get("balanced_accuracy"))
        if score is None:
            continue
        std = row.get("trial_campaign_balanced_accuracy_std", 1e9)
        candidates.append((float(score), float(std) if std is not None else 1e9, params, row))
    if not candidates:
        return {}
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def make_lightgbm(params: dict[str, Any], random_state: int):
    try:
        from lightgbm import LGBMClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("lightgbm is required for this experiment") from exc

    defaults = {
        "n_estimators": 160,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "n_jobs": 1,
    }
    defaults.update({key: value for key, value in params.items() if key in defaults and value is not None})
    return LGBMClassifier(
        **defaults,
        class_weight="balanced",
        random_state=random_state,
        force_col_wise=True,
        verbosity=-1,
    )


def as_model_matrix(X):
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float32, copy=False)
    return X


def fit_model(X, y, params: dict[str, Any], random_state: int, init_score=None):
    model = make_lightgbm(params, random_state)
    fit_kwargs = {}
    if init_score is not None:
        fit_kwargs["init_score"] = init_score
    model.fit(as_model_matrix(X), y, **fit_kwargs)
    return model


def raw_score(model, X) -> np.ndarray:
    raw = np.asarray(model.predict(as_model_matrix(X), raw_score=True), dtype=np.float64)
    if raw.ndim > 1:
        raw = raw[:, -1]
    return raw.reshape(-1)


def probability_from_raw(raw: np.ndarray) -> np.ndarray:
    return expit(np.clip(raw, -50.0, 50.0))


def predict_from_raw(raw: np.ndarray) -> np.ndarray:
    return (probability_from_raw(raw) >= CONFIG_PREDICTION_THRESHOLD).astype(int)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, label_encoder: LabelEncoder) -> dict[str, Any]:
    labels = np.arange(len(label_encoder.classes_))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "recall_by_class": {
            str(class_name): float(value)
            for class_name, value in zip(label_encoder.classes_, recalls)
        },
    }


def model_bytes(model) -> int:
    try:
        return int(len(model.booster_.model_to_string().encode("utf-8")))
    except Exception:
        return 0


def aggregate_federated_raw(models: list[dict[str, Any]], X, weighting: str) -> np.ndarray:
    contribution = np.zeros(X.shape[0], dtype=np.float64)
    weight_sum = 0.0
    for payload in models:
        if weighting == "uniform":
            weight = 1.0
        elif weighting == "examples":
            weight = float(payload["n_train"])
        else:
            raise ValueError(f"Unknown weighting: {weighting}")
        contribution += raw_score(payload["model"], X) * weight
        weight_sum += weight
    if weight_sum <= 0.0:
        return contribution
    return contribution / weight_sum


def evaluate_bayes_local_exact(
    semantic_module,
    target_train: pd.DataFrame,
    target_test: pd.DataFrame,
    target_campaign: str,
    params: dict[str, Any],
    label_encoder: LabelEncoder,
    fold_number: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    context_frame, meta_frame, split_type = split_context_and_meta(
        target_train,
        CONFIG_RANDOM_STATE + 100_000 + fold_number,
    )
    context_features = semantic_features(semantic_module, context_frame, target_campaign)
    meta_features = semantic_features(semantic_module, meta_frame, target_campaign)
    test_features = semantic_features(semantic_module, target_test, target_campaign)
    encoded_context, encoded_frames = encode_semantic_features(
        context_features,
        [context_features, meta_features, test_features],
    )
    y_context = label_encoder.transform(context_frame[TARGET_COLUMN])
    embedding_blocks, metadata = build_tabpfn_embedding_blocks(
        encoded_context,
        encoded_frames,
        train_block_count=1,
        y_train_context=y_context,
    )
    X_meta = embedding_blocks[1]
    X_test = embedding_blocks[2]
    y_meta = label_encoder.transform(meta_frame[TARGET_COLUMN])
    if len(np.unique(y_meta)) < 2:
        y_meta = label_encoder.transform(target_train[TARGET_COLUMN])
        target_train_features = semantic_features(semantic_module, target_train, target_campaign)
        encoded_context, encoded_frames = encode_semantic_features(
            context_features,
            [context_features, target_train_features, test_features],
        )
        embedding_blocks, metadata = build_tabpfn_embedding_blocks(
            encoded_context,
            encoded_frames,
            train_block_count=1,
            y_train_context=y_context,
        )
        X_meta = embedding_blocks[1]
        X_test = embedding_blocks[2]
        split_type = "fallback_full_train_as_meta"
    model = fit_model(X_meta, y_meta, params, CONFIG_RANDOM_STATE + fold_number)
    extra = {
        "lightgbm_params": params,
        "meta_train_size": CONFIG_META_TRAIN_SIZE,
        "context_meta_split_type": split_type,
        "n_tabpfn_context": int(len(context_frame)),
        "n_meta_train": int(len(y_meta)),
        **metadata,
    }
    return raw_score(model, X_test), extra


def evaluate_variant(
    *,
    experiment_name: str,
    variant: str,
    campaign_id: str,
    y_true: np.ndarray,
    raw: np.ndarray,
    label_encoder: LabelEncoder,
    n_train: int,
    n_support_train: int,
    n_clients: int,
    communication_cost: int,
    elapsed_seconds: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    y_pred = predict_from_raw(raw)
    result = {
        "experiment_name": experiment_name,
        "dataset": "dev_new_5",
        "target_campaign": campaign_id,
        "model": variant,
        "classifier": variant,
        "evaluation_scope": "dev_new_5_campaign_reverse_5fold",
        "train_fraction_target_campaign": CONFIG_TRAIN_SIZE,
        "test_fraction_target_campaign": 1.0 - CONFIG_TRAIN_SIZE,
        "k_folds": CONFIG_K_FOLDS,
        "reverse_kfold_train_folds": 1,
        "reverse_kfold_test_folds": CONFIG_K_FOLDS - 1,
        "support_train_fraction": CONFIG_SUPPORT_TRAIN_SIZE,
        "n_train_target_campaign": int(n_train),
        "n_train_support_campaigns": int(n_support_train),
        "n_test_target_campaign": int(len(y_true)),
        "n_clients": int(n_clients),
        "communication_cost": int(communication_cost),
        "server_received_raw_rows": int(n_train + n_support_train) if variant == "Combined" else 0,
        "server_received_raw_columns": None if variant == "Combined" else 0,
        "privacy_level": "centralized" if variant == "Combined" else "local_or_federated",
        "feature_stage": "extracted_features",
        "feature_selection": None,
        "feature_approach": "semantic_headers",
        "feature_representation": "tabpfn_global_embeddings",
        "source_feature_representation": "extracted_features/none/semantic_headers",
        "prediction_threshold": CONFIG_PREDICTION_THRESHOLD,
        "elapsed_seconds": float(elapsed_seconds),
        "status": "ok",
        "error": None,
    }
    result.update(metrics(y_true, y_pred, label_encoder))
    if extra:
        result.update(extra)
    return result


def aggregate_fold_results(fold_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_columns = ["balanced_accuracy", "macro_f1", "weighted_f1", "mcc"]
    aggregated = []
    group_keys = sorted({(row.get("target_campaign"), row.get("model")) for row in fold_results})
    for campaign_id, model_name in group_keys:
        rows = [
            row
            for row in fold_results
            if row.get("target_campaign") == campaign_id and row.get("model") == model_name and row.get("status") == "ok"
        ]
        if not rows:
            continue
        base = rows[-1].copy()
        base["evaluation_scope"] = "dev_new_5_campaign_reverse_5fold"
        base["fold_number"] = None
        base["fold_count"] = int(len(rows))
        base["fold_results"] = rows
        for column in metric_columns:
            values = [float(row[column]) for row in rows if row.get(column) is not None]
            base[column] = float(np.mean(values)) if values else None
            base[f"{column}_fold_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        base["elapsed_seconds"] = float(sum(float(row.get("elapsed_seconds") or 0.0) for row in rows))
        base["communication_cost"] = int(sum(int(row.get("communication_cost") or 0) for row in rows))
        base["n_test_target_campaign"] = int(sum(int(row.get("n_test_target_campaign") or 0) for row in rows))
        base["n_train_target_campaign_mean"] = float(np.mean([int(row.get("n_train_target_campaign") or 0) for row in rows]))
        base["n_train_support_campaigns_mean"] = float(np.mean([int(row.get("n_train_support_campaigns") or 0) for row in rows]))
        base["recall_by_class"] = rows[-1].get("recall_by_class")
        aggregated.append(base)
    return aggregated


def run_transfer_experiment(
    *,
    output_root: str,
    experiment_name: str,
    force_rerun: bool,
    federated_weighting: str,
    federated_variant_name: str,
    residual_variant_name: str,
    aggregate_test_fn: Callable[[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, str], tuple[np.ndarray, dict[str, Any]]],
    aggregate_train_fn: Callable[[list[dict[str, Any]], np.ndarray, np.ndarray, str], np.ndarray],
    extra_run_metadata: dict[str, Any] | None = None,
) -> None:
    """Run the reverse-5-fold dev_new_5 transfer experiment.

    `aggregate_test_fn(federated_models, X_target_train, y_target_train, X_target_test, target_campaign)`
    returns `(federated_raw_test, aggregation_extra)` and is timed as part of the
    "Federated" variant; `aggregation_extra` is merged into both the
    "Federated"/"Residual local" result rows. `aggregate_train_fn(federated_models,
    X_target_train, y_target_train, target_campaign)` returns `federated_raw_train`
    (used as the residual model's `init_score`) and is timed as part of the
    "Residual local" variant, matching the timing split in the original scripts.
    """
    output_path = PROJECT_ROOT / output_root
    if (output_path / "results.json").exists() and not force_rerun:
        print(f"Existing results found in {output_path}; set CONFIG_FORCE_RERUN=True to rerun.")
        return

    started_total = time.perf_counter()
    target_paths = [path for path in ordered_facebook_campaigns() if usable_for_10_90(path)]
    if len(target_paths) < 2:
        raise ValueError("dev_new_5 must have at least two usable campaigns.")

    params = load_best_lightgbm_params()
    semantic_module = load_semantic_headers_module()
    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_NAMES)

    fold_results = []
    metadata_by_campaign = []

    for target_position, target_path in enumerate(target_paths, start=1):
        target_campaign = target_path.stem
        target_frame = load_campaign(target_path)
        folds = reverse_campaign_kfold_splits(target_frame)

        candidate_support_payloads = []
        for position, path in enumerate(support_campaigns(target_path), start=1):
            frame = load_campaign(path)
            train = sample_support_train(frame, CONFIG_SUPPORT_TRAIN_SIZE, CONFIG_RANDOM_STATE + target_position * 1_000 + position)
            if train is None:
                continue
            candidate_support_payloads.append({"campaign": path.stem, "train": train})

        metadata_by_fold = []
        print(
            f"Target {target_position}/{len(target_paths)} {target_campaign}: "
            f"rows={len(target_frame)} candidate_supports={len(candidate_support_payloads)}"
        )

        for fold_number, target_train, target_test in folds:
            print(
                f"  Fold {fold_number}/{len(folds)}: train={len(target_train)} "
                f"test={len(target_test)}"
            )
            support_payloads, compatibility_scores = select_compatible_support_payloads(
                semantic_module,
                target_train,
                candidate_support_payloads,
                params,
                label_encoder,
            )

            started = time.perf_counter()
            y_target_test = label_encoder.transform(target_test[TARGET_COLUMN])
            local_raw, local_extra = evaluate_bayes_local_exact(
                semantic_module,
                target_train,
                target_test,
                target_campaign,
                params,
                label_encoder,
                fold_number,
            )
            fold_results.append(
                evaluate_variant(
                    experiment_name=experiment_name,
                    variant="Bayes local",
                    campaign_id=target_campaign,
                    y_true=y_target_test,
                    raw=local_raw,
                    label_encoder=label_encoder,
                    n_train=len(target_train),
                    n_support_train=0,
                    n_clients=1,
                    communication_cost=0,
                    elapsed_seconds=time.perf_counter() - started,
                    extra={
                        "fold_number": int(fold_number),
                        "lightgbm_params": params,
                        **local_extra,
                    },
                )
            )

            train_frames = [target_train] + [payload["train"] for payload in support_payloads]
            train_feature_parts = [semantic_features(semantic_module, target_train, target_campaign)]
            support_feature_parts = [
                semantic_features(semantic_module, payload["train"], payload["campaign"])
                for payload in support_payloads
            ]
            target_test_features = semantic_features(semantic_module, target_test, target_campaign)
            train_features = pd.concat(train_feature_parts + support_feature_parts, ignore_index=True)
            _encoded_train_reference, encoded_frames = encode_semantic_features(
                train_features,
                train_feature_parts + support_feature_parts + [target_test_features],
            )

            y_target_train = label_encoder.transform(target_train[TARGET_COLUMN])
            train_context = pd.concat(train_frames, ignore_index=True)
            y_train_context = label_encoder.transform(train_context[TARGET_COLUMN])
            embedding_blocks, embedding_metadata = build_tabpfn_embedding_blocks(
                _encoded_train_reference,
                encoded_frames,
                train_block_count=len(train_frames),
                y_train_context=y_train_context,
            )
            X_target_train = embedding_blocks[0]
            X_support_trains = embedding_blocks[1 : 1 + len(support_payloads)]
            X_target_test = embedding_blocks[-1]

            started = time.perf_counter()
            combined_train = train_context
            X_combined_train = np.vstack([X_target_train] + X_support_trains)
            y_combined_train = label_encoder.transform(combined_train[TARGET_COLUMN])
            combined_model = fit_model(
                X_combined_train,
                y_combined_train,
                params,
                CONFIG_RANDOM_STATE + target_position * 10_000 + 10 + fold_number,
            )
            combined_raw = raw_score(combined_model, X_target_test)
            fold_results.append(
                evaluate_variant(
                    experiment_name=experiment_name,
                    variant="Combined",
                    campaign_id=target_campaign,
                    y_true=y_target_test,
                    raw=combined_raw,
                    label_encoder=label_encoder,
                    n_train=len(target_train),
                    n_support_train=len(combined_train) - len(target_train),
                    n_clients=1 + len(support_payloads),
                    communication_cost=model_bytes(combined_model),
                    elapsed_seconds=time.perf_counter() - started,
                    extra={
                        "fold_number": int(fold_number),
                        "lightgbm_params": params,
                        "compatible_support_scores": compatibility_scores,
                        **embedding_metadata,
                    },
                )
            )

            started = time.perf_counter()
            local_model = fit_model(
                X_target_train,
                y_target_train,
                params,
                CONFIG_RANDOM_STATE + target_position * 10_000 + 100 + fold_number,
            )
            federated_models = [
                {
                    "campaign": target_campaign,
                    "model": local_model,
                    "n_train": len(target_train),
                }
            ]
            federated_model_bytes = model_bytes(local_model)
            for position, payload in enumerate(support_payloads, start=1):
                X_support = X_support_trains[position - 1]
                y_support = label_encoder.transform(payload["train"][TARGET_COLUMN])
                if len(np.unique(y_support)) < 2:
                    continue
                model = fit_model(
                    X_support,
                    y_support,
                    params,
                    CONFIG_RANDOM_STATE + target_position * 10_000 + 1_000 + fold_number * 100 + position,
                )
                federated_models.append({"campaign": payload["campaign"], "model": model, "n_train": len(payload["train"])})
                federated_model_bytes += model_bytes(model)
            federated_raw, aggregation_extra = aggregate_test_fn(
                federated_models, X_target_train, y_target_train, X_target_test, target_campaign
            )
            fold_results.append(
                evaluate_variant(
                    experiment_name=experiment_name,
                    variant=federated_variant_name,
                    campaign_id=target_campaign,
                    y_true=y_target_test,
                    raw=federated_raw,
                    label_encoder=label_encoder,
                    n_train=len(target_train),
                    n_support_train=sum(payload["n_train"] for payload in federated_models if payload["campaign"] != target_campaign),
                    n_clients=len(federated_models),
                    communication_cost=federated_model_bytes,
                    elapsed_seconds=time.perf_counter() - started,
                    extra={
                        "fold_number": int(fold_number),
                        **aggregation_extra,
                        "federated_clients": [payload["campaign"] for payload in federated_models],
                        "lightgbm_params": params,
                        "compatible_support_scores": compatibility_scores,
                        **embedding_metadata,
                    },
                )
            )

            started = time.perf_counter()
            train_global_raw = aggregate_train_fn(federated_models, X_target_train, y_target_train, target_campaign)
            residual_model = fit_model(
                X_target_train,
                y_target_train,
                params,
                CONFIG_RANDOM_STATE + target_position * 10_000 + 5_000 + fold_number,
                init_score=train_global_raw,
            )
            residual_raw = raw_score(residual_model, X_target_test)
            final_raw = federated_raw + CONFIG_RESIDUAL_ALPHA * residual_raw
            fold_results.append(
                evaluate_variant(
                    experiment_name=experiment_name,
                    variant=residual_variant_name,
                    campaign_id=target_campaign,
                    y_true=y_target_test,
                    raw=final_raw,
                    label_encoder=label_encoder,
                    n_train=len(target_train),
                    n_support_train=sum(payload["n_train"] for payload in federated_models if payload["campaign"] != target_campaign),
                    n_clients=len(federated_models),
                    communication_cost=federated_model_bytes + model_bytes(residual_model),
                    elapsed_seconds=time.perf_counter() - started,
                    extra={
                        "fold_number": int(fold_number),
                        **aggregation_extra,
                        "residual_alpha": CONFIG_RESIDUAL_ALPHA,
                        "federated_clients": [payload["campaign"] for payload in federated_models],
                        "lightgbm_params": params,
                        "compatible_support_scores": compatibility_scores,
                        **embedding_metadata,
                    },
                )
            )

            metadata_by_fold.append(
                {
                    "fold_number": int(fold_number),
                    "target_train_rows": int(len(target_train)),
                    "target_test_rows": int(len(target_test)),
                    "support_campaign_count": int(len(support_payloads)),
                    "support_campaigns": [payload["campaign"] for payload in support_payloads],
                    "compatible_support_scores": compatibility_scores,
                    "semantic_feature_count": int(_encoded_train_reference.shape[1]),
                    "semantic_encoded_feature_count": int(_encoded_train_reference.shape[1]),
                    **embedding_metadata,
                }
            )

        metadata_by_campaign.append(
            {
                "target_campaign": target_campaign,
                "target_file": str(target_path),
                "target_rows": int(len(target_frame)),
                "fold_count": int(len(folds)),
                "candidate_support_campaign_count": int(len(candidate_support_payloads)),
                "candidate_support_campaigns": [payload["campaign"] for payload in candidate_support_payloads],
                "fold_metadata": metadata_by_fold,
            }
        )

    results = aggregate_fold_results(fold_results)

    metadata = {
        "dataset": "dev_new_5",
        "campaigns": [path.stem for path in target_paths],
        "campaign_files": [str(path) for path in target_paths],
        "selection_method": "cpg_593 plus four fixed-seed Facebook campaigns outside original dev5",
        "random_state": CONFIG_RANDOM_STATE,
        "evaluation_protocol": "reverse_5fold_train_one_fold_test_four_folds",
        "k_folds": CONFIG_K_FOLDS,
        "meta_train_size": CONFIG_META_TRAIN_SIZE,
        "max_compatible_support_campaigns": int(CONFIG_MAX_COMPATIBLE_SUPPORT_CAMPAIGNS),
        "federated_weighting": federated_weighting,
        **(extra_run_metadata or {}),
        "campaign_metadata": metadata_by_campaign,
        "elapsed_seconds_total": float(time.perf_counter() - started_total),
    }
    for result in results:
        result["experiment_metadata"] = metadata
    for result in fold_results:
        result["experiment_metadata"] = metadata

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_path / "fold_results.json").write_text(json.dumps(fold_results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    normalize_for_csv(fold_results).to_csv(output_path / "fold_results.csv", index=False)
    save_results(output_path, results)

    print(f"dev_new_5 campaigns: {', '.join(metadata['campaigns'])}")
    for result in results:
        print(
            f"{result['target_campaign']} + {result['model']}: "
            f"ba={result['balanced_accuracy']:.4f} +/- {result.get('balanced_accuracy_fold_std', 0.0):.4f} "
            f"macro_f1={result['macro_f1']:.4f} +/- {result.get('macro_f1_fold_std', 0.0):.4f}"
        )
    print(f"Saved results to {output_path}")
