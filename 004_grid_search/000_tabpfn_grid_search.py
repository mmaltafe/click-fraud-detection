#!/usr/bin/env python3
"""
Run a detailed grid search for the standard local TabPFN evaluator.

The script reuses the campaign-level k-fold evaluation from:

    003_machine_learning_evaluation/002_tabpfn.py

Outputs:
    results/grid_search/tabpfn/{DATASET}/results.csv
    results/grid_search/tabpfn/{DATASET}/results.json
    results/grid_search/tabpfn/{DATASET}/summary.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VALID_DATASETS = {"dev5", "facebook50", "all50"}
PRIMARY_METRIC = "balanced_accuracy"
EVALUATION_METHOD = "tabpfn"
GRID_SEARCH_METHOD = "tabpfn_detailed"


# Pipeline configuration constants. Edit these values to change this script.
CONFIG_ENV_FILE = '.env'
CONFIG_EVALUATION_ROOT = 'results/machine_learning_evaluation'
CONFIG_EXTRACTED_ROOT = 'data/extracted_features'
CONFIG_SELECTED_ROOT = 'data/selected_features'
CONFIG_RAW_ROOT = 'data/raw'
CONFIG_OUTPUT_ROOT = 'results/grid_search/tabpfn'
CONFIG_DATASET = None
CONFIG_RANDOM_STATE = 42
CONFIG_TEST_SIZE = 0.3
CONFIG_MAX_ROWS = 0
CONFIG_MAX_CAMPAIGNS = 0
CONFIG_K_FOLDS = 5
CONFIG_FEATURE_STAGE = 'auto'
CONFIG_FEATURE_SELECTION = 'auto'
CONFIG_FEATURE_APPROACH = 'auto'
CONFIG_ALL_FEATURE_DATASETS = False
CONFIG_TOP_FEATURE_PIPELINES = 1
CONFIG_MAX_FEATURE_DATASETS = 0
CONFIG_MODEL_CACHE = 'models/tabpfn'
CONFIG_MODEL_PATH = None
CONFIG_ALLOW_BROWSER_LOGIN = False
CONFIG_DEVICE_GRID = 'auto'
CONFIG_IGNORE_PRETRAINING_LIMITS_GRID = 'false,true'
CONFIG_N_ESTIMATORS_GRID = '1,2,4,8'
CONFIG_MAX_CPU_TRAIN_ROWS_GRID = '500,1000,2000'
CONFIG_MAX_CPU_TEST_ROWS_GRID = '1000,2000'
CONFIG_ALLOW_CPU_LARGE_DATASET = False
CONFIG_MAX_DENSE_CELLS = 10000000
CONFIG_MAX_CONFIGS = 0
CONFIG_PLAN_ONLY = False
CONFIG_RETRY_FAILED = True
CONFIG_PROGRESS = True


def parse_args() -> argparse.Namespace:
    """Return script configuration from global constants; command-line args are intentionally ignored."""
    return argparse.Namespace(
        env_file=CONFIG_ENV_FILE,
        evaluation_root=CONFIG_EVALUATION_ROOT,
        extracted_root=CONFIG_EXTRACTED_ROOT,
        selected_root=CONFIG_SELECTED_ROOT,
        raw_root=CONFIG_RAW_ROOT,
        output_root=CONFIG_OUTPUT_ROOT,
        dataset=CONFIG_DATASET,
        random_state=CONFIG_RANDOM_STATE,
        test_size=CONFIG_TEST_SIZE,
        max_rows=CONFIG_MAX_ROWS,
        max_campaigns=CONFIG_MAX_CAMPAIGNS,
        k_folds=CONFIG_K_FOLDS,
        feature_stage=CONFIG_FEATURE_STAGE,
        feature_selection=CONFIG_FEATURE_SELECTION,
        feature_approach=CONFIG_FEATURE_APPROACH,
        all_feature_datasets=CONFIG_ALL_FEATURE_DATASETS,
        top_feature_pipelines=CONFIG_TOP_FEATURE_PIPELINES,
        max_feature_datasets=CONFIG_MAX_FEATURE_DATASETS,
        model_cache=CONFIG_MODEL_CACHE,
        model_path=CONFIG_MODEL_PATH,
        allow_browser_login=CONFIG_ALLOW_BROWSER_LOGIN,
        device_grid=CONFIG_DEVICE_GRID,
        ignore_pretraining_limits_grid=CONFIG_IGNORE_PRETRAINING_LIMITS_GRID,
        n_estimators_grid=CONFIG_N_ESTIMATORS_GRID,
        max_cpu_train_rows_grid=CONFIG_MAX_CPU_TRAIN_ROWS_GRID,
        max_cpu_test_rows_grid=CONFIG_MAX_CPU_TEST_ROWS_GRID,
        allow_cpu_large_dataset=CONFIG_ALLOW_CPU_LARGE_DATASET,
        max_dense_cells=CONFIG_MAX_DENSE_CELLS,
        max_configs=CONFIG_MAX_CONFIGS,
        plan_only=CONFIG_PLAN_ONLY,
        retry_failed=CONFIG_RETRY_FAILED,
        progress=CONFIG_PROGRESS,
    )

def read_env_value(env_file: Path, key: str) -> str:
    if not env_file.exists():
        raise FileNotFoundError(f"Environment file not found: {env_file}")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip('"').strip("'")
    raise KeyError(f"{key} was not found in {env_file}")


def parse_int_grid(value: str) -> list[int]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("integer grid cannot be empty")
    return [int(item) for item in values]


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


def parse_str_grid(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("string grid cannot be empty")
    return values


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def normalize_metric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_tabpfn_module(project_root: Path):
    evaluation_dir = project_root / "003_machine_learning_evaluation"
    if str(evaluation_dir) not in sys.path:
        sys.path.insert(0, str(evaluation_dir))
    path = evaluation_dir / "002_tabpfn.py"
    spec = importlib.util.spec_from_file_location("ml_eval_tabpfn_grid", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_for_csv(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = result.copy()
        for key, value in list(row.items()):
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        rows.append(row)
    return pd.DataFrame(rows)


def load_existing_results(output_path: Path) -> list[dict]:
    path = output_path / "results.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def summarize_results(results: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame([row for row in results if row.get("status") == "ok"])
    if frame.empty or PRIMARY_METRIC not in frame.columns:
        return pd.DataFrame()
    metrics = [column for column in (PRIMARY_METRIC, "macro_f1", "weighted_f1", "mcc", "fit_predict_seconds") if column in frame.columns]
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    groups = [column for column in ("dataset", "feature_stage", "feature_selection", "feature_approach", "campaign") if column in frame.columns]
    summary = frame.groupby(groups, dropna=False)[metrics].agg(["mean", "max", "std"]).reset_index()
    summary.columns = ["_".join(str(part) for part in column if part) if isinstance(column, tuple) else str(column) for column in summary.columns]
    return summary.sort_values(f"{PRIMARY_METRIC}_mean", ascending=False)


def save_results(output_path: Path, results: list[dict]) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    normalize_for_csv(results).to_csv(output_path / "results.csv", index=False)
    summary = summarize_results(results)
    if not summary.empty:
        summary.to_csv(output_path / "summary.csv", index=False)


def load_evaluation_rows(evaluation_root: Path, dataset: str) -> pd.DataFrame:
    path = evaluation_root / EVALUATION_METHOD / dataset / "results.json"
    if not path.exists():
        return pd.DataFrame()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()
    rows = []
    for row in data:
        item = dict(row)
        item["feature_selection"] = item.get("feature_selection") or "none"
        item[PRIMARY_METRIC] = normalize_metric(item.get(PRIMARY_METRIC))
        item["macro_f1"] = normalize_metric(item.get("macro_f1"))
        rows.append(item)
    return pd.DataFrame(rows)


def best_feature_pipelines(rows: pd.DataFrame, top_n: int) -> list[dict[str, Any]]:
    if rows.empty:
        return []
    candidates = rows[(rows.get("status") == "ok") & rows[PRIMARY_METRIC].notna()].copy()
    if candidates.empty:
        return []
    grouped = (
        candidates.groupby(["feature_stage", "feature_selection", "feature_approach"], dropna=False)
        .agg(source_balanced_accuracy=(PRIMARY_METRIC, "mean"), source_macro_f1=("macro_f1", "mean"), source_runs=(PRIMARY_METRIC, "count"))
        .reset_index()
        .sort_values(["source_balanced_accuracy", "source_macro_f1"], ascending=[False, False])
    )
    pipelines = []
    for _, row in grouped.head(max(1, top_n)).iterrows():
        selector = row["feature_selection"]
        pipelines.append(
            {
                "feature_stage": row["feature_stage"],
                "feature_selection": None if selector in {None, "none"} else selector,
                "feature_approach": row["feature_approach"],
                "source_evaluation_method": EVALUATION_METHOD,
                "source_balanced_accuracy": normalize_metric(row["source_balanced_accuracy"]),
                "source_macro_f1": normalize_metric(row["source_macro_f1"]),
                "source_runs": int(row["source_runs"]),
            }
        )
    return pipelines


def feature_selection_name(value: str | None) -> str:
    return value or "none"


def matches_filter(feature_dataset, args: argparse.Namespace) -> bool:
    if args.feature_stage != "auto" and feature_dataset.feature_stage != args.feature_stage:
        return False
    if args.feature_selection != "auto" and feature_selection_name(feature_dataset.feature_selection) != args.feature_selection:
        return False
    if args.feature_approach != "auto" and feature_dataset.feature_approach != args.feature_approach:
        return False
    return True


def matches_pipeline(feature_dataset, pipeline: dict[str, Any]) -> bool:
    return (
        feature_dataset.feature_stage == pipeline["feature_stage"]
        and (feature_dataset.feature_selection or None) == (pipeline["feature_selection"] or None)
        and feature_dataset.feature_approach == pipeline["feature_approach"]
    )


def select_feature_datasets(module, project_root: Path, args: argparse.Namespace, dataset: str) -> list[tuple[Any, dict[str, Any]]]:
    discovered = [
        module.cap_rows_by_campaign(item, args.max_rows, args.random_state)
        for item in module.discover_feature_datasets(project_root / args.extracted_root, project_root / args.selected_root, project_root / args.raw_root, dataset)
        if matches_filter(item, args)
    ]
    if not discovered:
        return []
    if args.all_feature_datasets or args.feature_stage != "auto" or args.feature_selection != "auto" or args.feature_approach != "auto":
        selected = [(item, {"feature_stage": item.feature_stage, "feature_selection": item.feature_selection, "feature_approach": item.feature_approach}) for item in discovered]
    else:
        pipelines = best_feature_pipelines(load_evaluation_rows(project_root / args.evaluation_root, dataset), args.top_feature_pipelines)
        selected = [(item, pipeline) for pipeline in pipelines for item in discovered if matches_pipeline(item, pipeline)]
        if not selected:
            selected = [(item, {"feature_stage": item.feature_stage, "feature_selection": item.feature_selection, "feature_approach": item.feature_approach}) for item in discovered]
    if args.max_feature_datasets > 0:
        selected = selected[: args.max_feature_datasets]
    return selected


def grid_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs = []
    for device, ignore_limits, n_estimators, max_train, max_test in product(
        parse_str_grid(args.device_grid),
        parse_bool_grid(args.ignore_pretraining_limits_grid),
        parse_int_grid(args.n_estimators_grid),
        parse_int_grid(args.max_cpu_train_rows_grid),
        parse_int_grid(args.max_cpu_test_rows_grid),
    ):
        configs.append(
            {
                "device": device,
                "ignore_pretraining_limits": ignore_limits,
                "n_estimators": n_estimators,
                "max_cpu_train_rows": max_train,
                "max_cpu_test_rows": max_test,
            }
        )
    if args.max_configs > 0 and len(configs) > args.max_configs:
        rng = np.random.default_rng(args.random_state)
        chosen = np.sort(rng.choice(np.arange(len(configs)), size=args.max_configs, replace=False))
        configs = [configs[int(index)] for index in chosen]
    return configs


def grid_key(dataset: str, feature_dataset, campaign_id: str, config: dict[str, Any]) -> str:
    return stable_json(
        {
            "grid_search_method": GRID_SEARCH_METHOD,
            "dataset": dataset,
            "feature_stage": feature_dataset.feature_stage,
            "feature_selection": feature_selection_name(feature_dataset.feature_selection),
            "feature_approach": feature_dataset.feature_approach,
            "campaign": campaign_id,
            "params": config,
        }
    )


def completed_grid_keys(results: list[dict], retry_failed: bool) -> set[str]:
    done = set()
    for row in results:
        if retry_failed and row.get("status") not in {"ok", "skipped"}:
            continue
        if row.get("grid_key"):
            done.add(row["grid_key"])
    return done


def config_args(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    payload = vars(args).copy()
    payload.update(config)
    payload["device"] = config["device"]
    payload["ignore_pretraining_limits"] = config["ignore_pretraining_limits"]
    payload["n_estimators"] = config["n_estimators"]
    return argparse.Namespace(**payload)


def annotate_result(result: dict, dataset: str, feature_dataset, campaign_id: str, config: dict[str, Any], source: dict[str, Any], elapsed: float) -> dict:
    enriched = result.copy()
    enriched["dataset"] = dataset
    enriched["grid_search_method"] = GRID_SEARCH_METHOD
    enriched["grid_model"] = "TabPFN"
    enriched["grid_params"] = config
    enriched["grid_key"] = grid_key(dataset, feature_dataset, campaign_id, config)
    enriched["campaign"] = campaign_id
    enriched["campaign_id"] = campaign_id
    enriched["source_evaluation_method"] = source.get("source_evaluation_method")
    enriched["source_balanced_accuracy"] = source.get("source_balanced_accuracy")
    enriched["source_macro_f1"] = source.get("source_macro_f1")
    enriched["source_runs"] = source.get("source_runs")
    enriched["grid_elapsed_seconds"] = elapsed
    return enriched


def failed_result(feature_dataset, dataset: str, campaign_id: str, config: dict[str, Any], source: dict[str, Any], error: Exception, elapsed: float) -> dict:
    result = {
        "feature_stage": feature_dataset.feature_stage,
        "feature_selection": feature_dataset.feature_selection,
        "feature_approach": feature_dataset.feature_approach,
        "classifier": "TabPFN",
        "evaluation_scope": "campaign",
        "campaign": campaign_id,
        "campaign_id": campaign_id,
        "dataset": dataset,
        "n_samples": int(len(feature_dataset.target)),
        "n_features": int(feature_dataset.X.shape[1]),
        "balanced_accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
        "mcc": None,
        "status": "failed",
        "error": str(error),
    }
    return annotate_result(result, dataset, feature_dataset, campaign_id, config, source, elapsed)


def campaign_jobs(module, feature_dataset, max_campaigns: int) -> list[tuple[str, np.ndarray]]:
    campaigns = module.campaign_indices(feature_dataset)
    return campaigns[:max_campaigns] if max_campaigns > 0 else campaigns


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    dataset = args.dataset or read_env_value(project_root / args.env_file, "DATASET")
    if dataset not in VALID_DATASETS:
        raise ValueError(f"Invalid dataset: {dataset}")

    module = load_tabpfn_module(project_root)
    selected_datasets = select_feature_datasets(module, project_root, args, dataset)
    if not selected_datasets:
        raise ValueError("No feature datasets were found for the requested filters.")

    configs = grid_configs(args)
    output_path = project_root / args.output_root / dataset
    results = load_existing_results(output_path)
    done = completed_grid_keys(results, args.retry_failed)

    plan = [
        {
            "grid_search_method": GRID_SEARCH_METHOD,
            "feature_stage": feature_dataset.feature_stage,
            "feature_selection": feature_selection_name(feature_dataset.feature_selection),
            "feature_approach": feature_dataset.feature_approach,
            "campaign": campaign_id,
            "params": stable_json(config),
        }
        for feature_dataset, _source in selected_datasets
        for campaign_id, _indices in campaign_jobs(module, feature_dataset, args.max_campaigns)
        for config in configs
    ]
    if args.plan_only:
        output_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(plan).to_csv(output_path / "plan.csv", index=False)
        print(f"Saved TabPFN grid-search plan with {len(plan)} jobs to {output_path / 'plan.csv'}")
        return

    module.configure_tabpfn_environment(project_root, args)
    model_path = module.resolve_model_path(project_root, args)
    total_jobs = len(plan)
    completed_now = 0
    if args.progress:
        print(f"TabPFN detailed grid search: {total_jobs} jobs planned for {dataset}", flush=True)

    for feature_dataset, source in selected_datasets:
        campaigns = campaign_jobs(module, feature_dataset, args.max_campaigns)
        if not campaigns:
            print(f"{feature_dataset.feature_approach}/{feature_selection_name(feature_dataset.feature_selection)}: no campaigns found; skipped")
            continue
        for campaign_id, indices in campaigns:
            campaign_dataset = module.subset_feature_dataset(feature_dataset, indices, campaign_id)
            for config in configs:
                key = grid_key(dataset, feature_dataset, campaign_id, config)
                if key in done:
                    continue
                started = time.perf_counter()
                try:
                    eval_args = config_args(args, config)
                    factory = lambda eval_args=eval_args: module.make_tabpfn_classifier(
                        eval_args.random_state,
                        eval_args.device,
                        eval_args.ignore_pretraining_limits,
                        model_path,
                        eval_args.n_estimators,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = module.evaluate_campaign_classifier(campaign_dataset, campaign_id, "TabPFN", factory, eval_args)
                    elapsed = time.perf_counter() - started
                    result = annotate_result(result, dataset, feature_dataset, campaign_id, config, source, elapsed)
                except Exception as exc:
                    elapsed = time.perf_counter() - started
                    result = failed_result(feature_dataset, dataset, campaign_id, config, source, exc, elapsed)
                results.append(result)
                done.add(key)
                completed_now += 1
                save_results(output_path, results)
                if args.progress:
                    selector = feature_selection_name(feature_dataset.feature_selection)
                    print(f"[{completed_now}/{total_jobs}] {feature_dataset.feature_approach}/{selector}/{campaign_id} + TabPFN {config}: {result['status']} ba={result.get(PRIMARY_METRIC)}", flush=True)

    save_results(output_path, results)
    print(f"Saved TabPFN detailed grid-search results to {output_path}")


if __name__ == "__main__":
    main()
