#!/usr/bin/env python3
"""
Run a staged grid search for the frozen TabPFN fine-tuning head.

Stage 1 runs a compact exploratory grid on a capped subset of campaigns/rows.
Then the script chooses whether a linear head or an MLP head looks more
promising and runs a more specific Stage 2 grid. The TabPFN base remains frozen;
only the final PyTorch head is trained.

Inputs:
    results/grid_search/tabpfn/{DATASET}/results.csv
    data/extracted_features/{approach}/{DATASET}
    data/selected_features/{selector}/{approach}/{DATASET}

Outputs:
    results/tabpfn_fine_tuning_staged/{DATASET}/stage1/results.csv
    results/tabpfn_fine_tuning_staged/{DATASET}/stage1/summary.csv
    results/tabpfn_fine_tuning_staged/{DATASET}/stage2/results.csv
    results/tabpfn_fine_tuning_staged/{DATASET}/stage2/summary.csv
    results/tabpfn_fine_tuning_staged/{DATASET}/staged_summary.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = PROJECT_ROOT / "005_tabpfn" / "000_fine_tune_head.py"


# Pipeline configuration constants. Edit these values to change this script.
CONFIG_ENV_FILE = ".env"
CONFIG_GRID_RESULTS_ROOT = "results/grid_search/tabpfn"
CONFIG_EXTRACTED_ROOT = "data/extracted_features"
CONFIG_SELECTED_ROOT = "data/selected_features"
CONFIG_RAW_ROOT = "data/raw"
CONFIG_OUTPUT_ROOT = "results/tabpfn_fine_tuning_staged"
CONFIG_MODEL_CACHE = "models/tabpfn"
CONFIG_MODEL_PATH = None
CONFIG_ALLOW_BROWSER_LOGIN = False
CONFIG_DATASET = None
CONFIG_RANDOM_STATE = 42
CONFIG_K_FOLDS = 5
CONFIG_MAX_DENSE_CELLS = 10_000_000
CONFIG_ALLOW_CPU_LARGE_DATASET = False
CONFIG_RETRY_ERRORS = True

# Stage 1: compact exploration.
CONFIG_STAGE1_MAX_CAMPAIGNS = 2
CONFIG_STAGE1_MAX_ROWS = 3000
CONFIG_STAGE1_HEAD_EPOCHS_GRID = "30,80"
CONFIG_STAGE1_HEAD_BATCH_SIZE_GRID = "128,256"
CONFIG_STAGE1_HEAD_LEARNING_RATE_GRID = "0.001,0.0003"
CONFIG_STAGE1_HEAD_WEIGHT_DECAY_GRID = "0.0001,0.001"
CONFIG_STAGE1_HEAD_HIDDEN_SIZE_GRID = "0,128,256"
CONFIG_STAGE1_HEAD_DROPOUT_GRID = "0.0,0.15"
CONFIG_STAGE1_USE_CLASS_WEIGHTS_GRID = "true,false"
CONFIG_STAGE1_MAX_CONFIGS = 0

# Stage 2: refined search after choosing linear vs MLP from Stage 1.
CONFIG_STAGE2_MAX_CAMPAIGNS = 0
CONFIG_STAGE2_MAX_ROWS = 0
CONFIG_STAGE2_LINEAR_HEAD_EPOCHS_GRID = "50,100,150"
CONFIG_STAGE2_LINEAR_HEAD_BATCH_SIZE_GRID = "128,256,512"
CONFIG_STAGE2_LINEAR_HEAD_LEARNING_RATE_GRID = "0.001,0.0003,0.0001"
CONFIG_STAGE2_LINEAR_HEAD_WEIGHT_DECAY_GRID = "0.0,0.0001,0.001,0.01"
CONFIG_STAGE2_LINEAR_HEAD_HIDDEN_SIZE_GRID = "0"
CONFIG_STAGE2_LINEAR_HEAD_DROPOUT_GRID = "0.0"
CONFIG_STAGE2_LINEAR_USE_CLASS_WEIGHTS_GRID = "true,false"
CONFIG_STAGE2_MLP_HEAD_EPOCHS_GRID = "80,150"
CONFIG_STAGE2_MLP_HEAD_BATCH_SIZE_GRID = "128,256"
CONFIG_STAGE2_MLP_HEAD_LEARNING_RATE_GRID = "0.001,0.0003,0.0001"
CONFIG_STAGE2_MLP_HEAD_WEIGHT_DECAY_GRID = "0.0001,0.001,0.01"
CONFIG_STAGE2_MLP_HEAD_HIDDEN_SIZE_GRID = "128,256,512"
CONFIG_STAGE2_MLP_HEAD_DROPOUT_GRID = "0.0,0.1,0.25,0.4"
CONFIG_STAGE2_MLP_USE_CLASS_WEIGHTS_GRID = "true,false"
CONFIG_STAGE2_MAX_CONFIGS = 96
CONFIG_FORCE_STAGE2_FAMILY = "auto"  # auto, linear, mlp
CONFIG_PLAN_ONLY = False


def load_base_module():
    spec = importlib.util.spec_from_file_location("tabpfn_frozen_head_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> Namespace:
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
        max_dense_cells=CONFIG_MAX_DENSE_CELLS,
        allow_cpu_large_dataset=CONFIG_ALLOW_CPU_LARGE_DATASET,
        retry_errors=CONFIG_RETRY_ERRORS,
        plan_only=CONFIG_PLAN_ONLY,
    )


def stage_args(args: Namespace, stage: str, max_campaigns: int, max_rows: int, grid: dict[str, Any]) -> Namespace:
    payload = vars(args).copy()
    payload.update(
        {
            "output_root": str(Path(args.output_root) / stage),
            "max_campaigns": max_campaigns,
            "max_rows": max_rows,
            "head_epochs_grid": grid["head_epochs_grid"],
            "head_batch_size_grid": grid["head_batch_size_grid"],
            "head_learning_rate_grid": grid["head_learning_rate_grid"],
            "head_weight_decay_grid": grid["head_weight_decay_grid"],
            "head_hidden_size_grid": grid["head_hidden_size_grid"],
            "head_dropout_grid": grid["head_dropout_grid"],
            "use_class_weights_grid": grid["use_class_weights_grid"],
            "max_configs": grid.get("max_configs", 0),
            "plan_only": args.plan_only,
        }
    )
    return Namespace(**payload)


def stage1_grid() -> dict[str, Any]:
    return {
        "head_epochs_grid": CONFIG_STAGE1_HEAD_EPOCHS_GRID,
        "head_batch_size_grid": CONFIG_STAGE1_HEAD_BATCH_SIZE_GRID,
        "head_learning_rate_grid": CONFIG_STAGE1_HEAD_LEARNING_RATE_GRID,
        "head_weight_decay_grid": CONFIG_STAGE1_HEAD_WEIGHT_DECAY_GRID,
        "head_hidden_size_grid": CONFIG_STAGE1_HEAD_HIDDEN_SIZE_GRID,
        "head_dropout_grid": CONFIG_STAGE1_HEAD_DROPOUT_GRID,
        "use_class_weights_grid": CONFIG_STAGE1_USE_CLASS_WEIGHTS_GRID,
        "max_configs": CONFIG_STAGE1_MAX_CONFIGS,
    }


def stage2_grid(family: str) -> dict[str, Any]:
    if family == "linear":
        return {
            "head_epochs_grid": CONFIG_STAGE2_LINEAR_HEAD_EPOCHS_GRID,
            "head_batch_size_grid": CONFIG_STAGE2_LINEAR_HEAD_BATCH_SIZE_GRID,
            "head_learning_rate_grid": CONFIG_STAGE2_LINEAR_HEAD_LEARNING_RATE_GRID,
            "head_weight_decay_grid": CONFIG_STAGE2_LINEAR_HEAD_WEIGHT_DECAY_GRID,
            "head_hidden_size_grid": CONFIG_STAGE2_LINEAR_HEAD_HIDDEN_SIZE_GRID,
            "head_dropout_grid": CONFIG_STAGE2_LINEAR_HEAD_DROPOUT_GRID,
            "use_class_weights_grid": CONFIG_STAGE2_LINEAR_USE_CLASS_WEIGHTS_GRID,
            "max_configs": CONFIG_STAGE2_MAX_CONFIGS,
        }
    return {
        "head_epochs_grid": CONFIG_STAGE2_MLP_HEAD_EPOCHS_GRID,
        "head_batch_size_grid": CONFIG_STAGE2_MLP_HEAD_BATCH_SIZE_GRID,
        "head_learning_rate_grid": CONFIG_STAGE2_MLP_HEAD_LEARNING_RATE_GRID,
        "head_weight_decay_grid": CONFIG_STAGE2_MLP_HEAD_WEIGHT_DECAY_GRID,
        "head_hidden_size_grid": CONFIG_STAGE2_MLP_HEAD_HIDDEN_SIZE_GRID,
        "head_dropout_grid": CONFIG_STAGE2_MLP_HEAD_DROPOUT_GRID,
        "use_class_weights_grid": CONFIG_STAGE2_MLP_USE_CLASS_WEIGHTS_GRID,
        "max_configs": CONFIG_STAGE2_MAX_CONFIGS,
    }


def staged_key(base, dataset: str, campaign_id: str, best, head_config: dict[str, Any], stage: str) -> str:
    payload = {
        "stage": stage,
        "dataset": dataset,
        "campaign": campaign_id,
        "classifier": base.CLASSIFIER_NAME,
        "best_tabpfn_config": base.best_config_to_dict(best),
        "head_config": head_config,
    }
    return base.stable_json(payload)


def run_stage(base, args: Namespace, dataset: str, best, feature_dataset, module, model_path: str, stage: str):
    output_path = PROJECT_ROOT / args.output_root / dataset
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "best_tabpfn_config.json").write_text(
        json.dumps(base.best_config_to_dict(best), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    results = base.remove_retryable_errors(base.load_existing_results(output_path), args.retry_errors)
    done = {
        str(row.get("fine_tune_grid_key"))
        for row in results
        if row.get("fine_tune_grid_key") and (not args.retry_errors or row.get("status") in {"ok", "skipped"})
    }

    campaigns = base.campaign_indices(feature_dataset)
    if args.max_campaigns > 0:
        campaigns = campaigns[: args.max_campaigns]
    if not campaigns:
        raise ValueError("No campaign index was found for the selected feature dataset.")

    head_configs = base.head_grid_configs(args)
    plan = [
        {"campaign": campaign_id, "head_config": base.stable_json(head_config)}
        for campaign_id, _indices in campaigns
        for head_config in head_configs
    ]
    if args.plan_only:
        pd.DataFrame(plan).to_csv(output_path / "plan.csv", index=False)
        print(f"Saved {stage} plan with {len(plan)} jobs to {output_path / 'plan.csv'}", flush=True)
        return results

    params = best.grid_params

    def factory():
        return module.make_tabpfn_classifier(
            random_state=args.random_state,
            device=params.get("device", "auto"),
            ignore_pretraining_limits=bool(params.get("ignore_pretraining_limits", False)),
            model_path=model_path,
            n_estimators=int(params.get("n_estimators", 8)),
        )

    print(f"{stage}: {len(head_configs)} head configs x {len(campaigns)} campaign(s) = {len(plan)} jobs", flush=True)
    completed_now = 0
    total_jobs = len(plan)
    for campaign_id, indices in campaigns:
        campaign_dataset = base.subset_feature_dataset(feature_dataset, indices, campaign_id)
        embedding_folds = None
        split_strategy = None
        effective_folds = None
        for head_config in head_configs:
            key = staged_key(base, dataset, campaign_id, best, head_config, stage)
            if key in done:
                completed_now += 1
                print(f"[{stage} {completed_now}/{total_jobs}] {campaign_id} {head_config}: skipped", flush=True)
                continue

            eval_args = base.args_with_head_config(args, head_config)
            started = time.perf_counter()
            try:
                if embedding_folds is None:
                    embedding_folds, split_strategy, effective_folds = base.prepare_campaign_embedding_folds(
                        campaign_dataset,
                        campaign_id,
                        factory,
                        eval_args,
                        best,
                    )
                result = base.evaluate_campaign_head_from_embeddings(
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
                result = base.failed_result(campaign_dataset, campaign_id, exc, best, head_config)
                result["dataset"] = dataset

            result["fine_tune_grid_key"] = key
            result["fine_tune_grid_method"] = "tabpfn_frozen_head_staged"
            result["fine_tune_stage"] = stage
            result["grid_model"] = base.CLASSIFIER_NAME
            result["stage_elapsed_seconds"] = float(time.perf_counter() - started)
            results.append(result)
            done.add(key)
            completed_now += 1
            base.save_fine_tune_results(output_path, results)
            print(
                f"[{stage} {completed_now}/{total_jobs}] {campaign_id} {head_config}: "
                f"{result.get('status')} ba={result.get(base.PRIMARY_METRIC)}",
                flush=True,
            )

    base.save_fine_tune_results(output_path, results)
    return results


def choose_stage2_family(results: list[dict]) -> tuple[str, dict[str, Any]]:
    ok_rows = [row for row in results if row.get("status") == "ok" and row.get("balanced_accuracy") is not None]
    if not ok_rows:
        raise ValueError("Stage 1 produced no successful rows.")

    frame = pd.DataFrame(ok_rows)
    frame["balanced_accuracy"] = pd.to_numeric(frame["balanced_accuracy"], errors="coerce")
    frame["macro_f1"] = pd.to_numeric(frame.get("macro_f1"), errors="coerce")
    frame["head_hidden_size"] = pd.to_numeric(frame["head_hidden_size"], errors="coerce").fillna(0)
    frame["head_family"] = np.where(frame["head_hidden_size"] <= 0, "linear", "mlp")
    ranked = (
        frame.groupby("head_family", dropna=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            runs=("balanced_accuracy", "count"),
        )
        .reset_index()
        .sort_values(["balanced_accuracy_mean", "macro_f1_mean", "runs"], ascending=[False, False, False])
    )
    selected = str(ranked.iloc[0]["head_family"])
    return selected, ranked.to_dict(orient="records")


def main() -> None:
    base = load_base_module()
    args = parse_args()
    dataset = args.dataset or base.read_env_value(PROJECT_ROOT / args.env_file, "DATASET")
    if dataset not in base.VALID_DATASETS:
        raise ValueError(f"Invalid DATASET={dataset!r}. Expected one of: {sorted(base.VALID_DATASETS)}")

    module = base.load_tabpfn_module()
    best = base.load_best_tabpfn_config(PROJECT_ROOT / args.grid_results_root, dataset)
    feature_dataset = base.load_best_feature_dataset(module, PROJECT_ROOT, args, dataset, best)

    base.configure_tabpfn_environment(PROJECT_ROOT, args)
    model_path = base.resolve_model_path(PROJECT_ROOT, args)

    root_output = PROJECT_ROOT / args.output_root / dataset
    root_output.mkdir(parents=True, exist_ok=True)

    stage1_args = stage_args(
        args,
        "stage1",
        CONFIG_STAGE1_MAX_CAMPAIGNS,
        CONFIG_STAGE1_MAX_ROWS,
        stage1_grid(),
    )
    stage1_dataset = base.cap_rows_by_campaign(feature_dataset, stage1_args.max_rows, stage1_args.random_state)
    stage1_results = run_stage(base, stage1_args, dataset, best, stage1_dataset, module, model_path, "stage1")

    if CONFIG_FORCE_STAGE2_FAMILY.lower() in {"linear", "mlp"}:
        family = CONFIG_FORCE_STAGE2_FAMILY.lower()
        family_ranking = [{"head_family": family, "forced": True}]
    else:
        family, family_ranking = choose_stage2_family(stage1_results)

    stage2_args = stage_args(
        args,
        "stage2",
        CONFIG_STAGE2_MAX_CAMPAIGNS,
        CONFIG_STAGE2_MAX_ROWS,
        stage2_grid(family),
    )
    stage2_dataset = base.cap_rows_by_campaign(feature_dataset, stage2_args.max_rows, stage2_args.random_state)
    stage2_results = run_stage(base, stage2_args, dataset, best, stage2_dataset, module, model_path, "stage2")

    summary = {
        "dataset": dataset,
        "best_tabpfn_config": base.best_config_to_dict(best),
        "stage1_output": str(PROJECT_ROOT / stage1_args.output_root / dataset),
        "stage2_output": str(PROJECT_ROOT / stage2_args.output_root / dataset),
        "selected_stage2_family": family,
        "stage1_family_ranking": family_ranking,
        "stage1_ok_rows": int(sum(row.get("status") == "ok" for row in stage1_results)),
        "stage2_ok_rows": int(sum(row.get("status") == "ok" for row in stage2_results)),
    }
    (root_output / "staged_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
