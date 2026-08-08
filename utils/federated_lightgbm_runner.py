from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = PROJECT_ROOT / "006_federated_learning" / "000_federated_lightgbm.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("federated_lightgbm_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {BASE_SCRIPT}")
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


def save_results(output_path: Path, results: list[dict]) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    normalize_for_csv(results).to_csv(output_path / "results.csv", index=False)


def stable_key(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def prepare_base_experiment(base, output_root: str, dataset_override: str | None = None):
    args = base.parse_args()
    args.output_root = output_root
    dataset = dataset_override or base.read_env_value(PROJECT_ROOT / args.env_file, "DATASET")
    if dataset not in base.VALID_DATASETS:
        raise ValueError(f"Invalid DATASET={dataset!r}")

    feature_datasets = base.discover_feature_datasets(
        PROJECT_ROOT / args.extracted_root,
        PROJECT_ROOT / args.selected_root,
        PROJECT_ROOT / args.raw_root,
        dataset,
    )
    if not feature_datasets:
        raise ValueError("No feature datasets found.")

    best_bayesian = None
    if args.use_best_bayesian_optimization:
        best_bayesian = base.load_best_bayesian_configuration(PROJECT_ROOT, args, dataset)
        args = base.apply_lightgbm_params_from_bayesian(args, best_bayesian)
        args = base.apply_tabpfn_params_from_bayesian(args, best_bayesian)
        feature_dataset = base.select_feature_dataset(feature_datasets, best_bayesian)
    else:
        feature_dataset = feature_datasets[0]

    feature_dataset = base.cap_rows_by_campaign(feature_dataset, args.max_rows, args.random_state)
    return args, dataset, feature_dataset, best_bayesian


def args_with_overrides(args: Namespace, overrides: dict[str, Any]) -> Namespace:
    payload = vars(args).copy()
    payload.update(overrides)
    return Namespace(**payload)


def run_federated_lightgbm_configs(
    *,
    experiment_name: str,
    output_root: str,
    configs: list[dict[str, Any]],
    dataset_override: str | None = None,
) -> None:
    base = load_base_module()
    args, dataset, feature_dataset, best_bayesian = prepare_base_experiment(base, output_root, dataset_override)
    output_path = PROJECT_ROOT / output_root / dataset
    results = load_existing_results(output_path)
    done = {
        stable_key(result.get("experiment_key", {}))
        for result in results
        if result.get("status") in {"ok", "error", "failed", "skipped"}
    }

    print(
        f"{experiment_name}: {feature_dataset.feature_stage}/"
        f"{feature_dataset.feature_selection or 'none'}/{feature_dataset.feature_approach}",
        flush=True,
    )

    for position, config in enumerate(configs, start=1):
        experiment_key = {
            "experiment_name": experiment_name,
            "dataset": dataset,
            "feature_stage": feature_dataset.feature_stage,
            "feature_selection": feature_dataset.feature_selection,
            "feature_approach": feature_dataset.feature_approach,
            "feature_representation": "tabpfn_global_embeddings"
            if getattr(args, "use_tabpfn_global_embeddings", False)
            else "source_features",
            "federated_component": "lightgbm_meta_classifier",
            "tabpfn_n_estimators": getattr(args, "tabpfn_n_estimators", None),
            "tabpfn_device": getattr(args, "device", None),
            "tabpfn_max_cpu_embedding_train_rows": getattr(args, "max_cpu_embedding_train_rows", None),
            "tabpfn_embedding_cache": getattr(args, "tabpfn_embedding_cache", None),
            "use_tabpfn_embedding_cache": getattr(args, "use_tabpfn_embedding_cache", None),
            "config": config,
        }
        key = stable_key(experiment_key)
        if key in done:
            print(f"[{position}/{len(configs)}] skipped {config}", flush=True)
            continue

        run_args = args_with_overrides(args, config)
        try:
            result = base.evaluate_feature_dataset(feature_dataset, dataset, run_args)
        except Exception as exc:
            result = base.failed_result(feature_dataset, exc, dataset)

        result["experiment_name"] = experiment_name
        result["experiment_key"] = experiment_key
        result["experiment_config"] = config
        result["grid_search_method"] = experiment_name
        result["grid_model"] = result.get("model") or result.get("classifier")
        result["grid_params"] = config
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
            f"[{position}/{len(configs)}] {result['status']} ba={result.get('balanced_accuracy')} config={config}",
            flush=True,
        )

    save_results(output_path, results)
    print(f"Saved {experiment_name} results to {output_path}", flush=True)
