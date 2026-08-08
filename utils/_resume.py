from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def output_files(output_path: Path) -> tuple[Path, Path]:
    return output_path / "results.csv", output_path / "results.json"


def load_existing_results(output_path: Path) -> list[dict]:
    _csv_path, json_path = output_files(output_path)
    if not json_path.exists():
        return []
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_results(output_path: Path, results: list[dict], normalize_for_csv: Callable[[list[dict]], pd.DataFrame]) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = output_files(output_path)
    normalize_for_csv(results).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def result_key(result: dict) -> tuple:
    return (
        result.get("config_hash"),
        result.get("feature_stage"),
        result.get("feature_selection"),
        result.get("feature_approach"),
        result.get("classifier") or result.get("model"),
        result.get("federated_algorithm"),
        result.get("evaluation_scope"),
        result.get("campaign") or result.get("campaign_id"),
    )


def completed_keys(results: Iterable[dict], current_config_hash: str) -> set[tuple]:
    return {
        result_key(result)
        for result in results
        if result.get("config_hash") == current_config_hash
        and result.get("status") in {"ok", "error", "failed", "skipped"}
    }


def add_resume_metadata(result: dict, config: dict, key_suffix: dict | None = None) -> dict:
    enriched = result.copy()
    enriched["config_hash"] = config_hash(config)
    enriched["config"] = config
    if key_suffix:
        enriched["resume_key"] = json.dumps(key_suffix, sort_keys=True, separators=(",", ":"), default=str)
    return enriched


def base_config(args: Namespace, exclude: set[str] | None = None) -> dict:
    exclude = exclude or set()
    return {
        key: value
        for key, value in sorted(vars(args).items())
        if key not in exclude
    }
