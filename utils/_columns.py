from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from utils._env import MISSING


CAMPAIGN_COLUMNS = (
    "campaign",
    "campaign_id",
    "campaignid",
    "campaignId",
    "Campaign",
    "CampaignId",
    "CampaignID",
)

TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".parquet", ".feather"}


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {normalize_name(column): column for column in columns}
    for candidate in candidates:
        candidate_name = normalize_name(candidate)
        if candidate_name in normalized:
            return normalized[candidate_name]
    return None


def is_table(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TABLE_SUFFIXES and not path.name.startswith(".")


def read_table_strict(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    separator = "\t" if suffix == ".tsv" else None
    return pd.read_csv(path, sep=separator, engine="python")


def read_table(path: Path) -> pd.DataFrame | None:
    try:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".feather":
            return pd.read_feather(path)

        separator = "\t" if suffix == ".tsv" else None
        return pd.read_csv(path, sep=separator, engine="python")
    except Exception as exc:
        print(f"WARNING: skipping unreadable table {path}: {exc}", file=sys.stderr)
        return None


def raw_files_for_dataset(dataset: str, dataset_path: Path) -> list[tuple[Path, str, str]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Raw dataset folder not found: {dataset_path}")

    files: list[tuple[Path, str, str]] = []
    if dataset == "all50":
        for traffic_source in sorted(path for path in dataset_path.iterdir() if path.is_dir()):
            for path in sorted(traffic_source.rglob("*")):
                if is_table(path):
                    campaign = path.parent.name if path.parent != traffic_source else path.stem
                    files.append((path, traffic_source.name, campaign))
        return files

    for path in sorted(dataset_path.rglob("*")):
        if is_table(path):
            campaign = path.parent.name if path.parent != dataset_path else path.stem
            files.append((path, MISSING, campaign))
    return files


def stringify_value(value: object) -> str:
    if pd.isna(value):
        return MISSING
    text = str(value).strip()
    return text if text else MISSING
