#!/usr/bin/env python3
"""
Create a reusable campaign LabelEncoder for the subdataset selected in .env.

Expected .env:
    DATASET=dev5

Input:
    data/raw/{DATASET}

Output:
    data/extracted_features/label_encoder/{DATASET}/campaign_label_encoder.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils._columns import CAMPAIGN_COLUMNS  # noqa: E402
from utils._env import VALID_DATASETS, read_env_value  # noqa: E402


@dataclass(frozen=True)
class CampaignEntry:
    traffic_source: str | None
    campaign: str
    source_path: str


# Pipeline configuration constants. Edit these values to change this script.
CONFIG_ENV_FILE = '.env'
CONFIG_RAW_ROOT = 'data/raw'
CONFIG_OUTPUT_ROOT = 'data/extracted_features/label_encoder'


def parse_args() -> argparse.Namespace:
    """Return script configuration from global constants; command-line args are intentionally ignored."""
    return argparse.Namespace(
        env_file=CONFIG_ENV_FILE,
        raw_root=CONFIG_RAW_ROOT,
        output_root=CONFIG_OUTPUT_ROOT,
    )


def campaign_column(columns: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in CAMPAIGN_COLUMNS:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def is_data_file(path: Path) -> bool:
    return path.is_file() and path.name != ".gitkeep" and not path.name.startswith(".")


def is_table(path: Path) -> bool:
    return path.suffix.lower() in {".csv", ".tsv", ".txt", ".parquet", ".feather"}


def read_table_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        return list(pd.read_parquet(path, columns=[]).columns)
    if path.suffix.lower() == ".feather":
        return list(pd.read_feather(path, columns=[]).columns)

    separator = "\t" if path.suffix.lower() == ".tsv" else None
    return list(pd.read_csv(path, sep=separator, engine="python", nrows=0).columns)


def campaigns_from_table(path: Path) -> list[str]:
    columns = read_table_columns(path)
    column = campaign_column(columns)
    if column is None:
        return []

    if path.suffix.lower() == ".parquet":
        table = pd.read_parquet(path, columns=[column])
    elif path.suffix.lower() == ".feather":
        table = pd.read_feather(path, columns=[column])
    else:
        separator = "\t" if path.suffix.lower() == ".tsv" else None
        table = pd.read_csv(path, sep=separator, engine="python", usecols=[column])

    return sorted(table[column].fillna("__missing__").astype(str).unique())


def campaign_name_from_path(path: Path) -> str:
    return path.stem if path.is_file() else path.name


def discover_flat_campaigns(dataset_path: Path) -> list[CampaignEntry]:
    entries: list[CampaignEntry] = []
    for child in sorted(dataset_path.iterdir()):
        if child.name.startswith("."):
            continue
        if child.is_dir():
            entries.append(CampaignEntry(None, campaign_name_from_path(child), str(child)))
            continue
        if not is_data_file(child):
            continue

        table_campaigns = campaigns_from_table(child) if is_table(child) else []
        if table_campaigns:
            entries.extend(CampaignEntry(None, campaign, str(child)) for campaign in table_campaigns)
        else:
            entries.append(CampaignEntry(None, campaign_name_from_path(child), str(child)))
    return entries


def discover_all50_campaigns(dataset_path: Path) -> list[CampaignEntry]:
    entries: list[CampaignEntry] = []
    traffic_sources = [path for path in dataset_path.iterdir() if path.is_dir() and path.name.upper().startswith("TS_")]

    for traffic_source_path in sorted(traffic_sources):
        for child in sorted(traffic_source_path.iterdir()):
            if child.name.startswith("."):
                continue
            if child.is_dir():
                campaign = campaign_name_from_path(child)
                entries.append(CampaignEntry(traffic_source_path.name, campaign, str(child)))
                continue
            if not is_data_file(child):
                continue

            table_campaigns = campaigns_from_table(child) if is_table(child) else []
            if table_campaigns:
                entries.extend(
                    CampaignEntry(traffic_source_path.name, campaign, str(child))
                    for campaign in table_campaigns
                )
            else:
                entries.append(
                    CampaignEntry(traffic_source_path.name, campaign_name_from_path(child), str(child))
                )

    return entries


def discover_campaigns(dataset: str, dataset_path: Path) -> list[CampaignEntry]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Raw dataset folder not found: {dataset_path}")

    entries = discover_all50_campaigns(dataset_path) if dataset == "all50" else discover_flat_campaigns(dataset_path)
    unique_entries = {
        (entry.traffic_source, entry.campaign): entry
        for entry in entries
    }
    return sorted(unique_entries.values(), key=lambda entry: ((entry.traffic_source or ""), entry.campaign))


def save_encoder(entries: list[CampaignEntry], output_path: Path) -> None:
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    campaigns = [entry.campaign if entry.campaign else "__missing__" for entry in entries]
    encoder = LabelEncoder()
    encoded_campaigns = encoder.fit_transform(campaigns)

    artifact = {
        "encoder": "sklearn.preprocessing.LabelEncoder",
        "classes": encoder.classes_.tolist(),
        "mapping": {
            entry.campaign: int(encoded_campaign)
            for entry, encoded_campaign in zip(entries, encoded_campaigns)
        },
    }
    (output_path / "campaign_label_encoder.json").write_text(
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    dataset = read_env_value(project_root / args.env_file, "DATASET")

    if dataset not in VALID_DATASETS:
        valid = ", ".join(sorted(VALID_DATASETS))
        raise ValueError(f"Invalid DATASET={dataset!r}. Expected one of: {valid}")

    dataset_path = project_root / args.raw_root / dataset
    entries = discover_campaigns(dataset, dataset_path)
    if not entries:
        raise ValueError(f"No campaigns were found in {dataset_path}. Run 000_get_kaggle_data first.")

    output_path = project_root / args.output_root / dataset
    save_encoder(entries, output_path)

    print(f"Saved campaign LabelEncoder with {len(entries)} campaigns to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
