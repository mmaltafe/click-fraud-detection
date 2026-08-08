#!/usr/bin/env python3
"""
Download the Kaggle click-fraud dataset and create raw subdatasets:

- data/raw/all50/TS_xxx: 50 smallest campaigns from each traffic source.
- data/raw/facebook50: 50 smallest campaigns from Facebook traffic source.
- data/raw/dev5: 5 random campaigns sampled from facebook50.

Traffic source folders follow the dataset convention TS_xx. TS_1/TS_01 is
Facebook. Campaigns are detected either as subdirectories/files inside TS_*
folders or as campaign columns in tabular files.
"""

from __future__ import annotations

import random
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DATASET_SLUG = "mmaltafe/click-fraud-detection"
OUTPUT_ROOT = "data/raw"
FACEBOOK_SOURCE_NUMBER = 1
MAX_ACCESSES_PER_CAMPAIGN = 10_000
DEFAULT_RANDOM_SEED = 42

KEEP_ORIGINAL = True

CAMPAIGN_COLUMNS = (
    "campaign",
    "campaign_id",
    "campaignid",
    "campaignId",
    "Campaign",
    "CampaignId",
    "CampaignID",
)


@dataclass(frozen=True)
class Campaign:
    traffic_source: str
    campaign_id: str
    accesses: int
    path: Path | None = None


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def download_dataset(download_dir: Path) -> Path:
    """Download with kagglehub when available, otherwise with kaggle CLI."""
    try:
        import kagglehub  # type: ignore

        downloaded = Path(kagglehub.dataset_download(DATASET_SLUG))
        local_copy = download_dir / "dataset"
        if local_copy.exists():
            shutil.rmtree(local_copy)
        shutil.copytree(downloaded, local_copy)
        return local_copy
    except ImportError:
        pass

    zip_path = download_dir / "dataset.zip"
    run(["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", str(download_dir), "--force"])

    downloaded_zips = sorted(download_dir.glob("*.zip"))
    if not downloaded_zips:
        raise FileNotFoundError("Kaggle CLI did not create a zip file.")
    downloaded_zips[0].rename(zip_path)

    extract_dir = download_dir / "dataset"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    return extract_dir


def normalize_ts_number(ts_name: str) -> int | None:
    suffix = ts_name.upper().removeprefix("TS_")
    try:
        return int(suffix)
    except ValueError:
        return None


def find_traffic_sources(dataset_root: Path) -> list[Path]:
    sources = [path for path in dataset_root.rglob("*") if path.is_dir() and path.name.upper().startswith("TS_")]
    if not sources:
        raise FileNotFoundError(f"No TS_* traffic source folders found under {dataset_root}.")
    return sorted(sources, key=lambda path: (normalize_ts_number(path.name) or 10**9, path.name))


def is_tabular(path: Path) -> bool:
    return path.suffix.lower() in {".csv", ".tsv", ".txt", ".parquet", ".feather"}


def read_table(path: Path):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    sep = "\t" if suffix == ".tsv" else None
    return pd.read_csv(path, sep=sep, engine="python")


def try_read_table(path: Path):
    try:
        return read_table(path)
    except Exception as exc:
        print(f"WARNING: skipping unreadable table {path}: {exc}", file=sys.stderr)
        return None


def campaign_column(columns: Iterable[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in CAMPAIGN_COLUMNS:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def count_rows(path: Path) -> int:
    if path.is_dir():
        return sum(count_rows(child) for child in path.rglob("*") if child.is_file())
    if is_tabular(path):
        table = try_read_table(path)
        if table is None:
            return 0
        return len(table)
    return 1


def discover_file_campaigns(ts_path: Path) -> list[Campaign]:
    children = [child for child in ts_path.iterdir() if child.name != ".DS_Store"]
    campaign_like = [child for child in children if child.is_dir() or is_tabular(child)]
    if len(campaign_like) <= 1:
        return []

    campaigns: list[Campaign] = []
    for child in sorted(campaign_like):
        accesses = count_rows(child)
        if accesses <= 0:
            print(f"WARNING: skipping campaign with no readable rows: {child}", file=sys.stderr)
            continue
        campaigns.append(
            Campaign(
                traffic_source=ts_path.name,
                campaign_id=child.stem if child.is_file() else child.name,
                accesses=accesses,
                path=child,
            )
        )
    return campaigns


def discover_tabular_campaigns(ts_path: Path) -> list[Campaign]:
    campaigns: list[Campaign] = []
    for table_path in sorted(path for path in ts_path.rglob("*") if path.is_file() and is_tabular(path)):
        table = try_read_table(table_path)
        if table is None:
            continue
        column = campaign_column(table.columns)
        if column is None:
            continue
        counts = table[column].astype(str).value_counts(dropna=False)
        campaigns.extend(
            Campaign(ts_path.name, campaign_id, int(accesses), path=table_path)
            for campaign_id, accesses in counts.items()
        )
    return campaigns


def discover_campaigns(ts_path: Path) -> list[Campaign]:
    file_campaigns = discover_file_campaigns(ts_path)
    if file_campaigns:
        return file_campaigns

    tabular_campaigns = discover_tabular_campaigns(ts_path)
    if tabular_campaigns:
        return tabular_campaigns

    raise ValueError(
        f"Could not detect campaigns in {ts_path}. Expected campaign folders/files or a campaign column."
    )


def select_50_smallest(campaigns: list[Campaign]) -> list[Campaign]:
    return sorted(campaigns, key=lambda item: (item.accesses, item.campaign_id))[:50]


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))


def copy_campaign(campaign: Campaign, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if campaign.path is None:
        raise ValueError(f"Campaign {campaign.campaign_id} does not have a source path.")

    target_name = safe_name(campaign.campaign_id)
    if campaign.path.is_dir():
        target = destination / target_name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(campaign.path, target)
        trim_campaign_directory(target, MAX_ACCESSES_PER_CAMPAIGN)
        return

    table = try_read_table(campaign.path)
    if table is None:
        print(f"WARNING: skipping campaign during copy: {campaign.campaign_id}", file=sys.stderr)
        return
    column = campaign_column(table.columns)
    if column is None:
        limited = table.head(MAX_ACCESSES_PER_CAMPAIGN)
    else:
        limited = table[table[column].astype(str) == str(campaign.campaign_id)].head(MAX_ACCESSES_PER_CAMPAIGN)

    target = destination / f"{target_name}.csv"
    limited.to_csv(target, index=False)


def trim_campaign_directory(campaign_dir: Path, max_accesses: int) -> None:
    """Trim tabular files inside copied directory once max_accesses rows are reached."""
    remaining = max_accesses
    for table_path in sorted(path for path in campaign_dir.rglob("*") if path.is_file() and is_tabular(path)):
        table = try_read_table(table_path)
        if table is None:
            table_path.unlink(missing_ok=True)
            continue
        if remaining <= 0:
            table.iloc[0:0].to_csv(table_path.with_suffix(".csv"), index=False)
            if table_path.suffix.lower() != ".csv":
                table_path.unlink()
            continue

        limited = table.head(remaining)
        remaining -= len(limited)
        target = table_path if table_path.suffix.lower() == ".csv" else table_path.with_suffix(".csv")
        limited.to_csv(target, index=False)
        if target != table_path:
            table_path.unlink()


def reset_output_dirs(output_root: Path, traffic_sources: Iterable[str]) -> None:
    for relative in ("dev5", "facebook50", "all50"):
        path = output_root / relative
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    for ts_name in traffic_sources:
        (output_root / "all50" / ts_name).mkdir(parents=True, exist_ok=True)


def prepare_subdatasets(dataset_root: Path, output_root: Path, seed: int) -> None:
    ts_paths = find_traffic_sources(dataset_root)
    campaigns_by_ts = {ts_path.name: select_50_smallest(discover_campaigns(ts_path)) for ts_path in ts_paths}

    reset_output_dirs(output_root, campaigns_by_ts.keys())

    facebook_campaigns: list[Campaign] | None = None
    for ts_name, campaigns in campaigns_by_ts.items():
        destination = output_root / "all50" / ts_name
        for campaign in campaigns:
            copy_campaign(campaign, destination)

        if normalize_ts_number(ts_name) == FACEBOOK_SOURCE_NUMBER:
            facebook_campaigns = campaigns

    if not facebook_campaigns:
        raise ValueError("Facebook traffic source TS_1/TS_01 was not found.")

    for campaign in facebook_campaigns:
        copy_campaign(campaign, output_root / "facebook50")

    random_generator = random.Random(seed)
    dev_campaigns = random_generator.sample(facebook_campaigns, k=min(5, len(facebook_campaigns)))
    for campaign in dev_campaigns:
        copy_campaign(campaign, output_root / "dev5")


def main() -> None:
    project_root = Path.cwd()
    output_root = (project_root / OUTPUT_ROOT).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    original_dir = output_root / "_original"
    if original_dir.exists():
        shutil.rmtree(original_dir)

    with tempfile.TemporaryDirectory(prefix="kaggle_click_fraud_") as temporary:
        temporary_dir = Path(temporary)
        dataset_root = download_dataset(temporary_dir)

        if KEEP_ORIGINAL:
            shutil.copytree(dataset_root, original_dir)
            dataset_root = original_dir

        prepare_subdatasets(dataset_root, output_root, DEFAULT_RANDOM_SEED)

    if not KEEP_ORIGINAL and original_dir.exists():
        shutil.rmtree(original_dir)

    print(f"Done. Subdatasets saved under {output_root}")


if __name__ == "__main__":
    main()
