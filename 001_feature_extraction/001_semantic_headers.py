#!/usr/bin/env python3
"""
Extract semantic HTTP-header features for the subdataset selected in .env.

Input:
    data/raw/{DATASET}

Output:
    data/extracted_features/semantic_headers/{DATASET}/semantic_headers.parquet
    data/extracted_features/semantic_headers/{DATASET}/target.parquet
"""

from __future__ import annotations

import argparse
import ipaddress
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlparse

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from user_agents import parse as parse_user_agent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils._columns import CAMPAIGN_COLUMNS  # noqa: E402
from utils._env import VALID_DATASETS, MISSING, read_env_value  # noqa: E402
from utils.target_utils import binary_target_series  # noqa: E402


TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".parquet", ".feather"}
TARGET_COLUMNS = ("Attack_type", "attack_type")

HEADER_ALIASES = {
    "accept": ("accept",),
    "accept_encoding": ("accept_encoding", "accept-encoding"),
    "accept_language": ("accept_language", "accept-language"),
    "cache_control": ("cache_control", "cache-control"),
    "cdn_loop": ("cdn_loop", "cdn-loop"),
    "cf_connecting_o2o": ("cf_connecting_o2o", "cf-connecting-o2o"),
    "ip_api_as": ("ip_api_as", "ip-api-as"),
    "ip_api_asname": ("ip_api_asname", "ip-api-asname"),
    "ip_api_isp": ("ip_api_isp", "ip-api-isp"),
    "ip_api_org": ("ip_api_org", "ip-api-org"),
    "ip_api_reverse": ("ip_api_reverse", "ip-api-reverse"),
    "pragma": ("pragma",),
    "priority": ("priority",),
    "referer": ("referer", "referrer"),
    "sec_ch_ua": ("sec_ch_ua", "sec-ch-ua"),
    "sec_ch_ua_mobile": ("sec_ch_ua_mobile", "sec-ch-ua-mobile"),
    "sec_ch_ua_platform": ("sec_ch_ua_platform", "sec-ch-ua-platform"),
    "sec_fetch_dest": ("sec_fetch_dest", "sec-fetch-dest"),
    "sec_fetch_mode": ("sec_fetch_mode", "sec-fetch-mode"),
    "sec_fetch_site": ("sec_fetch_site", "sec-fetch-site"),
    "sec_fetch_user": ("sec_fetch_user", "sec-fetch-user"),
    "upgrade_insecure_requests": ("upgrade_insecure_requests", "upgrade-insecure-requests"),
    "user_agent": ("user_agent", "user-agent"),
    "x_browser_channel": ("x_browser_channel", "x-browser-channel"),
    "x_browser_copyright": ("x_browser_copyright", "x-browser-copyright"),
    "x_browser_validation": ("x_browser_validation", "x-browser-validation"),
    "x_browser_year": ("x_browser_year", "x-browser-year"),
    "x_forwarded_for": ("x_forwarded_for", "x-forwarded-for"),
    "x_forwarded_port": ("x_forwarded_port", "x-forwarded-port"),
    "x_forwarded_proto": ("x_forwarded_proto", "x-forwarded-proto"),
    "x_forwarded_scheme": ("x_forwarded_scheme", "x-forwarded-scheme"),
    "x_has_set_referer": ("x_has_set_referer", "x-has-set-referer"),
    "x_requested_with": ("x_requested_with", "x-requested-with"),
    "x_scheme": ("x_scheme", "x-scheme"),
}


# Pipeline configuration constants. Edit these values to change this script.
CONFIG_ENV_FILE = '.env'
CONFIG_RAW_ROOT = 'data/raw'
CONFIG_OUTPUT_ROOT = 'data/extracted_features/semantic_headers'


def parse_args() -> argparse.Namespace:
    """Return script configuration from global constants; command-line args are intentionally ignored."""
    return argparse.Namespace(
        env_file=CONFIG_ENV_FILE,
        raw_root=CONFIG_RAW_ROOT,
        output_root=CONFIG_OUTPUT_ROOT,
    )


def normalized_column_lookup(columns: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for column in columns:
        normalized = normalize_name(column)
        lookup[normalized] = column
    return lookup


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def resolve_column(columns: list[str], canonical_name: str) -> str | None:
    lookup = normalized_column_lookup(columns)
    for alias in HEADER_ALIASES.get(canonical_name, (canonical_name,)):
        normalized = normalize_name(alias)
        if normalized in lookup:
            return lookup[normalized]
    return None


def campaign_column(columns: list[str]) -> str | None:
    lookup = normalized_column_lookup(columns)
    for candidate in CAMPAIGN_COLUMNS:
        normalized = normalize_name(candidate)
        if normalized in lookup:
            return lookup[normalized]
    return None


def target_column(columns: list[str]) -> str | None:
    lookup = normalized_column_lookup(columns)
    for candidate in TARGET_COLUMNS:
        normalized = normalize_name(candidate)
        if normalized in lookup:
            return lookup[normalized]
    return None


def is_table(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TABLE_SUFFIXES and not path.name.startswith(".")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)

    separator = "\t" if suffix == ".tsv" else None
    return pd.read_csv(path, sep=separator, engine="python")


def raw_files_for_dataset(dataset: str, dataset_path: Path) -> list[tuple[Path, str | None, str | None]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Raw dataset folder not found: {dataset_path}")

    files: list[tuple[Path, str | None, str | None]] = []
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
            files.append((path, None, campaign))
    return files


def value_series(table: pd.DataFrame, canonical_name: str) -> pd.Series:
    column = resolve_column(list(table.columns), canonical_name)
    if column is None:
        return pd.Series([MISSING] * len(table), index=table.index, dtype="object")
    return table[column].fillna(MISSING).astype(str).replace("", MISSING)


def boolean_header(series: pd.Series) -> pd.Series:
    normalized = series.fillna(MISSING).astype(str).str.strip().str.lower()
    return normalized.isin({"1", "?1", "true", "yes", "y"}).astype("int8")


def first_number(value: str, default: int = 0) -> int:
    match = re.search(r"\d+", value or "")
    return int(match.group(0)) if match else default


def split_csv_tokens(value: str) -> list[str]:
    if not value or value == MISSING:
        return []
    return [token.strip() for token in value.split(",") if token.strip()]


def extract_accept(value: str) -> dict[str, int]:
    tokens = split_csv_tokens(value)
    mime_tokens = [token.split(";", 1)[0].strip().lower() for token in tokens]
    return {
        "accept_mime_count": len(mime_tokens),
        "accept_has_html": int(any("text/html" == token for token in mime_tokens)),
        "accept_has_image": int(any(token.startswith("image/") for token in mime_tokens)),
        "accept_has_json": int(any("json" in token for token in mime_tokens)),
        "accept_has_xml": int(any("xml" in token for token in mime_tokens)),
        "accept_has_wildcard": int(any("*" in token for token in mime_tokens)),
        "accept_q_value_count": value.lower().count(";q=") if value != MISSING else 0,
    }


def extract_accept_encoding(value: str) -> dict[str, int]:
    tokens = {token.split(";", 1)[0].strip().lower() for token in split_csv_tokens(value)}
    return {
        "accept_encoding_count": len(tokens),
        "accept_encoding_has_gzip": int("gzip" in tokens),
        "accept_encoding_has_br": int("br" in tokens),
        "accept_encoding_has_deflate": int("deflate" in tokens),
        "accept_encoding_has_zstd": int("zstd" in tokens),
    }


def extract_accept_language(value: str) -> dict[str, int | str]:
    tokens = split_csv_tokens(value)
    first = tokens[0].split(";", 1)[0].strip() if tokens else MISSING
    parts = first.replace("_", "-").split("-", 1)
    primary = parts[0].lower() if parts and parts[0] else MISSING
    region = parts[1].upper() if len(parts) > 1 and parts[1] else MISSING
    return {
        "accept_language_primary": primary,
        "accept_language_region": region,
        "accept_language_count": len(tokens),
        "accept_language_q_value_count": value.lower().count(";q=") if value != MISSING else 0,
        "accept_language_has_wildcard": int(any(token.startswith("*") for token in tokens)),
    }


def extract_cache_control(value: str) -> dict[str, int]:
    lower = value.lower()
    return {
        "cache_control_has_no_cache": int("no-cache" in lower),
        "cache_control_has_no_store": int("no-store" in lower),
        "cache_control_has_public": int("public" in lower),
        "cache_control_has_private": int("private" in lower),
        "cache_control_max_age": first_number(lower, 0) if "max-age" in lower else 0,
    }


def extract_cdn_loop(value: str) -> dict[str, int | str]:
    if value == MISSING:
        return {"cdn_loop_provider": MISSING, "cdn_loop_count": 0}
    provider = value.split(";", 1)[0].strip().lower() or MISSING
    loops_match = re.search(r"loops\s*=\s*(\d+)", value, flags=re.IGNORECASE)
    return {
        "cdn_loop_provider": provider,
        "cdn_loop_count": int(loops_match.group(1)) if loops_match else 1,
    }


def extract_network_name(value: str, prefix: str) -> dict[str, int | str]:
    lower = value.lower() if value != MISSING else ""
    return {
        f"{prefix}_family": network_family(value),
        f"{prefix}_has_google": int("google" in lower),
        f"{prefix}_has_facebook": int("facebook" in lower or "meta" in lower),
        f"{prefix}_has_cloud": int("cloud" in lower or "hosting" in lower),
        f"{prefix}_length": 0 if value == MISSING else len(value),
    }


def network_family(value: str) -> str:
    if value == MISSING:
        return MISSING
    lower = value.lower()
    if "google" in lower:
        return "google"
    if "facebook" in lower or "meta" in lower:
        return "facebook"
    if "amazon" in lower or "aws" in lower:
        return "amazon"
    if "cloudflare" in lower:
        return "cloudflare"
    if "microsoft" in lower or "azure" in lower:
        return "microsoft"
    if "cloud" in lower or "hosting" in lower:
        return "cloud_hosting"
    return "other"


def extract_priority(value: str) -> dict[str, int]:
    lower = value.lower()
    urgency_match = re.search(r"u\s*=\s*(\d+)", lower)
    return {
        "priority_urgency": int(urgency_match.group(1)) if urgency_match else -1,
        "priority_has_incremental": int(bool(re.search(r"(^|,\s*)i($|,)", lower))),
    }


def extract_sec_ch_ua(value: str) -> dict[str, int]:
    lower = value.lower()
    brands = re.findall(r'"([^"]+)"\s*;\s*v\s*=', value)
    return {
        "sec_ch_ua_brand_count": len(brands),
        "sec_ch_ua_has_chromium": int("chromium" in lower),
        "sec_ch_ua_has_not_a_brand": int("not" in lower and "brand" in lower),
        "sec_ch_ua_has_android_webview": int("android webview" in lower),
        "sec_ch_ua_has_google_chrome": int("google chrome" in lower),
    }


def normalize_mobile(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"?1", "1", "true", "yes"}:
        return "mobile"
    if normalized in {"?0", "0", "false", "no"}:
        return "desktop"
    return MISSING


def normalize_platform(value: str) -> str:
    normalized = value.strip().strip('"').lower()
    if not normalized or normalized == MISSING:
        return MISSING
    if "android" in normalized:
        return "android"
    if "ios" in normalized or "iphone" in normalized or "ipad" in normalized:
        return "ios"
    if "windows" in normalized:
        return "windows"
    if "mac" in normalized:
        return "macos"
    if "linux" in normalized:
        return "linux"
    if "chrome os" in normalized or "chromium os" in normalized:
        return "chromeos"
    return normalized


def extract_sec_fetch(value: str, prefix: str) -> dict[str, str]:
    normalized = value.strip().lower()
    return {prefix: normalized if normalized else MISSING}


def extract_user_agent(value: str) -> dict[str, int | str]:
    if value == MISSING:
        return {
            "user_agent_browser_family": MISSING,
            "user_agent_os_family": MISSING,
            "user_agent_device_family": MISSING,
            "user_agent_browser_major": -1,
            "user_agent_is_bot": 0,
            "user_agent_is_mobile": 0,
            "user_agent_is_tablet": 0,
            "user_agent_is_pc": 0,
            "user_agent_is_app": 0,
            "user_agent_token_count": 0,
            "user_agent_length": 0,
        }

    parsed = parse_user_agent(value)
    lower = value.lower()
    browser_major = -1
    if parsed.browser.version and parsed.browser.version[0] is not None:
        browser_major = int(parsed.browser.version[0])

    return {
        "user_agent_browser_family": parsed.browser.family or MISSING,
        "user_agent_os_family": parsed.os.family or MISSING,
        "user_agent_device_family": parsed.device.family or MISSING,
        "user_agent_browser_major": browser_major,
        "user_agent_is_bot": int(parsed.is_bot),
        "user_agent_is_mobile": int(parsed.is_mobile),
        "user_agent_is_tablet": int(parsed.is_tablet),
        "user_agent_is_pc": int(parsed.is_pc),
        "user_agent_is_app": int(any(token in lower for token in (" wv", "gsa/", "fbav/", "instagram", "heytapbrowser"))),
        "user_agent_token_count": len(re.findall(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value)),
        "user_agent_length": len(value),
    }


def extract_referer(value: str) -> dict[str, int | str]:
    if value == MISSING:
        return {
            "referer_scheme": MISSING,
            "referer_domain": MISSING,
            "referer_path_depth": 0,
            "referer_query_param_count": 0,
            "referer_is_https": 0,
        }
    parsed = urlparse(value)
    path_depth = len([part for part in parsed.path.split("/") if part])
    return {
        "referer_scheme": parsed.scheme.lower() or MISSING,
        "referer_domain": parsed.netloc.lower() or MISSING,
        "referer_path_depth": path_depth,
        "referer_query_param_count": len(parse_qsl(parsed.query, keep_blank_values=True)),
        "referer_is_https": int(parsed.scheme.lower() == "https"),
    }


def extract_x_forwarded_for(value: str) -> dict[str, int | str]:
    ips = [part.strip() for part in value.split(",") if part.strip()] if value != MISSING else []
    versions: list[int] = []
    has_private = 0
    for raw_ip in ips:
        try:
            parsed = ipaddress.ip_address(raw_ip)
        except ValueError:
            continue
        versions.append(parsed.version)
        has_private = max(has_private, int(parsed.is_private))

    first_version = str(versions[0]) if versions else MISSING
    return {
        "x_forwarded_for_ip_count": len(ips),
        "x_forwarded_for_has_private_ip": has_private,
        "x_forwarded_for_has_ipv4": int(4 in versions),
        "x_forwarded_for_has_ipv6": int(6 in versions),
        "x_forwarded_for_first_ip_version": first_version,
    }


def normalize_protocol(*values: str) -> str:
    for value in values:
        normalized = value.strip().lower()
        if normalized in {"http", "https"}:
            return normalized
    return MISSING


def extract_x_requested_with(value: str) -> dict[str, int | str]:
    normalized = value.strip().lower()
    if not normalized or normalized == MISSING:
        return {
            "x_requested_with_family": MISSING,
            "x_requested_with_is_android_package": 0,
            "x_requested_with_length": 0,
        }
    family = "other"
    if "facebook" in normalized or "fb" in normalized:
        family = "facebook"
    elif "google" in normalized:
        family = "google"
    elif "youtube" in normalized:
        family = "youtube"
    elif "instagram" in normalized:
        family = "instagram"
    return {
        "x_requested_with_family": family,
        "x_requested_with_is_android_package": int(bool(re.match(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$", normalized))),
        "x_requested_with_length": len(value),
    }


def extract_row_features(row: pd.Series, source_columns: dict[str, str]) -> dict[str, int | str]:
    values = {
        canonical: str(row[column]) if column and pd.notna(row[column]) and str(row[column]) else MISSING
        for canonical, column in source_columns.items()
    }

    features: dict[str, int | str] = {}
    features.update(extract_accept(values["accept"]))
    features.update(extract_accept_encoding(values["accept_encoding"]))
    features.update(extract_accept_language(values["accept_language"]))
    features.update(extract_cache_control(values["cache_control"]))
    features.update(extract_cdn_loop(values["cdn_loop"]))
    features["cf_connecting_o2o_enabled"] = int(values["cf_connecting_o2o"] != MISSING)
    features.update(extract_network_name(values["ip_api_asname"], "ip_api_asname"))
    features.update(extract_network_name(values["ip_api_isp"], "ip_api_isp"))
    features.update(extract_network_name(values["ip_api_org"], "ip_api_org"))
    features.update(extract_network_name(values["ip_api_reverse"], "ip_api_reverse"))
    features["ip_api_as_number"] = first_number(values["ip_api_as"], 0)
    features["pragma_has_no_cache"] = int("no-cache" in values["pragma"].lower())
    features.update(extract_priority(values["priority"]))
    features.update(extract_sec_ch_ua(values["sec_ch_ua"]))
    features["sec_ch_ua_mobile_signal"] = normalize_mobile(values["sec_ch_ua_mobile"])
    features["sec_ch_ua_platform_signal"] = normalize_platform(values["sec_ch_ua_platform"])
    features.update(extract_sec_fetch(values["sec_fetch_dest"], "sec_fetch_dest"))
    features.update(extract_sec_fetch(values["sec_fetch_mode"], "sec_fetch_mode"))
    features.update(extract_sec_fetch(values["sec_fetch_site"], "sec_fetch_site"))
    features["sec_fetch_user_enabled"] = int(values["sec_fetch_user"] == "?1")
    features["upgrade_insecure_requests_enabled"] = int(values["upgrade_insecure_requests"] == "1")
    features.update(extract_user_agent(values["user_agent"]))
    features["x_browser_channel"] = values["x_browser_channel"].lower()
    features["x_browser_validation_length"] = 0 if values["x_browser_validation"] == MISSING else len(values["x_browser_validation"])
    features["x_browser_year"] = first_number(values["x_browser_year"], 0)
    features.update(extract_referer(values["referer"]))
    features.update(extract_x_forwarded_for(values["x_forwarded_for"]))
    protocol = normalize_protocol(values["x_forwarded_proto"], values["x_forwarded_scheme"], values["x_scheme"])
    features["forwarded_protocol"] = protocol
    features["forwarded_protocol_is_https"] = int(protocol == "https")
    features["x_forwarded_port"] = first_number(values["x_forwarded_port"], 0)
    features["x_forwarded_port_is_https"] = int(first_number(values["x_forwarded_port"], 0) == 443)
    features["x_has_set_referer_enabled"] = int(values["x_has_set_referer"] == "1")
    features.update(extract_x_requested_with(values["x_requested_with"]))
    return features


def add_campaign_context(table: pd.DataFrame, campaign: str | None, traffic_source: str | None) -> pd.DataFrame:
    table = table.copy()
    column = campaign_column(list(table.columns))
    table["_campaign"] = (
        table[column].fillna(MISSING).astype(str)
        if column is not None
        else (campaign or MISSING)
    )
    table["_traffic_source"] = traffic_source or MISSING
    return table


def split_target(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    column = target_column(list(table.columns))
    if column is None:
        target = binary_target_series([MISSING] * len(table), index=table.index)
        return table, target

    target = binary_target_series(table[column], index=table.index)
    return table.drop(columns=[column]), target


def extract_features_from_table(
    table: pd.DataFrame,
    campaign: str | None,
    traffic_source: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table, target = split_target(table)
    table = add_campaign_context(table, campaign, traffic_source)
    source_columns = {
        canonical: resolve_column(list(table.columns), canonical) or ""
        for canonical in HEADER_ALIASES
    }
    for canonical, column in list(source_columns.items()):
        if not column:
            table[f"__{canonical}"] = MISSING
            source_columns[canonical] = f"__{canonical}"

    rows = [extract_row_features(row, source_columns) for _, row in table.iterrows()]
    features = pd.DataFrame(rows)
    features.insert(0, "campaign", table["_campaign"].fillna(MISSING).astype(str).values)
    features.insert(1, "traffic_source", table["_traffic_source"].fillna(MISSING).astype(str).values)
    target_frame = pd.DataFrame({"attack_type": target.reset_index(drop=True)})
    return features, target_frame


def one_hot_encode(features: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = [
        column
        for column in features.columns
        if pd.api.types.is_object_dtype(features[column])
        or pd.api.types.is_string_dtype(features[column])
        or isinstance(features[column].dtype, pd.CategoricalDtype)
    ]
    if not categorical_columns:
        return features

    numeric = features.drop(columns=categorical_columns).reset_index(drop=True)
    categorical = features[categorical_columns].fillna(MISSING).astype(str)

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype="int8")
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype="int8")

    encoded = encoder.fit_transform(categorical)
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    encoded_frame = pd.DataFrame(encoded, columns=encoded_columns, index=numeric.index)
    return pd.concat([numeric, encoded_frame], axis=1)


def save_features(features: pd.DataFrame, target: pd.DataFrame, output_path: Path) -> None:
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path / "semantic_headers.parquet", index=False)
    target.to_parquet(output_path / "target.parquet", index=False)


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    dataset = read_env_value(project_root / args.env_file, "DATASET")

    if dataset not in VALID_DATASETS:
        valid = ", ".join(sorted(VALID_DATASETS))
        raise ValueError(f"Invalid DATASET={dataset!r}. Expected one of: {valid}")

    dataset_path = project_root / args.raw_root / dataset
    raw_files = raw_files_for_dataset(dataset, dataset_path)
    if not raw_files:
        raise ValueError(f"No raw table files were found in {dataset_path}. Run 000_get_kaggle_data first.")

    extracted_parts = []
    target_parts = []
    for path, traffic_source, campaign in raw_files:
        table = read_table(path)
        if table.empty:
            continue
        features, target = extract_features_from_table(table, campaign, traffic_source)
        extracted_parts.append(features)
        target_parts.append(target)

    if not extracted_parts:
        raise ValueError(f"No rows were found in raw table files under {dataset_path}.")

    extracted = pd.concat(extracted_parts, ignore_index=True)
    target = pd.concat(target_parts, ignore_index=True)
    encoded_features = one_hot_encode(extracted)

    output_path = project_root / args.output_root / dataset
    save_features(encoded_features, target, output_path)

    print(
        f"Saved semantic header features with {len(encoded_features)} rows and "
        f"{len(encoded_features.columns)} columns to {output_path}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
