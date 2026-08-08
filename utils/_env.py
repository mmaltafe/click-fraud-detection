from __future__ import annotations

from pathlib import Path


VALID_DATASETS = {"dev5", "facebook50", "all50"}
MISSING = "__missing__"
PRIMARY_METRIC = "balanced_accuracy"


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
