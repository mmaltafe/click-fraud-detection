from __future__ import annotations

import pandas as pd


ATTACK_LABEL = "attack"
LEGITIMATE_LABEL = "legitimate"
TARGET_COLUMN = "attack_type"
TARGET_COLUMNS = ("Attack_type", "attack_type")
LEGITIMATE_VALUES = {
    "legitimate",
    "legit",
    "benign",
    "normal",
    "valid",
    "human",
    "false",
    "0",
    "no",
}


def binary_attack_type(value: object) -> str:
    """Map any non-legitimate attack_type value to the binary attack class."""
    if pd.isna(value):
        return ATTACK_LABEL
    text = str(value).strip().lower()
    if text in LEGITIMATE_VALUES:
        return LEGITIMATE_LABEL
    return ATTACK_LABEL


def binary_target_series(values, index=None) -> pd.Series:
    return pd.Series(values, index=index).map(binary_attack_type).rename(TARGET_COLUMN)


def binary_target_frame(target: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(target, pd.Series):
        return pd.DataFrame({TARGET_COLUMN: target.map(binary_attack_type).reset_index(drop=True)})

    frame = target.copy()
    if TARGET_COLUMN in frame.columns:
        values = frame[TARGET_COLUMN]
    else:
        matching = [column for column in frame.columns if str(column).lower() == TARGET_COLUMN]
        values = frame[matching[0]] if matching else pd.Series([ATTACK_LABEL] * len(frame), index=frame.index)
    return pd.DataFrame({TARGET_COLUMN: values.map(binary_attack_type).reset_index(drop=True)})

