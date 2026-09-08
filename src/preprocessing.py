from __future__ import annotations

import numpy as np
import pandas as pd


DROP_COLUMNS = ["imei", "phone_number"]

INVALID_VALUES = {
    "mcc": [0],
    "mnc": [0],
    "cell_id": [-1],
    "lac": [65535],
}


def parse_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw telemetry dataset without changing scientific measurements.
    """
    out = df.copy()

    if "test_time" in out.columns:
        out["test_time"] = pd.to_datetime(
            out["test_time"],
            errors="coerce",
        )

    for column in DROP_COLUMNS:
        if column in out.columns:
            out = out.drop(columns=column)

    for column, invalid_values in INVALID_VALUES.items():
        if column in out.columns:
            out[column] = out[column].replace(invalid_values, np.nan)

    # Derived time variables.
    if "test_time" in out.columns:
        out["hour"] = out["test_time"].dt.hour
        out["dayofweek"] = out["test_time"].dt.dayofweek
        out["date"] = out["test_time"].dt.date

    # The original exploratory analysis uses speed > 0 as the operational
    # indicator for an active speed measurement.
    if "speed" in out.columns:
        out["speed_test_done"] = (out["speed"] > 0).astype("int8")

    return out


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory footprint while preserving analytical meaning.
    """
    out = df.copy()

    int_columns = out.select_dtypes(include=["int64"]).columns
    for column in int_columns:
        out[column] = pd.to_numeric(
            out[column],
            downcast="integer",
        )

    float_columns = out.select_dtypes(include=["float64"]).columns
    for column in float_columns:
        out[column] = pd.to_numeric(
            out[column],
            downcast="float",
        )

    categorical_columns = [
        "device",
        "device_model",
        "device_model_code",
        "device_id",
        "network_type",
        "operator",
        "url",
        "response_flag",
        "app_version",
    ]

    for column in categorical_columns:
        if column in out.columns:
            out[column] = out[column].astype("category")

    return out


def validate_dataset(df: pd.DataFrame) -> dict:
    """Return basic integrity checks."""
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_test_time": int(df["test_time"].isna().sum())
        if "test_time" in df.columns
        else None,
        "unique_devices": int(df["device_model"].nunique())
        if "device_model" in df.columns
        else None,
        "unique_cells": int(df["cell_id"].nunique(dropna=True))
        if "cell_id" in df.columns
        else None,
    }
