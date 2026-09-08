from __future__ import annotations

import numpy as np
import pandas as pd


DROP_COLUMNS = [
    "imei",
    "phone_number",
]

INVALID_VALUES = {
    "mcc": [0],
    "mnc": [0],
    "cell_id": [-1],
    "lac": [65535],
}


def parse_and_clean(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize raw telemetry fields without modifying measurements.
    """
    out = df.copy()

    if "test_time" in out.columns:
        out["test_time"] = pd.to_datetime(
            out["test_time"],
            errors="coerce",
        )

    # Remove sensitive identifiers before downstream analysis.
    for column in DROP_COLUMNS:
        if column in out.columns:
            out = out.drop(
                columns=column,
            )

    # Preserve network identifiers as nullable integers while normalizing
    # known sentinel values to missing data.
    for column, invalid_values in INVALID_VALUES.items():
        if column not in out.columns:
            continue

        series = pd.to_numeric(
            out[column],
            errors="coerce",
        )

        series = series.replace(
            invalid_values,
            np.nan,
        )

        out[column] = series.astype("Int64")

    if "test_time" in out.columns:
        out["hour"] = (
            out["test_time"]
            .dt.hour
            .astype("int8")
        )

        out["dayofweek"] = (
            out["test_time"]
            .dt.dayofweek
            .astype("int8")
        )

        out["date"] = (
            out["test_time"]
            .dt.normalize()
        )

    if "speed" in out.columns:
        out["speed_test_done"] = (
            out["speed"] > 0
        ).astype("int8")

    return out


def optimize_dtypes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reduce memory usage while preserving analytical semantics.
    """
    out = df.copy()

    integer_columns = out.select_dtypes(
        include=["int64"],
    ).columns

    for column in integer_columns:
        out[column] = pd.to_numeric(
            out[column],
            downcast="integer",
        )

    float_columns = out.select_dtypes(
        include=["float64"],
    ).columns

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

    # Convert repeated string dimensions to categorical storage.
    for column in categorical_columns:
        if column in out.columns:
            out[column] = out[column].astype(
                "category",
            )

    return out


def validate_dataset(
    df: pd.DataFrame,
) -> dict:
    """Return reproducibility-oriented data-integrity checks."""
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(
            df.duplicated().sum(),
        ),
        "missing_test_time": int(
            df["test_time"].isna().sum(),
        )
        if "test_time" in df.columns
        else None,
        "device_models": int(
            df["device_model"].nunique(),
        )
        if "device_model" in df.columns
        else None,
        "network_types": int(
            df["network_type"].nunique(),
        )
        if "network_type" in df.columns
        else None,
        "urls": int(
            df["url"].nunique(),
        )
        if "url" in df.columns
        else None,
        "unique_device_ids": int(
            df["device_id"].nunique(),
        )
        if "device_id" in df.columns
        else None,
        "unique_cells": int(
            df["cell_id"].nunique(
                dropna=True,
            ),
        )
        if "cell_id" in df.columns
        else None,
    }
