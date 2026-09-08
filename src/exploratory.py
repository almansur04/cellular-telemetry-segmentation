from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_urls(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute descriptive performance by web resource."""
    result = (
        df.groupby(
            "url",
            observed=True,
        )
        .agg(
            samples=("id", "count"),
            mean_download_mbps=(
                "page_download_speed",
                "mean",
            ),
            median_download_mbps=(
                "page_download_speed",
                "median",
            ),
            median_latency_ms=(
                "page_response_latency",
                "median",
            ),
            success_rate=(
                "response_flag",
                lambda x: (
                    x == "Success"
                ).mean(),
            ),
        )
        .reset_index()
        .sort_values(
            "mean_download_mbps",
            ascending=False,
        )
    )

    return result


def summarize_network_types(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute descriptive statistics by reported network type."""
    return (
        df.groupby(
            "network_type",
            observed=True,
        )
        .agg(
            samples=("id", "count"),
            mean_signal_dbm=(
                "signal_strength",
                "mean",
            ),
            mean_download_mbps=(
                "page_download_speed",
                "mean",
            ),
            median_download_mbps=(
                "page_download_speed",
                "median",
            ),
            mean_ping_ms=(
                "ping_time",
                "mean",
            ),
        )
        .reset_index()
    )


def summarize_temporal_patterns(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Compute hourly and weekday throughput summaries."""

    hourly = (
        df.groupby(
            "hour",
            observed=True,
        )
        .agg(
            mean_download_mbps=(
                "page_download_speed",
                "mean",
            ),
            median_download_mbps=(
                "page_download_speed",
                "median",
            ),
            samples=("id", "count"),
        )
        .reset_index()
    )

    weekday = (
        df.groupby(
            "dayofweek",
            observed=True,
        )
        .agg(
            mean_download_mbps=(
                "page_download_speed",
                "mean",
            ),
            median_download_mbps=(
                "page_download_speed",
                "median",
            ),
            samples=("id", "count"),
        )
        .reset_index()
    )

    return hourly, weekday


def dataset_summary(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Generate a compact dataset summary."""
    response_counts = (
        df["response_flag"]
        .value_counts(dropna=False)
        .to_dict()
    )

    app_counts = (
        df["app_version"]
        .value_counts(dropna=False)
        .to_dict()
    )

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "device_models": int(
            df["device_model"].nunique()
        ),
        "network_types": int(
            df["network_type"].nunique()
        ),
        "urls": int(
            df["url"].nunique()
        ),
        "unique_device_ids": int(
            df["device_id"].nunique()
        ),
        "unique_cells": int(
            df["cell_id"].nunique(
                dropna=True
            )
        ),
        "response_counts": {
            str(k): int(v)
            for k, v in response_counts.items()
        },
        "app_versions": {
            str(k): int(v)
            for k, v in app_counts.items()
        },
    }


def correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return exploratory Pearson correlations."""
    columns = [
        "signal_strength",
        "page_response_latency",
        "page_download_speed",
        "ping_time",
        "speed",
        "latitude",
        "longitude",
        "hour",
    ]

    columns = [
        c
        for c in columns
        if c in df.columns
    ]

    return df[columns].corr()