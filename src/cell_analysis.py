from __future__ import annotations

import pandas as pd


def add_session_failures(
    df: pd.DataFrame,
    latency_failure_ms: float,
    throughput_failure_mbps: float,
    throughput_urls: list[str],
) -> pd.DataFrame:
    """Create the deterministic session-level failure indicator."""
    out = df.copy()

    out["latency_failure"] = (
        out["page_response_latency"] > latency_failure_ms
    ).astype("int8")

    out["throughput_failure"] = (
        out["url"].isin(throughput_urls)
        & (out["page_download_speed"] < throughput_failure_mbps)
        & (out["speed_test_done"] == 1)
    ).astype("int8")

    out["sla_failure"] = (
        (out["latency_failure"] == 1)
        | (out["throughput_failure"] == 1)
    ).astype("int8")

    return out


def aggregate_cells(
    df: pd.DataFrame,
    min_sessions: int,
    failure_rate_threshold: float,
    coverage_rsrp_threshold_dbm: float,
) -> pd.DataFrame:
    """Aggregate session failures and classify cells as triage candidates."""

    required = ["cell_id", "signal_strength", "sla_failure", "id"]
    working = df.dropna(subset=["cell_id"]).copy()

    cell_stats = (
        working.groupby("cell_id", observed=True)
        .agg(
            total_sessions=("id", "count"),
            sla_failures=("sla_failure", "sum"),
            median_signal_dbm=("signal_strength", "median"),
        )
        .reset_index()
    )

    cell_stats = cell_stats[
        cell_stats["total_sessions"] >= min_sessions
    ].copy()

    cell_stats["failure_rate"] = (
        cell_stats["sla_failures"]
        / cell_stats["total_sessions"]
    )

    cell_stats["problematic"] = (
        cell_stats["failure_rate"] >= failure_rate_threshold
    )

    def classify(row: pd.Series) -> str:
        if row["failure_rate"] < failure_rate_threshold:
            return "Healthy"

        if row["median_signal_dbm"] < coverage_rsrp_threshold_dbm:
            return "Coverage candidate"

        return "Capacity/congestion candidate"

    cell_stats["triage_class"] = cell_stats.apply(
        classify,
        axis=1,
    )

    # The measured failure count is intentionally retained as an impact
    # metric rather than pretending that it is a causal infrastructure score.
    cell_stats["impact_count"] = cell_stats["sla_failures"]

    return cell_stats.sort_values(
        "impact_count",
        ascending=False,
    ).reset_index(drop=True)


def summarize_cell_results(
    cell_stats: pd.DataFrame,
    top_n: int = 50,
) -> dict:
    """Summarize network-wide severity and impact concentration."""
    active = len(cell_stats)
    problematic = int(cell_stats["problematic"].sum())

    top = cell_stats.head(top_n)
    total_failures = cell_stats["sla_failures"].sum()

    top_failure_share = (
        top["sla_failures"].sum() / total_failures
        if total_failures > 0
        else 0.0
    )

    return {
        "active_cells": int(active),
        "problematic_cells": problematic,
        "problematic_share": problematic / active if active else 0.0,
        "top_n": int(min(top_n, active)),
        "top_n_failure_share": float(top_failure_share),
        "total_failures": int(total_failures),
    }


def impact_severity_table(cell_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Separate severity and impact.

    Severity = failure rate.
    Impact   = absolute failure count.
    """
    result = cell_stats[
        [
            "cell_id",
            "total_sessions",
            "sla_failures",
            "failure_rate",
            "median_signal_dbm",
            "triage_class",
        ]
    ].copy()

    result["impact_rank"] = (
        result["sla_failures"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    result["severity_rank"] = (
        result["failure_rate"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return result.sort_values("impact_rank")
