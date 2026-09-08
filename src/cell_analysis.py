from __future__ import annotations

import numpy as np
import pandas as pd


def add_session_failures(
    df: pd.DataFrame,
    latency_failure_ms: float,
    throughput_failure_mbps: float,
    throughput_urls: list[str],
) -> pd.DataFrame:
    """Create deterministic session-level quality indicators."""
    out = df.copy()

    out["latency_failure"] = (
        out["page_response_latency"]
        > latency_failure_ms
    ).astype("int8")

    out["throughput_failure"] = (
        out["url"].isin(
            throughput_urls
        )
        & (
            out["page_download_speed"]
            < throughput_failure_mbps
        )
        & (
            out["speed_test_done"] == 1
        )
    ).astype("int8")

    out["quality_failure"] = (
        (
            out["latency_failure"]
            == 1
        )
        |
        (
            out["throughput_failure"]
            == 1
        )
    ).astype("int8")

    return out


def wilson_interval(
    failures: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute a Wilson confidence interval for a binomial proportion."""
    if total <= 0:
        return np.nan, np.nan

    from scipy.stats import norm

    z = norm.ppf(
        1 - (1 - confidence) / 2
    )

    p = failures / total

    denominator = (
        1
        + z**2 / total
    )

    center = (
        p
        + z**2 / (2 * total)
    ) / denominator

    margin = (
        z
        * np.sqrt(
            (
                p * (1 - p)
                / total
            )
            + (
                z**2
                / (4 * total**2)
            )
        )
        / denominator
    )

    return (
        max(
            0.0,
            center - margin,
        ),
        min(
            1.0,
            center + margin,
        ),
    )


def aggregate_cells(
    df: pd.DataFrame,
    min_sessions: int,
    failure_rate_threshold: float,
    coverage_rsrp_threshold_dbm: float,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Aggregate session-level quality indicators into cell-level triage metrics."""
    working = df.dropna(
        subset=[
            "cell_id",
            "id",
        ]
    ).copy()

    cell_stats = (
        working.groupby(
            "cell_id",
            observed=True,
        )
        .agg(
            total_sessions=(
                "id",
                "count",
            ),
            quality_failures=(
                "quality_failure",
                "sum",
            ),
            median_signal_dbm=(
                "signal_strength",
                "median",
            ),
        )
        .reset_index()
    )

    # Exclude low-volume cells whose failure rates are too unstable for triage.
    cell_stats = cell_stats[
        cell_stats[
            "total_sessions"
        ]
        >= min_sessions
    ].copy()

    cell_stats["failure_rate"] = (
        cell_stats[
            "quality_failures"
        ]
        / cell_stats[
            "total_sessions"
        ]
    )

    intervals = [
        wilson_interval(
            int(row.quality_failures),
            int(row.total_sessions),
            confidence=confidence,
        )
        for row in cell_stats.itertuples()
    ]

    cell_stats[
        "failure_rate_ci_low"
    ] = [
        interval[0]
        for interval in intervals
    ]

    cell_stats[
        "failure_rate_ci_high"
    ] = [
        interval[1]
        for interval in intervals
    ]

    # Require the entire lower confidence bound to clear the threshold
    # before treating a cell as statistically above the operational baseline.
    cell_stats[
        "significantly_above_threshold"
    ] = (
        cell_stats[
            "failure_rate_ci_low"
        ]
        >= failure_rate_threshold
    )

    cell_stats["problematic"] = (
        cell_stats[
            "failure_rate"
        ]
        >= failure_rate_threshold
    )

    def classify(
        row: pd.Series,
    ) -> str:
        """Assign a coverage or capacity-oriented operational triage class."""
        if (
            row["failure_rate"]
            < failure_rate_threshold
        ):
            return "Below threshold"

        if (
            row["median_signal_dbm"]
            < coverage_rsrp_threshold_dbm
        ):
            return "Coverage candidate"

        return "Capacity/congestion candidate"

    cell_stats[
        "triage_class"
    ] = cell_stats.apply(
        classify,
        axis=1,
    )

    # Use observed failure count as the impact measure for operational ranking.
    cell_stats[
        "impact_count"
    ] = cell_stats[
        "quality_failures"
    ]

    return (
        cell_stats
        .sort_values(
            [
                "impact_count",
                "failure_rate",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True,
        )
    )


def summarize_cell_results(
    cell_stats: pd.DataFrame,
    top_n: int = 50,
) -> dict:
    """Summarize cell severity, observed impact, and concentration."""
    active = len(
        cell_stats
    )

    problematic = int(
        cell_stats[
            "problematic"
        ].sum()
    )

    top = cell_stats.head(
        top_n
    )

    total_failures = int(
        cell_stats[
            "quality_failures"
        ].sum()
    )

    # Quantify how much observed failure volume is concentrated in the top cells.
    top_failure_share = (
        float(
            top[
                "quality_failures"
            ].sum()
            / total_failures
        )
        if total_failures > 0
        else 0.0
    )

    significant = int(
        cell_stats[
            "significantly_above_threshold"
        ].sum()
    )

    return {
        "active_cells": int(
            active
        ),
        "problematic_cells": problematic,
        "problematic_share": (
            problematic / active
            if active
            else 0.0
        ),
        "significantly_above_threshold_cells": (
            significant
        ),
        "significantly_above_threshold_share": (
            significant / active
            if active
            else 0.0
        ),
        "top_n": int(
            min(
                top_n,
                active,
            )
        ),
        "top_n_failure_share": (
            top_failure_share
        ),
        "total_failures": total_failures,
    }


def impact_severity_table(
    cell_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Separate cell impact ranking from failure-rate severity ranking."""
    result = cell_stats[
        [
            "cell_id",
            "total_sessions",
            "quality_failures",
            "failure_rate",
            "failure_rate_ci_low",
            "failure_rate_ci_high",
            "significantly_above_threshold",
            "median_signal_dbm",
            "triage_class",
        ]
    ].copy()

    result["impact_rank"] = (
        result[
            "quality_failures"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    result["severity_rank"] = (
        result[
            "failure_rate"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    return (
        result
        .sort_values(
            "impact_rank"
        )
        .reset_index(
            drop=True,
        )
    )


def get_top_cells(
    cell_stats: pd.DataFrame,
    top_n: int,
) -> set:
    """Return the top-N cells ranked by observed failure count."""
    return set(
        cell_stats.head(
            top_n
        )[
            "cell_id"
        ].tolist()
    )


def jaccard_similarity(
    set_a: set,
    set_b: set,
) -> float:
    """Compute Jaccard similarity between two cell sets."""
    union = set_a | set_b

    if not union:
        return 1.0

    return (
        len(set_a & set_b)
        / len(union)
    )


def classification_agreement(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    threshold: float,
) -> float:
    """
    Compare coverage/capacity assignments among cells above the
    same failure-rate threshold.
    """
    baseline_problematic = baseline[
        baseline["failure_rate"]
        >= threshold
    ].set_index(
        "cell_id"
    )

    candidate_problematic = candidate[
        candidate["failure_rate"]
        >= threshold
    ].set_index(
        "cell_id"
    )

    common = (
        baseline_problematic.index
        & candidate_problematic.index
    )

    if len(common) == 0:
        return np.nan

    baseline_class = (
        baseline_problematic.loc[
            common,
            "triage_class",
        ]
    )

    candidate_class = (
        candidate_problematic.loc[
            common,
            "triage_class",
        ]
    )

    # Reduce the operational classification to coverage vs. non-coverage.
    baseline_binary = (
        baseline_class
        == "Coverage candidate"
    )

    candidate_binary = (
        candidate_class
        == "Coverage candidate"
    )

    return float(
        (
            baseline_binary
            == candidate_binary
        ).mean()
    )


def threshold_sensitivity(
    df: pd.DataFrame,
    base_latency_ms: float,
    base_throughput_mbps: float,
    base_failure_rate: float,
    base_rsrp_dbm: float,
    throughput_urls: list[str],
    min_sessions: int,
    top_n: int,
    confidence: float,
    parameter: str,
    values: list[float],
) -> pd.DataFrame:
    """
    Run one-at-a-time threshold sensitivity analysis.

    Robustness is measured with the following parameter-specific quantities:

    latency / throughput:
        Top-N Jaccard similarity.

    failure_rate:
        Cells above threshold and cells statistically above threshold.

    rsrp:
        Coverage/capacity classification agreement.
    """
    baseline_session_df = add_session_failures(
        df,
        latency_failure_ms=base_latency_ms,
        throughput_failure_mbps=base_throughput_mbps,
        throughput_urls=throughput_urls,
    )

    baseline_cells = aggregate_cells(
        baseline_session_df,
        min_sessions=min_sessions,
        failure_rate_threshold=base_failure_rate,
        coverage_rsrp_threshold_dbm=base_rsrp_dbm,
        confidence=confidence,
    )

    baseline_top = get_top_cells(
        baseline_cells,
        top_n,
    )

    rows = []

    for value in values:
        latency = base_latency_ms
        throughput = base_throughput_mbps
        failure_rate = base_failure_rate
        rsrp = base_rsrp_dbm

        if parameter == "latency":
            latency = value

        elif parameter == "throughput":
            throughput = value

        elif parameter == "failure_rate":
            failure_rate = value

        elif parameter == "rsrp":
            rsrp = value

        else:
            raise ValueError(
                f"Unknown sensitivity parameter: {parameter}"
            )

        session_df = add_session_failures(
            df,
            latency_failure_ms=latency,
            throughput_failure_mbps=throughput,
            throughput_urls=throughput_urls,
        )

        cells = aggregate_cells(
            session_df,
            min_sessions=min_sessions,
            failure_rate_threshold=failure_rate,
            coverage_rsrp_threshold_dbm=rsrp,
            confidence=confidence,
        )

        summary = summarize_cell_results(
            cells,
            top_n=top_n,
        )

        row = {
            "parameter": parameter,
            "value": value,
            "active_cells": summary[
                "active_cells"
            ],
            "problematic_cells": summary[
                "problematic_cells"
            ],
            "problematic_share": summary[
                "problematic_share"
            ],
            "significantly_above_threshold_cells": (
                summary[
                    "significantly_above_threshold_cells"
                ]
            ),
            "significantly_above_threshold_share": (
                summary[
                    "significantly_above_threshold_share"
                ]
            ),
            "total_failures": summary[
                "total_failures"
            ],
        }

        if parameter in {
            "latency",
            "throughput",
        }:
            scenario_top = get_top_cells(
                cells,
                top_n,
            )

            row[
                "top_n_jaccard_vs_baseline"
            ] = jaccard_similarity(
                baseline_top,
                scenario_top,
            )

        elif parameter == "failure_rate":
            row[
                "top_n_jaccard_vs_baseline"
            ] = np.nan

        elif parameter == "rsrp":
            row[
                "top_n_jaccard_vs_baseline"
            ] = np.nan

            coverage_count = int(
                (
                    cells[
                        "triage_class"
                    ]
                    == "Coverage candidate"
                ).sum()
            )

            capacity_count = int(
                (
                    cells[
                        "triage_class"
                    ]
                    == "Capacity/congestion candidate"
                ).sum()
            )

            row[
                "coverage_candidates"
            ] = coverage_count

            row[
                "capacity_congestion_candidates"
            ] = capacity_count

            row[
                "triage_classification_agreement"
            ] = classification_agreement(
                baseline_cells,
                cells,
                threshold=base_failure_rate,
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )
