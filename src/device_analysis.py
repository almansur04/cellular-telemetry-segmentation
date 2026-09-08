from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp


def create_device_subset(
    df: pd.DataFrame,
    reference_url: str,
    allowed_networks: list[str],
    rsrp_min_dbm: float,
    min_device_sessions: int,
) -> pd.DataFrame:
    """
    Construct the restricted device-comparison regime.
    """
    subset = df[
        (df["url"] == reference_url)
        & (
            df["network_type"].isin(
                allowed_networks
            )
        )
        & (
            df["signal_strength"]
            >= rsrp_min_dbm
        )
        & (
            df["speed_test_done"] == 1
        )
    ].copy()

    counts = (
        subset["device_model"]
        .value_counts()
    )

    valid_devices = counts[
        counts > min_device_sessions
    ].index

    subset = subset[
        subset["device_model"].isin(
            valid_devices
        )
    ].copy()

    return subset


def kruskal_wallis_result(
    df: pd.DataFrame,
) -> dict[str, float]:
    """Run the omnibus Kruskal-Wallis test."""

    groups = [
        group[
            "page_download_speed"
        ].to_numpy()
        for _, group
        in df.groupby(
            "device_model",
            observed=True,
        )
    ]

    statistic, p_value = stats.kruskal(
        *groups
    )

    n = len(df)
    k = len(groups)

    epsilon_squared = (
        statistic - k + 1
    ) / (n - k)

    return {
        "n": float(n),
        "groups": float(k),
        "H": float(statistic),
        "p_value": float(p_value),
        "epsilon_squared": float(
            epsilon_squared
        ),
    }


def dunn_test(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Pairwise Dunn test with Bonferroni correction."""
    return sp.posthoc_dunn(
        df,
        val_col="page_download_speed",
        group_col="device_model",
        p_adjust="bonferroni",
    )


def cliffs_delta(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Compute Cliff's delta.

    Positive values indicate that x tends
    to exceed y.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    greater = 0
    less = 0

    for value in x:
        greater += np.sum(
            value > y
        )
        less += np.sum(
            value < y
        )

    denominator = (
        len(x) * len(y)
    )

    return (
        greater - less
    ) / denominator


def pairwise_effect_sizes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pairwise Cliff's delta."""

    groups = {
        name: group[
            "page_download_speed"
        ].to_numpy()
        for name, group
        in df.groupby(
            "device_model",
            observed=True,
        )
    }

    rows = []

    for device_a, device_b in itertools.combinations(
        groups.keys(),
        2,
    ):
        delta = cliffs_delta(
            groups[device_a],
            groups[device_b],
        )

        rows.append(
            {
                "device_a": device_a,
                "device_b": device_b,
                "cliffs_delta": delta,
                "absolute_cliffs_delta": abs(
                    delta
                ),
            }
        )

    return pd.DataFrame(rows)


def device_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Median, quartiles, IQR and sample size by device."""

    result = (
        df.groupby(
            "device_model",
            observed=True,
        )["page_download_speed"]
        .agg(
            median_speed_mbps="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
            sessions="count",
        )
        .reset_index()
    )

    result["iqr_mbps"] = (
        result["q3"]
        - result["q1"]
    )

    best_median = (
        result["median_speed_mbps"]
        .max()
    )

    result["relative_to_best_pct"] = (
        100
        * (
            1
            - result[
                "median_speed_mbps"
            ]
            / best_median
        )
    )

    return (
        result
        .sort_values(
            "median_speed_mbps",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def bootstrap_median_ci(
    values: np.ndarray,
    iterations: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval for a median."""

    rng = np.random.default_rng(
        random_state
    )

    values = np.asarray(values)
    sample_size = len(values)

    medians = np.empty(
        iterations
    )

    for i in range(iterations):
        sample = rng.choice(
            values,
            size=sample_size,
            replace=True,
        )

        medians[i] = np.median(
            sample
        )

    alpha = (
        1 - confidence
    ) / 2

    return (
        float(
            np.quantile(
                medians,
                alpha,
            )
        ),
        float(
            np.quantile(
                medians,
                1 - alpha,
            )
        ),
    )


def device_median_confidence_intervals(
    df: pd.DataFrame,
    iterations: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Bootstrap 95% CIs for device medians."""

    rows = []

    for device, group in df.groupby(
        "device_model",
        observed=True,
    ):
        low, high = (
            bootstrap_median_ci(
                group[
                    "page_download_speed"
                ].to_numpy(),
                iterations=iterations,
                confidence=confidence,
                random_state=random_state,
            )
        )

        rows.append(
            {
                "device_model": device,
                "median_mbps": group[
                    "page_download_speed"
                ].median(),
                "ci_low_mbps": low,
                "ci_high_mbps": high,
                "sessions": int(
                    len(group)
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "median_mbps",
            ascending=False,
        )
        .reset_index(drop=True)
    )