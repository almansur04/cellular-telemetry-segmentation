from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_publication_style() -> None:
    """Set restrained publication-style plotting defaults."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.22,
        }
    )


def save_figure(
    fig: plt.Figure,
    path: str | Path,
) -> None:
    """Save PDF and PNG copies."""
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        path.with_suffix(".png"),
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


def plot_framework(
    path: str | Path,
) -> None:
    """Draw the research workflow."""
    setup_publication_style()

    fig, ax = plt.subplots(
        figsize=(7.0, 2.4)
    )

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (
            0.2,
            "Raw\ntelemetry",
        ),
        (
            2.3,
            "Cleaning and\nvalidation",
        ),
        (
            4.4,
            "Workload and\nregime segmentation",
        ),
        (
            6.7,
            "Device-model\nanalysis",
        ),
        (
            8.8,
            "Session quality\nand cell triage",
        ),
    ]

    for x, text in boxes:
        patch = mpl.patches.FancyBboxPatch(
            (x, 1.0),
            1.55,
            1.05,
            boxstyle="round,pad=0.025",
            linewidth=0.9,
            edgecolor="0.25",
            facecolor="0.96",
        )

        ax.add_patch(
            patch
        )

        ax.text(
            x + 0.775,
            1.53,
            text,
            ha="center",
            va="center",
            fontsize=8.1,
        )

    arrow_positions = [
        (1.75, 2.3),
        (3.85, 4.4),
        (5.95, 6.7),
        (8.25, 8.8),
    ]

    for x1, x2 in arrow_positions:
        ax.annotate(
            "",
            xy=(x2, 1.53),
            xytext=(x1, 1.53),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=0.9,
                color="0.25",
            ),
        )

    ax.text(
        5.5,
        0.42,
        "Goal: reduce measurement heterogeneity before operational prioritization",
        ha="center",
        fontsize=8.5,
    )

    save_figure(
        fig,
        path,
    )


def plot_url_performance(
    url_stats: pd.DataFrame,
    path: str | Path,
) -> None:
    """Plot workload heterogeneity."""
    setup_publication_style()

    data = (
        url_stats
        .sort_values(
            "mean_download_mbps",
            ascending=True,
        )
        .copy()
    )

    labels = [
        str(label)
        .replace(
            "https://",
            "",
        )
        .rstrip("/")
        for label in data["url"]
    ]

    fig, ax = plt.subplots(
        figsize=(6.6, 3.7)
    )

    bars = ax.barh(
        labels,
        data["mean_download_mbps"],
        color="0.72",
        edgecolor="0.18",
        linewidth=0.6,
    )

    ax.set_xlabel(
        "Mean application-level download speed (Mbps)"
    )

    ax.set_title(
        "Application-level throughput varies by web resource"
    )

    ax.grid(
        axis="x"
    )

    max_value = data[
        "mean_download_mbps"
    ].max()

    for bar, value in zip(
        bars,
        data[
            "mean_download_mbps"
        ],
    ):
        ax.text(
            bar.get_width()
            + max_value * 0.015,
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=7.5,
        )

    save_figure(
        fig,
        path,
    )


def plot_device_boxplot(
    clean_room: pd.DataFrame,
    path: str | Path,
) -> None:
    """Plot device-model distributions."""
    setup_publication_style()

    ordered = (
        clean_room
        .groupby(
            "device_model",
            observed=True,
        )[
            "page_download_speed"
        ]
        .median()
        .sort_values(
            ascending=False
        )
        .index
    )

    values = [
        clean_room.loc[
            clean_room[
                "device_model"
            ] == device,
            "page_download_speed",
        ].to_numpy()
        for device in ordered
    ]

    fig, ax = plt.subplots(
        figsize=(7.0, 4.0)
    )

    ax.boxplot(
        values,
        patch_artist=True,
        showfliers=False,
        widths=0.62,
        medianprops={
            "color": "0.05",
            "linewidth": 1.1,
        },
        whiskerprops={
            "color": "0.25",
            "linewidth": 0.8,
        },
        capprops={
            "color": "0.25",
            "linewidth": 0.8,
        },
        boxprops={
            "facecolor": "0.88",
            "edgecolor": "0.25",
            "linewidth": 0.8,
        },
    )

    ax.set_xticks(
        range(
            1,
            len(ordered) + 1,
        )
    )

    ax.set_xticklabels(
        ordered,
        rotation=42,
        ha="right",
    )

    ax.set_xlabel(
        "Device model"
    )

    ax.set_ylabel(
        "Page download speed (Mbps)"
    )

    ax.set_title(
        "Device-model throughput in the restricted operating regime"
    )

    ax.grid(
        axis="y"
    )

    save_figure(
        fig,
        path,
    )


def plot_dunn_heatmap(
    dunn_matrix: pd.DataFrame,
    path: str | Path,
) -> None:
    """Plot lower-triangular Dunn-test evidence."""

    setup_publication_style()

    matrix = (
        dunn_matrix
        .astype(float)
        .copy()
    )

    values = -np.log10(
        np.clip(
            matrix,
            1e-300,
            1.0,
        )
    )

    mask = np.triu(
        np.ones_like(
            values,
            dtype=bool,
        )
    )

    fig, ax = plt.subplots(
        figsize=(6.7, 5.8)
    )

    image = ax.imshow(
        np.ma.array(
            values,
            mask=mask,
        ),
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",
    )

    ax.set_xticks(
        range(
            len(values.columns)
        )
    )

    ax.set_yticks(
        range(
            len(values.index)
        )
    )

    ax.set_xticklabels(
        values.columns,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        values.index
    )

    ax.set_title(
        "Pairwise Dunn tests with Bonferroni correction"
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.035,
        pad=0.03,
    )

    colorbar.set_label(
        r"$-\log_{10}(p_{\mathrm{adj}})$"
    )

    threshold = -np.log10(
        0.05
    )

    for i in range(
        values.shape[0]
    ):
        for j in range(
            values.shape[1]
        ):
            if i <= j:
                continue

            if (
                values.iloc[i, j]
                >= threshold
            ):
                ax.text(
                    j,
                    i,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

    save_figure(
        fig,
        path,
    )


def plot_adjusted_device_effects(
    effects: pd.DataFrame,
    path: str | Path,
) -> None:
    """
    Plot cell-adjusted device-model coefficients with 95% CIs.

    Coefficients are on the log1p throughput scale.
    """
    setup_publication_style()

    data = effects[
        effects[
            "device_model"
        ]
        != effects[
            "reference_device"
        ]
    ].copy()

    data = data.sort_values(
        "coefficient_log1p"
    )

    fig, ax = plt.subplots(
        figsize=(6.8, 4.4)
    )

    positions = np.arange(
        len(data)
    )

    x = data[
        "coefficient_log1p"
    ].to_numpy()

    low = data[
        "ci_low_log1p"
    ].to_numpy()

    high = data[
        "ci_high_log1p"
    ].to_numpy()

    left_error = x - low
    right_error = high - x

    ax.errorbar(
        x,
        positions,
        xerr=[
            left_error,
            right_error,
        ],
        fmt="o",
        markersize=4.2,
        color="0.15",
        ecolor="0.35",
        elinewidth=1.0,
        capsize=2.5,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=0.8,
        color="0.25",
    )

    ax.set_yticks(
        positions
    )

    ax.set_yticklabels(
        data[
            "device_model"
        ]
    )

    ax.set_xlabel(
        r"Cell-adjusted coefficient on $\log(1+\mathrm{speed})$"
    )

    ax.set_ylabel(
        "Device model"
    )

    ax.set_title(
        "Device-model associations after cell adjustment"
    )

    ax.grid(
        axis="x"
    )

    fig.text(
        0.01,
        0.005,
        "Reference device: SM-A266B; "
        "95% CIs use standard errors clustered by serving cell.",
        fontsize=7.5,
    )

    save_figure(
        fig,
        path,
    )


def plot_cell_pareto(
    cell_stats: pd.DataFrame,
    path: str | Path,
    top_n: int = 50,
) -> None:
    """Plot failure concentration among highest-impact cells."""

    setup_publication_style()

    data = (
        cell_stats
        .sort_values(
            "quality_failures",
            ascending=False,
        )
        .head(top_n)
        .copy()
    )

    data["rank"] = np.arange(
        1,
        len(data) + 1,
    )

    total_failures = (
        cell_stats[
            "quality_failures"
        ].sum()
    )

    data["cum_share"] = (
        data[
            "quality_failures"
        ].cumsum()
        / total_failures
        * 100
    )

    fig, ax1 = plt.subplots(
        figsize=(7.0, 4.0)
    )

    ax1.bar(
        data["rank"],
        data["quality_failures"],
        width=0.75,
        color="0.72",
        edgecolor="0.2",
        linewidth=0.5,
    )

    ax1.set_xlabel(
        "Cell rank by observed failure count"
    )

    ax1.set_ylabel(
        "Observed quality failures"
    )

    ax1.grid(
        axis="y"
    )

    ax2 = ax1.twinx()

    ax2.plot(
        data["rank"],
        data["cum_share"],
        linewidth=1.3,
        marker="o",
        markersize=2.8,
        color="0.10",
    )

    ax2.set_ylabel(
        "Cumulative share of all failures (%)"
    )

    ax2.set_ylim(
        0,
        max(
            100,
            data[
                "cum_share"
            ].max()
            * 1.12,
        ),
    )

    ax1.set_title(
        f"Failure concentration among the {len(data)} highest-impact cells"
    )

    save_figure(
        fig,
        path,
    )


def plot_impact_vs_severity(
    cell_stats: pd.DataFrame,
    path: str | Path,
) -> None:
    """Separate failure severity from observed impact volume."""

    setup_publication_style()

    fig, ax = plt.subplots(
        figsize=(6.6, 4.2)
    )

    below = (
        cell_stats[
            "failure_rate"
        ]
        < 0.05
    )

    ax.scatter(
        cell_stats.loc[
            below,
            "failure_rate",
        ]
        * 100,
        cell_stats.loc[
            below,
            "quality_failures",
        ],
        s=18,
        facecolors="none",
        edgecolors="0.45",
        linewidths=0.7,
        label="Below 5% threshold",
    )

    ax.scatter(
        cell_stats.loc[
            ~below,
            "failure_rate",
        ]
        * 100,
        cell_stats.loc[
            ~below,
            "quality_failures",
        ],
        s=22,
        facecolors="0.30",
        edgecolors="0.05",
        linewidths=0.5,
        label="At or above 5%",
    )

    ax.axvline(
        5,
        linestyle="--",
        linewidth=0.8,
        color="0.25",
    )

    ax.set_xlabel(
        "Cell failure rate (%)"
    )

    ax.set_ylabel(
        "Observed quality failures"
    )

    ax.set_title(
        "Cell severity versus observed user-impact volume"
    )

    ax.grid(
        True
    )

    ax.legend(
        frameon=False
    )

    save_figure(
        fig,
        path,
    )


def plot_ml_importance(
    importance: pd.DataFrame,
    path: str | Path,
) -> None:
    """Plot grouped held-out permutation importance."""

    setup_publication_style()

    data = (
        importance
        .sort_values(
            "r2_decrease",
            ascending=True,
        )
        .copy()
    )

    fig, ax = plt.subplots(
        figsize=(6.7, 3.9)
    )

    ax.barh(
        data["feature"],
        data["r2_decrease"],
        xerr=data[
            "permuted_r2_std"
        ],
        color="0.70",
        edgecolor="0.2",
        linewidth=0.5,
        capsize=2,
    )

    ax.set_xlabel(
        r"Decrease in held-out $R^2$ after permutation"
    )

    ax.set_ylabel(
        "Telemetry variable"
    )

    ax.set_title(
        "Held-out predictive importance"
    )

    ax.grid(
        axis="x"
    )

    save_figure(
        fig,
        path,
    )


def plot_threshold_sensitivity(
    sensitivity: pd.DataFrame,
    path: str | Path,
) -> None:
    """
    Plot stability of the top-50 cell set under
    one-at-a-time threshold changes.
    """
    setup_publication_style()

    data = sensitivity.copy()

    data["label"] = data[
        "value"
    ].map(
        lambda x: f"{x:g}"
    )

    fig, ax = plt.subplots(
        figsize=(6.7, 3.8)
    )

    x = np.arange(
        len(data)
    )

    ax.plot(
        x,
        data[
            "top_n_jaccard_vs_baseline"
        ],
        marker="o",
        markersize=4,
        linewidth=1.2,
        color="0.15",
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=0.7,
        color="0.35",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        data["label"]
    )

    ax.set_xlabel(
        "Alternative threshold value"
    )

    ax.set_ylabel(
        "Top-50 Jaccard similarity vs. baseline"
    )

    parameter = (
        data["parameter"]
        .iloc[0]
    )

    ax.set_title(
        f"Top-50 cell ranking sensitivity to {parameter} threshold"
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.grid(
        axis="y"
    )

    save_figure(
        fig,
        path,
    )