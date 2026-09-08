from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_publication_style() -> None:
    """Consistent restrained style for IEEE-style figures."""
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


def save_figure(fig: plt.Figure, path: str | Path) -> None:
    """Save both vector PDF and high-resolution PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = path.with_suffix(".pdf")
    png_path = path.with_suffix(".png")

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        png_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


def plot_framework(path: str | Path) -> None:
    """
    Minimal conceptual pipeline figure.
    Designed as a clean research-method figure rather than an infographic.
    """
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.2, "Raw\ntelemetry"),
        (2.2, "Cleaning and\nmeasurement checks"),
        (4.3, "Workload and regime\nsegmentation"),
        (6.5, "Device-model\nanalysis"),
        (8.2, "Cell impact and\ntriage"),
    ]

    for x, text in boxes:
        patch = mpl.patches.FancyBboxPatch(
            (x, 1.0),
            1.45,
            1.0,
            boxstyle="round,pad=0.025",
            linewidth=0.9,
            edgecolor="0.25",
            facecolor="0.97",
        )
        ax.add_patch(patch)
        ax.text(
            x + 0.725,
            1.5,
            text,
            ha="center",
            va="center",
            fontsize=8.2,
        )

    for start in [1.65, 3.65, 5.75, 7.95]:
        ax.annotate(
            "",
            xy=(start + 0.48, 1.5),
            xytext=(start, 1.5),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=0.9,
                color="0.25",
            ),
        )

    ax.text(
        5,
        0.45,
        "Goal: reduce measurement heterogeneity before network-quality prioritization",
        ha="center",
        fontsize=8.5,
    )

    save_figure(fig, path)


def plot_url_performance(url_stats: pd.DataFrame, path: str | Path) -> None:
    """Plot application/workload heterogeneity."""
    setup_publication_style()

    data = url_stats.sort_values(
        "avg_download_mbps",
        ascending=True,
    )

    labels = [
        label.replace("https://", "").rstrip("/")
        for label in data["url"]
    ]

    fig, ax = plt.subplots(figsize=(6.6, 3.6))

    ax.barh(
        labels,
        data["avg_download_mbps"],
        edgecolor="0.15",
        linewidth=0.6,
        color="0.72",
    )

    ax.set_xlabel("Mean application-level download speed (Mbps)")
    ax.set_title("Measured throughput varies strongly by web resource")
    ax.grid(axis="x")

    for i, value in enumerate(data["avg_download_mbps"]):
        ax.text(
            value + data["avg_download_mbps"].max() * 0.015,
            i,
            f"{value:.2f}",
            va="center",
            fontsize=7.5,
        )

    save_figure(fig, path)


def plot_device_boxplot(
    clean_room: pd.DataFrame,
    path: str | Path,
) -> None:
    """Publication-style device throughput distribution."""
    setup_publication_style()

    ordered = (
        clean_room.groupby("device_model", observed=True)[
            "page_download_speed"
        ]
        .median()
        .sort_values(ascending=False)
        .index
    )

    values = [
        clean_room.loc[
            clean_room["device_model"] == device,
            "page_download_speed",
        ].to_numpy()
        for device in ordered
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    box = ax.boxplot(
        values,
        patch_artist=True,
        showfliers=False,
        widths=0.62,
        medianprops=dict(color="0.05", linewidth=1.1),
        whiskerprops=dict(color="0.25", linewidth=0.8),
        capprops=dict(color="0.25", linewidth=0.8),
        boxprops=dict(
            facecolor="0.88",
            edgecolor="0.25",
            linewidth=0.8,
        ),
    )

    ax.set_xticks(range(1, len(ordered) + 1))
    ax.set_xticklabels(
        ordered,
        rotation=42,
        ha="right",
    )

    ax.set_ylabel("Page download speed (Mbps)")
    ax.set_xlabel("Device model")
    ax.set_title(
        "Device-model throughput under a restricted favorable operating regime"
    )

    ax.grid(axis="y")

    save_figure(fig, path)


def plot_dunn_heatmap(
    dunn_matrix: pd.DataFrame,
    path: str | Path,
) -> None:
    """
    Plot -log10 adjusted p-values instead of raw p-values.

    This avoids a matrix filled with visually meaningless '0.000' labels.
    """
    setup_publication_style()

    matrix = dunn_matrix.copy().astype(float)
    values = -np.log10(np.clip(matrix, 1e-300, 1.0))

    mask = np.triu(np.ones_like(values, dtype=bool))

    fig, ax = plt.subplots(figsize=(6.6, 5.8))

    image = ax.imshow(
        np.ma.array(values, mask=mask),
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",
    )

    ax.set_xticks(range(len(values.columns)))
    ax.set_yticks(range(len(values.index)))

    ax.set_xticklabels(
        values.columns,
        rotation=45,
        ha="right",
    )
    ax.set_yticklabels(values.index)

    ax.set_title(
        "Pairwise Dunn tests with Bonferroni correction"
    )

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(r"$-\log_{10}(p_{\mathrm{adj}})$")

    threshold = -np.log10(0.05)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if i <= j:
                continue

            value = values.iloc[i, j]

            if value >= threshold:
                ax.text(
                    j,
                    i,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

    save_figure(fig, path)


def plot_cell_pareto(
    cell_stats: pd.DataFrame,
    path: str | Path,
    top_n: int = 50,
) -> None:
    """
    Pareto-style impact plot without arbitrary 80% cutoff.
    """
    setup_publication_style()

    data = (
        cell_stats.sort_values(
            "sla_failures",
            ascending=False,
        )
        .head(top_n)
        .copy()
    )

    data["rank"] = np.arange(1, len(data) + 1)

    total_failures = cell_stats["sla_failures"].sum()
    data["cum_share"] = (
        data["sla_failures"].cumsum()
        / total_failures
        * 100
    )

    fig, ax1 = plt.subplots(figsize=(7.0, 4.0))

    ax1.bar(
        data["rank"],
        data["sla_failures"],
        width=0.75,
        color="0.72",
        edgecolor="0.2",
        linewidth=0.5,
    )

    ax1.set_xlabel("Cell rank by observed failure count")
    ax1.set_ylabel("Observed quality failures")
    ax1.grid(axis="y")

    ax2 = ax1.twinx()

    ax2.plot(
        data["rank"],
        data["cum_share"],
        linewidth=1.3,
        marker="o",
        markersize=2.8,
        color="0.10",
    )

    ax2.set_ylabel("Cumulative share of failures (%)")
    ax2.set_ylim(0, max(100, data["cum_share"].max() * 1.12))

    ax1.set_title(
        f"Failure concentration among the {len(data)} highest-impact cells"
    )

    save_figure(fig, path)


def plot_impact_vs_severity(
    cell_stats: pd.DataFrame,
    path: str | Path,
) -> None:
    """Separate cell severity from user-impact volume."""
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    healthy = cell_stats["failure_rate"] < 0.05

    ax.scatter(
        cell_stats.loc[healthy, "failure_rate"] * 100,
        cell_stats.loc[healthy, "sla_failures"],
        s=18,
        facecolors="none",
        edgecolors="0.45",
        linewidths=0.7,
        label="Below threshold",
    )

    ax.scatter(
        cell_stats.loc[~healthy, "failure_rate"] * 100,
        cell_stats.loc[~healthy, "sla_failures"],
        s=22,
        facecolors="0.30",
        edgecolors="0.05",
        linewidths=0.5,
        label="Above threshold",
    )

    ax.axvline(
        5,
        linestyle="--",
        linewidth=0.8,
        color="0.25",
    )

    ax.set_xlabel("Cell failure rate (%)")
    ax.set_ylabel("Observed failure count")
    ax.set_title("Cell severity versus user-impact volume")
    ax.grid(True)
    ax.legend(frameon=False)

    save_figure(fig, path)


def plot_ml_importance(
    importance: pd.DataFrame,
    path: str | Path,
) -> None:
    """Plot grouped permutation importance."""
    setup_publication_style()

    data = importance.sort_values(
        "importance_mean_r2_drop",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    ax.barh(
        data["feature"],
        data["importance_mean_r2_drop"],
        xerr=data["importance_std_r2_drop"],
        color="0.70",
        edgecolor="0.2",
        linewidth=0.5,
        capsize=2,
    )

    ax.set_xlabel(r"Permutation importance: decrease in $R^2$")
    ax.set_title("Held-out predictive importance of telemetry variables")
    ax.grid(axis="x")

    save_figure(fig, path)
