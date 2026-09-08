from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.device_analysis import (
    create_device_subset,
    device_median_confidence_intervals,
    device_summary,
    dunn_test,
    kruskal_wallis_result,
    pairwise_effect_sizes,
)
from src.io_utils import (
    ensure_output_dirs,
    load_config,
    save_csv,
    save_json,
)
from src.plotting import (
    plot_device_boxplot,
    plot_dunn_heatmap,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_output_dirs(config)

    df = pd.read_parquet(
        config["data"]["cleaned_parquet_path"]
    )

    da = config["device_analysis"]

    clean_room = create_device_subset(
        df,
        reference_url=da["reference_url"],
        allowed_networks=da["allowed_networks"],
        rsrp_min_dbm=da["rsrp_min_dbm"],
        min_device_sessions=da["min_device_sessions"],
    )

    summary = device_summary(clean_room)
    ci = device_median_confidence_intervals(clean_room)
    kw = kruskal_wallis_result(clean_room)
    dunn = dunn_test(clean_room)
    effects = pairwise_effect_sizes(clean_room)

    save_csv(
        summary,
        Path(config["output"]["tables"]) / "device_summary.csv",
    )

    save_csv(
        ci,
        Path(config["output"]["tables"]) / "device_median_ci.csv",
    )

    save_csv(
        effects,
        Path(config["output"]["tables"]) / "device_cliffs_delta.csv",
    )

    dunn.to_csv(
        Path(config["output"]["tables"]) / "dunn_bonferroni.csv"
    )

    save_json(
        kw,
        Path(config["output"]["metrics"]) / "kruskal_wallis.json",
    )

    plot_device_boxplot(
        clean_room,
        Path(config["output"]["figures"]) / "02_device_throughput",
    )

    plot_dunn_heatmap(
        dunn,
        Path(config["output"]["figures"]) / "03_dunn_bonferroni",
    )

    print("\nRestricted device-analysis subset:")
    print(f"{len(clean_room):,} sessions")

    print("\nKruskal-Wallis:")
    print(kw)

    print("\nDevice summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
