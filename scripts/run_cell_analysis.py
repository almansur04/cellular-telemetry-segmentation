from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cell_analysis import (
    add_session_failures,
    aggregate_cells,
    impact_severity_table,
    summarize_cell_results,
)
from src.io_utils import (
    ensure_output_dirs,
    load_config,
    save_csv,
    save_json,
)
from src.plotting import (
    plot_cell_pareto,
    plot_impact_vs_severity,
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

    ca = config["cell_analysis"]

    df = add_session_failures(
        df,
        latency_failure_ms=ca["latency_failure_ms"],
        throughput_failure_mbps=ca["throughput_failure_mbps"],
        throughput_urls=ca["throughput_urls"],
    )

    cell_stats = aggregate_cells(
        df,
        min_sessions=ca["minimum_sessions_per_cell"],
        failure_rate_threshold=ca["failure_rate_threshold"],
        coverage_rsrp_threshold_dbm=ca["rsrp_coverage_threshold_dbm"],
    )

    summary = summarize_cell_results(
        cell_stats,
        top_n=ca["top_n_cells"],
    )

    table = impact_severity_table(cell_stats)

    save_csv(
        cell_stats,
        Path(config["output"]["tables"]) / "cell_statistics.csv",
    )

    save_csv(
        table,
        Path(config["output"]["tables"]) / "cell_impact_severity.csv",
    )

    save_json(
        summary,
        Path(config["output"]["metrics"]) / "cell_summary.json",
    )

    plot_cell_pareto(
        cell_stats,
        Path(config["output"]["figures"]) / "05_cell_failure_concentration",
        top_n=ca["top_n_cells"],
    )

    plot_impact_vs_severity(
        cell_stats,
        Path(config["output"]["figures"]) / "06_cell_impact_vs_severity",
    )

    print("\nCell summary:")
    print(summary)

    print("\nTop 10 cells:")
    print(
        table.head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
