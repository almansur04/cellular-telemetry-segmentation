from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Make repository-level src imports available when running this script directly.
sys.path.append(
    str(
        Path(__file__)
        .resolve()
        .parents[1]
    )
)

from src.cell_analysis import (
    add_session_failures,
    aggregate_cells,
    impact_severity_table,
    summarize_cell_results,
)
from src.io_utils import (
    ensure_output_dirs,
    load_config,
    load_cleaned_parquet,
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

    config = load_config(
        args.config
    )

    ensure_output_dirs(
        config
    )

    # Reuse the cleaned dataset produced by the shared preprocessing pipeline.
    df = load_cleaned_parquet(
        config["data"][
            "cleaned_parquet_path"
        ]
    )

    ca = config[
        "cell_analysis"
    ]

    # Convert session telemetry into deterministic latency and throughput
    # failure indicators using the configured operational thresholds.
    df = add_session_failures(
        df,
        latency_failure_ms=ca[
            "latency_failure_ms"
        ],
        throughput_failure_mbps=ca[
            "throughput_failure_mbps"
        ],
        throughput_urls=ca[
            "throughput_urls"
        ],
    )

    # Aggregate session-level failures into cell-level severity and
    # coverage/capacity triage metrics.
    cell_stats = aggregate_cells(
        df,
        min_sessions=ca[
            "minimum_sessions_per_cell"
        ],
        failure_rate_threshold=ca[
            "failure_rate_threshold"
        ],
        coverage_rsrp_threshold_dbm=ca[
            "rsrp_coverage_threshold_dbm"
        ],
        confidence=ca[
            "confidence_level"
        ],
    )

    summary = summarize_cell_results(
        cell_stats,
        top_n=ca[
            "top_n_cells"
        ],
    )

    # Keep impact ranking and failure-rate severity available as separate
    # views so operational volume is not conflated with proportional severity.
    table = impact_severity_table(
        cell_stats
    )

    # Persist full-resolution outputs for reproducible downstream analysis.
    save_csv(
        cell_stats,
        Path(
            config["output"][
                "tables"
            ]
        )
        / "cell_statistics.csv",
    )

    save_csv(
        table,
        Path(
            config["output"][
                "tables"
            ]
        )
        / "cell_impact_severity.csv",
    )

    save_json(
        summary,
        Path(
            config["output"][
                "metrics"
            ]
        )
        / "cell_summary.json",
    )

    # Generate complementary views of failure concentration and cell severity.
    plot_cell_pareto(
        cell_stats,
        Path(
            config["output"][
                "figures"
            ]
        )
        / "06_cell_failure_concentration",
        top_n=ca[
            "top_n_cells"
        ],
    )

    plot_impact_vs_severity(
        cell_stats,
        Path(
            config["output"][
                "figures"
            ]
        )
        / "07_cell_impact_vs_severity",
    )

    print(
        "\nCell summary:"
    )

    print(
        summary
    )

    print(
        "\nTop 10 cells:"
    )

    print(
        table.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
