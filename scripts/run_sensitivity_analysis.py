from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# Make repository-level src imports available when running this script directly.
sys.path.append(
    str(
        Path(__file__)
        .resolve()
        .parents[1]
    )
)

from src.cell_analysis import (
    threshold_sensitivity,
)
from src.io_utils import (
    ensure_output_dirs,
    load_config,
    load_cleaned_parquet,
    save_csv,
)
from src.plotting import (
    plot_threshold_sensitivity,
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

    # Use the cleaned Parquet artifact produced by the main analysis pipeline.
    df = load_cleaned_parquet(
        config["data"][
            "cleaned_parquet_path"
        ]
    )

    ca = config[
        "cell_analysis"
    ]

    sensitivity_cfg = ca[
        "sensitivity"
    ]

    # Evaluate one threshold family at a time while holding all other
    # cell-analysis parameters at their configured baseline values.
    scenarios = [
        (
            "latency",
            sensitivity_cfg[
                "latency_thresholds_ms"
            ],
        ),
        (
            "throughput",
            sensitivity_cfg[
                "throughput_thresholds_mbps"
            ],
        ),
        (
            "failure_rate",
            sensitivity_cfg[
                "failure_rate_thresholds"
            ],
        ),
        (
            "rsrp",
            sensitivity_cfg[
                "rsrp_thresholds_dbm"
            ],
        ),
    ]

    all_results = []

    for parameter, values in scenarios:
        print(
            f"\nSensitivity analysis: {parameter}"
        )

        result = threshold_sensitivity(
            df,
            base_latency_ms=ca[
                "latency_failure_ms"
            ],
            base_throughput_mbps=ca[
                "throughput_failure_mbps"
            ],
            base_failure_rate=ca[
                "failure_rate_threshold"
            ],
            base_rsrp_dbm=ca[
                "rsrp_coverage_threshold_dbm"
            ],
            throughput_urls=ca[
                "throughput_urls"
            ],
            min_sessions=ca[
                "minimum_sessions_per_cell"
            ],
            top_n=ca[
                "top_n_cells"
            ],
            confidence=ca[
                "confidence_level"
            ],
            parameter=parameter,
            values=values,
        )

        all_results.append(
            result
        )

        output_name = (
            f"sensitivity_{parameter}.csv"
        )

        # Persist parameter-specific results alongside the corresponding figure.
        save_csv(
            result,
            Path(
                config["output"][
                    "tables"
                ]
            )
            / output_name,
        )

        plot_threshold_sensitivity(
            result,
            Path(
                config["output"][
                    "figures"
                ]
            )
            / f"09_sensitivity_{parameter}",
        )

        print(
            result.to_string(
                index=False
            )
        )

    # Combine all one-at-a-time sensitivity runs into a single analysis artifact.
    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    save_csv(
        combined,
        Path(
            config["output"][
                "tables"
            ]
        )
        / "threshold_sensitivity_all.csv",
    )

    print(
        "\nCorrected threshold sensitivity analysis completed."
    )


if __name__ == "__main__":
    main()
