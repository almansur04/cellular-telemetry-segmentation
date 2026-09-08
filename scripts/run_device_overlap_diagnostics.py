from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(
    str(
        Path(__file__).resolve().parents[1]
    )
)

from src.device_analysis import (
    create_device_subset,
)
from src.io_utils import (
    ensure_output_dirs,
    load_config,
    load_cleaned_parquet,
    save_csv,
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

    df = load_cleaned_parquet(
        config["data"][
            "cleaned_parquet_path"
        ]
    )

    da = config[
        "device_analysis"
    ]

    clean_room = create_device_subset(
        df,
        reference_url=da[
            "reference_url"
        ],
        allowed_networks=da[
            "allowed_networks"
        ],
        rsrp_min_dbm=da[
            "rsrp_min_dbm"
        ],
        min_device_sessions=da[
            "min_device_sessions"
        ],
    )

    # --------------------------------------------------------
    # Device x Cell counts
    # --------------------------------------------------------
    counts = pd.crosstab(
        clean_room["cell_id"],
        clean_room["device_model"],
    )

    save_csv(
        counts.reset_index(),
        Path(
            config["output"][
                "tables"
            ]
        )
        / "device_cell_counts.csv",
    )

    # Number of distinct device models observed per cell.
    device_models_per_cell = (
        clean_room.groupby(
            "cell_id",
            observed=True,
        )["device_model"]
        .nunique()
    )

    overlap_summary = (
        device_models_per_cell
        .value_counts()
        .sort_index()
        .rename_axis(
            "device_models_in_cell"
        )
        .reset_index(
            name="number_of_cells"
        )
    )

    save_csv(
        overlap_summary,
        Path(
            config["output"][
                "tables"
            ]
        )
        / "cell_device_overlap_summary.csv",
    )

    # --------------------------------------------------------
    # Device-pair co-occurrence
    # --------------------------------------------------------
    devices = sorted(
        clean_room[
            "device_model"
        ].astype(str).unique()
    )

    rows = []

    for i, device_a in enumerate(
        devices
    ):
        for device_b in devices[
            i + 1:
        ]:
            cells_a = set(
                counts.index[
                    counts[device_a] > 0
                ]
            )

            cells_b = set(
                counts.index[
                    counts[device_b] > 0
                ]
            )

            common = (
                cells_a
                & cells_b
            )

            union = (
                cells_a
                | cells_b
            )

            jaccard = (
                len(common)
                / len(union)
                if union
                else 0.0
            )

            rows.append(
                {
                    "device_a": device_a,
                    "device_b": device_b,
                    "cells_device_a": len(
                        cells_a
                    ),
                    "cells_device_b": len(
                        cells_b
                    ),
                    "common_cells": len(
                        common
                    ),
                    "jaccard_cell_overlap": jaccard,
                }
            )

    pairwise = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "common_cells",
                "jaccard_cell_overlap",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    save_csv(
        pairwise,
        Path(
            config["output"][
                "tables"
            ]
        )
        / "device_pair_cell_overlap.csv",
    )

    # --------------------------------------------------------
    # Basic summary
    # --------------------------------------------------------
    cells_total = len(
        device_models_per_cell
    )

    cells_with_multiple_devices = int(
        (
            device_models_per_cell
            >= 2
        ).sum()
    )

    print(
        "\nDevice/cell overlap diagnostics"
    )

    print(
        f"Restricted sessions: "
        f"{len(clean_room):,}"
    )

    print(
        f"Cells represented: "
        f"{cells_total:,}"
    )

    print(
        "Cells containing >=2 device models: "
        f"{cells_with_multiple_devices:,} "
        f"({cells_with_multiple_devices / cells_total * 100:.1f}%)"
    )

    print(
        "\nNumber of device models per cell:"
    )

    print(
        overlap_summary.to_string(
            index=False
        )
    )

    print(
        "\nHighest-overlap device pairs:"
    )

    print(
        pairwise.head(20).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()