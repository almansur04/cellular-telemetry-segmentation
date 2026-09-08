from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(
    str(
        Path(__file__)
        .resolve()
        .parents[1]
    )
)

from src.exploratory import (
    correlations,
    dataset_summary,
    summarize_network_types,
    summarize_temporal_patterns,
    summarize_urls,
)
from src.io_utils import (
    ensure_output_dirs,
    load_config,
    load_raw_csv,
    save_csv,
    save_dataframe,
    save_json,
)
from src.plotting import (
    plot_framework,
    plot_url_performance,
)
from src.preprocessing import (
    optimize_dtypes,
    parse_and_clean,
    validate_dataset,
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

    raw_path = config[
        "data"
    ]["csv_path"]

    print(
        f"Loading raw data from: {raw_path}"
    )

    df = load_raw_csv(
        raw_path
    )

    print(
        f"Raw shape: {df.shape}"
    )

    cleaned = parse_and_clean(
        df
    )

    cleaned = optimize_dtypes(
        cleaned
    )

    integrity = validate_dataset(
        cleaned
    )

    print(
        "\nDataset integrity:"
    )
    print(
        integrity
    )

    save_dataframe(
        cleaned,
        config["data"][
            "cleaned_parquet_path"
        ],
    )

    summary = dataset_summary(
        cleaned
    )

    save_json(
        summary,
        Path(
            config["output"][
                "metrics"
            ]
        )
        / "dataset_summary.json",
    )

    urls = summarize_urls(
        cleaned
    )

    networks = summarize_network_types(
        cleaned
    )

    hourly, weekday = (
        summarize_temporal_patterns(
            cleaned
        )
    )

    corr = correlations(
        cleaned
    )

    save_csv(
        urls,
        Path(
            config["output"][
                "tables"
            ]
        )
        / "url_summary.csv",
    )

    save_csv(
        networks,
        Path(
            config["output"][
                "tables"
            ]
        )
        / "network_summary.csv",
    )

    save_csv(
        hourly,
        Path(
            config["output"][
                "tables"
            ]
        )
        / "hourly_summary.csv",
    )

    save_csv(
        weekday,
        Path(
            config["output"][
                "tables"
            ]
        )
        / "weekday_summary.csv",
    )

    corr.to_csv(
        Path(
            config["output"][
                "tables"
            ]
        )
        / "correlations.csv"
    )

    plot_framework(
        Path(
            config["output"][
                "figures"
            ]
        )
        / "01_framework"
    )

    plot_url_performance(
        urls,
        Path(
            config["output"][
                "figures"
            ]
        )
        / "00_url_performance",
    )

    print(
        "\nPreprocessing and exploratory analysis completed."
    )

    print(
        "Cleaned data saved to:",
        config["data"][
            "cleaned_parquet_path"
        ],
    )


if __name__ == "__main__":
    main()