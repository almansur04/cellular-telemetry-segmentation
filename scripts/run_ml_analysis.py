from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.io_utils import (
    ensure_output_dirs,
    load_config,
    save_csv,
    save_json,
)
from src.ml_analysis import (
    DEFAULT_FEATURES,
    build_model,
    evaluate_model,
    grouped_permutation_importance,
    random_split,
    temporal_split,
)
from src.plotting import plot_ml_importance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_output_dirs(config)

    data_path = Path(config["data"]["cleaned_parquet_path"])

    if not data_path.exists():
        raise FileNotFoundError(
            f"Cleaned Parquet not found: {data_path}. "
            "Run run_all.py first."
        )

    df = pd.read_parquet(data_path)

    feature_columns = DEFAULT_FEATURES

    working = df.dropna(
        subset=feature_columns + ["page_download_speed"]
    ).copy()

    if config["ml"]["use_temporal_split"]:
        train, test = temporal_split(
            working,
            test_fraction=config["ml"]["test_size"],
        )
        split_name = "temporal"
    else:
        train, test = random_split(
            working,
            test_fraction=config["ml"]["test_size"],
            random_state=config["project"]["random_seed"],
        )
        split_name = "random"

    model = build_model(
        n_estimators=config["ml"]["n_estimators"],
        max_depth=config["ml"]["max_depth"],
        random_state=config["project"]["random_seed"],
    )

    metrics = evaluate_model(
        model,
        train,
        test,
        feature_columns,
    )

    importance = grouped_permutation_importance(
        model,
        test,
        feature_columns,
        repeats=config["ml"]["permutation_repeats"],
        random_state=config["project"]["random_seed"],
    )

    metrics["split_type"] = split_name
    metrics["train_rows"] = len(train)
    metrics["test_rows"] = len(test)

    save_json(
        metrics,
        Path(config["output"]["metrics"]) / "ml_metrics.json",
    )

    save_csv(
        importance,
        Path(config["output"]["tables"]) / "ml_feature_importance.csv",
    )

    plot_ml_importance(
        importance,
        Path(config["output"]["figures"]) / "04_ml_feature_importance",
    )

    print("ML analysis completed.")
    print(metrics)
    print(importance)


if __name__ == "__main__":
    main()
