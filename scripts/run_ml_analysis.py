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

from src.io_utils import (
    ensure_output_dirs,
    load_config,
    load_cleaned_parquet,
    save_csv,
    save_json,
)
from src.ml_analysis import (
    build_rf_model,
    build_ridge_model,
    evaluate_model_with_ci,
    grouped_permutation_importance,
    random_split,
    temporal_split,
)
from src.plotting import (
    plot_ml_importance,
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

    # Use the cleaned Parquet artifact shared by the analysis pipeline.
    df = load_cleaned_parquet(
        config["data"][
            "cleaned_parquet_path"
        ]
    )

    feature_columns = (
        config["ml"]["features"]
    )

    required_columns = (
        feature_columns
        + [
            "page_download_speed",
            "test_time",
        ]
    )

    working = df.dropna(
        subset=required_columns
    ).copy()

    ml_cfg = config[
        "ml"
    ]

    # Select a chronological split when evaluating on future observations.
    if ml_cfg[
        "use_temporal_split"
    ]:
        train, test = temporal_split(
            working,
            test_fraction=ml_cfg[
                "test_size"
            ],
        )

        split_name = "temporal"

    else:
        train, test = random_split(
            working,
            test_fraction=ml_cfg[
                "test_size"
            ],
            random_state=config[
                "project"
            ]["random_seed"],
        )

        split_name = "random"

    seed = config[
        "project"
    ]["random_seed"]

    print(
        f"ML split: {split_name}"
    )

    print(
        f"Training rows: {len(train):,}"
    )

    print(
        f"Test rows: {len(test):,}"
    )

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------
    ridge = build_ridge_model(
        alpha=ml_cfg[
            "ridge"
        ]["alpha"]
    )

    ridge_metrics = evaluate_model_with_ci(
        ridge,
        train,
        test,
        feature_columns,
        iterations=1000,
        confidence=0.95,
        random_state=seed,
    )

    ridge_metrics.update(
        {
            "model": "Ridge baseline",
            "split_type": split_name,
            "train_rows": len(train),
            "test_rows": len(test),
        }
    )

    # ------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------
    rf_cfg = ml_cfg[
        "random_forest"
    ]

    rf = build_rf_model(
        n_estimators=rf_cfg[
            "n_estimators"
        ],
        max_depth=rf_cfg[
            "max_depth"
        ],
        min_samples_leaf=rf_cfg[
            "min_samples_leaf"
        ],
        random_state=seed,
    )

    rf_metrics = evaluate_model_with_ci(
        rf,
        train,
        test,
        feature_columns,
        iterations=1000,
        confidence=0.95,
        random_state=seed,
    )

    rf_metrics.update(
        {
            "model": "Random Forest",
            "split_type": split_name,
            "train_rows": len(train),
            "test_rows": len(test),
        }
    )

    model_comparison = [
        ridge_metrics,
        rf_metrics,
    ]

    # Persist comparable model metrics for reproducible reporting.
    model_comparison_df = pd.DataFrame(
        model_comparison
    )

    save_csv(
        model_comparison_df,
        Path(
            config["output"][
                "tables"
            ]
        )
        / "ml_model_comparison.csv",
    )

    save_json(
        rf_metrics,
        Path(
            config["output"][
                "metrics"
            ]
        )
        / "ml_random_forest_metrics.json",
    )

    save_json(
        ridge_metrics,
        Path(
            config["output"][
                "metrics"
            ]
        )
        / "ml_ridge_metrics.json",
    )

    # ------------------------------------------------------------------
    # Permutation importance
    # ------------------------------------------------------------------
    # Measure feature contribution on held-out data rather than training data.
    importance = grouped_permutation_importance(
        rf,
        test,
        feature_columns,
        repeats=ml_cfg[
            "permutation_repeats"
        ],
        random_state=seed,
    )

    save_csv(
        importance,
        Path(
            config["output"][
                "tables"
            ]
        )
        / "ml_feature_importance.csv",
    )

    plot_ml_importance(
        importance,
        Path(
            config["output"][
                "figures"
            ]
        )
        / "08_ml_feature_importance",
    )

    print(
        "\nModel comparison:"
    )

    print(
        model_comparison_df.to_string(
            index=False
        )
    )

    print(
        "\nPermutation importance:"
    )

    print(
        importance.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
