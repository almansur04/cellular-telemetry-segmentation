from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(
    str(
        Path(__file__)
        .resolve()
        .parents[1]
    )
)

from src.adjusted_device_analysis import (
    extract_device_effects,
    fit_cell_fixed_effects_model,
    fit_within_cell_model,
    model_summary,
    overlap_summary,
)
from src.device_analysis import (
    create_device_subset,
)
from src.io_utils import (
    ensure_output_dirs,
    load_config,
    load_cleaned_parquet,
    save_csv,
    save_json,
)
from src.plotting import (
    plot_adjusted_device_effects,
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

    reference_device = da[
        "reference_device"
    ]

    reference_network = da[
        "reference_network"
    ]

    # ========================================================
    # Cell-overlap diagnostics
    # ========================================================
    overlap = overlap_summary(
        clean_room
    )

    save_json(
        overlap,
        Path(
            config["output"][
                "metrics"
            ]
        )
        / "device_cell_overlap.json",
    )

    # ========================================================
    # Primary cell fixed-effects model
    # ========================================================
    print(
        "\nFitting primary cell fixed-effects model..."
    )

    result_all, data_all = (
        fit_cell_fixed_effects_model(
            clean_room,
            reference_device=reference_device,
            reference_network=reference_network,
        )
    )

    effects_all = extract_device_effects(
        result_all,
        reference_device=reference_device,
        analysis_label="All represented cells",
    )

    summary_all = model_summary(
        result_all,
        data_all,
        analysis_label="All represented cells",
    )

    # ========================================================
    # Within-cell robustness model
    # ========================================================
    print(
        "\nFitting within-cell robustness model..."
    )

    result_multi, data_multi = (
        fit_within_cell_model(
            clean_room,
            reference_device=reference_device,
            reference_network=reference_network,
        )
    )

    effects_multi = extract_device_effects(
        result_multi,
        reference_device=reference_device,
        analysis_label="Multi-device cells only",
    )

    summary_multi = model_summary(
        result_multi,
        data_multi,
        analysis_label="Multi-device cells only",
    )

    # ========================================================
    # Save coefficients
    # ========================================================
    combined_effects = pd.concat(
        [
            effects_all,
            effects_multi,
        ],
        ignore_index=True,
    )

    save_csv(
        combined_effects,
        Path(
            config["output"][
                "tables"
            ]
        )
        / "cell_adjusted_device_effects.csv",
    )

    model_comparison = pd.DataFrame(
        [
            summary_all,
            summary_multi,
        ]
    )

    save_csv(
        model_comparison,
        Path(
            config["output"][
                "tables"
            ]
        )
        / "cell_adjusted_model_comparison.csv",
    )

    save_json(
        summary_all,
        Path(
            config["output"][
                "metrics"
            ]
        )
        / "cell_adjusted_device_model.json",
    )

    save_json(
        summary_multi,
        Path(
            config["output"][
                "metrics"
            ]
        )
        / "within_cell_device_model.json",
    )

    # ========================================================
    # Plot primary adjusted effects
    # ========================================================
    plot_adjusted_device_effects(
        effects_all,
        Path(
            config["output"][
                "figures"
            ]
        )
        / "04_adjusted_device_effects",
    )

    # ========================================================
    # Human-readable coefficient summaries
    # ========================================================
    for result, filename, title in [
        (
            result_all,
            "cell_adjusted_device_model.txt",
            "Primary cell fixed-effects model",
        ),
        (
            result_multi,
            "within_cell_device_model.txt",
            "Within-cell robustness model",
        ),
    ]:
        text_path = (
            Path(
                config["output"][
                    "metrics"
                ]
            )
            / filename
        )

        with text_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                title
                + "\n"
            )
            f.write(
                "=" * len(title)
                + "\n\n"
            )
            f.write(
                "The full statsmodels summary is intentionally "
                "not used as the primary reported result because "
                "large fixed-effect joint Wald tests can become "
                "rank-deficient. Device-specific clustered "
                "coefficients and confidence intervals are "
                "reported separately.\n\n"
            )

            f.write(
                "N observations: "
                f"{int(result.nobs)}\n"
            )

            f.write(
                "R-squared: "
                f"{result.rsquared:.6f}\n"
            )

            f.write(
                "Adjusted R-squared: "
                f"{result.rsquared_adj:.6f}\n"
            )

    print(
        "\nDevice/cell overlap:"
    )
    print(
        overlap
    )

    print(
        "\nPrimary model:"
    )
    print(
        summary_all
    )

    print(
        "\nWithin-cell robustness model:"
    )
    print(
        summary_multi
    )

    print(
        "\nPrimary adjusted effects:"
    )
    print(
        effects_all.to_string(
            index=False
        )
    )

    print(
        "\nWithin-cell adjusted effects:"
    )
    print(
        effects_multi.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()