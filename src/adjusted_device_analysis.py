from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def prepare_adjusted_dataset(
    clean_room: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the restricted device-analysis dataset.

    Required controls:
      - device model
      - reported signal strength
      - network type
      - hour
      - day of week
      - serving cell

    Response:
      log1p(page_download_speed)
    """
    required = [
        "page_download_speed",
        "device_model",
        "signal_strength",
        "network_type",
        "hour",
        "dayofweek",
        "cell_id",
    ]

    missing = [
        column
        for column in required
        if column not in clean_room.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data = clean_room[
        required
    ].dropna().copy()

    data["device_model"] = (
        data["device_model"]
        .astype(str)
    )

    data["network_type"] = (
        data["network_type"]
        .astype(str)
    )

    data["cell_id"] = (
        data["cell_id"]
        .astype(str)
    )

    data["log_download_speed"] = np.log1p(
        data["page_download_speed"]
    )

    return data


def _fit_fixed_effects(
    data: pd.DataFrame,
    reference_device: str,
    reference_network: str,
):
    """
    Fit device-model regression with serving-cell fixed effects
    and standard errors clustered by cell.
    """
    if reference_device not in set(
        data["device_model"]
    ):
        raise ValueError(
            f"Reference device '{reference_device}' "
            "is absent from the analysis data."
        )

    if reference_network not in set(
        data["network_type"]
    ):
        raise ValueError(
            f"Reference network '{reference_network}' "
            "is absent from the analysis data."
        )

    formula = (
        "log_download_speed ~ "
        f"C(device_model, "
        f"Treatment(reference='{reference_device}')) "
        "+ signal_strength "
        f"+ C(network_type, "
        f"Treatment(reference='{reference_network}')) "
        "+ hour "
        "+ dayofweek "
        "+ C(cell_id)"
    )

    model = smf.ols(
        formula=formula,
        data=data,
    )

    result = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": data["cell_id"]
        },
    )

    return result


def fit_cell_fixed_effects_model(
    clean_room: pd.DataFrame,
    reference_device: str = "SM-A266B",
    reference_network: str = "5G",
):
    """
    Primary adjusted model.

    All represented serving cells are retained.
    """
    data = prepare_adjusted_dataset(
        clean_room
    )

    result = _fit_fixed_effects(
        data,
        reference_device=reference_device,
        reference_network=reference_network,
    )

    return result, data


def prepare_within_cell_dataset(
    clean_room: pd.DataFrame,
) -> pd.DataFrame:
    """
    Restrict the dataset to cells in which at least two device
    models are observed.

    This directly tests whether device-model associations remain
    where within-cell comparisons are possible.
    """
    data = prepare_adjusted_dataset(
        clean_room
    )

    device_counts = (
        data.groupby(
            "cell_id",
            observed=True,
        )["device_model"]
        .nunique()
    )

    multi_device_cells = (
        device_counts[
            device_counts >= 2
        ]
        .index
    )

    return data[
        data["cell_id"].isin(
            multi_device_cells
        )
    ].copy()


def fit_within_cell_model(
    clean_room: pd.DataFrame,
    reference_device: str = "SM-A266B",
    reference_network: str = "5G",
):
    """
    Robustness model restricted to cells containing at least two
    device models.
    """
    data = prepare_within_cell_dataset(
        clean_room
    )

    result = _fit_fixed_effects(
        data,
        reference_device=reference_device,
        reference_network=reference_network,
    )

    return result, data


def extract_device_effects(
    result,
    reference_device: str,
    analysis_label: str,
) -> pd.DataFrame:
    """
    Extract device-specific coefficients and 95% clustered CIs.

    Coefficients are on the log1p throughput scale.
    """
    rows = []

    conf = result.conf_int()

    for parameter in result.params.index:
        if (
            "C(device_model"
            not in parameter
        ):
            continue

        coefficient = float(
            result.params[parameter]
        )

        standard_error = float(
            result.bse[parameter]
        )

        p_value = float(
            result.pvalues[parameter]
        )

        ci_low = float(
            conf.loc[
                parameter,
                0,
            ]
        )

        ci_high = float(
            conf.loc[
                parameter,
                1,
            ]
        )

        device_name = (
            parameter
            .split("T.")[-1]
            .rstrip("]")
            .strip("'")
            .strip('"')
        )

        # This is a descriptive transformed-scale quantity,
        # not a causal percentage effect.
        relative_change_pct = (
            100
            * (
                np.exp(
                    coefficient
                )
                - 1
            )
        )

        rows.append(
            {
                "analysis": analysis_label,
                "device_model": device_name,
                "reference_device": reference_device,
                "coefficient_log1p": coefficient,
                "std_error": standard_error,
                "p_value": p_value,
                "ci_low_log1p": ci_low,
                "ci_high_log1p": ci_high,
                "approx_relative_change_pct": (
                    float(
                        relative_change_pct
                    )
                ),
            }
        )

    reference_row = pd.DataFrame(
        [
            {
                "analysis": analysis_label,
                "device_model": reference_device,
                "reference_device": reference_device,
                "coefficient_log1p": 0.0,
                "std_error": np.nan,
                "p_value": np.nan,
                "ci_low_log1p": 0.0,
                "ci_high_log1p": 0.0,
                "approx_relative_change_pct": 0.0,
            }
        ]
    )

    output = pd.concat(
        [
            reference_row,
            pd.DataFrame(rows),
        ],
        ignore_index=True,
    )

    return (
        output
        .sort_values(
            "coefficient_log1p",
            ascending=True,
        )
        .reset_index(drop=True)
    )


def model_summary(
    result,
    data: pd.DataFrame,
    analysis_label: str,
) -> dict:
    """Return compact model metadata."""
    return {
        "analysis": analysis_label,
        "observations": int(
            result.nobs
        ),
        "cells": int(
            data["cell_id"].nunique()
        ),
        "r_squared": float(
            result.rsquared
        ),
        "adjusted_r_squared": float(
            result.rsquared_adj
        ),
        "residual_df": float(
            result.df_resid
        ),
    }


def overlap_summary(
    clean_room: pd.DataFrame,
) -> dict:
    """Summarize cell/device overlap."""
    data = prepare_adjusted_dataset(
        clean_room
    )

    counts = (
        data.groupby(
            "cell_id",
            observed=True,
        )["device_model"]
        .nunique()
    )

    total_cells = len(
        counts
    )

    multi_device_cells = int(
        (counts >= 2).sum()
    )

    return {
        "represented_cells": int(
            total_cells
        ),
        "multi_device_cells": (
            multi_device_cells
        ),
        "multi_device_cell_share": (
            multi_device_cells
            / total_cells
            if total_cells
            else 0.0
        ),
    }