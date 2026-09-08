from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(
    categorical: list[str],
    numeric: list[str],
) -> ColumnTransformer:
    """Build categorical one-hot encoding with passthrough numeric features."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                categorical,
            ),
            (
                "numeric",
                "passthrough",
                numeric,
            ),
        ],
    )


def build_rf_model(
    n_estimators: int = 300,
    max_depth: int = 12,
    min_samples_leaf: int = 2,
    random_state: int = 42,
) -> Pipeline:
    """Build a Random Forest regression pipeline."""
    categorical = [
        "network_type",
        "device_model_code",
        "url",
    ]

    numeric = [
        "signal_strength",
        "hour",
        "dayofweek",
        "latitude",
        "longitude",
    ]

    transformer = build_preprocessor(
        categorical,
        numeric,
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        [
            ("preprocess", transformer),
            ("model", model),
        ],
    )


def build_ridge_model(
    alpha: float = 1.0,
) -> Pipeline:
    """Build a regularized linear regression baseline."""
    categorical = [
        "network_type",
        "device_model_code",
        "url",
    ]

    numeric = [
        "signal_strength",
        "hour",
        "dayofweek",
        "latitude",
        "longitude",
    ]

    transformer = build_preprocessor(
        categorical,
        numeric,
    )

    model = Ridge(
        alpha=alpha,
    )

    return Pipeline(
        [
            ("preprocess", transformer),
            ("model", model),
        ],
    )


def temporal_split(
    df: pd.DataFrame,
    timestamp_col: str = "test_time",
    test_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split observations chronologically into training and test sets."""
    # Preserve temporal ordering to avoid future observations leaking into training.
    ordered = (
        df.sort_values(
            timestamp_col,
        )
        .reset_index(
            drop=True,
        )
    )

    split_index = int(
        len(ordered)
        * (1.0 - test_fraction),
    )

    return (
        ordered.iloc[
            :split_index
        ].copy(),
        ordered.iloc[
            split_index:
        ].copy(),
    )


def random_split(
    df: pd.DataFrame,
    test_fraction: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a reproducible random train/test split."""
    test = df.sample(
        frac=test_fraction,
        random_state=random_state,
    )

    train = df.drop(
        index=test.index,
    )

    return (
        train.copy(),
        test.copy(),
    )


def evaluate_model(
    model: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "page_download_speed",
) -> dict[str, float]:
    """Fit a regression model and evaluate held-out predictions."""
    X_train = train[
        feature_columns
    ]

    y_train = train[
        target_column
    ]

    X_test = test[
        feature_columns
    ]

    y_test = test[
        target_column
    ]

    model.fit(
        X_train,
        y_train,
    )

    prediction = model.predict(
        X_test,
    )

    return {
        "r2": float(
            r2_score(
                y_test,
                prediction,
            )
        ),
        "mae_mbps": float(
            mean_absolute_error(
                y_test,
                prediction,
            )
        ),
        "rmse_mbps": float(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    prediction,
                )
            )
        ),
    }


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    iterations: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap a regression metric on held-out predictions.

    Each held-out observation is treated as an independent resampling unit.
    """
    rng = np.random.default_rng(
        random_state,
    )

    y_true = np.asarray(
        y_true,
    )

    y_pred = np.asarray(
        y_pred,
    )

    n = len(y_true)

    values = []

    for _ in range(iterations):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        yt = y_true[
            indices
        ]

        yp = y_pred[
            indices
        ]

        if metric == "r2":
            value = r2_score(
                yt,
                yp,
            )
        elif metric == "mae":
            value = mean_absolute_error(
                yt,
                yp,
            )
        elif metric == "rmse":
            value = np.sqrt(
                mean_squared_error(
                    yt,
                    yp,
                )
            )
        else:
            raise ValueError(
                f"Unknown metric: {metric}"
            )

        values.append(
            value
        )

    alpha = (
        1 - confidence
    ) / 2

    return (
        float(
            np.quantile(
                values,
                alpha,
            )
        ),
        float(
            np.quantile(
                values,
                1 - alpha,
            )
        ),
    )


def evaluate_model_with_ci(
    model: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "page_download_speed",
    iterations: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate held-out predictions and estimate bootstrap confidence intervals."""
    X_train = train[
        feature_columns
    ]

    y_train = train[
        target_column
    ].to_numpy()

    X_test = test[
        feature_columns
    ]

    y_test = test[
        target_column
    ].to_numpy()

    model.fit(
        X_train,
        y_train,
    )

    prediction = model.predict(
        X_test
    )

    metrics = {
        "r2": float(
            r2_score(
                y_test,
                prediction,
            )
        ),
        "mae_mbps": float(
            mean_absolute_error(
                y_test,
                prediction,
            )
        ),
        "rmse_mbps": float(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    prediction,
                )
            )
        ),
    }

    for metric in (
        "r2",
        "mae",
        "rmse",
    ):
        low, high = bootstrap_metric_ci(
            y_test,
            prediction,
            metric=metric,
            iterations=iterations,
            confidence=confidence,
            random_state=random_state,
        )

        metrics[
            f"{metric}_ci_low"
        ] = low

        metrics[
            f"{metric}_ci_high"
        ] = high

    return metrics


def grouped_permutation_importance(
    model: Pipeline,
    test: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "page_download_speed",
    repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute held-out permutation importance for original telemetry fields.

    Importance is measured as the decrease in R² after permutation.
    """
    rng = np.random.default_rng(
        random_state,
    )

    X = test[
        feature_columns
    ].copy()

    y = test[
        target_column
    ].to_numpy()

    baseline_prediction = model.predict(
        X
    )

    baseline_r2 = r2_score(
        y,
        baseline_prediction,
    )

    rows = []

    for column in feature_columns:
        permuted_scores = []

        for _ in range(repeats):
            shuffled = X.copy()

            values = shuffled[
                column
            ].to_numpy(
                copy=True
            )

            # Permute one original field while keeping all other features fixed.
            rng.shuffle(
                values
            )

            shuffled[
                column
            ] = values

            prediction = model.predict(
                shuffled
            )

            permuted_r2 = r2_score(
                y,
                prediction,
            )

            permuted_scores.append(
                permuted_r2
            )

        mean_permuted_r2 = float(
            np.mean(
                permuted_scores
            )
        )

        std_permuted_r2 = float(
            np.std(
                permuted_scores,
                ddof=1,
            )
        )

        rows.append(
            {
                "feature": column,
                "baseline_r2": float(
                    baseline_r2
                ),
                "permuted_r2_mean": (
                    mean_permuted_r2
                ),
                "permuted_r2_std": (
                    std_permuted_r2
                ),
                "r2_decrease": float(
                    baseline_r2
                    - mean_permuted_r2
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "r2_decrease",
            ascending=False,
        )
        .reset_index(
            drop=True,
        )
    )
