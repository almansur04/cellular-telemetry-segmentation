from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


DEFAULT_FEATURES = [
    "signal_strength",
    "network_type",
    "device_model_code",
    "url",
    "hour",
    "dayofweek",
    "ping_time",
    "speed",
]


def build_model(
    n_estimators: int = 300,
    max_depth: int = 12,
    random_state: int = 42,
) -> Pipeline:
    """Build an RF model with statistically appropriate categorical encoding."""
    categorical = [
        "network_type",
        "device_model_code",
        "url",
    ]

    numeric = [
        "signal_strength",
        "hour",
        "dayofweek",
        "ping_time",
        "speed",
    ]

    transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
            (
                "numeric",
                "passthrough",
                numeric,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        [
            ("preprocess", transformer),
            ("model", model),
        ]
    )


def temporal_split(
    df: pd.DataFrame,
    timestamp_col: str = "test_time",
    test_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split."""
    ordered = df.sort_values(timestamp_col).reset_index(drop=True)
    split_index = int(len(ordered) * (1.0 - test_fraction))

    return (
        ordered.iloc[:split_index].copy(),
        ordered.iloc[split_index:].copy(),
    )


def random_split(
    df: pd.DataFrame,
    test_fraction: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random train/test split."""
    test = df.sample(
        frac=test_fraction,
        random_state=random_state,
    )
    train = df.drop(index=test.index)

    return train.copy(), test.copy()


def evaluate_model(
    model: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "page_download_speed",
) -> dict[str, float]:
    """Fit and evaluate the predictive model."""
    X_train = train[feature_columns]
    y_train = train[target_column]

    X_test = test[feature_columns]
    y_test = test[target_column]

    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    return {
        "r2": float(r2_score(y_test, prediction)),
        "mae_mbps": float(mean_absolute_error(y_test, prediction)),
        "rmse_mbps": float(np.sqrt(mean_squared_error(y_test, prediction))),
    }


def grouped_permutation_importance(
    model: Pipeline,
    test: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "page_download_speed",
    repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Permute original dataframe columns one at a time.

    This avoids the interpretability problem of treating label-encoded
    categories as ordered numerical variables.
    """
    rng = np.random.default_rng(random_state)

    X = test[feature_columns].copy()
    y = test[target_column].to_numpy()

    baseline_prediction = model.predict(X)
    baseline_r2 = r2_score(y, baseline_prediction)

    rows: list[dict[str, Any]] = []

    for column in feature_columns:
        scores = []

        for _ in range(repeats):
            shuffled = X.copy()

            values = shuffled[column].to_numpy(copy=True)
            rng.shuffle(values)
            shuffled[column] = values

            prediction = model.predict(shuffled)
            score = r2_score(y, prediction)
            scores.append(baseline_r2 - score)

        rows.append(
            {
                "feature": column,
                "importance_mean_r2_drop": float(np.mean(scores)),
                "importance_std_r2_drop": float(np.std(scores, ddof=1)),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("importance_mean_r2_drop", ascending=False)
        .reset_index(drop=True)
    )
