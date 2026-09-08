from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# Anchor relative paths to the repository root for reproducible execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    """
    Resolve configured paths relative to the project root.

    Absolute paths are preserved; relative paths are resolved from the
    repository root.
    """
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration from a resolved project path."""
    path = resolve_project_path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dirs(
    config: dict[str, Any],
) -> None:
    """Create configured output directories if they do not exist."""
    for key in (
        "figures",
        "tables",
        "metrics",
    ):
        path = resolve_project_path(
            config["output"][key]
        )
        path.mkdir(
            parents=True,
            exist_ok=True,
        )


def load_raw_csv(
    path: str | Path,
) -> pd.DataFrame:
    """Load the raw cellular telemetry dataset."""
    path = resolve_project_path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input data not found: {path}\n"
            "Place webbrowsing.csv in the repository data/ directory."
        )

    print(
        f"Reading dataset: {path}"
    )

    # Disable chunk-wise type inference for consistent schema handling.
    return pd.read_csv(
        path,
        low_memory=False,
    )


def load_cleaned_parquet(
    path: str | Path,
) -> pd.DataFrame:
    """Load the cleaned Parquet dataset."""
    path = resolve_project_path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned Parquet not found: {path}\n"
            "Run scripts/run_all.py first."
        )

    return pd.read_parquet(
        path
    )


def save_dataframe(
    df: pd.DataFrame,
    path: str | Path,
) -> None:
    """Persist a DataFrame as a Parquet dataset."""
    path = resolve_project_path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        path,
        index=False,
    )


def save_json(
    payload: dict[str, Any],
    path: str | Path,
) -> None:
    """Persist metrics or metadata as JSON."""
    path = resolve_project_path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            default=str,
        )


def save_csv(
    df: pd.DataFrame,
    path: str | Path,
) -> None:
    """Persist tabular analysis results as CSV."""
    path = resolve_project_path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )
