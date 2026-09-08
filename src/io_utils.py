from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dirs(config: dict[str, Any]) -> None:
    """Create output directories declared in the configuration."""
    for key in ("figures", "tables", "metrics"):
        Path(config["output"][key]).mkdir(parents=True, exist_ok=True)


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    """Load raw telemetry CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input data not found: {path}")

    df = pd.read_csv(
        path,
        low_memory=False,
    )
    return df


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame, creating parent directories when needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Save JSON metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save tabular results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
