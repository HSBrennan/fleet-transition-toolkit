"""Helpers for loading and transforming input CSV datasets."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = WORKSPACE_ROOT

_DATASET_LABELS = {
    "Input Data": "REHIP Model_V9(Input data) copy.csv",
    "Model": "REHIP Model_V9(Model) copy.csv",
    "Sheet1": "REHIP Model_V9(Sheet1) copy.csv",
    "Sheet3": "REHIP Model_V9(Sheet3) copy.csv",
}


def list_available_datasets() -> Dict[str, Path]:
    return {
        label: DATASETS_DIR / filename for label, filename in _DATASET_LABELS.items()
    }


@lru_cache(maxsize=16)
def load_dataset(label: str, usecols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    datasets = list_available_datasets()
    if label not in datasets:
        raise ValueError(f"Dataset '{label}' not recognised. Available: {', '.join(datasets)}")

    path = datasets[label]
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path, usecols=usecols)
    return df


def melt_time_series(
    df: pd.DataFrame, id_vars: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """Convert a wide-year structure into tidy long format."""

    if df.empty:
        return df

    if id_vars is None:
        id_vars = [df.columns[0]]

    value_vars = [c for c in df.columns if c not in id_vars]

    return df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="Year",
        value_name="Value",
    )

