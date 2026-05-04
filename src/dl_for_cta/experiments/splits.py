from __future__ import annotations

import numpy as np
import pandas as pd

from dl_for_cta.config.schema import TrainingConfig


def _end_exclusive(date_text: str) -> pd.Timestamp:
    return pd.Timestamp(date_text) + pd.Timedelta(days=1)


def split_train_valid_test(
    frame: pd.DataFrame,
    training: TrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_start = pd.Timestamp(training.train_start)
    train_end = _end_exclusive(training.first_train_end)
    valid_start = pd.Timestamp(training.validation_start)
    valid_end = _end_exclusive(training.validation_end)
    test_start = pd.Timestamp(training.first_test_start)

    train = frame[(frame["datetime"] >= train_start) & (frame["datetime"] < train_end)].copy()
    valid = frame[(frame["datetime"] >= valid_start) & (frame["datetime"] < valid_end)].copy()
    test = frame[frame["datetime"] >= test_start].copy()
    return train, valid, test


def train_fill_values(train: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    clean = train[columns].replace([np.inf, -np.inf], np.nan)
    values = clean.median(numeric_only=True).fillna(0.0)
    return {str(key): float(value) for key, value in values.items()}


def apply_fill_values(frame: pd.DataFrame, columns: list[str], values: dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    numeric = out[columns].apply(pd.to_numeric, errors="coerce")
    out[columns] = numeric.replace([np.inf, -np.inf], np.nan).fillna(values).fillna(0.0)
    return out
