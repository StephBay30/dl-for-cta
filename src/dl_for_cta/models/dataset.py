from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dl_for_cta.config.schema import ModelConfig


EXCLUDED_COLUMNS = {
    "order_book_id",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "total_turnover",
}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    cols = []
    for col in frame.columns:
        if col in EXCLUDED_COLUMNS or col == "target_ret" or col.startswith("target_ret_"):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def add_weighted_target(frame: pd.DataFrame, model: ModelConfig) -> pd.DataFrame:
    out = frame.copy()
    weights = np.asarray(model.target_weights, dtype=float)
    weights = weights / weights.sum()
    target = np.zeros(len(out), dtype=float)
    for horizon, weight in zip(model.target_horizons, weights):
        target += weight * out[f"target_ret_{horizon}"].to_numpy(dtype=float)
    out["target_ret"] = target
    return out


class SequenceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, columns: list[str], sequence_length: int) -> None:
        frame = frame.dropna(subset=columns + ["target_ret"]).copy()
        self.x = []
        self.y = []
        for _, part in frame.groupby("order_book_id", sort=False):
            values = part[columns].to_numpy(dtype=np.float32)
            targets = part["target_ret"].to_numpy(dtype=np.float32)
            for idx in range(sequence_length - 1, len(part)):
                start = idx - sequence_length + 1
                self.x.append(values[start : idx + 1])
                self.y.append(targets[start : idx + 1])
        if not self.x:
            raise ValueError("No valid sequences after dropping missing feature/target rows.")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])
