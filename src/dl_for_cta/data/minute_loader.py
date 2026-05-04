from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from dl_for_cta.config.schema import DataConfig


REQUIRED_COLUMNS = {
    "order_book_id",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "total_turnover",
}


logger = logging.getLogger(__name__)


def _validate_n_min_bar(n_min_bar: int) -> None:
    if not isinstance(n_min_bar, int) or n_min_bar < 1:
        raise ValueError(f"n_min_bar must be a positive integer, got {n_min_bar!r}.")


def _symbol_files(root: Path, symbol: str, start_date: str, end_date: str) -> list[Path]:
    symbol_dir = root / symbol
    if not symbol_dir.exists():
        raise FileNotFoundError(f"Missing symbol directory: {symbol_dir}")
    files = sorted(symbol_dir.glob("*.parquet"))
    return [p for p in files if start_date <= p.stem <= end_date]


def aggregate_minute_bars(frame: pd.DataFrame, n_min_bar: int) -> pd.DataFrame:
    _validate_n_min_bar(n_min_bar)
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.sort_values(["order_book_id", "datetime"]).reset_index(drop=True)
    if n_min_bar == 1:
        return out

    parts = []
    for (_, session), part in out.groupby(["order_book_id", out["datetime"].dt.date], sort=False):
        part = part.sort_values("datetime").reset_index(drop=True)
        complete_rows = len(part) // n_min_bar * n_min_bar
        if complete_rows == 0:
            continue
        part = part.iloc[:complete_rows].copy()
        part["_bar_group"] = np.arange(len(part)) // n_min_bar
        grouped = part.groupby("_bar_group", sort=False).agg(
            {
                "order_book_id": "first",
                "datetime": "last",
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "total_turnover": "sum",
            }
        )
        parts.append(grouped.reset_index(drop=True))
    if not parts:
        raise ValueError(f"No complete {n_min_bar}-minute bars after aggregation.")
    return pd.concat(parts, ignore_index=True).sort_values(["order_book_id", "datetime"]).reset_index(drop=True)


def inspect_minute_data(config: DataConfig) -> pd.DataFrame:
    root = Path(config.min_bar_root)
    logger.info("Inspecting minute data root=%s symbols=%s dates=%s..%s", root, config.symbols, config.start_date, config.end_date)
    rows = []
    for symbol in config.symbols:
        files = _symbol_files(root, symbol, config.start_date, config.end_date)
        if not files:
            rows.append({"symbol": symbol, "files": 0, "start": None, "end": None, "rows_first_file": 0})
            continue
        first = pd.read_parquet(files[0])
        missing = sorted(REQUIRED_COLUMNS.difference(first.columns))
        rows.append(
            {
                "symbol": symbol,
                "files": len(files),
                "start": files[0].stem,
                "end": files[-1].stem,
                "rows_first_file": len(first),
                "missing_columns": ",".join(missing),
            }
        )
    return pd.DataFrame(rows)


def load_minute_bars(config: DataConfig, symbols: list[str] | None = None) -> pd.DataFrame:
    _validate_n_min_bar(config.n_min_bar)
    root = Path(config.min_bar_root)
    selected = symbols or config.symbols
    logger.info(
        "Loading minute bars root=%s symbols=%s dates=%s..%s n_min_bar=%d",
        root,
        selected,
        config.start_date,
        config.end_date,
        config.n_min_bar,
    )
    frames = []
    for symbol in selected:
        files = _symbol_files(root, symbol, config.start_date, config.end_date)
        logger.info("Loading symbol=%s files=%d", symbol, len(files))
        for file in files:
            frame = pd.read_parquet(file)
            missing = REQUIRED_COLUMNS.difference(frame.columns)
            if missing:
                raise ValueError(f"{file} missing columns: {sorted(missing)}")
            frames.append(frame)
    if not frames:
        raise ValueError("No minute bars found for configured symbols/date range.")

    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["order_book_id", "datetime"]).reset_index(drop=True)
    raw_rows = len(df)
    df = aggregate_minute_bars(df, config.n_min_bar)
    logger.info(
        "Loaded bars rows=%d raw_rows=%d symbols=%d n_min_bar=%d",
        len(df),
        raw_rows,
        df["order_book_id"].nunique(),
        config.n_min_bar,
    )
    return df
