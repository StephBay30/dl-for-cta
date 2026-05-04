from __future__ import annotations

import numpy as np
import pandas as pd

from dl_for_cta.config.schema import FeaturesConfig, ModelConfig


def _zscore(series: pd.Series, window: int) -> pd.Series:
    min_periods = min(window, max(2, window // 5))
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def _macd(close: pd.Series, fast: int, slow: int) -> pd.Series:
    fast_ema = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    vol = close.pct_change().ewm(span=slow, adjust=False, min_periods=slow).std()
    return (fast_ema - slow_ema) / close.shift(1).mul(vol).replace(0.0, np.nan)


def _add_for_symbol(frame: pd.DataFrame, features: FeaturesConfig, model: ModelConfig) -> pd.DataFrame:
    frame = frame.sort_values("datetime").copy()
    close = frame["close"]
    returns = close.pct_change()
    frame["ret_1"] = returns

    for window in features.return_windows:
        frame[f"ret_{window}"] = close.pct_change(window)
        frame[f"mom_gap_{window}"] = frame[f"ret_{window}"] - frame["ret_1"]

    for window in features.vol_windows:
        min_periods = min(window, max(2, window // 5))
        frame[f"vol_{window}"] = returns.rolling(window, min_periods=min_periods).std(ddof=0)

    safe_close = close.replace(0.0, np.nan)
    frame["bar_range"] = (frame["high"] - frame["low"]) / safe_close
    session = frame["datetime"].dt.date
    session_high = frame.groupby(session)["high"].cummax()
    session_low = frame.groupby(session)["low"].cummin()
    frame["intraday_range"] = (session_high - session_low) / safe_close
    for window in features.vol_windows:
        rolling_high = frame["high"].rolling(window, min_periods=min(window, max(2, window // 5))).max()
        rolling_low = frame["low"].rolling(window, min_periods=min(window, max(2, window // 5))).min()
        frame[f"range_{window}"] = (rolling_high - rolling_low) / safe_close
    frame["short_reversal"] = -frame["ret_1"]

    for window in features.volume_z_windows:
        frame[f"volume_z_{window}"] = _zscore(frame["volume"], window)
        frame[f"turnover_z_{window}"] = _zscore(frame["total_turnover"], window)

    if features.use_time_of_day:
        minutes = frame["datetime"].dt.hour * 60 + frame["datetime"].dt.minute
        day_min = minutes - minutes.groupby(session).transform("min")
        denom = day_min.groupby(session).transform("max").replace(0, np.nan)
        phase = 2.0 * np.pi * (day_min / denom)
        frame["tod_sin"] = np.sin(phase)
        frame["tod_cos"] = np.cos(phase)

    if features.use_macd:
        for fast, slow in features.macd_pairs:
            frame[f"macd_{fast}_{slow}"] = _macd(close, fast, slow)

    for horizon in model.target_horizons:
        frame[f"target_ret_{horizon}"] = close.shift(-horizon) / close - 1.0

    return frame


def build_basic_features(df: pd.DataFrame, features: FeaturesConfig, model: ModelConfig) -> pd.DataFrame:
    parts = [_add_for_symbol(frame, features, model) for _, frame in df.groupby("order_book_id", sort=False)]
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["order_book_id", "datetime"]).reset_index(drop=True)
