from __future__ import annotations

import pandas as pd

from dl_for_cta.backtest.metrics import summarize_returns


def annualized_periods_per_year(annualization_minutes: int, n_min_bar: int) -> int:
    if n_min_bar < 1:
        raise ValueError(f"n_min_bar must be a positive integer, got {n_min_bar!r}.")
    if annualization_minutes < 1:
        raise ValueError(f"annualization_minutes must be positive, got {annualization_minutes!r}.")
    return int(annualization_minutes / n_min_bar * 252)


def run_position_backtest(
    frame: pd.DataFrame,
    *,
    position_col: str,
    cost_bps_single_side: float,
    periods_per_year: int,
    execution_lag_minutes: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for symbol, part in frame.sort_values("datetime").groupby("order_book_id", sort=False):
        part = part.copy()
        ret_next = part["close"].shift(-execution_lag_minutes) / part["close"] - 1.0
        turnover = part[position_col].diff().abs().fillna(part[position_col].abs())
        cost = turnover * cost_bps_single_side / 10000.0
        part["strategy_return_before_cost"] = part[position_col] * ret_next
        part["strategy_return_after_cost"] = part["strategy_return_before_cost"] - cost
        part["turnover"] = turnover
        part["trade_count"] = (turnover > 0).astype(int)
        part["order_book_id"] = symbol
        parts.append(part)
    detailed = pd.concat(parts, ignore_index=True)
    portfolio = detailed.groupby("datetime", as_index=True)["strategy_return_after_cost"].mean().sort_index()
    metrics = summarize_returns(portfolio, periods_per_year)
    metrics["turnover"] = float(detailed["turnover"].mean())
    metrics["trade_count"] = float(detailed["trade_count"].sum())
    return detailed, pd.DataFrame([metrics])
