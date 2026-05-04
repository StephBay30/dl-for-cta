from __future__ import annotations

import pandas as pd

from dl_for_cta.backtest.engine import annualized_periods_per_year, run_position_backtest


def test_signal_t_uses_next_minute_return() -> None:
    frame = pd.DataFrame(
        {
            "order_book_id": ["x", "x", "x"],
            "datetime": pd.date_range("2020-01-01 09:31", periods=3, freq="min"),
            "close": [100.0, 101.0, 103.02],
            "position": [1.0, -1.0, -1.0],
        }
    )
    detailed, _ = run_position_backtest(
        frame,
        position_col="position",
        cost_bps_single_side=0.0,
        periods_per_year=240 * 252,
    )
    assert round(float(detailed.loc[0, "strategy_return_before_cost"]), 4) == 0.01
    assert round(float(detailed.loc[1, "strategy_return_before_cost"]), 4) == -0.02


def test_annualized_periods_per_year_scales_by_bar_minutes() -> None:
    assert annualized_periods_per_year(240, 1) == 240 * 252
    assert annualized_periods_per_year(240, 5) == 48 * 252
