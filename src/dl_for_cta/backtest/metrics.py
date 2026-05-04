from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_returns(returns: pd.Series, periods_per_year: int) -> dict[str, float]:
    clean = returns.fillna(0.0)
    equity = (1.0 + clean).cumprod()
    years = max(len(clean) / periods_per_year, 1e-12)
    annual_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if len(equity) else 0.0
    annual_vol = float(clean.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
    }
