from __future__ import annotations

import pandas as pd

from dl_for_cta.data.minute_loader import aggregate_minute_bars


def _bars() -> pd.DataFrame:
    rows = []
    for symbol in ["a", "b"]:
        for day in ["2020-01-01", "2020-01-02"]:
            for idx, ts in enumerate(pd.date_range(f"{day} 09:31", periods=6, freq="min")):
                base = 100 * (1 if symbol == "a" else 2) + idx
                rows.append(
                    {
                        "order_book_id": symbol,
                        "datetime": ts,
                        "open": float(base),
                        "high": float(base + 1),
                        "low": float(base - 1),
                        "close": float(base + 0.5),
                        "volume": idx + 1,
                        "total_turnover": (idx + 1) * 10,
                    }
                )
    return pd.DataFrame(rows)


def test_aggregate_minute_bars_keeps_one_minute_bars_unchanged() -> None:
    frame = _bars()

    out = aggregate_minute_bars(frame, 1)

    assert len(out) == len(frame)
    assert out["datetime"].tolist() == frame.sort_values(["order_book_id", "datetime"])["datetime"].tolist()


def test_aggregate_minute_bars_builds_complete_five_minute_ohlcv() -> None:
    frame = _bars().query("order_book_id == 'a' and datetime.dt.date == @pd.Timestamp('2020-01-01').date()")

    out = aggregate_minute_bars(frame, 5)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["datetime"] == pd.Timestamp("2020-01-01 09:35")
    assert row["open"] == 100.0
    assert row["high"] == 105.0
    assert row["low"] == 99.0
    assert row["close"] == 104.5
    assert row["volume"] == 15
    assert row["total_turnover"] == 150


def test_aggregate_minute_bars_does_not_cross_symbol_or_date() -> None:
    out = aggregate_minute_bars(_bars(), 5)

    assert len(out) == 4
    assert out.groupby(["order_book_id", out["datetime"].dt.date]).size().tolist() == [1, 1, 1, 1]


def test_aggregate_minute_bars_drops_incomplete_tail() -> None:
    out = aggregate_minute_bars(_bars().query("order_book_id == 'a'"), 5)

    assert len(out) == 2
    assert out["datetime"].tolist() == [pd.Timestamp("2020-01-01 09:35"), pd.Timestamp("2020-01-02 09:35")]
