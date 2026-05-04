from __future__ import annotations

import pandas as pd

from dl_for_cta.config.schema import TrainingConfig
from dl_for_cta.experiments.splits import apply_fill_values, split_train_valid_test, train_fill_values


def test_train_valid_test_splits_do_not_overlap() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-01 09:31", "2020-01-02 09:31", "2020-01-03 09:31"]),
            "x": [1.0, 2.0, 3.0],
        }
    )
    training = TrainingConfig(
        train_start="2020-01-01",
        first_train_end="2020-01-01",
        validation_start="2020-01-02",
        validation_end="2020-01-02",
        first_test_start="2020-01-03",
    )
    train, valid, test = split_train_valid_test(frame, training)
    assert train["datetime"].dt.date.astype(str).tolist() == ["2020-01-01"]
    assert valid["datetime"].dt.date.astype(str).tolist() == ["2020-01-02"]
    assert test["datetime"].dt.date.astype(str).tolist() == ["2020-01-03"]


def test_fill_values_are_fit_on_train_only() -> None:
    train = pd.DataFrame({"x": [1.0, 3.0], "y": [10.0, 20.0]})
    valid = pd.DataFrame({"x": [None], "y": [None]})
    values = train_fill_values(train, ["x", "y"])
    out = apply_fill_values(valid, ["x", "y"], values)
    assert out.loc[0, "x"] == 2.0
    assert out.loc[0, "y"] == 15.0
