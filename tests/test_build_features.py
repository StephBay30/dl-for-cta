from __future__ import annotations

from pathlib import Path

import pandas as pd

from dl_for_cta.config.schema import (
    BacktestConfig,
    CpdConfig,
    DataConfig,
    ExperimentConfig,
    FeaturesConfig,
    ModelConfig,
    OutputsConfig,
    ProjectConfig,
    SearchConfig,
    ThresholdConfig,
    TrainingConfig,
)
from dl_for_cta.features import build_features


def _config(*, cpd_enabled: bool = True, cpd_windows: list[int] | None = None) -> ExperimentConfig:
    return ExperimentConfig(
        project=ProjectConfig(),
        data=DataConfig(min_bar_root="x", symbols=["a"], start_date="2020-01-01", end_date="2020-01-02"),
        features=FeaturesConfig(),
        cpd=CpdConfig(enabled=cpd_enabled, windows=cpd_windows or [5]),
        model=ModelConfig(),
        threshold=ThresholdConfig(),
        training=TrainingConfig(),
        search=SearchConfig(),
        backtest=BacktestConfig(),
        outputs=OutputsConfig(experiment_name="unit"),
        raw={},
    )


def _basic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_book_id": ["a", "a"],
            "datetime": pd.date_range("2020-01-01 09:31", periods=2, freq="min"),
            "close": [100.0, 101.0],
            "ret_1": [0.0, 0.01],
        }
    )


def _cpd_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_book_id": ["a", "a"],
            "datetime": pd.date_range("2020-01-01 09:31", periods=2, freq="min"),
            "cp_score_5": [0.1, 0.2],
            "cp_loc_5": [0.3, 0.4],
            "cp_score_10": [0.5, 0.6],
            "cp_loc_10": [0.7, 0.8],
        }
    )


def test_build_and_save_cpd_features_writes_cpd_only_parquet(monkeypatch) -> None:
    saved = {}

    def fake_to_parquet(self, path, index=False):  # noqa: ANN001
        saved["path"] = path
        saved["index"] = index
        saved["columns"] = list(self.columns)

    monkeypatch.setattr(build_features, "feature_cache_dir", lambda config: Path("cache") / "features" / "unit")
    monkeypatch.setattr(build_features, "load_basic_features", lambda config: _basic_frame())
    monkeypatch.setattr(build_features, "build_cpd_features_for_symbol", lambda frame, config, symbol: _cpd_frame())
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    paths = build_features.build_and_save_cpd_features(_config(cpd_windows=[5, 10]))

    assert paths == [Path("cache") / "features" / "unit" / "cpd_features.parquet"]
    assert saved["path"] == paths[0]
    assert saved["index"] is False
    assert saved["columns"] == ["order_book_id", "datetime", "cp_score_5", "cp_loc_5", "cp_score_10", "cp_loc_10"]


def test_load_model_features_returns_basic_when_cpd_disabled(monkeypatch) -> None:
    basic = _basic_frame()

    monkeypatch.setattr(build_features, "load_basic_features", lambda config: basic)

    out = build_features.load_model_features(_config(cpd_enabled=False))

    assert out.equals(basic)


def test_load_model_features_merges_requested_cpd_windows(monkeypatch) -> None:
    basic = _basic_frame()
    cpd = _cpd_frame()

    monkeypatch.setattr(build_features, "load_basic_features", lambda config: basic)
    monkeypatch.setattr(build_features, "cpd_feature_path", lambda config: Path(__file__))
    monkeypatch.setattr(pd, "read_parquet", lambda path: cpd)

    out = build_features.load_model_features(_config(cpd_windows=[5]))

    assert "ret_1" in out.columns
    assert "cp_score_5" in out.columns
    assert "cp_loc_5" in out.columns
    assert "cp_score_10" not in out.columns
    assert out["cp_score_5"].tolist() == [0.1, 0.2]


def test_load_model_features_errors_when_requested_cpd_window_is_missing(monkeypatch) -> None:
    basic = _basic_frame()
    cpd = _cpd_frame()

    monkeypatch.setattr(build_features, "load_basic_features", lambda config: basic)
    monkeypatch.setattr(build_features, "cpd_feature_path", lambda config: Path(__file__))
    monkeypatch.setattr(pd, "read_parquet", lambda path: cpd)

    try:
        build_features.load_model_features(_config(cpd_windows=[20]))
    except FileNotFoundError as exc:
        assert "cp_score_20" in str(exc)
        assert "run build-cpd" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing CPD window.")
