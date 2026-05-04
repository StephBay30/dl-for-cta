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


def _config(
    *,
    cpd_enabled: bool = True,
    cpd_windows: list[int] | None = None,
    cpd_n_jobs: int = 1,
    cpd_resume: bool = True,
) -> ExperimentConfig:
    return ExperimentConfig(
        project=ProjectConfig(),
        data=DataConfig(min_bar_root="x", symbols=["a"], start_date="2020-01-01", end_date="2020-01-02"),
        features=FeaturesConfig(),
        cpd=CpdConfig(enabled=cpd_enabled, windows=cpd_windows or [5], n_jobs=cpd_n_jobs, resume=cpd_resume),
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
            "order_book_id": ["a", "a", "a"],
            "datetime": [
                pd.Timestamp("2020-01-01 09:31"),
                pd.Timestamp("2020-01-01 09:32"),
                pd.Timestamp("2020-01-02 09:31"),
            ],
            "close": [100.0, 101.0, 102.0],
            "ret_1": [0.0, 0.01, 0.02],
        }
    )


def _basic_frame_two_symbols() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_book_id": ["a", "a", "b", "b"],
            "datetime": list(pd.date_range("2020-01-01 09:31", periods=2, freq="min")) * 2,
            "close": [100.0, 101.0, 200.0, 201.0],
            "ret_1": [0.0, 0.01, 0.0, 0.005],
        }
    )


def _cpd_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_book_id": ["a", "a", "a"],
            "datetime": [
                pd.Timestamp("2020-01-01 09:31"),
                pd.Timestamp("2020-01-01 09:32"),
                pd.Timestamp("2020-01-02 09:31"),
            ],
            "cp_score_5": [0.1, 0.2, 0.3],
            "cp_loc_5": [0.3, 0.4, 0.5],
            "cp_score_10": [0.5, 0.6, 0.7],
            "cp_loc_10": [0.7, 0.8, 0.9],
        }
    )


def _shard_path(symbol: str, window: int) -> Path:
    return Path("cache") / "cpd" / "unit" / symbol / f"window_{window}.parquet"


def _install_parquet_store(monkeypatch):  # noqa: ANN001
    store = {}
    original_exists = Path.exists

    def fake_to_parquet(self, path, index=False):  # noqa: ANN001
        store[Path(path)] = {"frame": self.copy(), "index": index}

    def fake_read_parquet(path):  # noqa: ANN001
        return store[Path(path)]["frame"].copy()

    def fake_exists(self):  # noqa: ANN001
        path = Path(self)
        if path in store:
            return True
        return original_exists(self)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(Path, "exists", fake_exists)
    return store


def test_build_and_save_basic_features_writes_single_parquet(monkeypatch) -> None:
    saved = {}

    def fake_to_parquet(self, path, index=False):  # noqa: ANN001
        saved["path"] = path
        saved["index"] = index
        saved["columns"] = list(self.columns)

    monkeypatch.setattr(build_features, "feature_cache_dir", lambda config: Path("cache") / "features" / "unit")
    monkeypatch.setattr(build_features, "load_minute_bars", lambda config: _basic_frame()[["order_book_id", "datetime", "close"]])
    monkeypatch.setattr(build_features, "build_basic_features", lambda bars, features, model: _basic_frame())
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    path = build_features.build_and_save_basic_features(_config())

    assert path == Path("cache") / "features" / "unit" / "basic_features.parquet"
    assert saved["path"] == path
    assert saved["index"] is False


def test_build_and_save_cpd_features_writes_single_cpd_parquet(monkeypatch) -> None:
    saved = []

    def fake_to_parquet(self, path, index=False):  # noqa: ANN001
        saved.append({"path": path, "index": index, "columns": list(self.columns), "rows": len(self)})

    monkeypatch.setattr(build_features, "feature_cache_dir", lambda config: Path("cache") / "features" / "unit")
    monkeypatch.setattr(build_features, "cpd_shard_path", lambda config, symbol, window: _shard_path(symbol, window))
    monkeypatch.setattr(build_features, "load_basic_features", lambda config: _basic_frame())
    monkeypatch.setattr(build_features, "build_cpd_features_for_symbol", lambda frame, config, symbol: _cpd_frame())
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    paths = build_features.build_and_save_cpd_features(_config(cpd_windows=[5, 10]))

    assert paths == [Path("cache") / "features" / "unit" / "cpd_features.parquet"]
    assert saved[-1]["path"] == paths[0]
    assert saved[-1]["index"] is False
    assert saved[-1]["columns"] == ["order_book_id", "datetime", "cp_score_5", "cp_loc_5", "cp_score_10", "cp_loc_10"]
    assert saved[-1]["rows"] == 3


def test_build_and_save_cpd_features_keeps_outer_symbol_loop_with_parallel_idx(monkeypatch) -> None:
    calls = []
    saved = {}

    def fake_build(frame, config, symbol):  # noqa: ANN001
        calls.append((symbol, config.n_jobs))
        out = frame[["order_book_id", "datetime"]].copy()
        out["cp_score_5"] = 0.1
        out["cp_loc_5"] = 0.2
        return out

    def fake_to_parquet(self, path, index=False):  # noqa: ANN001
        saved["rows"] = len(self)
        saved["columns"] = list(self.columns)

    monkeypatch.setattr(build_features, "feature_cache_dir", lambda config: Path("cache") / "features" / "unit")
    monkeypatch.setattr(build_features, "cpd_shard_path", lambda config, symbol, window: _shard_path(symbol, window))
    monkeypatch.setattr(build_features, "load_basic_features", lambda config: _basic_frame_two_symbols())
    monkeypatch.setattr(build_features, "build_cpd_features_for_symbol", fake_build)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    build_features.build_and_save_cpd_features(_config(cpd_windows=[5], cpd_n_jobs=2))

    assert calls == [("a", 2), ("b", 2)]
    assert saved["rows"] == 4
    assert saved["columns"] == ["order_book_id", "datetime", "cp_score_5", "cp_loc_5"]


def test_build_and_save_cpd_features_resumes_valid_shard(monkeypatch) -> None:
    store = _install_parquet_store(monkeypatch)
    shard_path = _shard_path("a", 5)
    store[shard_path] = {"frame": _cpd_frame()[["order_book_id", "datetime", "cp_score_5", "cp_loc_5"]], "index": False}

    def fail_build(frame, config, symbol):  # noqa: ANN001
        raise AssertionError("CPD should not rebuild a valid shard.")

    monkeypatch.setattr(build_features, "feature_cache_dir", lambda config: Path("cache") / "features" / "unit")
    monkeypatch.setattr(build_features, "cpd_shard_path", lambda config, symbol, window: _shard_path(symbol, window))
    monkeypatch.setattr(build_features, "load_basic_features", lambda config: _basic_frame())
    monkeypatch.setattr(build_features, "build_cpd_features_for_symbol", fail_build)

    paths = build_features.build_and_save_cpd_features(_config(cpd_windows=[5]))

    final_path = paths[0]
    assert final_path in store
    assert store[final_path]["frame"]["cp_score_5"].tolist() == [0.1, 0.2, 0.3]


def test_build_and_save_cpd_features_rebuilds_invalid_shard(monkeypatch) -> None:
    store = _install_parquet_store(monkeypatch)
    shard_path = _shard_path("a", 5)
    store[shard_path] = {"frame": _cpd_frame()[["order_book_id", "datetime", "cp_score_5"]], "index": False}
    calls = []

    def fake_build(frame, config, symbol):  # noqa: ANN001
        calls.append((symbol, config.windows))
        return _cpd_frame()[["order_book_id", "datetime", "cp_score_5", "cp_loc_5"]]

    monkeypatch.setattr(build_features, "feature_cache_dir", lambda config: Path("cache") / "features" / "unit")
    monkeypatch.setattr(build_features, "cpd_shard_path", lambda config, symbol, window: _shard_path(symbol, window))
    monkeypatch.setattr(build_features, "load_basic_features", lambda config: _basic_frame())
    monkeypatch.setattr(build_features, "build_cpd_features_for_symbol", fake_build)

    build_features.build_and_save_cpd_features(_config(cpd_windows=[5]))

    assert calls == [("a", [5])]
    assert list(store[shard_path]["frame"].columns) == ["order_book_id", "datetime", "cp_score_5", "cp_loc_5"]


def test_build_and_save_cpd_features_rebuilds_when_resume_disabled(monkeypatch) -> None:
    store = _install_parquet_store(monkeypatch)
    shard_path = _shard_path("a", 5)
    store[shard_path] = {"frame": _cpd_frame()[["order_book_id", "datetime", "cp_score_5", "cp_loc_5"]], "index": False}
    calls = []

    def fake_build(frame, config, symbol):  # noqa: ANN001
        calls.append((symbol, config.windows))
        out = _cpd_frame()[["order_book_id", "datetime", "cp_score_5", "cp_loc_5"]].copy()
        out["cp_score_5"] = [0.9, 0.8, 0.7]
        return out

    monkeypatch.setattr(build_features, "feature_cache_dir", lambda config: Path("cache") / "features" / "unit")
    monkeypatch.setattr(build_features, "cpd_shard_path", lambda config, symbol, window: _shard_path(symbol, window))
    monkeypatch.setattr(build_features, "load_basic_features", lambda config: _basic_frame())
    monkeypatch.setattr(build_features, "build_cpd_features_for_symbol", fake_build)

    build_features.build_and_save_cpd_features(_config(cpd_windows=[5], cpd_resume=False))

    assert calls == [("a", [5])]
    assert store[shard_path]["frame"]["cp_score_5"].tolist() == [0.9, 0.8, 0.7]


def test_build_and_save_cpd_features_writes_symbol_window_shards(monkeypatch) -> None:
    store = _install_parquet_store(monkeypatch)
    calls = []

    def fake_build(frame, config, symbol):  # noqa: ANN001
        window = config.windows[0]
        calls.append((symbol, window))
        out = frame[["order_book_id", "datetime"]].copy()
        out[f"cp_score_{window}"] = window
        out[f"cp_loc_{window}"] = 0.1 if symbol == "a" else 0.2
        return out

    monkeypatch.setattr(build_features, "feature_cache_dir", lambda config: Path("cache") / "features" / "unit")
    monkeypatch.setattr(build_features, "cpd_shard_path", lambda config, symbol, window: _shard_path(symbol, window))
    monkeypatch.setattr(build_features, "load_basic_features", lambda config: _basic_frame_two_symbols())
    monkeypatch.setattr(build_features, "build_cpd_features_for_symbol", fake_build)

    paths = build_features.build_and_save_cpd_features(_config(cpd_windows=[5, 10]))

    shard_paths = {_shard_path(symbol, window) for symbol in ("a", "b") for window in (5, 10)}
    assert shard_paths.issubset(store)
    assert calls == [("a", 5), ("a", 10), ("b", 5), ("b", 10)]
    assert list(store[paths[0]]["frame"].columns) == [
        "order_book_id",
        "datetime",
        "cp_score_5",
        "cp_loc_5",
        "cp_score_10",
        "cp_loc_10",
    ]
    assert len(store[paths[0]]["frame"]) == 4


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
    assert out["cp_score_5"].tolist() == [0.1, 0.2, 0.3]


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
