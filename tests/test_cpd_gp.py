from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from dl_for_cta.config.schema import CpdConfig
from dl_for_cta.features.cpd_gp import build_cpd_features_for_symbol
from dl_for_cta.features.cpd_gp import compute_cpd_window


def test_gp_cpd_detects_known_change() -> None:
    rng = np.random.default_rng(7)
    before = rng.normal(0.0, 0.2, 20)
    after = rng.normal(1.5, 0.2, 20)
    result = compute_cpd_window(np.r_[before, after], max_iter=100)
    assert result.success
    assert result.score > 0.6
    assert 0.0 <= result.loc <= 1.0


def test_gp_cpd_optimizer_does_not_emit_runtime_warnings() -> None:
    returns = np.r_[np.zeros(20), np.linspace(-10.0, 10.0, 20)]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        compute_cpd_window(returns, max_iter=20, standardize=False)
    assert not [warning for warning in record if issubclass(warning.category, RuntimeWarning)]


def test_build_cpd_features_does_not_write_symbol_cache() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01 09:31", periods=12, freq="min"),
            "close": np.linspace(100.0, 101.0, 12),
        }
    )
    config = CpdConfig(windows=[5], min_valid_points=5, max_optimizer_iter=5, cache_dir="unused-cache-dir")
    out = build_cpd_features_for_symbol(frame, config, symbol="x")
    assert "cp_score_5" in out.columns


def test_build_cpd_features_for_symbol_parallel_matches_single_process() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01 09:31", periods=16, freq="min"),
            "close": np.r_[np.linspace(100.0, 101.0, 8), np.linspace(102.0, 103.0, 8)],
        }
    )
    single = CpdConfig(windows=[5], min_valid_points=5, max_optimizer_iter=5, n_jobs=1)
    parallel = CpdConfig(windows=[5], min_valid_points=5, max_optimizer_iter=5, n_jobs=2)

    single_out = build_cpd_features_for_symbol(frame, single, symbol="x", show_progress=False)
    parallel_out = build_cpd_features_for_symbol(frame, parallel, symbol="x", show_progress=False)

    assert list(parallel_out.columns) == list(single_out.columns)
    assert np.allclose(parallel_out["cp_score_5"], single_out["cp_score_5"], equal_nan=True)
    assert np.allclose(parallel_out["cp_loc_5"], single_out["cp_loc_5"], equal_nan=True)


def test_cpd_config_rejects_invalid_n_jobs() -> None:
    try:
        CpdConfig(n_jobs=0)
    except ValueError as exc:
        assert "n_jobs" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid n_jobs.")
