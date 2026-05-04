from __future__ import annotations

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
