from __future__ import annotations

import pandas as pd

from dl_for_cta.config.schema import FeaturesConfig, ModelConfig
from dl_for_cta.features.basic import build_basic_features
from dl_for_cta.models.dataset import feature_columns


def test_forward_target_is_not_used_as_feature() -> None:
    frame = pd.DataFrame(
        {
            "order_book_id": ["x"] * 5,
            "datetime": pd.date_range("2020-01-01 09:31", periods=5, freq="min"),
            "open": [1, 2, 3, 4, 5],
            "high": [1, 2, 3, 4, 5],
            "low": [1, 2, 3, 4, 5],
            "close": [1, 2, 3, 4, 5],
            "volume": [10, 11, 12, 13, 14],
            "total_turnover": [10, 11, 12, 13, 14],
        }
    )
    out = build_basic_features(
        frame,
        FeaturesConfig(return_windows=[1], vol_windows=[2], volume_z_windows=[2]),
        ModelConfig(target_horizons=[1], target_weights=[1.0]),
    )
    assert "target_ret_1" in out.columns
    assert "target_ret_1" not in feature_columns(out)
