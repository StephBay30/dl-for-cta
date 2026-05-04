from __future__ import annotations

from dl_for_cta.models.threshold import HardThreshold


def test_hard_threshold_only_rebalances_on_large_changes() -> None:
    positions = HardThreshold(0.2).apply([0.0, 0.1, 0.19, 0.25, 0.51, 0.55])
    assert positions == [0.0, 0.0, 0.0, 0.25, 0.51, 0.51]
