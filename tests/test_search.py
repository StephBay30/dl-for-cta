from __future__ import annotations

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
from dl_for_cta.experiments.search import apply_candidate, expand_search_grid, filter_cpd_feature_columns


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        project=ProjectConfig(),
        data=DataConfig(min_bar_root="x", symbols=["a"], start_date="2020-01-01", end_date="2020-01-02"),
        features=FeaturesConfig(),
        cpd=CpdConfig(windows=[60, 240]),
        model=ModelConfig(hidden_size=64),
        threshold=ThresholdConfig(initial_value=0.05),
        training=TrainingConfig(learning_rate=0.001),
        search=SearchConfig(grid={"learning_rate": [0.001, 0.01], "hidden_size": [8, 16]}),
        backtest=BacktestConfig(),
        outputs=OutputsConfig(),
        raw={},
    )


def test_expand_search_grid_builds_cartesian_product() -> None:
    grid = expand_search_grid(_config())
    assert len(grid) == 4
    assert {"hidden_size": 8, "learning_rate": 0.001} in grid


def test_apply_candidate_updates_config_copies() -> None:
    model, threshold, training, cpd = apply_candidate(
        _config(),
        {"hidden_size": 8, "threshold_initial_value": 0.2, "learning_rate": 0.01, "cpd_windows": [60]},
    )
    assert model.hidden_size == 8
    assert threshold.initial_value == 0.2
    assert training.learning_rate == 0.01
    assert cpd.windows == [60]


def test_filter_cpd_feature_columns_keeps_candidate_windows_only() -> None:
    cols = ["ret_1", "cp_score_60", "cp_loc_60", "cp_score_240", "cp_loc_240"]
    assert filter_cpd_feature_columns(cols, [60]) == ["ret_1", "cp_score_60", "cp_loc_60"]
