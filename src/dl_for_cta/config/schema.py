from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "a_share_index_minute_gp_cpd_dmn"
    seed: int = 42
    device: str = "cuda"


@dataclass(frozen=True)
class DataConfig:
    min_bar_root: str
    symbols: list[str]
    start_date: str
    end_date: str
    n_min_bar: int = 1
    datetime_col: str = "datetime"
    price_col: str = "close"
    open_col: str = "open"
    volume_col: str = "volume"
    turnover_col: str = "total_turnover"


@dataclass(frozen=True)
class FeaturesConfig:
    return_windows: list[int] = field(default_factory=lambda: [1, 5, 15, 30, 60, 120, 240])
    vol_windows: list[int] = field(default_factory=lambda: [30, 60, 240])
    volume_z_windows: list[int] = field(default_factory=lambda: [240, 1200])
    use_time_of_day: bool = True
    use_macd: bool = True
    macd_pairs: list[list[int]] = field(default_factory=lambda: [[8, 24], [16, 48], [32, 96]])


@dataclass(frozen=True)
class CpdConfig:
    enabled: bool = True
    method: str = "gp_sigmoid_changepoint"
    windows: list[int] = field(default_factory=lambda: [60, 240, 1200])
    kernel: str = "matern32"
    standardize_window: bool = True
    cache_dir: str = "cache/cpd"
    resume: bool = True
    min_valid_points: int = 30
    max_optimizer_iter: int = 100
    fallback: str = "previous_value"


@dataclass(frozen=True)
class ModelConfig:
    name: str = "dmn_lstm"
    sequence_length: int = 63
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    target_horizons: list[int] = field(default_factory=lambda: [1, 5, 30])
    target_weights: list[float] = field(default_factory=lambda: [1 / 3, 1 / 3, 1 / 3])
    position_activation: str = "tanh"


@dataclass(frozen=True)
class ThresholdConfig:
    enabled: bool = True
    learnable: bool = True
    shared_across_symbols: bool = True
    initial_value: float = 0.05
    min_value: float = 0.0
    max_value: float = 1.0
    train_gate: str = "soft"
    backtest_gate: str = "hard"


@dataclass(frozen=True)
class TrainingConfig:
    split: str = "expanding_window"
    train_start: str = "2015-01-05"
    first_train_end: str = "2019-12-31"
    validation_start: str = "2020-01-01"
    validation_end: str = "2020-12-31"
    first_test_start: str = "2021-01-01"
    test_window_years: int = 1
    expand_by_years: int = 1
    batch_size: int = 256
    epochs: int = 100
    early_stopping_patience: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    loss: str = "negative_sharpe_with_turnover"
    turnover_penalty: float = 0.0


@dataclass(frozen=True)
class SearchConfig:
    grid: dict[str, list[Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestConfig:
    execution_lag_minutes: int = 1
    cost_bps_single_side: float = 5.0
    portfolio_weighting: str = "equal"
    annualization_minutes: int = 240
    save_positions: bool = True
    save_equity_curve: bool = True


@dataclass(frozen=True)
class OutputsConfig:
    root: str = "outputs"
    experiment_name: str = "gp_cpd_dmn_minute"
    save_config_snapshot: bool = True

    @property
    def experiment_dir(self) -> Path:
        return Path(self.root) / self.experiment_name


@dataclass(frozen=True)
class ExperimentConfig:
    project: ProjectConfig
    data: DataConfig
    features: FeaturesConfig
    cpd: CpdConfig
    model: ModelConfig
    threshold: ThresholdConfig
    training: TrainingConfig
    search: SearchConfig
    backtest: BacktestConfig
    outputs: OutputsConfig
    raw: dict[str, Any]
