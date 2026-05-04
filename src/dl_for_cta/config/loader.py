from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, TypeVar

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


T = TypeVar("T")


def _read_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8-sig"))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(path: Path) -> dict[str, Any]:
    raw = _read_toml(path)
    base_value = raw.pop("base", None)
    if base_value is None:
        return raw
    if not isinstance(base_value, str):
        raise TypeError("Config field 'base' must be a string path.")

    base_path = Path(base_value)
    if not base_path.is_absolute():
        base_path = path.parent / base_path

    base_raw = _read_toml(base_path)
    if "base" in base_raw:
        raise ValueError("Nested base configs are not supported.")
    return _deep_merge(base_raw, raw)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"Config section [{name}] must be a table.")
    return value


def _build(cls: type[T], values: dict[str, Any]) -> T:
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = _load_raw_config(config_path)

    return ExperimentConfig(
        project=_build(ProjectConfig, _section(raw, "project")),
        data=_build(DataConfig, _section(raw, "data")),
        features=_build(FeaturesConfig, _section(raw, "features")),
        cpd=_build(CpdConfig, _section(raw, "cpd")),
        model=_build(ModelConfig, _section(raw, "model")),
        threshold=_build(ThresholdConfig, _section(raw, "threshold")),
        training=_build(TrainingConfig, _section(raw, "training")),
        search=_build(SearchConfig, {"grid": _section(raw, "search")}),
        backtest=_build(BacktestConfig, _section(raw, "backtest")),
        outputs=_build(OutputsConfig, _section(raw, "outputs")),
        raw=raw,
    )
