from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path

import pandas as pd

from dl_for_cta.config.schema import ExperimentConfig
from dl_for_cta.data.minute_loader import load_minute_bars
from dl_for_cta.features.basic import build_basic_features
from dl_for_cta.features.cpd_gp import build_cpd_features_for_symbol
from dl_for_cta.utils.paths import ensure_dir


logger = logging.getLogger(__name__)


KEY_COLUMNS = ["order_book_id", "datetime"]


def feature_cache_dir(config: ExperimentConfig) -> Path:
    return ensure_dir(Path("cache") / "features" / config.outputs.experiment_name)


def basic_feature_path(config: ExperimentConfig) -> Path:
    return feature_cache_dir(config) / "basic_features.parquet"


def cpd_feature_path(config: ExperimentConfig) -> Path:
    return feature_cache_dir(config) / "cpd_features.parquet"


def cpd_shard_dir(config: ExperimentConfig, symbol: str) -> Path:
    return ensure_dir(Path(config.cpd.cache_dir) / config.outputs.experiment_name / symbol)


def cpd_shard_path(config: ExperimentConfig, symbol: str, window: int) -> Path:
    return cpd_shard_dir(config, symbol) / f"window_{window}.parquet"


def build_and_save_basic_features(config: ExperimentConfig) -> Path:
    logger.info("Building basic features experiment=%s", config.outputs.experiment_name)
    bars = load_minute_bars(config.data)
    features = build_basic_features(bars, config.features, config.model)
    out = basic_feature_path(config)
    features.to_parquet(out, index=False)
    logger.info("Saved basic features rows=%d cols=%d path=%s", len(features), len(features.columns), out)
    return out


def load_basic_features(config: ExperimentConfig) -> pd.DataFrame:
    path = basic_feature_path(config)
    if not path.exists():
        logger.info("Basic feature file missing; building path=%s", path)
        build_and_save_basic_features(config)
    logger.info("Loading basic features path=%s", path)
    return pd.read_parquet(path)


def _thin_cpd_input(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["order_book_id", "datetime", "close"]].copy()


def _cpd_shard_columns(window: int) -> list[str]:
    return [*KEY_COLUMNS, f"cp_score_{window}", f"cp_loc_{window}"]


def _valid_cpd_shard(shard: pd.DataFrame, expected_keys: pd.DataFrame, window: int) -> bool:
    cols = _cpd_shard_columns(window)
    if any(col not in shard.columns for col in cols):
        return False
    if len(shard) != len(expected_keys):
        return False
    actual_keys = shard[KEY_COLUMNS].reset_index(drop=True)
    return actual_keys.equals(expected_keys.reset_index(drop=True))


def _load_or_build_cpd_shard(frame: pd.DataFrame, config: ExperimentConfig, symbol: str, window: int) -> pd.DataFrame:
    path = cpd_shard_path(config, symbol, window)
    cols = _cpd_shard_columns(window)
    expected_keys = frame[KEY_COLUMNS].reset_index(drop=True)

    if config.cpd.resume and path.exists():
        try:
            shard = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[cpd] shard unreadable; rebuilding symbol=%s window=%d path=%s error=%s",
                symbol,
                window,
                path,
                exc,
            )
        else:
            if _valid_cpd_shard(shard, expected_keys, window):
                logger.info("[cpd] resume hit symbol=%s window=%d path=%s", symbol, window, path)
                return shard[cols]
            logger.warning("[cpd] shard invalid; rebuilding symbol=%s window=%d path=%s", symbol, window, path)

    logger.info("[cpd] shard build symbol=%s window=%d rows=%d path=%s", symbol, window, len(frame), path)
    shard_config = replace(config.cpd, windows=[window])
    shard = build_cpd_features_for_symbol(frame, shard_config, symbol=symbol)
    shard = shard[cols]
    shard.to_parquet(path, index=False)
    logger.info("[cpd] shard saved symbol=%s window=%d rows=%d path=%s", symbol, window, len(shard), path)
    return shard


def build_and_save_cpd_features(config: ExperimentConfig) -> list[Path]:
    logger.info(
        "Building CPD features experiment=%s windows=%s n_jobs=%d",
        config.outputs.experiment_name,
        config.cpd.windows,
        config.cpd.n_jobs,
    )
    basic = load_basic_features(config)
    cpd_frames = []
    paths = []
    for symbol, frame in basic.groupby("order_book_id", sort=False):
        logger.info("Building CPD features symbol=%s rows=%d", symbol, len(frame))
        cpd_input = _thin_cpd_input(frame).sort_values("datetime").reset_index(drop=True)
        symbol_cpd = cpd_input[KEY_COLUMNS].copy()
        for window in config.cpd.windows:
            shard = _load_or_build_cpd_shard(cpd_input, config, str(symbol), window)
            symbol_cpd = symbol_cpd.merge(shard, on=KEY_COLUMNS, how="left")
        cpd_frames.append(symbol_cpd)
    cpd = pd.concat(cpd_frames, ignore_index=True)
    out = cpd_feature_path(config)
    cpd.to_parquet(out, index=False)
    paths.append(out)
    logger.info("Saved CPD features rows=%d cols=%d path=%s", len(cpd), len(cpd.columns), out)
    return paths


def _cpd_columns_for_windows(frame: pd.DataFrame, windows: list[int], path: Path) -> list[str]:
    cols = KEY_COLUMNS.copy()
    missing = []
    for window in windows:
        for col in (f"cp_score_{window}", f"cp_loc_{window}"):
            if col in frame.columns:
                cols.append(col)
            else:
                missing.append(col)
    if missing:
        raise FileNotFoundError(
            f"Missing CPD columns {missing} in {path}; run build-cpd with cpd.windows including {windows} first."
        )
    return cols


def load_model_features(config: ExperimentConfig) -> pd.DataFrame:
    basic = load_basic_features(config)
    if not config.cpd.enabled:
        return basic

    path = cpd_feature_path(config)
    if not path.exists():
        raise FileNotFoundError(f"Missing CPD feature file {path}; run build-cpd first.")
    logger.info("Loading CPD features path=%s windows=%s", path, config.cpd.windows)
    cpd = pd.read_parquet(path)
    cpd_cols = _cpd_columns_for_windows(cpd, config.cpd.windows, path)
    return basic.merge(cpd[cpd_cols], on=KEY_COLUMNS, how="left")
