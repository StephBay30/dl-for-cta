from __future__ import annotations

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


def build_and_save_cpd_features(config: ExperimentConfig) -> list[Path]:
    logger.info("Building CPD features experiment=%s windows=%s", config.outputs.experiment_name, config.cpd.windows)
    basic = load_basic_features(config)
    cpd_frames = []
    for symbol, frame in basic.groupby("order_book_id", sort=False):
        logger.info("Building CPD features symbol=%s rows=%d", symbol, len(frame))
        cpd_frames.append(build_cpd_features_for_symbol(frame, config.cpd, symbol=str(symbol)))
    cpd = pd.concat(cpd_frames, ignore_index=True)
    out = cpd_feature_path(config)
    cpd.to_parquet(out, index=False)
    logger.info("Saved CPD features rows=%d cols=%d path=%s", len(cpd), len(cpd.columns), out)
    return [out]


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
