from __future__ import annotations

import logging

from dl_for_cta.config.schema import ExperimentConfig
from dl_for_cta.experiments import run_backtest, run_cpd, run_features, run_train


logger = logging.getLogger(__name__)


def run(config: ExperimentConfig) -> dict[str, object]:
    logger.info("[pipeline] start")
    features = run_features.run(config)
    cpd = run_cpd.run(config)
    checkpoint = run_train.run(config)
    metrics = run_backtest.run(config)
    logger.info("[pipeline] done")
    return {"features": features, "cpd": cpd, "checkpoint": checkpoint, "metrics": metrics}
