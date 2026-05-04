from __future__ import annotations

import logging

from dl_for_cta.config.schema import ExperimentConfig
from dl_for_cta.features.build_features import build_and_save_basic_features


logger = logging.getLogger(__name__)


def run(config: ExperimentConfig) -> str:
    logger.info("Starting basic feature build")
    path = str(build_and_save_basic_features(config))
    logger.info("Finished basic feature build output=%s", path)
    return path
