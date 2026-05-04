from __future__ import annotations

import logging

from dl_for_cta.config.schema import ExperimentConfig
from dl_for_cta.features.build_features import build_and_save_cpd_features


logger = logging.getLogger(__name__)


def run(config: ExperimentConfig) -> list[str]:
    logger.info("Starting CPD feature build")
    paths = [str(path) for path in build_and_save_cpd_features(config)]
    logger.info("Finished CPD feature build outputs=%s", paths)
    return paths
