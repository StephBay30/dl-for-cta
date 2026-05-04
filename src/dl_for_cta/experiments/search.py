from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any

from dl_for_cta.config.schema import (
    CpdConfig,
    ExperimentConfig,
    ModelConfig,
    ThresholdConfig,
    TrainingConfig,
)


MODEL_KEYS = {"hidden_size", "dropout", "sequence_length", "target_horizons", "target_weights"}
THRESHOLD_KEYS = {"threshold_initial_value"}
TRAINING_KEYS = {"learning_rate", "turnover_penalty"}
CPD_KEYS = {"cpd_windows"}


def expand_search_grid(config: ExperimentConfig) -> list[dict[str, Any]]:
    grid = config.search.grid
    if not grid:
        return [{}]
    keys = sorted(grid)
    values = []
    for key in keys:
        candidates = grid[key]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"[search].{key} must be a non-empty list.")
        values.append(candidates)
    return [dict(zip(keys, combo)) for combo in product(*values)]


def apply_candidate(
    config: ExperimentConfig,
    candidate: dict[str, Any],
) -> tuple[ModelConfig, ThresholdConfig, TrainingConfig, CpdConfig]:
    unknown = set(candidate).difference(MODEL_KEYS | THRESHOLD_KEYS | TRAINING_KEYS | CPD_KEYS)
    if unknown:
        raise ValueError(f"Unsupported search keys: {sorted(unknown)}")

    model_updates = {key: candidate[key] for key in MODEL_KEYS if key in candidate}
    threshold_updates = {}
    if "threshold_initial_value" in candidate:
        threshold_updates["initial_value"] = candidate["threshold_initial_value"]
    training_updates = {key: candidate[key] for key in TRAINING_KEYS if key in candidate}
    cpd_updates = {}
    if "cpd_windows" in candidate:
        cpd_updates["windows"] = candidate["cpd_windows"]

    return (
        replace(config.model, **model_updates),
        replace(config.threshold, **threshold_updates),
        replace(config.training, **training_updates),
        replace(config.cpd, **cpd_updates),
    )


def filter_cpd_feature_columns(columns: list[str], cpd_windows: list[int]) -> list[str]:
    allowed = {f"cp_score_{window}" for window in cpd_windows} | {f"cp_loc_{window}" for window in cpd_windows}
    out = []
    for col in columns:
        if col.startswith("cp_score_") or col.startswith("cp_loc_"):
            if col in allowed:
                out.append(col)
        else:
            out.append(col)
    return out
