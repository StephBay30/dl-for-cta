from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import torch

from dl_for_cta.backtest.engine import annualized_periods_per_year, run_position_backtest
from dl_for_cta.config.schema import ExperimentConfig
from dl_for_cta.experiments.splits import apply_fill_values, split_train_valid_test
from dl_for_cta.features.build_features import load_model_features
from dl_for_cta.models.dmn_lstm import DmnLstm
from dl_for_cta.models.threshold import HardThreshold
from dl_for_cta.utils.paths import ensure_dir


logger = logging.getLogger(__name__)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _predict_symbol(model: DmnLstm, part, cols: list[str], seq_len: int, device: torch.device) -> np.ndarray:
    values = part[cols].to_numpy(dtype=np.float32)
    raw = np.full(len(part), np.nan, dtype=float)
    model.eval()
    with torch.no_grad():
        for idx in range(seq_len - 1, len(part)):
            window = torch.from_numpy(values[idx - seq_len + 1 : idx + 1]).unsqueeze(0).to(device)
            raw[idx] = float(model(window)[0, -1].detach().cpu())
    return raw


def run(config: ExperimentConfig) -> str:
    out_dir = ensure_dir(config.outputs.experiment_dir)
    checkpoint_path = out_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint {checkpoint_path}; run train first.")
    logger.info("[backtest] loading checkpoint=%s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cols = list(checkpoint["feature_columns"])

    df = load_model_features(config)
    periods_per_year = annualized_periods_per_year(config.backtest.annualization_minutes, config.data.n_min_bar)
    logger.info("[backtest] n_min_bar=%d periods_per_year=%d", config.data.n_min_bar, periods_per_year)
    _, _, test = split_train_valid_test(df, config.training)
    if test.empty:
        raise ValueError("Test frame is empty; check first_test_start.")
    logger.info("[backtest] test rows=%d symbols=%d start=%s", len(test), test["order_book_id"].nunique(), config.training.first_test_start)
    fill_values = checkpoint.get("fill_values")
    if fill_values is None:
        med = df[cols].median(numeric_only=True).fillna(0.0)
        fill_values = {str(key): float(value) for key, value in med.items()}
    test = apply_fill_values(test, cols, fill_values)

    device = _device(config.project.device)
    model_params = checkpoint.get("model_params", {})
    model = DmnLstm(
        input_size=len(cols),
        hidden_size=int(model_params.get("hidden_size", config.model.hidden_size)),
        num_layers=int(model_params.get("num_layers", config.model.num_layers)),
        dropout=float(model_params.get("dropout", config.model.dropout)),
        position_activation=str(model_params.get("position_activation", config.model.position_activation)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    threshold = HardThreshold(float(checkpoint["threshold_value"]))
    sequence_length = int(model_params.get("sequence_length", config.model.sequence_length))
    logger.info("[backtest] device=%s sequence_length=%d threshold=%.6f", device, sequence_length, threshold.theta)

    parts = []
    for _, part in test.groupby("order_book_id", sort=False):
        part = part.sort_values("datetime").copy()
        logger.info("[backtest] predicting symbol=%s rows=%d", part["order_book_id"].iloc[0], len(part))
        part["raw_position"] = _predict_symbol(model, part, cols, sequence_length, device)
        part["position"] = threshold.apply(part["raw_position"].fillna(0.0).to_numpy())
        parts.append(part)
    pred = pd.concat(parts, ignore_index=True)

    detailed, metrics = run_position_backtest(
        pred,
        position_col="position",
        cost_bps_single_side=config.backtest.cost_bps_single_side,
        periods_per_year=periods_per_year,
        execution_lag_minutes=config.backtest.execution_lag_minutes,
    )
    positions_path = out_dir / "test_positions.parquet"
    metrics_path = out_dir / "test_metrics.csv"
    if config.backtest.save_positions:
        detailed.to_parquet(positions_path, index=False)
        logger.info("[backtest] saved positions path=%s rows=%d", positions_path, len(detailed))
    metrics.to_csv(metrics_path, index=False)
    logger.info("[backtest] saved metrics path=%s metrics=%s", metrics_path, metrics.iloc[0].to_dict())
    return str(metrics_path)
