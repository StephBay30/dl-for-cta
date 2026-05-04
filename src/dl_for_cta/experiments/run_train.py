from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dl_for_cta.backtest.engine import annualized_periods_per_year, run_position_backtest
from dl_for_cta.config.schema import ExperimentConfig, ModelConfig, ThresholdConfig, TrainingConfig
from dl_for_cta.experiments.search import apply_candidate, expand_search_grid, filter_cpd_feature_columns
from dl_for_cta.experiments.splits import apply_fill_values, split_train_valid_test, train_fill_values
from dl_for_cta.features.build_features import load_model_features
from dl_for_cta.models.dataset import SequenceDataset, add_weighted_target, feature_columns
from dl_for_cta.models.dmn_lstm import DmnLstm
from dl_for_cta.models.losses import negative_sharpe_loss
from dl_for_cta.models.threshold import HardThreshold, LearnableThreshold
from dl_for_cta.utils.paths import ensure_dir
from dl_for_cta.utils.seed import set_seed


logger = logging.getLogger(__name__)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _make_model(model_config: ModelConfig, input_size: int, device: torch.device) -> DmnLstm:
    return DmnLstm(
        input_size=input_size,
        hidden_size=model_config.hidden_size,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        position_activation=model_config.position_activation,
    ).to(device)


def _predict_symbol(model: DmnLstm, part: pd.DataFrame, cols: list[str], seq_len: int, device: torch.device) -> np.ndarray:
    values = part[cols].to_numpy(dtype=np.float32)
    raw = np.full(len(part), np.nan, dtype=float)
    model.eval()
    with torch.no_grad():
        for idx in range(seq_len - 1, len(part)):
            window = torch.from_numpy(values[idx - seq_len + 1 : idx + 1]).unsqueeze(0).to(device)
            raw[idx] = float(model(window)[0, -1].detach().cpu())
    return raw


def _validation_metrics(
    model: DmnLstm,
    threshold: LearnableThreshold,
    valid: pd.DataFrame,
    cols: list[str],
    model_config: ModelConfig,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    parts = []
    hard_threshold = HardThreshold(float(threshold.value().detach().cpu()))
    for _, part in valid.groupby("order_book_id", sort=False):
        part = part.sort_values("datetime").copy()
        part["raw_position"] = _predict_symbol(model, part, cols, model_config.sequence_length, device)
        if config.threshold.enabled:
            part["position"] = hard_threshold.apply(part["raw_position"].fillna(0.0).to_numpy())
        else:
            part["position"] = part["raw_position"].fillna(0.0)
        parts.append(part)
    if not parts:
        return {"sharpe": float("-inf"), "annual_return": 0.0, "annual_volatility": 0.0, "max_drawdown": 0.0}

    _, metrics = run_position_backtest(
        pd.concat(parts, ignore_index=True),
        position_col="position",
        cost_bps_single_side=config.backtest.cost_bps_single_side,
        periods_per_year=annualized_periods_per_year(config.backtest.annualization_minutes, config.data.n_min_bar),
        execution_lag_minutes=config.backtest.execution_lag_minutes,
    )
    return {key: float(metrics.iloc[0][key]) for key in metrics.columns}


def _checkpoint_payload(
    model: DmnLstm,
    threshold: LearnableThreshold,
    cols: list[str],
    fill_values: dict[str, float],
    model_config: ModelConfig,
    threshold_config: ThresholdConfig,
    training_config: TrainingConfig,
    candidate: dict[str, Any],
    candidate_id: str,
    epoch: int,
    valid_metrics: dict[str, float],
    raw_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "threshold_state": threshold.state_dict(),
        "threshold_value": float(threshold.value().detach().cpu()),
        "feature_columns": cols,
        "fill_values": fill_values,
        "model_params": {
            "input_size": len(cols),
            "hidden_size": model_config.hidden_size,
            "num_layers": model_config.num_layers,
            "dropout": model_config.dropout,
            "position_activation": model_config.position_activation,
            "sequence_length": model_config.sequence_length,
        },
        "threshold_params": {
            "initial_value": threshold_config.initial_value,
            "min_value": threshold_config.min_value,
            "max_value": threshold_config.max_value,
        },
        "training_params": {
            "learning_rate": training_config.learning_rate,
            "turnover_penalty": training_config.turnover_penalty,
        },
        "candidate": candidate,
        "candidate_id": candidate_id,
        "epoch": epoch,
        "valid_metrics": valid_metrics,
        "config": raw_config,
    }


def run(config: ExperimentConfig) -> str:
    set_seed(config.project.seed)
    device = _device(config.project.device)
    logger.info("[train] device=%s", device)
    periods_per_year = annualized_periods_per_year(config.backtest.annualization_minutes, config.data.n_min_bar)
    logger.info("[train] n_min_bar=%d periods_per_year=%d", config.data.n_min_bar, periods_per_year)
    base_df = load_model_features(config)
    all_candidates = expand_search_grid(config)
    logger.info("[train] candidates=%d", len(all_candidates))
    out_dir = ensure_dir(config.outputs.experiment_dir)
    candidates_dir = ensure_dir(out_dir / "candidates")

    search_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    best_global_sharpe = float("-inf")
    best_global_path: Path | None = None

    for candidate_idx, candidate in enumerate(all_candidates):
        candidate_id = f"candidate_{candidate_idx:03d}"
        model_config, threshold_config, training_config, cpd_config = apply_candidate(config, candidate)
        logger.info("[train] start %s params=%s", candidate_id, candidate)
        df = add_weighted_target(base_df, model_config)
        cols = filter_cpd_feature_columns(feature_columns(df), cpd_config.windows)
        train, valid, _ = split_train_valid_test(df, training_config)
        if train.empty:
            raise ValueError(f"{candidate_id}: training frame is empty; check date config.")
        if valid.empty:
            raise ValueError(f"{candidate_id}: validation frame is empty; check date config.")

        fill_values = train_fill_values(train, cols)
        train = apply_fill_values(train, cols, fill_values)
        valid = apply_fill_values(valid, cols, fill_values)
        dataset = SequenceDataset(train, cols, model_config.sequence_length)
        loader = DataLoader(dataset, batch_size=training_config.batch_size, shuffle=True)
        logger.info(
            "[train] %s rows train=%d valid=%d features=%d sequences=%d",
            candidate_id,
            len(train),
            len(valid),
            len(cols),
            len(dataset),
        )

        model = _make_model(model_config, len(cols), device)
        threshold = LearnableThreshold(
            threshold_config.initial_value,
            threshold_config.min_value,
            threshold_config.max_value,
        ).to(device)
        params = list(model.parameters()) + (list(threshold.parameters()) if threshold_config.learnable else [])
        optim = torch.optim.Adam(params, lr=training_config.learning_rate, weight_decay=training_config.weight_decay)

        candidate_dir = ensure_dir(candidates_dir / candidate_id)
        best_candidate_sharpe = float("-inf")
        best_candidate_epoch = -1
        best_candidate_path = candidate_dir / "best_epoch.pt"
        patience = 0

        for epoch in range(training_config.epochs):
            model.train()
            losses = []
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                optim.zero_grad()
                raw = model(x)
                positions = threshold(raw) if threshold_config.enabled else raw
                loss = negative_sharpe_loss(positions, y, turnover_penalty=training_config.turnover_penalty)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optim.step()
                losses.append(float(loss.detach().cpu()))

            train_loss = float(np.mean(losses))
            metrics = _validation_metrics(model, threshold, valid, cols, model_config, config, device)
            valid_sharpe = float(metrics["sharpe"])
            validation_rows.append(
                {
                    "candidate_id": candidate_id,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **{f"valid_{key}": value for key, value in metrics.items()},
                    **candidate,
                }
            )

            if valid_sharpe > best_candidate_sharpe:
                best_candidate_sharpe = valid_sharpe
                best_candidate_epoch = epoch
                patience = 0
                is_best = True
                torch.save(
                    _checkpoint_payload(
                        model,
                        threshold,
                        cols,
                        fill_values,
                        model_config,
                        threshold_config,
                        training_config,
                        candidate,
                        candidate_id,
                        epoch,
                        metrics,
                        config.raw,
                    ),
                    best_candidate_path,
                )
            else:
                patience += 1
                is_best = False

            logger.info(
                "[train] %s epoch=%d/%d train_loss=%.6f valid_after_cost_sharpe=%.6f "
                "threshold=%.6f best_epoch=%d best_valid_sharpe=%.6f %s",
                candidate_id,
                epoch + 1,
                training_config.epochs,
                train_loss,
                valid_sharpe,
                float(threshold.value().detach().cpu()),
                best_candidate_epoch + 1,
                best_candidate_sharpe,
                "NEW_BEST" if is_best else "",
            )

            if patience >= training_config.early_stopping_patience:
                logger.info("[train] %s early_stop patience=%d", candidate_id, patience)
                break

        search_rows.append(
            {
                "candidate_id": candidate_id,
                "best_epoch": best_candidate_epoch,
                "best_valid_sharpe": best_candidate_sharpe,
                "checkpoint": str(best_candidate_path),
                **candidate,
            }
        )
        if best_candidate_sharpe > best_global_sharpe:
            best_global_sharpe = best_candidate_sharpe
            best_global_path = best_candidate_path
        logger.info(
            "[train] done %s best_epoch=%d best_valid_sharpe=%.6f",
            candidate_id,
            best_candidate_epoch + 1,
            best_candidate_sharpe,
        )

    if best_global_path is None:
        raise RuntimeError("No candidate checkpoint was produced.")

    best_model_path = out_dir / "best_model.pt"
    shutil.copy2(best_global_path, best_model_path)
    pd.DataFrame(search_rows).to_csv(out_dir / "search_results.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(out_dir / "validation_metrics.csv", index=False)
    logger.info("[train] selected=%s best_valid_sharpe=%.6f", best_model_path, best_global_sharpe)
    return str(best_model_path)
