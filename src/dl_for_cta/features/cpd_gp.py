from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from tqdm.auto import tqdm

from dl_for_cta.config.schema import CpdConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CpdResult:
    score: float
    loc: float
    nlml_matern: float
    nlml_cp: float
    success: bool


def matern32_kernel(x: np.ndarray, lengthscale: float, variance: float) -> np.ndarray:
    dist = np.abs(x[:, None] - x[None, :])
    scaled = np.sqrt(3.0) * dist / max(lengthscale, 1e-8)
    return variance * (1.0 + scaled) * np.exp(-scaled)


def _nlml(y: np.ndarray, kernel: np.ndarray, noise: float) -> float:
    n = len(y)
    cov = kernel + (noise * noise + 1e-7) * np.eye(n)
    try:
        chol = np.linalg.cholesky(cov)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
        log_det = 2.0 * np.log(np.diag(chol)).sum()
        return float(0.5 * y @ alpha + 0.5 * log_det + 0.5 * n * np.log(2.0 * np.pi))
    except np.linalg.LinAlgError:
        return 1e12


def _matern_objective(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    lengthscale, variance, noise = np.exp(params)
    return _nlml(y, matern32_kernel(x, lengthscale, variance), noise)


def _cp_kernel(params: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    l1, v1, l2, v2, noise, c_raw, s_raw = params
    length1, variance1, length2, variance2 = np.exp([l1, v1, l2, v2])
    c = float(expit(c_raw))
    steepness = float(np.exp(s_raw))
    gate_after = expit(steepness * (x - c))
    gate_before = 1.0 - gate_after
    k_before = matern32_kernel(x, length1, variance1)
    k_after = matern32_kernel(x, length2, variance2)
    kernel = k_before * np.outer(gate_before, gate_before) + k_after * np.outer(gate_after, gate_after)
    return kernel, float(np.exp(noise))


def _cp_objective(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    kernel, noise = _cp_kernel(params, x)
    return _nlml(y, kernel, noise)


def _standardize(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std


def compute_cpd_window(
    returns: np.ndarray,
    *,
    max_iter: int = 100,
    standardize: bool = True,
) -> CpdResult:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return CpdResult(0.5, 0.5, np.nan, np.nan, False)
    y = _standardize(values) if standardize else values
    x = np.linspace(0.0, 1.0, len(y))

    base_init = np.log(np.ones(3, dtype=float))
    base_res = minimize(
        _matern_objective,
        base_init,
        args=(x, y),
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    nlml_m = float(base_res.fun)
    base_params = np.exp(base_res.x if base_res.success else base_init)

    cp_init = np.array(
        [
            np.log(base_params[0]),
            np.log(base_params[1]),
            np.log(base_params[0]),
            np.log(base_params[1]),
            np.log(base_params[2]),
            logit(0.5),
            np.log(1.0),
        ],
        dtype=float,
    )
    bounds = [
        (-8.0, 5.0),
        (-8.0, 5.0),
        (-8.0, 5.0),
        (-8.0, 5.0),
        (-10.0, 2.0),
        (logit(1e-3), logit(1.0 - 1e-3)),
        (np.log(1e-2), np.log(1e3)),
    ]
    cp_res = minimize(
        _cp_objective,
        cp_init,
        args=(x, y),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter},
    )

    if not cp_res.success:
        retry = cp_init.copy()
        retry = np.log(np.ones(7, dtype=float))
        retry[5] = logit(0.5)
        cp_res = minimize(
            _cp_objective,
            retry,
            args=(x, y),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter},
        )

    nlml_cp = float(cp_res.fun)
    cp_success = bool(cp_res.success)
    loc = float(expit(cp_res.x[5])) if cp_success and np.all(np.isfinite(cp_res.x)) else 0.5
    score = float(expit(nlml_m - nlml_cp)) if np.isfinite(nlml_m) and np.isfinite(nlml_cp) else 0.5
    success = bool(cp_success and np.isfinite(nlml_m) and np.isfinite(nlml_cp))
    return CpdResult(score, loc, nlml_m, nlml_cp, success)


def build_cpd_features_for_symbol(
    frame: pd.DataFrame,
    config: CpdConfig,
    *,
    symbol: str,
) -> pd.DataFrame:
    frame = frame.sort_values("datetime").copy()
    returns = frame["close"].pct_change().to_numpy(dtype=float)
    result = pd.DataFrame({"order_book_id": symbol, "datetime": frame["datetime"].to_numpy()})

    for window in config.windows:
        logger.info("[cpd] start symbol=%s window=%d rows=%d", symbol, window, len(frame))
        scores = np.full(len(frame), np.nan)
        locs = np.full(len(frame), np.nan)
        last_score = 0.5
        last_loc = 0.5
        fallback_count = 0
        iterator = tqdm(
            range(window - 1, len(frame)),
            desc=f"cpd {symbol} w={window}",
            unit="bar",
        )
        for idx in iterator:
            sample = returns[idx - window + 1 : idx + 1]
            if np.isfinite(sample).sum() < max(config.min_valid_points, 5):
                continue
            cpd = compute_cpd_window(sample, max_iter=config.max_optimizer_iter, standardize=config.standardize_window)
            if cpd.success:
                last_score, last_loc = cpd.score, cpd.loc
            elif config.fallback == "previous_value":
                fallback_count += 1
                if fallback_count == 1 or fallback_count % 100 == 0:
                    logger.warning(
                        "[cpd] fallback previous_value symbol=%s window=%d fallback_count=%d idx=%d "
                        "last_score=%.6f last_loc=%.6f",
                        symbol,
                        window,
                        fallback_count,
                        idx,
                        last_score,
                        last_loc,
                    )
                last_loc = min(1.0, last_loc + 1.0 / window)
            else:
                last_score, last_loc = cpd.score, cpd.loc
            scores[idx] = last_score
            locs[idx] = last_loc
        result[f"cp_score_{window}"] = scores
        result[f"cp_loc_{window}"] = locs
        logger.info("[cpd] done symbol=%s window=%d fallback_count=%d", symbol, window, fallback_count)

    return result
