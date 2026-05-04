from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from tqdm.auto import tqdm

from dl_for_cta.config.schema import CpdConfig


logger = logging.getLogger(__name__)


MATERN_BOUNDS = [(-8.0, 5.0), (-8.0, 5.0), (-10.0, 2.0)]
CP_BOUNDS = [
    (-8.0, 5.0),
    (-8.0, 5.0),
    (-8.0, 5.0),
    (-8.0, 5.0),
    (-10.0, 2.0),
    (logit(1e-3), logit(1.0 - 1e-3)),
    (np.log(1e-2), np.log(1e3)),
]

_WORKER_RETURNS: np.ndarray | None = None
_WORKER_WINDOW: int | None = None
_WORKER_MAX_ITER: int | None = None
_WORKER_STANDARDIZE: bool | None = None
_WORKER_MIN_VALID_POINTS: int | None = None


@dataclass(frozen=True)
class CpdResult:
    score: float
    loc: float
    nlml_matern: float
    nlml_cp: float
    success: bool


@dataclass
class _CpdApplyState:
    last_score: float = 0.5
    last_loc: float = 0.5
    fallback_count: int = 0


def matern32_kernel(x: np.ndarray, lengthscale: float, variance: float) -> np.ndarray:
    dist = np.abs(x[:, None] - x[None, :])
    scaled = np.sqrt(3.0) * dist / max(lengthscale, 1e-8)
    return variance * (1.0 + scaled) * np.exp(-scaled)


def _nlml(y: np.ndarray, kernel: np.ndarray, noise: float) -> float:
    if not np.all(np.isfinite(kernel)) or not np.isfinite(noise):
        return 1e12
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
    if not np.all(np.isfinite(params)):
        return 1e12
    lengthscale, variance, noise = np.exp(params)
    return _nlml(y, matern32_kernel(x, lengthscale, variance), noise)


def _cp_kernel(params: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    if not np.all(np.isfinite(params)):
        return np.full((len(x), len(x)), np.nan), np.nan
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
        bounds=MATERN_BOUNDS,
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
    cp_res = minimize(
        _cp_objective,
        cp_init,
        args=(x, y),
        method="L-BFGS-B",
        bounds=CP_BOUNDS,
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
            bounds=CP_BOUNDS,
            options={"maxiter": max_iter},
        )

    nlml_cp = float(cp_res.fun)
    cp_success = bool(cp_res.success)
    loc = float(expit(cp_res.x[5])) if cp_success and np.all(np.isfinite(cp_res.x)) else 0.5
    score = float(expit(nlml_m - nlml_cp)) if np.isfinite(nlml_m) and np.isfinite(nlml_cp) else 0.5
    success = bool(cp_success and np.isfinite(nlml_m) and np.isfinite(nlml_cp))
    return CpdResult(score, loc, nlml_m, nlml_cp, success)


def _compute_cpd_for_idx(
    idx: int,
    returns: np.ndarray,
    *,
    window: int,
    min_valid_points: int,
    max_iter: int,
    standardize: bool,
) -> tuple[int, CpdResult | None]:
    sample = returns[idx - window + 1 : idx + 1]
    if np.isfinite(sample).sum() < max(min_valid_points, 5):
        return idx, None
    return idx, compute_cpd_window(sample, max_iter=max_iter, standardize=standardize)


def _init_cpd_idx_worker(
    returns: np.ndarray,
    window: int,
    max_iter: int,
    standardize: bool,
    min_valid_points: int,
) -> None:
    global _WORKER_RETURNS, _WORKER_WINDOW, _WORKER_MAX_ITER, _WORKER_STANDARDIZE, _WORKER_MIN_VALID_POINTS
    _WORKER_RETURNS = returns
    _WORKER_WINDOW = window
    _WORKER_MAX_ITER = max_iter
    _WORKER_STANDARDIZE = standardize
    _WORKER_MIN_VALID_POINTS = min_valid_points


def _compute_cpd_idx_worker(idx: int) -> tuple[int, CpdResult | None]:
    if (
        _WORKER_RETURNS is None
        or _WORKER_WINDOW is None
        or _WORKER_MAX_ITER is None
        or _WORKER_STANDARDIZE is None
        or _WORKER_MIN_VALID_POINTS is None
    ):
        raise RuntimeError("CPD worker was not initialized.")
    return _compute_cpd_for_idx(
        idx,
        _WORKER_RETURNS,
        window=_WORKER_WINDOW,
        min_valid_points=_WORKER_MIN_VALID_POINTS,
        max_iter=_WORKER_MAX_ITER,
        standardize=_WORKER_STANDARDIZE,
    )


def _cpd_chunksize(total: int, n_jobs: int) -> int:
    return max(1, min(32, total // max(n_jobs * 16, 1) or 1))


def _apply_cpd_results(
    results: Iterable[tuple[int, CpdResult | None]],
    *,
    scores: np.ndarray,
    locs: np.ndarray,
    symbol: str,
    window: int,
    fallback: str,
    state: _CpdApplyState | None = None,
    index_to_position: dict[int, int] | None = None,
) -> _CpdApplyState:
    state = state or _CpdApplyState()
    for idx, cpd in results:
        if cpd is None:
            continue
        pos = index_to_position[idx] if index_to_position is not None else idx
        if cpd.success:
            state.last_score, state.last_loc = cpd.score, cpd.loc
        elif fallback == "previous_value":
            state.fallback_count += 1
            if state.fallback_count == 1 or state.fallback_count % 100 == 0:
                logger.warning(
                    "[cpd] fallback previous_value symbol=%s window=%d fallback_count=%d idx=%d "
                    "last_score=%.6f last_loc=%.6f",
                    symbol,
                    window,
                    state.fallback_count,
                    idx,
                    state.last_score,
                    state.last_loc,
                )
            state.last_loc = min(1.0, state.last_loc + 1.0 / window)
        else:
            state.last_score, state.last_loc = cpd.score, cpd.loc
        scores[pos] = state.last_score
        locs[pos] = state.last_loc
    return state


def _compute_window_results(
    idxs: range,
    returns: np.ndarray,
    config: CpdConfig,
    *,
    symbol: str,
    window: int,
    show_progress: bool,
) -> Iterable[tuple[int, CpdResult | None]]:
    if config.n_jobs == 1:
        iterator = tqdm(
            idxs,
            desc=f"cpd {symbol} w={window}",
            unit="bar",
            disable=not show_progress,
        )
        for idx in iterator:
            yield _compute_cpd_for_idx(
                idx,
                returns,
                window=window,
                min_valid_points=config.min_valid_points,
                max_iter=config.max_optimizer_iter,
                standardize=config.standardize_window,
            )
        return

    try:
        with ProcessPoolExecutor(
            max_workers=config.n_jobs,
            initializer=_init_cpd_idx_worker,
            initargs=(returns, window, config.max_optimizer_iter, config.standardize_window, config.min_valid_points),
        ) as executor:
            iterator = executor.map(_compute_cpd_idx_worker, idxs, chunksize=_cpd_chunksize(len(idxs), config.n_jobs))
            yield from tqdm(
                iterator,
                total=len(idxs),
                desc=f"cpd {symbol} w={window}",
                unit="bar",
                disable=not show_progress,
            )
    except PermissionError:
        logger.warning("[cpd] multiprocessing unavailable; falling back to n_jobs=1 symbol=%s window=%d", symbol, window)
        iterator = tqdm(
            idxs,
            desc=f"cpd {symbol} w={window}",
            unit="bar",
            disable=not show_progress,
        )
        for idx in iterator:
            yield _compute_cpd_for_idx(
                idx,
                returns,
                window=window,
                min_valid_points=config.min_valid_points,
                max_iter=config.max_optimizer_iter,
                standardize=config.standardize_window,
            )


def build_cpd_features_for_symbol(
    frame: pd.DataFrame,
    config: CpdConfig,
    *,
    symbol: str,
    show_progress: bool = True,
) -> pd.DataFrame:
    frame = frame.sort_values("datetime").copy()
    returns = frame["close"].pct_change().to_numpy(dtype=float)
    result = pd.DataFrame({"order_book_id": symbol, "datetime": frame["datetime"].to_numpy()})

    for window in config.windows:
        logger.info("[cpd] start symbol=%s window=%d rows=%d n_jobs=%d", symbol, window, len(frame), config.n_jobs)
        scores = np.full(len(frame), np.nan)
        locs = np.full(len(frame), np.nan)
        results = _compute_window_results(
            range(window - 1, len(frame)),
            returns,
            config,
            symbol=symbol,
            window=window,
            show_progress=show_progress,
        )
        fallback_count = _apply_cpd_results(
            results,
            scores=scores,
            locs=locs,
            symbol=symbol,
            window=window,
            fallback=config.fallback,
        ).fallback_count
        result[f"cp_score_{window}"] = scores
        result[f"cp_loc_{window}"] = locs
        logger.info("[cpd] done symbol=%s window=%d fallback_count=%d", symbol, window, fallback_count)

    return result
