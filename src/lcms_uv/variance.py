from __future__ import annotations

import numpy as np

from .tracks import split_consecutive_segments


FIT_MIN_SIGNAL = 0.0
FIT_MIN_VARIANCE = 0.0
FIT_IRLS_MAX_ITERS = 25
FIT_IRLS_RTOL = 1e-6


def collect_centered_d2_pairs(
    tracks: list[tuple[float, np.ndarray, np.ndarray]],
    min_points: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect pointwise `(x_i, vhat_i)` pairs from contiguous track segments."""
    x_all: list[np.ndarray] = []
    v_all: list[np.ndarray] = []

    for _mz, scans, intensity in tracks:
        if scans.size < min_points:
            continue

        for seg in split_consecutive_segments(scans):
            if seg.size < min_points:
                continue
            y = intensity[seg].astype(float)
            d2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
            v = (d2 * d2) / 6.0
            x = (y[2:] + y[1:-1] + y[:-2]) / 3.0
            m = np.isfinite(x) & np.isfinite(v) & (x > 0.0) & (v > 0.0)
            if np.any(m):
                x_all.append(x[m])
                v_all.append(v[m])

    if not x_all:
        return np.array([]), np.array([])
    return np.concatenate(x_all), np.concatenate(v_all)


def _nnls_3var(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    ws = np.sqrt(np.maximum(w, 1e-24))
    xw = x * ws[:, None]
    yw = y * ws

    best = np.zeros(3, dtype=float)
    best_sse = float(np.dot(yw, yw))

    for mask in range(1, 8):
        active = np.array([(mask & 1) > 0, (mask & 2) > 0, (mask & 4) > 0], dtype=bool)
        xa = xw[:, active]
        if xa.shape[1] == 0:
            continue
        coef_a, *_ = np.linalg.lstsq(xa, yw, rcond=None)
        if np.any(coef_a < 0.0):
            continue
        coef = np.zeros(3, dtype=float)
        coef[active] = coef_a
        resid = yw - xw @ coef
        sse = float(np.dot(resid, resid))
        if sse < best_sse:
            best = coef
            best_sse = sse

    return best


def quadratic_variance_model(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Evaluate `sigma0_sq + alpha*x + beta*x^2`."""
    return coef[0] + coef[1] * x + coef[2] * x * x


def fit_quadratic_variance(
    x: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit the quadratic variance model on all valid proxy points via IRLS + NNLS."""
    m = (
        np.isfinite(x)
        & np.isfinite(v)
        & (x > FIT_MIN_SIGNAL)
        & (v > FIT_MIN_VARIANCE)
    )
    xx = np.asarray(x[m], dtype=float)
    vv = np.asarray(v[m], dtype=float)
    if xx.size < 3:
        raise RuntimeError(
            "Not enough valid points for fit; need at least 3 (X,V) pairs "
            f"with X > {FIT_MIN_SIGNAL:g} and V > {FIT_MIN_VARIANCE:g}."
        )

    a = np.column_stack([np.ones_like(xx), xx, xx * xx])
    model_floor = max(1e-12, 1e-6 * float(np.median(vv)))

    coef = _nnls_3var(a, vv, np.ones_like(vv))
    converged = False
    n_irls_iters = 0

    for n_irls_iters in range(1, FIT_IRLS_MAX_ITERS + 1):
        pred_for_w = np.maximum(quadratic_variance_model(xx, coef), model_floor)
        w = 1.0 / (pred_for_w * pred_for_w)
        coef_new = _nnls_3var(a, vv, w)

        denom = max(float(np.linalg.norm(coef)), model_floor)
        rel_change = float(np.linalg.norm(coef_new - coef) / denom)
        coef = coef_new
        if rel_change <= FIT_IRLS_RTOL:
            converged = True
            break

    pred = quadratic_variance_model(xx, coef)
    pred_for_w = np.maximum(pred, model_floor)
    rel_resid = (pred - vv) / pred_for_w

    diag = {
        "relative_rmse": float(np.sqrt(np.mean(rel_resid * rel_resid))),
        "mean_abs_rel_error": float(np.mean(np.abs(vv - pred) / np.maximum(vv, 1e-12))),
        "n_points_used": int(xx.size),
        "fit_min_signal": float(FIT_MIN_SIGNAL),
        "fit_min_variance": float(FIT_MIN_VARIANCE),
        "n_points_excluded": int(np.size(x) - xx.size),
        "fit_weight_scheme": "irls_inverse_model_variance_sq",
        "fit_model_floor": float(model_floor),
        "fit_irls_iterations": int(n_irls_iters),
        "fit_irls_converged": bool(converged),
    }
    return coef, diag
