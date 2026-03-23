from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tracks import DEFAULT_MZ_BIN_WIDTH, DEFAULT_MZ_TOLERANCE_PPM, extract_sparse_tracks
from .variance import (
    collect_centered_d2_pairs,
    fit_quadratic_variance,
    fit_quadratic_variance_with_isotonic_pilot,
    quadratic_variance_model,
)


@dataclass(frozen=True)
class VarianceFit:
    """Fitted quadratic variance model and calibration metadata."""

    sigma0_sq: float
    alpha: float
    beta: float
    unit_scale: float
    n_scans: int
    n_tracks: int
    n_pairs: int
    n_fit_points: int


def quadratic_vst(x: np.ndarray, sigma0_sq: float, alpha: float, beta: float) -> np.ndarray:
    """Apply the analytic VST for `sigma0_sq + alpha*x + beta*x^2`."""
    eps = 1e-12
    if beta <= eps:
        if alpha <= eps:
            return x / np.sqrt(max(sigma0_sq, eps))
        inner = np.maximum(alpha * x + sigma0_sq, eps)
        y = (2.0 / alpha) * np.sqrt(inner)
        y0 = (2.0 / alpha) * np.sqrt(max(sigma0_sq, eps))
        return y - y0

    var = np.maximum(sigma0_sq + alpha * x + beta * x * x, eps)
    root_beta = np.sqrt(beta)
    arg = np.maximum(2.0 * root_beta * np.sqrt(var) + 2.0 * beta * x + alpha, eps)
    y = np.log(arg) / root_beta

    arg0 = max(2.0 * root_beta * np.sqrt(max(sigma0_sq, eps)) + alpha, eps)
    y0 = np.log(arg0) / root_beta
    return y - y0


def unit_variance_vst(
    x: np.ndarray,
    sigma0_sq: float,
    alpha: float,
    beta: float,
    unit_scale: float,
) -> np.ndarray:
    return float(unit_scale) * quadratic_vst(
        x,
        sigma0_sq=sigma0_sq,
        alpha=alpha,
        beta=beta,
    )


def estimate_unit_scale(
    tracks: list[tuple[float, np.ndarray, np.ndarray]],
    sigma0_sq: float,
    alpha: float,
    beta: float,
    min_points: int,
) -> tuple[float, dict]:
    """Calibrate the VST so the transformed centered-d2 proxy has median variance ~1."""
    vst_tracks = []
    for mz_value, scans, y in tracks:
        y_vst = quadratic_vst(y.astype(float), sigma0_sq=sigma0_sq, alpha=alpha, beta=beta)
        vst_tracks.append((mz_value, scans, y_vst))

    x_vst, v_vst = collect_centered_d2_pairs(vst_tracks, min_points=min_points)
    if x_vst.size == 0:
        return 1.0, {
            "unit_variance_level_pre_scale": float("nan"),
            "unit_variance_level_post_scale": float("nan"),
            "unit_variance_estimator": "unavailable",
        }

    level = float(np.median(v_vst))
    level = max(level, 1e-12)
    unit_scale = float(1.0 / np.sqrt(level))
    return unit_scale, {
        "unit_variance_level_pre_scale": float(level),
        "unit_variance_level_post_scale": float(level * unit_scale * unit_scale),
        "unit_variance_estimator": "median_pointwise_variance_proxy",
        "unit_variance_point_n": int(v_vst.size),
    }


def estimate_vst_from_file(
    path: str | Path,
    mz_bin_width: float = DEFAULT_MZ_BIN_WIDTH,
    min_intensity: float = 0.0,
    min_points: int = 5,
    same_scan_aggregation: str = "max",
    calibrate_unit_variance: bool = True,
    mz_tolerance_ppm: float = DEFAULT_MZ_TOLERANCE_PPM,
    fit_method: str = "isotonic_pilot",
) -> tuple[VarianceFit, dict]:
    """Estimate the variance model and calibrated VST directly from one LC-MS file."""
    tracks, n_scans = extract_sparse_tracks(
        path=path,
        mz_bin_width=mz_bin_width,
        min_intensity=min_intensity,
        min_points=min_points,
        same_scan_aggregation=same_scan_aggregation,
        mz_tolerance_ppm=mz_tolerance_ppm,
    )
    if not tracks:
        raise RuntimeError("No tracks found; relax track extraction thresholds.")

    x, v = collect_centered_d2_pairs(tracks=tracks, min_points=min_points)
    if x.size == 0:
        raise RuntimeError("No centered rank-2 pairs found; relax segment thresholds.")

    if fit_method == "isotonic_pilot":
        coef, diag = fit_quadratic_variance_with_isotonic_pilot(x=x, v=v)
    elif fit_method == "irls":
        coef, diag = fit_quadratic_variance(x=x, v=v)
    else:
        raise ValueError(f"Unsupported fit_method: {fit_method}")

    sigma0_sq = float(coef[0])
    alpha = float(coef[1])
    beta = float(coef[2])

    if calibrate_unit_variance:
        unit_scale, unit_meta = estimate_unit_scale(
            tracks=tracks,
            sigma0_sq=sigma0_sq,
            alpha=alpha,
            beta=beta,
            min_points=min_points,
        )
    else:
        unit_scale = 1.0
        unit_meta = {
            "unit_variance_level_pre_scale": float("nan"),
            "unit_variance_level_post_scale": float("nan"),
            "unit_variance_estimator": "disabled",
        }

    fit = VarianceFit(
        sigma0_sq=sigma0_sq,
        alpha=alpha,
        beta=beta,
        unit_scale=float(unit_scale),
        n_scans=int(n_scans),
        n_tracks=int(len(tracks)),
        n_pairs=int(x.size),
        n_fit_points=int(diag["n_points_used"]),
    )

    pred = quadratic_variance_model(x, coef)
    ratio = pred / v

    diagnostics = {
        "fit_method": fit_method,
        "relative_rmse": diag["relative_rmse"],
        "mean_abs_rel_error": diag.get("mean_abs_rel_error"),
        "mean_abs_scaled_error": diag.get("mean_abs_scaled_error"),
        "n_fit_points": int(diag["n_points_used"]),
        "n_points_used": int(diag["n_points_used"]),
        "fit_min_signal": diag["fit_min_signal"],
        "fit_min_variance": diag["fit_min_variance"],
        "n_points_excluded": int(diag["n_points_excluded"]),
        "fit_weight_scheme": diag["fit_weight_scheme"],
        "fit_model_floor": diag.get("fit_model_floor"),
        "fit_pilot_floor": diag.get("fit_pilot_floor"),
        "fit_irls_iterations": (
            int(diag["fit_irls_iterations"]) if diag.get("fit_irls_iterations") is not None else None
        ),
        "fit_irls_converged": (
            bool(diag["fit_irls_converged"]) if diag.get("fit_irls_converged") is not None else None
        ),
        "n_pairs": int(x.size),
        "mz_bin_width_da": float(mz_bin_width),
        "mz_tolerance_ppm": float(mz_tolerance_ppm),
        "median_pred_over_v": float(np.median(ratio)),
        "p90_pred_over_v": float(np.quantile(ratio, 0.9)),
        "p99_pred_over_v": float(np.quantile(ratio, 0.99)),
        "fraction_pred_gt_v": float(np.mean(pred > v)),
        "fraction_pred_gt_10x_v": float(np.mean(pred > 10.0 * v)),
        "fraction_pred_gt_100x_v": float(np.mean(pred > 100.0 * v)),
    }
    diagnostics.update(unit_meta)
    return fit, diagnostics
