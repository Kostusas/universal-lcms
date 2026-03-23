from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (ROOT, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from lcms_uv.variance import estimate_isotonic_pilot_variance
from lcms_uv.variance import fit_quadratic_variance, fit_quadratic_variance_with_isotonic_pilot, quadratic_variance_model


class IsotonicPilotVarianceTest(unittest.TestCase):
    def test_estimate_isotonic_pilot_variance_recovers_monotone_trend_on_synthetic_data(self) -> None:
        rng = np.random.default_rng(7)
        x = np.exp(np.linspace(np.log(1.0), np.log(1.0e4), 2000))
        true_variance = 5.0 + 1.5 * x + 0.02 * x * x
        noisy_variance = true_variance * np.exp(rng.normal(0.0, 1.0, size=x.shape))

        perm = rng.permutation(x.size)
        x_shuffled = x[perm]
        noisy_shuffled = noisy_variance[perm]
        true_shuffled = true_variance[perm]

        pilot = estimate_isotonic_pilot_variance(x_shuffled, noisy_shuffled)

        self.assertEqual(pilot.shape, x_shuffled.shape)
        self.assertTrue(np.all(np.isfinite(pilot)))
        self.assertTrue(np.all(pilot > 0.0))

        order = np.argsort(x_shuffled, kind="mergesort")
        self.assertTrue(np.all(np.diff(pilot[order]) >= 0.0))

        raw_log_error = np.median(np.abs(np.log(noisy_shuffled) - np.log(true_shuffled)))
        pilot_log_error = np.median(np.abs(np.log(pilot) - np.log(true_shuffled)))
        self.assertLess(pilot_log_error, raw_log_error)

    def test_estimate_isotonic_pilot_variance_handles_ties_and_invalid_points(self) -> None:
        x = np.array([1.0, 1.0, 2.0, 2.0, 4.0, np.nan, 0.0, 3.0])
        v = np.array([2.0, 8.0, 3.0, 27.0, 64.0, 5.0, 7.0, 9.0])

        pilot = estimate_isotonic_pilot_variance(x, v)

        self.assertTrue(np.isnan(pilot[5]))
        self.assertTrue(np.isnan(pilot[6]))
        self.assertAlmostEqual(float(pilot[0]), float(pilot[1]))
        self.assertAlmostEqual(float(pilot[2]), float(pilot[3]))

        valid = np.isfinite(x) & np.isfinite(v) & (x > 0.0) & (v > 0.0)
        order = np.argsort(x[valid], kind="mergesort")
        self.assertTrue(np.all(np.diff(pilot[valid][order]) >= 0.0))

    def test_quadratic_fit_recovers_synthetic_parameters_under_mild_noise(self) -> None:
        rng = np.random.default_rng(123)
        x = np.exp(np.linspace(np.log(1.0), np.log(1.0e4), 4000))
        true_coef = np.array([5.0, 1.5, 0.02])
        true_variance = quadratic_variance_model(x, true_coef)
        observed_variance = true_variance * np.exp(rng.normal(0.0, 0.2, size=x.shape))

        baseline_coef, baseline_diag = fit_quadratic_variance(x, observed_variance)
        pilot_coef, pilot_diag = fit_quadratic_variance_with_isotonic_pilot(x, observed_variance)

        baseline_rel = np.abs(baseline_coef - true_coef) / np.abs(true_coef)
        pilot_rel = np.abs(pilot_coef - true_coef) / np.abs(true_coef)

        self.assertLess(float(np.mean(baseline_rel)), 0.05)
        self.assertLess(float(np.mean(pilot_rel)), 0.05)
        self.assertLess(float(np.max(baseline_rel)), 0.1)
        self.assertLess(float(np.max(pilot_rel)), 0.1)
        self.assertEqual(baseline_diag["fit_weight_scheme"], "irls_inverse_model_variance_sq")
        self.assertEqual(pilot_diag["fit_weight_scheme"], "fixed_inverse_pilot_variance_sq")
        self.assertEqual(pilot_diag["fit_pilot_source"], "isotonic_log_variance")

    def test_isotonic_pilot_weighted_fit_handles_moderate_contamination(self) -> None:
        rng = np.random.default_rng(123)
        x = np.exp(np.linspace(np.log(1.0), np.log(1.0e4), 4000))
        true_coef = np.array([5.0, 1.5, 0.02])
        true_variance = quadratic_variance_model(x, true_coef)

        observed_variance = true_variance * np.exp(rng.normal(0.0, 0.3, size=x.shape))
        outlier_mask = rng.random(x.size) < 0.001
        observed_variance[outlier_mask] *= 100.0

        baseline_coef, _ = fit_quadratic_variance(x, observed_variance)
        pilot_coef, pilot_diag = fit_quadratic_variance_with_isotonic_pilot(x, observed_variance)

        baseline_rel = np.abs(baseline_coef - true_coef) / np.abs(true_coef)
        pilot_rel = np.abs(pilot_coef - true_coef) / np.abs(true_coef)
        baseline_pred = quadratic_variance_model(x, baseline_coef)
        pilot_pred = quadratic_variance_model(x, pilot_coef)

        baseline_curve_error = np.median(np.abs(baseline_pred - true_variance) / true_variance)
        pilot_curve_error = np.median(np.abs(pilot_pred - true_variance) / true_variance)
        baseline_frac_gt_10x = np.mean(baseline_pred > 10.0 * observed_variance)
        pilot_frac_gt_10x = np.mean(pilot_pred > 10.0 * observed_variance)

        self.assertLess(float(np.mean(baseline_rel)), 0.1)
        self.assertLess(float(np.mean(pilot_rel)), 0.08)
        self.assertLess(float(np.mean(pilot_rel)), float(np.mean(baseline_rel)))
        self.assertLess(float(pilot_curve_error), float(baseline_curve_error))
        self.assertLess(float(baseline_frac_gt_10x), 0.01)
        self.assertLess(float(pilot_frac_gt_10x), 0.01)
        self.assertEqual(pilot_diag["fit_pilot_source"], "isotonic_log_variance")

    def test_isotonic_pilot_weighted_fit_is_less_inflated_under_harsh_stress_contamination(self) -> None:
        rng = np.random.default_rng(123)
        x = np.exp(np.linspace(np.log(1.0), np.log(1.0e4), 4000))
        true_coef = np.array([5.0, 1.5, 0.02])
        true_variance = quadratic_variance_model(x, true_coef)

        observed_variance = true_variance * np.exp(rng.normal(0.0, 0.6, size=x.shape))
        outlier_mask = rng.random(x.size) < 0.005
        observed_variance[outlier_mask] *= 1.0e5

        baseline_coef, baseline_diag = fit_quadratic_variance(x, observed_variance)
        pilot_coef, pilot_diag = fit_quadratic_variance_with_isotonic_pilot(x, observed_variance)

        baseline_pred = quadratic_variance_model(x, baseline_coef)
        pilot_pred = quadratic_variance_model(x, pilot_coef)

        baseline_log_error = np.median(np.abs(np.log(baseline_pred) - np.log(true_variance)))
        pilot_log_error = np.median(np.abs(np.log(pilot_pred) - np.log(true_variance)))
        baseline_overprediction = np.median(baseline_pred / true_variance)
        pilot_overprediction = np.median(pilot_pred / true_variance)

        self.assertLess(pilot_log_error, baseline_log_error)
        self.assertLess(pilot_overprediction, baseline_overprediction)
        self.assertEqual(pilot_diag["fit_weight_scheme"], "fixed_inverse_pilot_variance_sq")
        self.assertEqual(pilot_diag["fit_pilot_source"], "isotonic_log_variance")
        self.assertEqual(baseline_diag["fit_weight_scheme"], "irls_inverse_model_variance_sq")


if __name__ == "__main__":
    unittest.main()
