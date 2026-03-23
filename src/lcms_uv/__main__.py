from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import DEFAULT_MZ_BIN_WIDTH, estimate_vst_from_file


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Estimate LC-MS quadratic variance model and unit-variance transform."
    )
    p.add_argument("file", type=str, help="Input .mzML or .mzXML file.")
    p.add_argument("--json-out", type=str, default="", help="Optional path to save JSON result.")
    p.add_argument(
        "--mz-bin-width",
        type=float,
        default=DEFAULT_MZ_BIN_WIDTH,
        help="Coarse m/z bin width in Da used before internal ppm-based mass-track merging.",
    )
    p.add_argument("--min-intensity", type=float, default=0.0)
    p.add_argument(
        "--min-points",
        type=int,
        default=5,
        help="Minimum points required for both track retention and contiguous segments.",
    )
    p.add_argument("--same-scan-aggregation", choices=["max", "sum"], default="max")
    p.add_argument(
        "--no-unit-variance",
        action="store_true",
        help="Disable post-fit unit-variance scaling calibration.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    fit, diagnostics = estimate_vst_from_file(
        path=args.file,
        mz_bin_width=args.mz_bin_width,
        min_intensity=args.min_intensity,
        min_points=args.min_points,
        same_scan_aggregation=args.same_scan_aggregation,
        calibrate_unit_variance=not bool(args.no_unit_variance),
    )

    result = {
        "file": str(Path(args.file).resolve()),
        "model": {
            "sigma0_sq": fit.sigma0_sq,
            "alpha": fit.alpha,
            "beta": fit.beta,
            "unit_scale": fit.unit_scale,
            "variance_formula": "v(x) = sigma0_sq + alpha*x + beta*x^2",
            "vst_formula": "z(x) = unit_scale * quadratic_vst(x, sigma0_sq, alpha, beta)",
        },
        "summary": {
            "n_scans": fit.n_scans,
            "n_tracks": fit.n_tracks,
            "n_pairs": fit.n_pairs,
            "n_fit_points": fit.n_fit_points,
        },
        "diagnostics": diagnostics,
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")


if __name__ == "__main__":
    main()
