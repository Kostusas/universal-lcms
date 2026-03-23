from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from lxml import etree
from pyteomics import mzml, mzxml

from lcms_uv import DEFAULT_MZ_BIN_WIDTH, DEFAULT_MZ_TOLERANCE_PPM, estimate_vst_from_file

QUANTILES = (
    ("p01", 0.01),
    ("p10", 0.10),
    ("p50", 0.50),
    ("p90", 0.90),
    ("p99", 0.99),
    ("p999", 0.999),
)


def iter_ms1_spectra(path: str | Path):
    in_path = Path(path)
    suffix = in_path.suffix.lower()

    if suffix == ".mzml":
        with mzml.MzML(str(in_path)) as reader:
            for spec in reader:
                if int(spec.get("ms level", 0)) != 1:
                    continue
                yield spec
        return

    if suffix == ".mzxml":
        with mzxml.MzXML(str(in_path)) as reader:
            for spec in reader:
                if int(spec.get("msLevel", 0)) != 1:
                    continue
                yield spec
        return

    raise ValueError(f"Unsupported extension: {in_path.suffix}. Use .mzML or .mzXML")


def _quantile_summary(
    values: list[float] | np.ndarray,
    *,
    include_max: bool = False,
    include_span: bool = False,
) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}

    payload = {
        "n": int(arr.size),
    }
    for label, q in QUANTILES:
        payload[label] = float(np.quantile(arr, q))
    if include_max:
        payload["max"] = float(np.max(arr))
    if include_span:
        payload["span"] = float(np.max(arr) - np.min(arr))
    return payload


def _local_name(tag: str) -> str:
    return etree.QName(tag).localname


def _child_cvparam_names(elem: etree._Element) -> list[str]:
    names: list[str] = []
    for child in elem:
        if _local_name(child.tag) != "cvParam":
            continue
        name = child.get("name")
        if name:
            names.append(name)
    return sorted(names)


def _child_component_terms(parent: etree._Element, component_name: str) -> list[str]:
    values: list[str] = []
    for child in parent:
        if _local_name(child.tag) != component_name:
            continue
        values.extend(_child_cvparam_names(child))
    return sorted(set(values))


def _scan_time_unit_name(path: str | Path) -> str | None:
    in_path = Path(path)
    if in_path.suffix.lower() != ".mzml":
        return None

    with in_path.open("rb") as handle:
        for _event, elem in etree.iterparse(handle, events=("end",), huge_tree=True):
            if _local_name(elem.tag) != "cvParam":
                elem.clear()
                continue
            if elem.get("name") == "scan start time":
                return elem.get("unitName") or elem.get("unitAccession")
            elem.clear()
    return None


def _scan_time_scale(unit_name: str | None) -> float:
    if unit_name is None:
        return 1.0

    token = unit_name.strip().lower()
    if token.startswith("sec"):
        return 1.0
    if token.startswith("min"):
        return 60.0
    if token.startswith("hour"):
        return 3600.0
    return 1.0


def _duration_to_seconds(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if not isinstance(value, str):
        return None

    token = value.strip()
    if not token:
        return None
    if not token.startswith("PT"):
        try:
            return float(token)
        except ValueError:
            return None

    token = token[2:]
    minutes = 0.0
    seconds = 0.0
    if "M" in token:
        head, token = token.split("M", 1)
        minutes = float(head) if head else 0.0
    if token.endswith("S"):
        seconds = float(token[:-1]) if token[:-1] else 0.0
    return (60.0 * minutes) + seconds


def _scan_start_time_seconds(spec: dict, scale: float) -> float | None:
    if "scanList" in spec:
        scan_list = spec.get("scanList") or {}
        scans = scan_list.get("scan") or []
        if scans:
            value = scans[0].get("scan start time")
            if value is not None:
                return float(value) * scale

    if "retentionTime" in spec:
        return _duration_to_seconds(spec.get("retentionTime"))
    return None


def _scalar(spec: dict, *keys: str) -> float | None:
    for key in keys:
        value = spec.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _polarity(spec: dict) -> str | None:
    if "positive scan" in spec:
        return "positive"
    if "negative scan" in spec:
        return "negative"
    polarity = spec.get("polarity")
    if isinstance(polarity, str):
        token = polarity.strip()
        if token:
            return token
    return None


def extract_run_metadata(path: str | Path) -> dict:
    in_path = Path(path)
    payload = {
        "path": str(in_path.resolve()),
        "name": in_path.name,
        "suffix": in_path.suffix.lower(),
        "size_bytes": int(in_path.stat().st_size),
    }

    if in_path.suffix.lower() != ".mzml":
        return payload

    source_files = []
    software = []
    instruments = []
    file_content_terms: list[str] = []

    with in_path.open("rb") as handle:
        context = etree.iterparse(handle, events=("start", "end"), huge_tree=True)
        for event, elem in context:
            name = _local_name(elem.tag)

            if event == "start" and name == "spectrumList":
                break

            if event != "end":
                continue

            if name == "fileContent":
                file_content_terms = _child_cvparam_names(elem)
            elif name == "sourceFile":
                source_files.append(
                    {
                        "id": elem.get("id"),
                        "name": elem.get("name"),
                        "location": elem.get("location"),
                        "terms": _child_cvparam_names(elem),
                    }
                )
            elif name == "software":
                software.append(
                    {
                        "id": elem.get("id"),
                        "version": elem.get("version"),
                        "terms": _child_cvparam_names(elem),
                    }
                )
            elif name == "instrumentConfiguration":
                component_list = None
                software_ref = None
                for child in elem:
                    child_name = _local_name(child.tag)
                    if child_name == "componentList":
                        component_list = child
                    elif child_name == "softwareRef":
                        software_ref = child.get("ref")

                instruments.append(
                    {
                        "id": elem.get("id"),
                        "instrument_terms": _child_cvparam_names(elem),
                        "ion_sources": _child_component_terms(component_list, "source") if component_list is not None else [],
                        "analyzers": _child_component_terms(component_list, "analyzer") if component_list is not None else [],
                        "detectors": _child_component_terms(component_list, "detector") if component_list is not None else [],
                        "software_ref": software_ref,
                    }
                )

    payload.update(
        {
            "format": "mzML",
            "file_content_terms": file_content_terms,
            "source_files": source_files,
            "software": software,
            "instrument_configurations": instruments,
            "instrument_names": sorted(
                {
                    name
                    for item in instruments
                    for name in item.get("instrument_terms", [])
                }
            ),
        }
    )
    return payload


def summarize_ms1_run(path: str | Path) -> dict:
    unit_name = _scan_time_unit_name(path)
    time_scale = _scan_time_scale(unit_name)

    scan_times: list[float] = []
    point_counts: list[float] = []
    positive_point_counts: list[float] = []
    base_peaks: list[float] = []
    tics: list[float] = []
    mz_lows: list[float] = []
    mz_highs: list[float] = []
    scan_positive_medians: list[float] = []
    scan_positive_q95: list[float] = []
    positive_intensity_chunks: list[np.ndarray] = []
    polarities: set[str] = set()

    global_max_intensity = 0.0
    total_points = 0
    total_positive_points = 0

    for spec in iter_ms1_spectra(path):
        mz_array = np.asarray(spec.get("m/z array", spec.get("mz array", [])), dtype=float)
        intensity_array = np.asarray(spec.get("intensity array", []), dtype=float)
        n_points = int(min(mz_array.size, intensity_array.size))

        point_counts.append(float(n_points))
        total_points += n_points

        if n_points == 0:
            positive_point_counts.append(0.0)
        else:
            mz_slice = mz_array[:n_points]
            intensity_slice = intensity_array[:n_points]
            positive = intensity_slice[np.isfinite(intensity_slice) & (intensity_slice > 0.0)]

            positive_point_counts.append(float(positive.size))
            total_positive_points += int(positive.size)
            if positive.size:
                global_max_intensity = max(global_max_intensity, float(np.max(positive)))
                positive_intensity_chunks.append(positive)
                scan_positive_medians.append(float(np.median(positive)))
                scan_positive_q95.append(float(np.quantile(positive, 0.95)))

            mz_low = _scalar(spec, "lowest observed m/z", "lowMz")
            if mz_low is None and np.any(np.isfinite(mz_slice)):
                mz_low = float(np.nanmin(mz_slice))
            if mz_low is not None:
                mz_lows.append(mz_low)

            mz_high = _scalar(spec, "highest observed m/z", "highMz")
            if mz_high is None and np.any(np.isfinite(mz_slice)):
                mz_high = float(np.nanmax(mz_slice))
            if mz_high is not None:
                mz_highs.append(mz_high)

            base_peak = _scalar(spec, "base peak intensity", "basePeakIntensity")
            if base_peak is None and positive.size:
                base_peak = float(np.max(positive))
            if base_peak is not None:
                base_peaks.append(base_peak)

            tic = _scalar(spec, "total ion current", "totIonCurrent")
            if tic is None and positive.size:
                tic = float(np.sum(positive))
            if tic is not None:
                tics.append(tic)

        scan_time = _scan_start_time_seconds(spec, scale=time_scale)
        if scan_time is not None:
            scan_times.append(scan_time)

        polarity = _polarity(spec)
        if polarity:
            polarities.add(polarity)

    deltas = np.diff(np.asarray(scan_times, dtype=float)) if len(scan_times) >= 2 else np.array([], dtype=float)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
    rates = 1.0 / deltas if deltas.size else np.array([], dtype=float)
    positive_intensity = (
        np.concatenate(positive_intensity_chunks)
        if positive_intensity_chunks
        else np.array([], dtype=float)
    )

    return {
        "ms1_scan_count": int(len(point_counts)),
        "scan_time_unit": "second" if scan_times else (unit_name or "unknown"),
        "scan_start_time_seconds": _quantile_summary(scan_times, include_span=True),
        "scan_delta_seconds": _quantile_summary(deltas),
        "scan_rate_hz": _quantile_summary(rates),
        "centroids_per_scan": _quantile_summary(point_counts),
        "positive_centroids_per_scan": _quantile_summary(positive_point_counts),
        "lowest_observed_mz": _quantile_summary(mz_lows),
        "highest_observed_mz": _quantile_summary(mz_highs),
        "positive_intensity": _quantile_summary(positive_intensity, include_max=True),
        "base_peak_intensity": _quantile_summary(base_peaks, include_max=True),
        "total_ion_current": _quantile_summary(tics, include_max=True),
        "positive_intensity_median_per_scan": _quantile_summary(scan_positive_medians, include_max=True),
        "positive_intensity_q95_per_scan": _quantile_summary(scan_positive_q95, include_max=True),
        "polarity_modes": sorted(polarities),
        "global_max_intensity": float(global_max_intensity),
        "total_centroids": int(total_points),
        "total_positive_centroids": int(total_positive_points),
    }


def survey_file(
    path: str | Path,
    *,
    include_fit: bool = True,
    mz_bin_width: float = DEFAULT_MZ_BIN_WIDTH,
    min_intensity: float = 0.0,
    min_points: int = 5,
    same_scan_aggregation: str = "max",
    calibrate_unit_variance: bool = True,
    mz_tolerance_ppm: float = DEFAULT_MZ_TOLERANCE_PPM,
) -> dict:
    in_path = Path(path)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file": extract_run_metadata(in_path),
        "scan_statistics": summarize_ms1_run(in_path),
    }

    if not include_fit:
        return report

    fit, diagnostics = estimate_vst_from_file(
        path=in_path,
        mz_bin_width=mz_bin_width,
        min_intensity=min_intensity,
        min_points=min_points,
        same_scan_aggregation=same_scan_aggregation,
        calibrate_unit_variance=calibrate_unit_variance,
        mz_tolerance_ppm=mz_tolerance_ppm,
    )

    report["fit_settings"] = {
        "mz_bin_width_da": float(mz_bin_width),
        "mz_tolerance_ppm": float(mz_tolerance_ppm),
        "min_intensity": float(min_intensity),
        "min_points": int(min_points),
        "same_scan_aggregation": same_scan_aggregation,
        "calibrate_unit_variance": bool(calibrate_unit_variance),
    }
    report["variance_fit"] = {
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
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Survey LC-MS QC files: metadata, compact scan/intensity stats, and variance-fit diagnostics."
    )
    parser.add_argument("files", nargs="+", help="Input .mzML or .mzXML files.")
    parser.add_argument("--json-out", type=str, default="", help="Optional path to save the JSON report.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation level.")
    parser.add_argument("--skip-errors", action="store_true", help="Keep going if one file fails.")
    parser.add_argument("--no-fit", action="store_true", help="Skip the variance-fit step.")
    parser.add_argument("--mz-bin-width", type=float, default=DEFAULT_MZ_BIN_WIDTH)
    parser.add_argument("--mz-tolerance-ppm", type=float, default=DEFAULT_MZ_TOLERANCE_PPM)
    parser.add_argument("--min-intensity", type=float, default=0.0)
    parser.add_argument("--min-points", type=int, default=5)
    parser.add_argument("--same-scan-aggregation", choices=["max", "sum"], default="max")
    parser.add_argument("--no-unit-variance", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    reports = []
    errors = []

    for raw_path in args.files:
        path = Path(raw_path)
        try:
            reports.append(
                survey_file(
                    path,
                    include_fit=not bool(args.no_fit),
                    mz_bin_width=args.mz_bin_width,
                    mz_tolerance_ppm=args.mz_tolerance_ppm,
                    min_intensity=args.min_intensity,
                    min_points=args.min_points,
                    same_scan_aggregation=args.same_scan_aggregation,
                    calibrate_unit_variance=not bool(args.no_unit_variance),
                )
            )
        except Exception as exc:
            if not args.skip_errors:
                raise
            errors.append({"file": str(path.resolve()), "error": str(exc)})

    payload = {
        "n_files": len(reports),
        "include_fit": not bool(args.no_fit),
        "files": reports,
    }
    if errors:
        payload["errors"] = errors

    text = json.dumps(payload, indent=args.indent)
    print(text)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")


if __name__ == "__main__":
    main()
