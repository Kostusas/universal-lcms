from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from pyteomics import mzml, mzxml


DEFAULT_MZ_BIN_WIDTH = 1e-3
DEFAULT_MZ_TOLERANCE_PPM = 5.0

Track = tuple[float, np.ndarray, np.ndarray]


def iter_ms1_arrays(path: str | Path):
    """Yield `(mz, intensity)` arrays for each MS1 scan in an mzML/mzXML file."""
    in_path = Path(path)
    suffix = in_path.suffix.lower()

    if suffix == ".mzml":
        with mzml.MzML(str(in_path)) as reader:
            for spec in reader:
                if int(spec.get("ms level", 0)) != 1:
                    continue
                mz_array = np.asarray(spec.get("m/z array", []), dtype=float)
                intensity_array = np.asarray(spec.get("intensity array", []), dtype=float)
                n = min(mz_array.size, intensity_array.size)
                yield mz_array[:n], intensity_array[:n]
        return

    if suffix == ".mzxml":
        with mzxml.MzXML(str(in_path)) as reader:
            for spec in reader:
                if int(spec.get("msLevel", 0)) != 1:
                    continue
                mz_array = np.asarray(spec.get("m/z array", spec.get("mz array", [])), dtype=float)
                intensity_array = np.asarray(spec.get("intensity array", []), dtype=float)
                n = min(mz_array.size, intensity_array.size)
                yield mz_array[:n], intensity_array[:n]
        return

    raise ValueError(f"Unsupported extension: {in_path.suffix}. Use .mzML or .mzXML")


def split_consecutive_segments(scans: np.ndarray) -> list[np.ndarray]:
    """Return index segments for consecutive scan runs within a sparse track."""
    if scans.size == 0:
        return []
    cuts = np.where(np.diff(scans) != 1)[0]
    idx = np.arange(scans.size, dtype=np.int32)
    return [seg for seg in np.split(idx, cuts + 1) if seg.size > 0]


def _ppm_gap(a: float, b: float, ppm: float) -> float:
    return (ppm * 1e-6) * max(abs(a), abs(b))


def _within_ppm(center: float, value: float, ppm: float) -> bool:
    return abs(value - center) <= _ppm_gap(center, value, ppm)


def _cluster_sorted_mz(sorted_mz: np.ndarray, ppm: float) -> list[np.ndarray]:
    if sorted_mz.size == 0:
        return []

    groups: list[list[int]] = [[0]]
    center = float(sorted_mz[0])
    n_group = 1
    for i in range(1, sorted_mz.size):
        value = float(sorted_mz[i])
        if _within_ppm(center, value, ppm):
            groups[-1].append(i)
            n_group += 1
            center += (value - center) / float(n_group)
        else:
            groups.append([i])
            center = value
            n_group = 1
    return [np.asarray(group, dtype=np.int32) for group in groups]


def _aggregate_same_scan(scans: np.ndarray, intensity: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scans, kind="mergesort")
    scans_sorted = scans[order]
    intensity_sorted = intensity[order]

    unique_scans, starts = np.unique(scans_sorted, return_index=True)
    values = np.empty(unique_scans.size, dtype=float)

    for i, start in enumerate(starts):
        stop = starts[i + 1] if i + 1 < starts.size else scans_sorted.size
        block = intensity_sorted[start:stop]
        values[i] = float(np.max(block)) if mode == "max" else float(np.sum(block))

    return unique_scans.astype(np.int32), values


def _consensus_mz(mz: np.ndarray, intensity: np.ndarray) -> float:
    weights = np.maximum(intensity, 0.0)
    if float(np.sum(weights)) > 0.0:
        return float(np.average(mz, weights=weights))
    return float(np.mean(mz))


def extract_sparse_tracks(
    path: str | Path,
    mz_bin_width: float = DEFAULT_MZ_BIN_WIDTH,
    min_intensity: float = 0.0,
    min_points: int = 5,
    same_scan_aggregation: str = "max",
    mz_tolerance_ppm: float = DEFAULT_MZ_TOLERANCE_PPM,
) -> tuple[list[Track], int]:
    """Build sparse mass tracks using coarse Da bins plus ppm-based merging."""
    if mz_bin_width <= 0.0:
        raise ValueError("mz_bin_width must be positive.")
    if mz_tolerance_ppm <= 0.0:
        raise ValueError("mz_tolerance_ppm must be positive.")
    if same_scan_aggregation not in {"max", "sum"}:
        raise ValueError("same_scan_aggregation must be 'max' or 'sum'.")

    coarse_tree: dict[int, list[tuple[float, int, float]]] = defaultdict(list)
    n_scans = 0

    for scan_idx, (mz_array, intensity_array) in enumerate(iter_ms1_arrays(path)):
        n_scans += 1
        if mz_array.size == 0:
            continue

        mask = (
            np.isfinite(mz_array)
            & np.isfinite(intensity_array)
            & (mz_array > 0.0)
            & (intensity_array >= float(min_intensity))
        )
        if not np.any(mask):
            continue

        mz_sel = mz_array[mask]
        intensity_sel = intensity_array[mask]
        coarse_idx = np.floor(mz_sel / float(mz_bin_width)).astype(np.int64)
        for mz_value, idx, intensity in zip(mz_sel, coarse_idx, intensity_sel, strict=False):
            coarse_tree[int(idx)].append((float(mz_value), int(scan_idx), float(intensity)))

    if not coarse_tree:
        return [], n_scans

    coarse_entries = []
    for idx in sorted(coarse_tree):
        values = coarse_tree[idx]
        mz_values = np.asarray([v[0] for v in values], dtype=float)
        scans = np.asarray([v[1] for v in values], dtype=np.int32)
        intensity = np.asarray([v[2] for v in values], dtype=float)
        coarse_entries.append(
            (
                idx,
                mz_values,
                scans,
                intensity,
                _consensus_mz(mz_values, intensity),
            )
        )

    merged_groups: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    cur_mz: list[np.ndarray] = []
    cur_scans: list[np.ndarray] = []
    cur_intensity: list[np.ndarray] = []
    cur_center: float | None = None
    cur_weighted_sum = 0.0
    cur_weight_sum = 0.0
    cur_count = 0
    cur_mean_sum = 0.0

    for _idx, mz_values, scans, intensity, entry_center in coarse_entries:
        if cur_center is None or _within_ppm(cur_center, entry_center, mz_tolerance_ppm):
            cur_mz.append(mz_values)
            cur_scans.append(scans)
            cur_intensity.append(intensity)
            weights = np.maximum(intensity, 0.0)
            cur_weighted_sum += float(np.sum(mz_values * weights))
            cur_weight_sum += float(np.sum(weights))
            cur_mean_sum += float(np.sum(mz_values))
            cur_count += int(mz_values.size)
            cur_center = (
                cur_weighted_sum / cur_weight_sum
                if cur_weight_sum > 0.0
                else cur_mean_sum / float(cur_count)
            )
            continue

        merged_groups.append((np.concatenate(cur_mz), np.concatenate(cur_scans), np.concatenate(cur_intensity)))
        cur_mz = [mz_values]
        cur_scans = [scans]
        cur_intensity = [intensity]
        cur_center = entry_center
        weights = np.maximum(intensity, 0.0)
        cur_weighted_sum = float(np.sum(mz_values * weights))
        cur_weight_sum = float(np.sum(weights))
        cur_mean_sum = float(np.sum(mz_values))
        cur_count = int(mz_values.size)

    if cur_mz:
        merged_groups.append((np.concatenate(cur_mz), np.concatenate(cur_scans), np.concatenate(cur_intensity)))

    tracks: list[Track] = []
    for mz_values, scans, intensity in merged_groups:
        order = np.argsort(mz_values, kind="mergesort")
        mz_sorted = mz_values[order]
        scans_sorted = scans[order]
        intensity_sorted = intensity[order]

        for group in _cluster_sorted_mz(mz_sorted, mz_tolerance_ppm):
            group_scans, group_intensity = _aggregate_same_scan(scans_sorted[group], intensity_sorted[group], same_scan_aggregation)
            if group_scans.size < min_points:
                continue
            tracks.append(
                (
                    _consensus_mz(mz_sorted[group], intensity_sorted[group]),
                    group_scans,
                    group_intensity,
                )
            )

    tracks.sort(key=lambda item: item[0])
    return tracks, n_scans
