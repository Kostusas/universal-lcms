from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SRC, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from survey_qc_files import survey_file


DEFAULT_MANIFEST = ROOT / "notes" / "metabolomics_workbench_candidate_manifest.json"
DEFAULT_DOWNLOAD_DIR = Path("/tmp") / f"universal-lcms-workbench-downloads-{os.environ.get('USER', 'user')}"
DEFAULT_OUTPUT_DIR = Path.home() / "universal-lcms-workbench-runs"
DEFAULT_DOWNLOAD_RETRIES = 6
DEFAULT_DOWNLOAD_RETRY_BASE_SECONDS = 5.0
RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    token = token.strip("._")
    return token or "item"


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def _candidate_pool(manifest: dict, group: str) -> list[dict]:
    if group == "candidates":
        return list(manifest.get("candidates", []))
    if group == "reserve_candidates":
        return list(manifest.get("reserve_candidates", []))
    if group == "all":
        return list(manifest.get("candidates", [])) + list(manifest.get("reserve_candidates", []))
    raise ValueError(f"Unsupported group: {group}")


def _download_basename(candidate: dict) -> str:
    raw_name = Path(candidate["source"]["file_path_in_archive"]).name
    stem = _slugify(Path(raw_name).stem)
    suffix = Path(raw_name).suffix.lower()
    parts = [
        _slugify(candidate["study_id"]),
        _slugify(candidate["analysis_id"]),
        stem,
    ]
    return "_".join(parts) + suffix


def _output_basename(index: int, candidate: dict) -> str:
    return f"{index:03d}_{_slugify(candidate['study_id'])}_{_slugify(candidate['label'])}.json"


def _retry_delay_seconds(base_seconds: float, attempt: int) -> float:
    return base_seconds * (2 ** (attempt - 1)) + random.uniform(0.0, base_seconds)


def _is_retryable_download_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code in RETRYABLE_HTTP_STATUS_CODES
    return isinstance(exc, (URLError, TimeoutError, OSError))


def _download_candidate(
    candidate: dict,
    download_dir: Path,
    *,
    overwrite: bool,
    max_attempts: int,
    retry_base_seconds: float,
) -> dict:
    download_dir.mkdir(parents=True, exist_ok=True)
    out_path = download_dir / _download_basename(candidate)
    expected_size = int(candidate["size_bytes"])

    if out_path.exists() and not overwrite:
        actual_size = int(out_path.stat().st_size)
        if actual_size == expected_size:
            return {
                "status": "reused_existing",
                "local_path": str(out_path.resolve()),
                "size_bytes_expected": expected_size,
                "size_bytes_actual": actual_size,
                "downloaded_at_utc": None,
            }

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    form = {
        "A": candidate["source"]["archive_name"],
        "F": candidate["source"]["file_path_in_archive"],
    }
    request = Request(
        candidate["source"]["download_url"],
        data=urlencode(form).encode("utf-8"),
        headers={"User-Agent": "universal-lcms-workbench-runner/1.0"},
    )

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        tmp_path.unlink(missing_ok=True)
        try:
            with urlopen(request, timeout=300) as response, tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
            break
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            last_error = exc
            if attempt >= max_attempts or not _is_retryable_download_error(exc):
                raise
            delay_seconds = _retry_delay_seconds(retry_base_seconds, attempt)
            print(
                f"[download retry {attempt}/{max_attempts}] "
                f"{candidate['label']} failed with {type(exc).__name__}: {exc}. "
                f"Retrying in {delay_seconds:.1f}s...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay_seconds)

    if last_error is not None and not tmp_path.exists() and not out_path.exists():
        raise RuntimeError(f"Download failed for {candidate['label']}") from last_error

    actual_size = int(tmp_path.stat().st_size)
    if expected_size and actual_size != expected_size:
        raise RuntimeError(
            f"Downloaded size mismatch for {candidate['label']}: "
            f"expected {expected_size} bytes, got {actual_size} bytes."
        )

    tmp_path.replace(out_path)
    return {
        "status": "downloaded",
        "local_path": str(out_path.resolve()),
        "size_bytes_expected": expected_size,
        "size_bytes_actual": actual_size,
        "downloaded_at_utc": _now_utc(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download one Metabolomics Workbench candidate from the manifest and run the survey pipeline."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--group", choices=["candidates", "reserve_candidates", "all"], default="candidates")
    parser.add_argument("--index", type=int, required=True, help="Zero-based candidate index within the selected group.")
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--download-retries", type=int, default=DEFAULT_DOWNLOAD_RETRIES)
    parser.add_argument("--download-retry-base-seconds", type=float, default=DEFAULT_DOWNLOAD_RETRY_BASE_SECONDS)
    parser.add_argument("--overwrite-download", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--no-fit", action="store_true")
    parser.add_argument("--fit-method", choices=["isotonic_pilot", "irls"], default="isotonic_pilot")
    parser.add_argument("--mz-bin-width", type=float, default=0.001)
    parser.add_argument("--mz-tolerance-ppm", type=float, default=5.0)
    parser.add_argument("--min-intensity", type=float, default=0.0)
    parser.add_argument("--min-points", type=int, default=5)
    parser.add_argument("--same-scan-aggregation", choices=["max", "sum"], default="max")
    parser.add_argument("--no-unit-variance", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    pool = _candidate_pool(manifest, args.group)

    if args.index < 0 or args.index >= len(pool):
        raise IndexError(f"Candidate index {args.index} is out of range for group {args.group} (n={len(pool)}).")

    candidate = pool[args.index]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _output_basename(args.index, candidate)

    if output_path.exists() and not args.overwrite_output:
        payload = {
            "status": "skipped_existing_output",
            "output_path": str(output_path),
            "candidate_index": args.index,
            "candidate_label": candidate["label"],
        }
        print(json.dumps(payload, indent=2))
        return

    download = _download_candidate(
        candidate,
        args.download_dir.resolve(),
        overwrite=bool(args.overwrite_download),
        max_attempts=max(1, int(args.download_retries)),
        retry_base_seconds=max(0.0, float(args.download_retry_base_seconds)),
    )

    survey = survey_file(
        download["local_path"],
        include_fit=not bool(args.no_fit),
        fit_method=args.fit_method,
        mz_bin_width=args.mz_bin_width,
        min_intensity=args.min_intensity,
        min_points=args.min_points,
        same_scan_aggregation=args.same_scan_aggregation,
        calibrate_unit_variance=not bool(args.no_unit_variance),
        mz_tolerance_ppm=args.mz_tolerance_ppm,
    )

    payload = {
        "generated_at_utc": _now_utc(),
        "manifest_path": str(manifest_path),
        "candidate_group": args.group,
        "candidate_index": args.index,
        "candidate_count": len(pool),
        "candidate": candidate,
        "download": download,
        "survey": survey,
    }

    text = json.dumps(payload, indent=2)
    print(text)
    output_path.write_text(text + "\n")


if __name__ == "__main__":
    main()
