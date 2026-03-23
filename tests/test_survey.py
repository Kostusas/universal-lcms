from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SRC, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from survey_qc_files import extract_run_metadata, summarize_ms1_run, survey_file


class SurveySmokeTest(unittest.TestCase):
    def test_extract_run_metadata_from_local_qc(self) -> None:
        metadata = extract_run_metadata(ROOT / "test-data" / "QC2_rtclip.mzML")
        self.assertEqual(metadata["format"], "mzML")
        self.assertIn("ZenoTOF 7600", metadata["instrument_names"])
        self.assertGreaterEqual(len(metadata["software"]), 1)

    def test_summarize_ms1_run_from_local_qc(self) -> None:
        summary = summarize_ms1_run(ROOT / "test-data" / "QC2_rtclip.mzML")
        self.assertEqual(summary["ms1_scan_count"], 4201)
        self.assertEqual(summary["scan_time_unit"], "second")
        self.assertGreater(summary["scan_delta_seconds"]["p50"], 0.0)
        self.assertGreater(summary["base_peak_intensity"]["max"], 3.0e5)
        self.assertLessEqual(summary["positive_intensity"]["p01"], summary["positive_intensity"]["p10"])
        self.assertLessEqual(summary["positive_intensity"]["p10"], summary["positive_intensity"]["p50"])
        self.assertLessEqual(summary["positive_intensity"]["p50"], summary["positive_intensity"]["p90"])
        self.assertLessEqual(summary["positive_intensity"]["p90"], summary["positive_intensity"]["p99"])
        self.assertLessEqual(summary["positive_intensity"]["p99"], summary["positive_intensity"]["p999"])

    def test_survey_file_keeps_fit_and_omits_rt_segments(self) -> None:
        report = survey_file(ROOT / "test-data" / "QC2_rtclip.mzML", include_fit=False)
        self.assertIn("scan_statistics", report)
        self.assertNotIn("rt_segments", report)
        self.assertIn("positive_intensity", report["scan_statistics"])


if __name__ == "__main__":
    unittest.main()
