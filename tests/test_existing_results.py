"""Optional regressions against local datasets; never replace saved reports."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

from pipeline.paths import PROJECT_ROOT


class ExistingResultsTests(unittest.TestCase):
    @unittest.skipUnless(
        (PROJECT_ROOT / "LaserScans/comparison_results/all_comparisons.json").is_file(),
        "Local scanner exports and comparison results are distributed separately",
    )
    def test_all_saved_laser_comparisons_still_verify_without_rewriting_reports(self):
        from pipeline import verify_laser_comparisons as verification

        report_path = verification.OUT / "verification.json"
        original = report_path.read_bytes()
        captured = []

        def capture_report(path, text, *args, **kwargs):
            self.assertEqual(path, report_path)
            captured.append(json.loads(text))
            return len(text)

        with patch.object(sys, "argv", ["verify_laser_comparisons"]), \
                patch.object(Path, "write_text", capture_report), \
                redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            verification.main()
        self.assertEqual(stopped.exception.code, 0, captured)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["status"], "complete_pass")
        self.assertEqual(captured[0]["verified_comparisons"], 13)
        self.assertEqual(captured[0]["raw_original_files_verified"], 24)
        self.assertEqual(report_path.read_bytes(), original)

    @unittest.skipUnless(
        (PROJECT_ROOT / "2026-10-29/dense_results/models_report.json").is_file(),
        "Local dense results are distributed separately",
    )
    def test_dense_geometry_and_saved_models_without_rewriting_report(self):
        from pipeline import dense_laser_pipeline as dense

        report_path = PROJECT_ROOT / "2026-10-29/dense_results/verification.json"
        original = report_path.read_bytes()
        with patch.object(dense, "dump") as save, redirect_stdout(io.StringIO()):
            runpy.run_module("pipeline.verify_dense_laser", run_name="__main__")
        save.assert_called_once()
        self.assertEqual(save.call_args.args[0], report_path)
        result = save.call_args.args[1]
        self.assertEqual(result["verified_frames"], 21)
        self.assertEqual(result["verified_models"], 11)
        self.assertEqual(report_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
