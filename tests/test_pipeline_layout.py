"""Check relocated entrypoints without running reconstruction or writing results."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, mock_open, patch

from pipeline.paths import PROJECT_ROOT
from pipeline.project_config import load_project_config
from pipeline import run_dense_laser_pipeline as dense
from pipeline import run_laser_reference_comparison as reference


class PipelineLayoutTests(unittest.TestCase):
    def test_dense_driver_resolves_all_stages_and_preserves_resume_options(self):
        for recompute in (False, True):
            with self.subTest(recompute=recompute):
                arguments = ["run_dense_laser_pipeline", "--blender", sys.executable]
                if recompute:
                    arguments.append("--recompute")
                with patch.object(sys, "argv", arguments), \
                        patch.object(dense.subprocess, "run", return_value=Mock(returncode=0)) as run, \
                        patch.object(Path, "mkdir"), \
                        patch.object(Path, "open", mock_open()), \
                        patch.object(Path, "write_text"), redirect_stdout(io.StringIO()):
                    dense.main()
                self.assertEqual(run.call_count, 5)
                for call in run.call_args_list:
                    command = call.args[0]
                    script = next(Path(arg) for arg in command if str(arg).endswith(".py"))
                    self.assertTrue(script.is_file(), script)
                    self.assertEqual(script.parent, PROJECT_ROOT / "pipeline")
                    self.assertEqual(call.kwargs["cwd"], PROJECT_ROOT)
                stereo = run.call_args_list[0].args[0]
                self.assertIn("--all", stereo)
                self.assertEqual("--resume" in stereo, not recompute)

    def test_dense_driver_stops_on_failed_stage(self):
        with patch.object(sys, "argv", ["dense", "--blender", sys.executable]), \
                patch.object(dense.subprocess, "run", return_value=Mock(returncode=1)) as run, \
                patch.object(Path, "mkdir"), patch.object(Path, "open", mock_open()), \
                patch.object(Path, "write_text"), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "stereo failed"):
                dense.main()
        self.assertEqual(run.call_count, 1)

    def test_reference_driver_preserves_filters_and_optional_stages(self):
        cases = [
            (["--include-single", "--objects", "Metal"], 3),
            (["--no-plots", "--objects", "Metal"], 2),
            (["--self-check"], 1),
        ]
        for arguments, count in cases:
            with self.subTest(arguments=arguments), \
                    patch.object(sys, "argv", ["reference", *arguments]), \
                    patch.object(reference.subprocess, "run") as run:
                reference.main()
            self.assertEqual(run.call_count, count)
            self.assertEqual(run.call_args_list[0].args[0][2:], arguments)
            for call in run.call_args_list:
                script = Path(call.args[0][1])
                self.assertTrue(script.is_file(), script)
                self.assertEqual(script.parent, PROJECT_ROOT / "pipeline")
                self.assertEqual(call.kwargs["cwd"], PROJECT_ROOT)
                self.assertTrue(call.kwargs["check"])

    def test_config_path_is_independent_of_working_directory(self):
        expected = load_project_config(PROJECT_ROOT / "project_config.example.yaml")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                actual = load_project_config(Path("project_config.example.yaml"))
            finally:
                os.chdir(previous)
        self.assertEqual(actual, expected)

    def test_reference_self_check_can_run_as_a_direct_script_from_elsewhere(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "pipeline/run_laser_reference_comparison.py"),
                 "--self-check"],
                cwd=directory, text=True, capture_output=True, timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Known rigid transform and metric checks passed", result.stdout)

    def test_reference_self_check_can_run_as_a_module(self):
        result = subprocess.run(
            [sys.executable, "-m", "pipeline.run_laser_reference_comparison", "--self-check"],
            cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
