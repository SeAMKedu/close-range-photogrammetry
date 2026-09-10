# Calibration baseline: 2026-05-19 dataset

Environment: `.venv`, Python 3.11.16. All dependencies from `requirements.txt`
installed successfully; `uv pip check` and OpenCV/NumPy/Open3D/PyYAML imports passed.

Run from the project directory:

```powershell
.venv/Scripts/python.exe -u run_calibration_baseline.py
```

Local settings are in `project_config.yaml`. Intrinsics use
`chessboards/intrinsics_img_sets`; stereo uses `chessboards/extrinsics_img_sets_NEW`.
Reference camera: Basler 60gc. Board: 16 x 11 inner corners, 7 mm squares.
The printed board label in the inspected image confirms 12 x 17 squares and 7 mm.

| Calibration | RMS (px) | Images/shared frames | Baseline threshold |
| --- | ---: | ---: | --- |
| Basler 60gc intrinsic | 0.201136 | 15/15 | < 0.5: pass |
| Basler 60gm intrinsic | 0.250141 | 14/15 | < 0.5: pass |
| Hikrobot intrinsic | 0.370944 | 13/15 | < 0.5: pass |
| Basler 60gc to Basler 60gm | 0.216739 | 14/15 | < 1.0: pass |
| Basler 60gc to Hikrobot | 0.245693 | 14/15 | < 1.0: pass |

These thresholds are baseline screening criteria, not proof of measurement accuracy.
All object and NEW stereo JPEGs were readable and matched the corresponding
intrinsic calibration resolution (Basler: 1600 x 1200; Hikrobot: 4096 x 3000).
Representative generated corner overlays were visually inspected, not every image.

Intrinsic detection failures: Basler 60gm frame 3; Hikrobot frames 11 and 12.
Hikrobot source frame 7 failed in the NEW stereo set; after synchronization this
is logical reference frame 13. The stereo calibration uses the other 14 frames.

The original file prefixes were not synchronized across camera directories. In
the NEW set, both target cameras have a cyclic source-frame offset of +9 relative
to Basler 60gc. For example, Basler 60gc frame 1 corresponds to Basler 60gm frame
10 and Hikrobot frame 10. The configuration records the diagnosed cycle length
and offsets. Applying the offsets reduced stereo RMS from 83.375/103.660 px to
0.217/0.246 px. The accepted translations from Basler 60gc are approximately
[-135.74, -5.21, 8.89] mm to Basler 60gm and [-68.58, -0.83, -31.53] mm to
Hikrobot.

Outputs under `2026-05-19/calibration_files`:

- Three camera-specific intrinsic JSON files.
- `baseline_report.json`: settings, versions, commit, metrics and acceptance=false.
- `extrinsics_baseline_candidate.json`: accepted candidate produced by the runner.
- `extrinsics_result.json`: accepted stereo result for reconstruction.
- `debug/intrinsics` and `debug/extrinsics`: newly generated corner overlays.
- `stereo_sync_diagnostic.json`: inferred mappings for both supplied stereo sets.

The calibration passed the baseline thresholds and `extrinsics_result.json` was
published. Original images and supplied overlays were preserved. The mapping was
inferred by clustering camera transforms calculated from solvePnP board poses and
verified by repeating fixed-intrinsic stereo calibration. The original stereo set
also forms a geometrically consistent calibration after its own offsets, but the
configured and accepted baseline uses the NEW set. Do not transfer these
calibrations to the 2026-05-07 dataset without confirming the same camera geometry
and optical settings.

The runner reuses the existing calibration functions without GUI windows. It saves
overlays under calibration outputs and exits with failure if baseline thresholds
are not met. A rerun replaces candidate outputs and reports; archive these first
if comparing runs. The input datasets, scanner reference and virtual environment
are excluded from Git through `.gitignore`.
