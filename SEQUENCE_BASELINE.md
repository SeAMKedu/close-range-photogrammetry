# Basler sequence baseline: terminal_green frames 1-15

This pilot reconstructs and filters the first 15 synchronized object frames
using the accepted Basler 60gc to Basler 60gm calibration. The object dataset
contains 65 frames in total; this is a registration-readiness pilot, not yet the
complete turntable sequence.

Run from the project directory:

```powershell
.venv/Scripts/python.exe -u run_sequence_baseline.py
```

The sparse reconstruction uses SIFT, a 0.75 Lowe ratio and a 2.0 px maximum
reprojection error. Filtering uses 10 statistical neighbours and a 2.0 standard
deviation ratio. Radius filtering and voxel downsampling remain disabled.

## Result

All 15 requested frames succeeded.

| Metric | Value |
| --- | ---: |
| Feature matches | 2,263 |
| Triangulated points | 2,124 |
| Filtered points | 2,041 |
| Frames failed or missing | 0 |
| Processing time | approximately 11 s |

| Frame | Matches | Triangulated | Filtered | Removed | Median Z (mm) | Z span (mm) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 129 | 124 | 123 | 1 | 440.18 | 12.80 |
| 2 | 126 | 116 | 111 | 5 | 440.21 | 24.86 |
| 3 | 128 | 118 | 114 | 4 | 439.94 | 9.98 |
| 4 | 124 | 117 | 110 | 7 | 439.88 | 14.94 |
| 5 | 130 | 121 | 120 | 1 | 439.80 | 24.50 |
| 6 | 122 | 114 | 109 | 5 | 439.65 | 10.76 |
| 7 | 108 | 101 | 92 | 9 | 439.70 | 13.89 |
| 8 | 102 | 100 | 94 | 6 | 439.85 | 10.21 |
| 9 | 142 | 131 | 121 | 10 | 442.43 | 19.55 |
| 10 | 137 | 127 | 118 | 9 | 442.89 | 17.80 |
| 11 | 179 | 171 | 168 | 3 | 443.70 | 17.50 |
| 12 | 174 | 165 | 162 | 3 | 443.68 | 47.87 |
| 13 | 242 | 231 | 228 | 3 | 451.38 | 25.14 |
| 14 | 221 | 206 | 201 | 5 | 451.88 | 28.57 |
| 15 | 199 | 182 | 170 | 12 | 451.84 | 20.58 |

The automatic review flags frames 7 and 8 for their low point counts and frame
12 for its large filtered depth span. Frame 13 has the highest point count and
is useful as a strong comparison. Before registration, manually inspect the
filtered clouds and debug overlays for frames 7, 8, 12 and 13.

Outputs are under
`2026-05-19/reconstruction_results_basler_sequence/terminal_green`:

- `sequence_report.json` contains the complete machine-readable report.
- `sequence_summary.csv` contains the compact frame comparison.
- Each `frame_N` directory contains sparse and filtered NPZ/PLY clouds plus
  feature-match and reprojection debug images.

If frames 7, 8, 12 and 13 pass visual inspection, the next step is a registration
pilot for frames 1-15. After registration is verified, extend reconstruction to
the remaining frames 16-65.

The registration pilot has now been run and is documented in
`REGISTRATION_BASELINE.md`.
