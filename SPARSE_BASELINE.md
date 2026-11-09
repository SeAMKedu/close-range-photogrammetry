# Sparse baseline: terminal_green frame 1

The first sparse reconstruction baseline uses the accepted 2026-05-19
calibration and the synchronized object images whose filename timestamps differ
by approximately 163 ms between the two Basler cameras.

Run from the project directory:

```powershell
.venv/Scripts/python.exe -u run_sparse_baseline.py
```

Settings are stored in the ignored local `project_config.yaml`:

- Object: `terminal_green`
- Frame: `1`
- Reference: Basler 60gc
- Target: Basler 60gm
- Feature method: SIFT
- Lowe ratio: 0.75
- Maximum reprojection error: 2.0 px

Result:

| Metric | Value |
| --- | ---: |
| Ratio-test feature matches | 129 |
| Triangulated points passing geometric checks | 124 |
| Finite points | 124 |
| Processing time | 0.45 s |
| Median depth | 440.18 mm |
| 5th to 95th percentile depth | 437.98 to 444.73 mm |

The match and reprojection overlays were inspected. Correspondences are mainly
on the label, printed details, markings and lower structural edges. Smooth green
surfaces remain sparsely covered, consistent with the known SIFT limitation.
Most points form a narrow depth group around 440 mm. One point at 552.39 mm is a
clear depth outlier and should be handled by the existing frame filtering stage.

Outputs are under
`2026-05-19/reconstruction_results_baseline/terminal_green/frame_1`:

- `sparse_point_cloud.npz` and `.ply`
- `frame_info.json`
- `baseline_report.json`
- feature-match and reprojection debug images

The baseline is accepted as a successful sparse reconstruction. It establishes
that calibration, image synchronization, matching and triangulation work for this
camera pair and frame. It does not yet establish complete surface coverage or
measurement accuracy.

## Frame-level filtering

Run from the project directory:

```powershell
.venv/Scripts/python.exe -u run_filter_baseline.py
```

The sparse cloud uses gentle statistical filtering: 10 neighbours and a 2.0
standard-deviation ratio. Voxel downsampling and radius filtering are disabled.
The full pipeline's radius setting (6 neighbours within 2 mm) was tested but was
too aggressive for this sparse frame, retaining only 37 of 124 points.

The accepted filtering retained 123 of 124 points. It removed the isolated depth
outlier at 552.39 mm while retaining the main surface group. The filtered depth
range is 434.67 to 447.47 mm and its median is 440.18 mm.

Additional outputs in the same frame directory:

- `filtered_point_cloud.npz` and `.ply`
- `filter_report.json`

The frame 1 result passed visual inspection. The next 15-frame reconstruction
and filtering pilot is documented in `SEQUENCE_BASELINE.md`.
