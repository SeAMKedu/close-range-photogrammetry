# Full Basler pipeline: terminal_green frames 1-65

The accepted Basler 60gc to Basler 60gm pipeline was run for every synchronized
`terminal_green` frame. The run includes sparse reconstruction, frame-level
statistical filtering, sequential registration and dominant-plane cleanup.

Run the complete pipeline from the project directory:

```powershell
.venv/Scripts/python.exe -u run_full_pipeline.py
```

## Result

| Metric | Value |
| --- | ---: |
| Requested frames | 65 |
| Reconstructed frames | 65 |
| Registered frames | 65 |
| Failed frames | 0 |
| Feature matches | 10,579 |
| Triangulated points | 9,639 |
| Frame-filtered points | 9,145 |
| Plane-filtered registered points | 8,806 |
| Plane-filter retention | 96.29% |
| Complete pipeline time | approximately 110 s |

The registered-cloud plane filter uses a 1.5 mm perpendicular distance from a
RANSAC-fitted dominant plane. It removes 339 points. In the cleaned cloud, the
median plane residual is 0.21 mm, the 95th percentile is 0.77 mm and the 99th
percentile is 1.24 mm.

All 64 registration transitions passed the configured numerical quality gates.
The ranges were:

- 51-125 RANSAC inliers per transition
- 93.0-100.0% RANSAC inliers
- 0.062-0.690 mm RANSAC RMSE
- 0.822-1.000 ICP fitness
- 0.513-1.570 mm ICP inlier RMSE

Transitions 18 -> 19, 33 -> 34 and 49 -> 50 contain approximately 90-degree
object reorientations. Their large rigid transforms are expected rather than
registration drift. Frame 50 has the lowest ICP fitness (0.822), while frame 23
has the highest ICP RMSE (1.570 mm); both remain inside the review thresholds.

Outputs are under
`2026-05-19/reconstruction_results_basler_full/terminal_green`:

- `full_pipeline_report.json` summarizes the whole run.
- `sequence_report.json` and `sequence_summary.csv` describe reconstruction.
- `registered/registration_summary.json` describes registration quality.
- `registered/registered_filter_report.json` describes plane filtering.
- `registered/plane_filtered_registered_point_cloud.ply` is the final cloud for
  visual review.

## Manual acceptance gate

Open the final plane-filtered PLY and confirm that it contains one coherent
surface without rotated duplicates, obvious registration seams, or removal of
valid surface regions. Spot-check the after-ICP debug clouds for source frames
19, 34 and 50, which cover all three 90-degree reorientation boundaries. Frame
23 is an additional useful check for the largest ICP RMSE.

If these checks pass, the 65-frame Basler reconstruction and registration stage
is accepted. The next development stage is comparison against the profile
scanner reference data and quantitative surface-error measurement.

The preliminary profile-scanner alignment and its manual acceptance gate are
documented in `PROFILE_COMPARISON.md`.
