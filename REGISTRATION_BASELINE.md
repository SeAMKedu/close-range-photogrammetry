# Registration baseline: terminal_green frames 1-15

The first 15 filtered Basler frame clouds were registered sequentially into the
coordinate system of frame 1. Temporal SIFT matches provide 3D-3D
correspondences, RANSAC estimates the rigid transform, and point-to-point ICP
refines each consecutive frame alignment.

Run from the project directory:

```powershell
.venv/Scripts/python.exe -u run_registration_baseline.py
```

## Result

All 15 frames registered successfully. The merged cloud contains all 2,041
filtered input points; voxel downsampling is disabled for this baseline.

| Transition | 3D pairs | Inliers | Inliers (%) | RANSAC RMSE (mm) | ICP fitness | ICP RMSE (mm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 -> 2 | 71 | 69 | 97.2 | 0.136 | 0.964 | 0.960 |
| 2 -> 3 | 77 | 74 | 96.1 | 0.328 | 0.974 | 1.092 |
| 3 -> 4 | 82 | 82 | 100.0 | 0.105 | 0.991 | 0.998 |
| 4 -> 5 | 70 | 70 | 100.0 | 0.108 | 0.933 | 1.065 |
| 5 -> 6 | 72 | 71 | 98.6 | 0.155 | 0.972 | 1.116 |
| 6 -> 7 | 69 | 69 | 100.0 | 0.278 | 0.978 | 0.896 |
| 7 -> 8 | 62 | 62 | 100.0 | 0.178 | 0.894 | 1.023 |
| 8 -> 9 | 71 | 71 | 100.0 | 0.121 | 0.983 | 1.227 |
| 9 -> 10 | 82 | 82 | 100.0 | 0.151 | 0.983 | 0.881 |
| 10 -> 11 | 90 | 89 | 98.9 | 0.165 | 0.911 | 0.842 |
| 11 -> 12 | 109 | 106 | 97.2 | 0.394 | 0.938 | 0.894 |
| 12 -> 13 | 101 | 98 | 97.0 | 0.253 | 0.974 | 0.987 |
| 13 -> 14 | 123 | 117 | 95.1 | 0.143 | 0.965 | 0.592 |
| 14 -> 15 | 109 | 108 | 99.1 | 0.392 | 0.976 | 0.659 |

No transition crossed the automatic review thresholds of fewer than 10 RANSAC
inliers, ICP fitness below 0.5, or ICP RMSE above 2.0 mm. The aligned frame
centroids span approximately 2.57 x 6.52 x 0.97 mm, so no obvious cumulative
drift is visible in this numerical check. The merged bounding-box span is
approximately 77.38 x 50.81 x 48.32 mm. The large Z span is associated mainly
with the already flagged depth range in frame 12.

Outputs are under
`2026-05-19/reconstruction_results_basler_sequence/terminal_green/registered`:

- `registered_sparse_point_cloud.ply` and `.npz` contain the merged cloud.
- `frame_transforms.json` contains every transform into frame 1 coordinates.
- `registration_report.json` is the original frame-by-frame report.
- `registration_summary.json` adds quality metrics and review decisions.
- `debug/feature_matches` contains temporal inlier overlays.
- `debug/before_icp` and `debug/after_icp` contain pairwise alignment clouds.

Before extending to all 65 object frames, manually inspect the merged PLY. Also
spot-check the frame 7-to-8 match overlay (lowest ICP fitness) and frame 8-to-9
after-ICP cloud (highest ICP RMSE). If they look correct, the 15-frame
registration baseline is accepted.

The merged-cloud cleanup now uses dominant-plane filtering rather than a global
Z limit, because the measured surface is tilted in camera coordinates. With a
1.5 mm perpendicular threshold, the 15-frame pilot retains 2,005 of 2,041
points (98.24%). The complete 65-frame run is documented in `FULL_PIPELINE.md`.
