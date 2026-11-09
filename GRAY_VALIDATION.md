# Accepted terminal-gray validation

Date: 2026-09-16

## Pipeline result

The accepted Basler pipeline was run independently for all 72 `terminal_gray`
frames. All frames reconstructed and registered without failures or automatic
registration-review flags.

| Stage | Result |
| --- | ---: |
| Reconstructed frames | 72 / 72 |
| Registered frames | 72 / 72 |
| Feature matches | 8,604 |
| Triangulated points | 7,957 |
| Frame-filtered points | 7,400 |
| Dominant-plane filtered points | 6,587 |
| Final filter retention | 89.01% |
| Final plane residual median | 0.565 mm |
| Final plane residual 95th percentile | 1.178 mm |

## Accepted scanner comparison

The cleaned cloud was aligned to `terminal_grey.csv`. The best automatic
orientation has 98.63% XY coverage. Its score margin over the second candidate
is only 0.0046, so the overlay and error map were inspected manually. Both
passed visual review on 2026-09-16.

| Metric | Accepted result |
| --- | ---: |
| Covered points | 6,497 / 6,587 |
| Nearest-reference 3D median | 0.651 mm |
| Nearest-reference 3D RMSE | 1.254 mm |
| Nearest-reference 3D 95th percentile | 1.177 mm |
| Signed height mean | -0.002 mm |
| Signed height RMSE | 0.709 mm |
| Absolute height mean | 0.591 mm |
| Absolute height 95th percentile | 1.081 mm |
| Absolute height 99th percentile | 1.508 mm |
| Height error at or below 1.00 mm | 92.04% |
| Height error at or below 1.50 mm | 98.95% |

These are post-alignment surface-agreement figures, not an independent measure
of absolute calibration accuracy.

## Visual review

`alignment_overlay_blender.ply` was reviewed with scanner samples in blue and
photogrammetry samples in orange. The outline and asymmetric details were
accepted despite the small numerical margin between the two best orientation
candidates.

`photogrammetry_error_map_blender.ply` was also accepted. Blue means approximately zero
height error, green approximately 0.75 mm, red 1.5 mm or greater, and magenta a
point outside the scanner footprint. No disqualifying systematic displacement
was observed.

The files are under
`2026-05-19/reconstruction_results_basler_gray_full/terminal_gray/registered/profile_reference_comparison`.
