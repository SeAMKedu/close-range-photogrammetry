# Marker A/B test: terminal gray

Date: 2026-09-16

The same Basler reconstruction pipeline was run for unmarked `terminal_gray`
and marked `marked_terminal_gray`. Both were compared against the same
`terminal_grey.csv` profile-scanner reference. The unmarked alignment is
accepted; the marked alignment remains preliminary until visual review.

The frame counts differ (72 versus 70), so density is compared per frame.

| Metric | Unmarked | Marked | Change |
| --- | ---: | ---: | ---: |
| Triangulated points per frame | 110.5 | 207.6 | 1.88x |
| Final points per frame | 91.5 | 180.3 | 1.97x |
| Final plane residual median | 0.565 mm | 0.200 mm | -64.6% |
| Scanner XY coverage | 98.63% | 99.56% | +0.92 percentage points |
| Nearest-reference 3D RMSE | 1.254 mm | 0.509 mm | -59.4% |
| Height RMSE | 0.709 mm | 0.418 mm | -41.1% |
| Absolute height 95th percentile | 1.081 mm | 0.836 mm | -22.6% |
| Height error at or below 1 mm | 92.04% | 96.94% | +4.90 percentage points |

The preliminary result strongly indicates that the artificial markings nearly
double useful sparse-point density and improve surface agreement. This
conclusion becomes accepted only after the marked overlay and error map pass
visual review; ICP-based figures remain post-alignment metrics rather than
absolute calibration accuracy.

Marked review files are under
`2026-05-19/reconstruction_results_basler_marked_gray_full/marked_terminal_gray/registered/profile_reference_comparison`.
