# Terminal-green surface agreement report

Date: 2026-09-16

## Result

The cleaned 65-frame Basler photogrammetry cloud was aligned to the
`terminal_green.csv` profile-scanner reference. The overlay and error map were
visually accepted. The comparison covers 8,761 of 8,806 photogrammetry points
(99.49%).

| Metric | Result |
| --- | ---: |
| Nearest-reference 3D median | 0.403 mm |
| Nearest-reference 3D RMSE | 0.501 mm |
| Nearest-reference 3D 95th percentile | 0.863 mm |
| Signed height mean | -0.001 mm |
| Signed height RMSE | 0.417 mm |
| Absolute height mean | 0.303 mm |
| Absolute height 95th percentile | 0.820 mm |
| Absolute height 99th percentile | 1.353 mm |
| Height error at or below 0.25 mm | 53.68% |
| Height error at or below 0.50 mm | 81.85% |
| Height error at or below 1.00 mm | 97.53% |
| Height error at or below 1.50 mm | 99.43% |
| Points above 1.50 mm | 50 |

The signed mean close to zero indicates no meaningful average height bias after
alignment. Most samples agree within one millimetre; the remaining larger
differences are spatially limited rather than a whole-cloud offset.

## Scope and limitations

- These are post-alignment surface-agreement figures. ICP estimated the rigid
  alignment from the same two surfaces, so the result is not an independent
  absolute-position or calibration-accuracy test.
- The photogrammetry result is a sparse, feature-biased sample rather than a
  uniform surface scan.
- The profile scanner sampling is anisotropic: approximately 0.0599 mm in X and
  1.01129 mm in Y. Nearest-neighbour metrics include that sampling limitation.
- The final dominant-plane filter retained 96.29% of the registered points and
  removed points farther than 1.5 mm from the dominant surface.
- No project acceptance tolerance has been specified. The report therefore
  records measured agreement but does not declare a formal pass or fail.

## Reproducibility

Run the green comparison from the project root with:

```powershell
.venv/Scripts/python.exe -u compare_profile_reference.py
```

The numerical source is
`2026-05-19/reconstruction_results_basler_full/terminal_green/registered/profile_reference_comparison/profile_comparison_report.json`.
