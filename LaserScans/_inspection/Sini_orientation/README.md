# Sini orientation investigation — 2026-11-06

## Correction applied

The primary Sini single and threepiece comparisons have now been replaced. A fixed reflection in the leveled horizontal coordinate plane is expressed in the native photogrammetry coordinate system and applied before the same multistart rigid registration used previously. The fitted pose remains proper and unit-scale; the total mapping has determinant -1. Both transforms, the conversion basis and its Sini-only scope are saved in each `registration.json`. Raw PCD/CSV/SUR/SRF files and dense reconstruction PLYs are unchanged.

| Current result | Median (mm) | RMSE (mm) | P95 (mm) |
| --- | ---: | ---: | ---: |
| Sini single | 0.129 | 1.252 | 0.527 |
| Sini threepiece | 0.137 | 1.688 | 0.725 |

The full alignment search gives slightly different results from the earlier local diagnostic below. Updated outputs retain every point. The original mismatched outputs are archived in `superseded_before_correction.zip`; the figures and diagnostic JSON files below describe that historical investigation. Do not use them as current result tables. The origin of the coordinate convention mismatch is still unresolved, so no automatic parity correction is applied to other objects.

![Both corrected models, alongside the laser scan and archived mismatched results](corrected_height_correspondence.png)

## Historical diagnosis

The user's observation is confirmed: the current comparison places the long, narrow recess opposite the short, wide recess of the reference scan. These features are visible directly in the native laser height data and the aligned photogrammetry height data. Matching the outer footprint was insufficient to validate this alignment.

![Identical-coordinate height maps: raw laser, current photogrammetry and reflected diagnostic](height_orientation.png)

The left panel is the raw laser scan, the middle panel is the existing aligned photogrammetry, and the right panel is a diagnostic horizontal-axis reversal followed by rigid refinement. Both recesses correspond in the right panel. Every panel uses the same plotting axes and height colour scale. The plotting axis order does not change between panels.

For `Sini_threepiece`, the stored comparison has a P95 of 3.621 mm. The diagnostic has P95 0.760 mm, median 0.137 mm and RMSE 1.693 mm, retaining all 203,583 photogrammetry points. Outliers remain; the reduction in the mismatched recess patches does not establish that all residuals are reconstruction errors.

The original registration constrained the rotation determinant to +1 and therefore could not reconcile a handedness mismatch. The diagnostic transform has determinant -1. It must not be described as a rigid pose correction. The exact source of the mismatch (acquisition/export axis convention or an earlier reconstruction coordinate convention) has not yet been established. A reduced residual alone does not justify reflecting measurement data as a general fitting option.

The original Sini accuracy interpretation was withdrawn and the mismatched outputs have now been replaced as described above. Their archived copies must not be used as current accuracy results. Other scans should be checked for the same coordinate-convention issue; symmetry can conceal a reversed axis.

`diagnostic.json` records eight axis-sign hypotheses for the original fused model, their fitted transforms and all-point statistics. `single_diagnostic.json` tests the same horizontal-reversal hypothesis locally on the original single model. These are historical investigation outputs only. From the project root, use `.venv/Scripts/python.exe pipeline/run_laser_reference_comparison.py --objects Sini --include-single --force-registration` to reproduce the corrected primary results.
