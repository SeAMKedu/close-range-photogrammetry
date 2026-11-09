# Laser reference comparison

> **Sini correction — 2026-11-06:** The single and threepiece comparisons have been recomputed with a fixed horizontal handedness conversion before rigid registration, matching the long/narrow and short/wide recesses. The current arrays, summaries, figures, Blender comparisons and error videos replace the mismatched results. P95 is now 0.527 mm (single) and 0.725 mm (threepiece). This is a documented, feature-supported Sini-specific conversion; the acquisition/export origin of the mismatch remains unresolved and no conversion is inferred for other objects. Raw scans and reconstruction geometry are preserved. See [the investigation and correction record](LaserScans/_inspection/Sini_orientation/README.md).

The files in [LaserScans](LaserScans/) can be used directly as reference measurements. A triangle mesh is unnecessary: the PCD exports already contain organized XYZ point clouds. [pipeline/laser_reference_io.py](pipeline/laser_reference_io.py) reads the binary PCD geometry, checks every finite coordinate and the missing-data mask against the companion CSV, and confirms millimetre units from the paired SUR header. Original exports remain unchanged. SRF files are retained and hashed, but are not needed for geometry import.

The importer preserves all finite samples and native holes. Missing PCD cells contain NaNs; blank CSV cells are missing observations, not zero-height points. Converted ordinary PLY clouds, organized NPZ arrays and provenance reports are in [reference_import](LaserScans/reference_import/).

## Scan identities

The filenames do not identify objects. This mapping follows visual shape inspection and the user's confirmation that Harmu was scanned on opposite faces:

| Filename frame | Object |
| --- | --- |
| 0000 | Musti |
| 0001 | Sini |
| 0002 | Metal |
| 0003 | Harmu, photographed face; use as reference |
| 0004 | Harmu, opposite face; diagnostic only |
| 0005 | Piiri |

Scan 0003 has the rounded bosses protruding into the triangular recesses that are visible in the photogrammetry images. Scan 0004 lacks these features. The [annotated native-height comparison](LaserScans/_inspection/harmu_face/harmu_face_features.png) records this feature-based identification. The selection uses surface geometry and the user's opposite-face confirmation, not the smaller reconstruction residual. The two scans are not combined or treated as a repeatability experiment. The [scan contact sheet](LaserScans/_inspection/scan_contact_sheet.png) shows all native height maps.

## Running the comparison

Run from the project root after restoring the scanner exports and generating or restoring the dense models. These inputs and the generated results are excluded from Git; see [the data setup instructions](README.md#data-and-generated-files).

```powershell
.\.venv\Scripts\python.exe pipeline/laser_reference_io.py --audit
.\.venv\Scripts\python.exe pipeline/run_laser_reference_comparison.py
```

The wrapper [pipeline/run_laser_reference_comparison.py](pipeline/run_laser_reference_comparison.py) runs the unchanged geometric comparison, applies the required face interpretation with [pipeline/update_laser_reference_labels.py](pipeline/update_laser_reference_labels.py), then rerenders the labelled figures. Use this entrypoint to keep opposite-face diagnostics out of the accuracy summary. The default processes the dense three-position models. Add `--include-single` to include the single-position models. Use `--objects Metal` or `--scan-frames 3 4` to select comparisons. `--force-registration` recomputes alignment; otherwise matching cached transforms are reused. Generated results for the selected comparisons are overwritten. Source scans and photogrammetry models are preserved.

To reproduce the complete set used by the numerical verification and Blender exports, first run the import audit above, then compare all single-position and three-position models without object or scan filters:

```powershell
.\.venv\Scripts\python.exe pipeline/run_laser_reference_comparison.py --include-single
```

This requires all eleven dense models and all six scanner export sets, including the paired PCD, CSV and SUR files. Retain the original SRF files as well for the documented provenance checks. It produces thirteen model/scan comparisons: eleven eligible corresponding-surface comparisons and two opposite-face diagnostics. The default three-position-only run is insufficient for the complete verification and eleven-model Blender export below.

[accuracy_summary.csv](LaserScans/comparison_results/accuracy_summary.csv) is the primary table of corresponding-surface comparisons. [summary.csv](LaserScans/comparison_results/summary.csv) and `all_comparisons.json` retain all numerical results, including the explicitly labelled opposite-face diagnostics. Each comparison folder contains the registration transform and diagnostics, full `comparison_report.json`, individual distances in `distances.npz`, aligned and colour-coded PLY clouds, and PNG/PDF figures with top/side overlays, error maps and cumulative distributions.

## Interpreting the results

The fitted registration permits only rigid rotation and translation: no scale fitting or fitted reflection. Sini first receives a fixed horizontal handedness conversion supported by its asymmetric recesses. Its saved composite transform therefore has determinant -1, while the separately recorded fitted rigid transform has determinant +1. Other models retain their original coordinate conventions. Subsampling, robust losses and trimmed correspondences affect alignment estimation only. Evaluation retains every input photogrammetry point, including large discrepancies.

Forward distances measure each photogrammetry point's nearest 3D scanner sample. Reverse distances measure each scanner sample's nearest photogrammetry point and therefore reflect missing coverage as well as disagreement. Neither direction measures complete unseen object surfaces.

Signed height differences compare Z against the nearest scanner sample in XY without interpolation. Supported-point counts and exclusion counts accompany a primary 0.30 mm XY radius and sensitivity results at 0.15 and 0.50 mm. These subsets are distinct from the full-cloud 3D statistics.

Corresponding-face comparisons remain provisional post-alignment surface agreement. Harmu scan 0004 is excluded from accuracy interpretation because it measures a different physical surface. Symmetry and existing reconstruction flags still require attention. The Gocator 2650 export records approximately 0.0712 × 0.195628 mm sampling and 0.0008 mm Z quantization; these are not scanner accuracy specifications. Scanner calibration and uncertainty have not been supplied, so the reported deviations are not calibrated uncertainty bounds.

The paper was not altered during this comparison work.

## Interactive comparison models

Open [the Blender overview](LaserScans/comparison_results/blender/Kaikki_kappaleet_virhevertailu.blend) to inspect the five fused models together. The same folder contains eleven individual comparison files, including the single-position reconstructions, with PNG previews. Each file has a colour-coded deviation scene and a second scene showing the aligned laser and photogrammetry clouds together. All models use the same seven distance bands; the point attribute `error_mm` retains the numerical distance, including values above the highest colour threshold.

The exporter preserves the aligned scanner coordinates and does not recompute registration or remove points. Display spheres are viewing aids, not uncertainty bounds. The overview translates each paired cloud together for layout only. [Finnish viewing instructions](LaserScans/comparison_results/blender/LUE_MINUT.md) explain navigation, interpretation, the full-precision NPZ/CSV/PLY exports and rebuild commands. [Saved-file verification](LaserScans/comparison_results/blender/verification.json) checks the Blender geometry, distance attributes, colours, units and visibility against the source results.

After the complete `--include-single` comparison above, run these commands from the project root. Continue to asset generation only after numerical verification passes. Replace the Blender executable path if your installation differs:

```powershell
.\.venv\Scripts\python.exe pipeline/verify_laser_comparisons.py
.\.venv\Scripts\python.exe pipeline/prepare_laser_comparison_assets.py
& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --python-exit-code 1 --python pipeline/export_laser_comparison_blender.py
& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --python-exit-code 1 --python pipeline/verify_laser_comparison_blender.py
```

The asset preparation selects the eleven eligible comparisons. The exporter writes eleven individual Blender files and the five-model overview under `LaserScans/comparison_results/blender/`. The last command verifies those saved files against the original scanner data and numerical comparison results, using the scanner hashes from the numerical verification report.

## Measured results

All 13 comparisons were computed correctly: eleven compare corresponding surfaces, and two compare the single-position and fused Harmu models against the opposite face in scan 0004. Those two results are retained only as diagnostics. The primary table below includes the five fused models against corresponding surfaces and uses every point in each photogrammetry cloud. Values are forward nearest-3D sample distances after rigid alignment, in millimetres.

| Model / scan | Median | RMSE | 95th percentile | 99th percentile | Within 1 mm | Reverse 95th percentile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Harmu / 0003](LaserScans/comparison_results/Harmu_threepiece__scan_0003/Harmu_threepiece__scan_0003_comparison.png) | 0.106 | 0.407 | 0.397 | 1.108 | 98.93% | 0.350 |
| [Metal / 0002](LaserScans/comparison_results/Metal_threepiece__scan_0002/Metal_threepiece__scan_0002_comparison.png) | 0.096 | 0.567 | 0.538 | 3.679 | 96.94% | 0.283 |
| [Musti / 0000](LaserScans/comparison_results/Musti_threepiece__scan_0000/Musti_threepiece__scan_0000_comparison.png) | 0.123 | 0.514 | 0.402 | 0.856 | 99.23% | 0.716 |
| [Piiri / 0005](LaserScans/comparison_results/Piiri_threepiece__scan_0005/Piiri_threepiece__scan_0005_comparison.png) | 0.175 | 1.931 | 4.651 | 7.848 | 79.90% | 5.724 |
| [Sini / 0001](LaserScans/comparison_results/Sini_threepiece__scan_0001/Sini_threepiece__scan_0001_comparison.png) | 0.137 | 1.688 | 0.725 | 10.501 | 95.75% | 0.759 |

A 95th percentile is a percentile of the sampled distances, not a 95% confidence interval. Reverse distances use the laser samples as the query cloud and also reflect missing photogrammetric coverage. The point samples are neither independent nor uniformly distributed over the complete object.

The opposite-face Harmu fused diagnostic has median 0.117 mm, RMSE 0.567 mm and 95th percentile 0.489 mm. These values are numerically valid distances between different surfaces, but they are not a photogrammetry accuracy measurement. They illustrate why a small best-fit residual alone cannot establish the correct reference face. Metal and Musti have near-equal orientation alternatives caused by their symmetric geometry, and Musti retains the earlier reconstruction review flag.

Sini's earlier large recess patches were caused by matching the asymmetric recesses to the wrong sides; those values have been superseded. The corrected single model has median 0.129 mm, RMSE 1.252 mm and P95 0.527 mm. The corrected threepiece result still has a high RMSE because isolated height bands/outliers remain in the photogrammetry cloud. No points were removed to obtain the improved P95. Piiri retains its previous result; this Sini correction does not establish the axis convention of any other scan. Metal includes reconstructed points near its holes where the laser has no corresponding top-surface samples. Residual differences do not allocate every discrepancy to one instrument.

For comparison, the single-position Harmu model against scan 0003 has median 0.100 mm, RMSE 0.174 mm and 95th percentile 0.322 mm. Fusion increases coverage but does not guarantee lower deviations. The corresponding-surface single-position and fused results are in [accuracy_summary.csv](LaserScans/comparison_results/accuracy_summary.csv); [summary.csv](LaserScans/comparison_results/summary.csv) also retains the opposite-face diagnostics with their statuses.

## Verification and review

[Independent numerical verification](LaserScans/comparison_results/verification.json) passed all 13 computations. This does not make the two opposite-face comparisons suitable for accuracy assessment: eleven corresponding-surface comparisons are eligible for the primary summary. Verification recomputed the reported statistics from exported aligned PLY coordinates and the original reference PCDs, checked all three XY support radii, verified rigid transforms and point counts, and checked all 24 original scanner-export hashes. [pipeline/verify_laser_comparisons.py](pipeline/verify_laser_comparisons.py) reproduces these checks.

For a fresh run, complete the import audit and the unfiltered `--include-single` comparison above first. This verification expects all thirteen comparisons and fails if any report is missing or stale.

```powershell
.\.venv\Scripts\python.exe pipeline/verify_laser_comparisons.py
```

Top and side overlays were inspected for correspondence of the object outline, holes and raised regions. This is an initial numerical and visual comparison, not formal metrological acceptance. Neither an application tolerance nor the reference scanner calibration/uncertainty budget has been supplied. No normality assumption, confidence interval or claimed certified accuracy is added to the measured distributions.
