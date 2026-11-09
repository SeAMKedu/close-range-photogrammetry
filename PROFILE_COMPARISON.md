# Accepted profile-scanner comparison: terminal_green

The cleaned 65-frame Basler reconstruction was aligned to
`Profile scanner reference/terminal_green.csv`. The scanner CSV contains a
53 x 1,353 height grid with 45,565 valid XYZ samples. Its measurement spacing is
0.0599 mm in X, 1.01129 mm in Y and 0.0076 mm in Z.

Run from the project directory:

```powershell
.venv/Scripts/python.exe -u pipeline/compare_profile_reference.py
```

The alignment tests eight axis-swap and direction candidates, initializes each
from the dominant surface planes and bounding boxes, and refines each candidate
with point-to-point ICP. The selected candidate is a 180-degree in-plane
rotation without mirroring. It covers 99.49% of the photogrammetry points. Its
score exceeds the second-best candidate by 0.046, which is a clear numerical
preference in this candidate set.

## Accepted post-alignment metrics

The alignment overlay and error map passed visual review on 2026-09-16. ICP
uses the compared surfaces themselves, so the values describe post-alignment
surface agreement and are not an independent absolute calibration validation.

| Metric | Value |
| --- | ---: |
| Photogrammetry points | 8,806 |
| Points over valid scanner area | 8,761 |
| XY coverage | 99.49% |
| Nearest-3D median | 0.403 mm |
| Nearest-3D RMSE | 0.501 mm |
| Nearest-3D 95th percentile | 0.863 mm |
| Signed Z mean | -0.001 mm |
| Signed Z median | -0.008 mm |
| Signed Z RMSE | 0.417 mm |
| Absolute Z mean | 0.303 mm |
| Absolute Z 95th percentile | 0.820 mm |
| Absolute Z 99th percentile | 1.353 mm |

## Visual alignment review

The Blender-friendly overlay was reviewed from multiple directions. The object
outline, cut-outs and distinctive regions overlap in corresponding physical
locations, and no systematic lateral shift or incorrect orientation was seen.

In `photogrammetry_error_map.ply`, blue means approximately zero error,
green is approximately 0.75 mm, red is 1.5 mm or greater, and magenta marks a
point outside the scanner's valid footprint. The reviewed map did not show a
systematic shifted pattern.

Blender does not shade color attributes on a mesh made only of isolated
vertices. For Blender, import `photogrammetry_error_map_blender.ply` with
Vertex Colors set to sRGB, switch to Object Mode, deselect the object, and use
Solid Shading with Color set to Attribute. This variant represents each sample
as a 0.12 mm colored octahedron and preserves the same error values.
The corresponding `alignment_overlay_blender.ply` uses small colored
octahedra as well: scanner samples are blue and photogrammetry samples orange.

Outputs are under
`2026-05-19/reconstruction_results_basler_full/terminal_green/registered/profile_reference_comparison`:

- `alignment_overlay.ply`
- `alignment_overlay_blender.ply`
- `photogrammetry_error_map.ply`
- `photogrammetry_error_map_blender.ply`
- `aligned_photogrammetry.ply`
- `scanner_reference.ply`
- `profile_comparison_report.json`

The machine-readable report status is
`accepted_post_alignment_surface_comparison`. See `ACCURACY_REPORT.md` for the
final interpretation and limitations.
