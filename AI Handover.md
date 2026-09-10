# 3D-Laatu.AI Close-Range Photogrammetry Pipeline – AI Handover

## Purpose of this document

This document provides background information for an AI assistant supporting further development of the close-range photogrammetry pipeline in the 3D-Laatu.AI project.

The colleague continuing the work has access to the Git repository. **Always treat the current Git repository as the source of truth for the actual implementation.** This document explains the reasoning behind the implementation, experiments already carried out, known problems, discarded assumptions, and likely next development directions.

The software is still under active research and development. Some scripts are part of the main reconstruction pipeline, while others are experimental.

---

# 1. Project objective

The overall research objective is to investigate whether a relatively inexpensive fixed multi-camera system can be used for close-range photogrammetric 3D measurement and eventually manufacturing quality inspection.

The current laboratory system uses three fixed cameras. A target object is captured synchronously by all cameras. Between synchronized captures, the object can move and rotate relative to the cameras.

This is important: **the cameras remain fixed, but the object pose changes between frames**.

A central research question is how much additional 3D information can be obtained by observing the same object in several different positions and orientations.

The longer-term intention is to compare the camera-based reconstruction against other 3D measurement sources, including industrial profile scanning and commercial photogrammetry software such as RealityScan and Polycam.

---

# 2. Development environment

The pipeline has been developed primarily in Python and run directly from VS Code during the research/development phase.

The tested Python version has been:

```text
Python 3.11.9
```

Dependencies are installed through:

```bash
pip install -r requirements.txt
```

Important libraries include at least:

```text
OpenCV
NumPy
Open3D
PyYAML
```

The code intentionally uses modern Python conventions such as:

```python
def function(
    path: Path,
    points: np.ndarray,
) -> float:
```

and `dataclass` structures for related data.

`pathlib.Path` is preferred over older `os.path`, `listdir`, and string-based path manipulation.

---

# 3. Project configuration

Common project-specific information is centralized in:

```text
project_config.yaml
```

A private local `project_config.yaml` should not be committed if it contains machine-specific paths or usernames.

The repository should instead contain something such as:

```text
project_config.example.yaml
```

The user copies this to:

```text
project_config.yaml
```

and edits local paths.

`project_config.yaml` currently controls information such as:

```text
img_root
camera_names
reference_camera_index
checkerboard geometry
reconstruction output folder
debug settings
visual hull settings, if experimental visual hull scripts are used
```

Debug settings have been used extensively to restrict reconstruction to selected cameras and frames.

Typical concept:

```yaml
debug:
  enabled: true
  target_camera_indices: [2]
  frame_keys_to_use: ["1"]
```

When debugging is disabled, the normal full dataset should be processed.

---

# 4. Image data organization

The expected input organization is approximately:

```text
img_root/
│
├── calibration_files/
│
├── chessboards/
│   ├── intrinsics_data_set/
│   │   ├── cam0/
│   │   ├── cam1/
│   │   └── cam2/
│   │
│   └── extrinsics_data_set/
│       ├── cam0/
│       ├── cam1/
│       └── cam2/
│
└── object/
    ├── obj0/
    │   ├── cam0/
    │   ├── cam1/
    │   └── cam2/
    │
    └── obj1/
        ├── cam0/
        ├── cam1/
        └── cam2/
```

Synchronized filenames do not need to be identical.

Synchronization is based on the prefix before the first underscore.

Example:

```text
cam0/1_obj0_xxxx.jpg
cam1/1_obj0_yyyy.jpg
cam2/1_obj0_zzzz.jpg
```

These are all interpreted as:

```text
frame 1
```

Natural sorting must be used because ordinary alphabetical sorting would order:

```text
1
10
11
2
```

incorrectly.

---

# 5. Calibration

## Intrinsic calibration

Script:

```text
calibrate_cam_with_chessboards.py
```

Each camera is calibrated individually with multiple checkerboard images using OpenCV.

Relevant parameters include:

```text
camera matrix
distortion coefficients
rvecs / tvecs for calibration views
OpenCV RMS
mean reprojection error
```

The checkerboard dimensions used in one dataset were 16 × 9 squares with 7.5 mm squares; verify the actual current configuration from Git/configuration rather than assuming these values.

Calibration quality observed during good calibration runs has been roughly:

```text
intrinsic OpenCV RMS ≈ 0.2 px
mean reprojection error < 0.1 px
```

These were considered good.

Blurred checkerboard images should be avoided even if OpenCV technically detects the corners.

---

## Extrinsic / stereo calibration

Script:

```text
calibrate_stereo_with_chessboards.py
```

Common synchronized checkerboard images are used to determine camera-to-camera rotations and translations.

A major debugging incident occurred here:

One camera pair had approximately:

```text
rms_reprojection_error ≈ 64.6 px
```

while another had approximately:

```text
rms_reprojection_error ≈ 0.28 px
```

The first calibration was clearly invalid despite checkerboard corner detection appearing visually correct.

This bad extrinsic calibration caused an additional crescent-like false 3D surface in sparse reconstructions.

Therefore:

**Always inspect stereo RMS values before debugging later reconstruction stages.**

A very high stereo RMS should be treated as a calibration failure rather than compensated for downstream.

---

# 6. Current main reconstruction pipeline

The current intended main pipeline is:

```text
calibrate_cam_with_chessboards.py
        ↓
calibrate_stereo_with_chessboards.py
        ↓
reconstruct_sparse_point_cloud.py
        ↓
filter_point_cloud.py
        ↓
register_sparse_frames.py
        ↓
filter_registered_point_cloud.py
        ↓
mesh_from_registered_point_cloud.py
```

`filter_point_cloud.py` MUST run before `register_sparse_frames.py`.

`register_sparse_frames.py` expects:

```text
frame_X/filtered_point_cloud.npz
```

and will otherwise fail with a message similar to:

```text
FileNotFoundError:
No filtered frame clouds found ...
```

---

# 7. Frame-wise sparse reconstruction

Script:

```text
reconstruct_sparse_point_cloud.py
```

Important historical lesson:

The first implementation accumulated points from all frames using approximately:

```python
np.vstack(all_points)
```

This was incorrect because the object changes pose between frames.

The resulting cloud looked geometrically chaotic because points belonging to different object poses were being interpreted as one object.

This has now been changed conceptually to:

```text
one synchronized frame
        ↓
one independent sparse point cloud
```

Typical result structure is:

```text
reconstruction_results/
  obj0/
    frame_1/
      sparse_point_cloud.npz
      sparse_point_cloud.ply
      frame_info.json
      debug/
        feature_matches/
        reprojection/
    frame_2/
      ...
```

The sparse reconstruction uses feature matching, currently primarily SIFT, followed by triangulation and reprojection filtering.

---

# 8. Important sparse reconstruction finding: object edges

A major early problem was that the reconstructed object shape was not recognizable.

`cv2.drawMatches` debug images revealed the reason:

**SIFT correspondences were concentrated on textured interior regions and there were almost no reliable correspondences on the external silhouette edges.**

Therefore the sparse point cloud did not describe the global object outline.

This is not necessarily a bug in the triangulation implementation. It is a limitation of feature-based sparse reconstruction on low-texture plastic components.

Experiments included:

```text
dark/matte background
image masking
largest connected component / contour concepts
manual texture markings on the plastic component
```

Manual markings substantially improved feature availability and reconstruction density.

However, a sparse SIFT reconstruction should still not be expected to produce a complete surface.

---

# 9. Frame-level filtering

Script:

```text
filter_point_cloud.py
```

Each frame is filtered independently before registration.

Methods include:

```text
non-finite point removal
optional depth filtering
statistical outlier removal
radius outlier removal
optional voxel downsampling
```

Radius outlier removal produced a particularly visible improvement in early experiments.

The script produces:

```text
frame_X/
  filtered_point_cloud.npz
  filtered_point_cloud.ply
```

These filtered frame clouds are the input to frame registration.

---

# 10. Frame registration

Script:

```text
register_sparse_frames.py
```

The purpose is to exploit the fact that the same object is captured in different positions and orientations.

The current registration concept is:

```text
images from frame i
        +
images from frame i+1
        ↓
temporal SIFT matching
        ↓
3D–3D correspondences
        ↓
RANSAC rigid transform
        ↓
point-to-point ICP refinement
        ↓
frame transform
```

Important implementation details:

```text
SIFT feature matching is performed between frames.
Features are also triangulated to obtain 3D–3D correspondences.
A rigid transform is estimated using RANSAC + SVD.
Open3D point-to-point ICP refines the initial transform.
```

Example current settings have included roughly:

```text
sift_nfeatures = 8000
stereo_lowe_ratio = 0.75
temporal_lowe_ratio = 0.70
max_reprojection_error_px = 2.0

ransac_iterations = 2000
ransac_inlier_threshold = 3.0
ransac_min_inliers = 6

icp_max_correspondence_distance = 5.0
icp_max_iterations = 100
```

Units of the 3D thresholds follow calibration units. If checkerboard square dimensions were specified in millimetres, the 3D values are millimetres.

---

## Current registration strategy

At the time of this handover, registration is sequential:

```text
frame 1 ← frame 2 ← frame 3 ← frame 4 ...
```

Pairwise transforms are accumulated into the first successfully registered frame coordinate system.

This means drift is theoretically possible.

A future alternative worth testing is:

```text
frame 1 ← frame 2
frame 1 ← frame 3
frame 1 ← frame 4
...
```

or a pose-graph/global optimization approach.

Do not assume this change is necessary without testing.

---

## Observed registration quality

One tested dataset produced typically:

```text
150–270 temporal 3D correspondences per frame pair
very high RANSAC inlier ratios
RANSAC RMSE roughly 0.2–0.5 mm
ICP fitness commonly 0.95–1.0
ICP inlier RMSE usually below 1 mm
```

Large inter-frame rotations were initially suspected to be registration errors.

Inspection of the source images showed that this assumption was wrong.

Examples included approximately:

```text
frame 9 → 10: >60° actual physical rotation
frame 12 → 13: >90° actual physical rotation
```

Therefore:

**Do not reject large rotations purely because the rotation angle is large. The object can genuinely rotate substantially between frames.**

Registration validity must be evaluated against the source images and geometric errors, not an arbitrary small-angle assumption.

---

# 11. Registered object-level point cloud

The registration stage produces approximately:

```text
reconstruction_results/
  obj0/
    registered/
      registered_sparse_point_cloud.npz
      registered_sparse_point_cloud.ply
      frame_transforms.json
      registration_report.json
      debug/
```

`frame_transforms.json` is important because it provides the frame-to-common-coordinate-system transforms.

`registration_report.json` stores metrics such as:

```text
temporal 3D correspondence count
RANSAC inlier count
RANSAC RMSE
ICP fitness
ICP inlier RMSE
pairwise transform
transform to reference frame
```

Debug outputs include feature match images and point clouds before/after ICP.

---

# 12. Filtering the registered cloud

Script:

```text
filter_registered_point_cloud.py
```

This is a second filtering stage at object level.

It is intentionally separate from frame-level filtering.

Why both are needed:

```text
frame filtering
    → removes bad points before registration

registered-cloud filtering
    → handles accumulated and overlapping points after registration
```

The registered cloud is generally denser because several object poses contribute observations.

Available operations include:

```text
non-finite removal
optional depth filtering
optional voxel downsampling
statistical outlier removal
radius outlier removal
```

Produces:

```text
registered/
  filtered_registered_point_cloud.npz
  filtered_registered_point_cloud.ply
```

---

# 13. Object-level mesh generation

Script:

```text
mesh_from_registered_point_cloud.py
```

The main object-level mesh should be generated from:

```text
filtered_registered_point_cloud.npz
```

rather than from a single frame.

Supported methods:

```text
Alpha Shape
Ball Pivoting
Poisson reconstruction
```

Frame-specific meshing still exists for debugging/comparison but is not part of the main reconstruction pipeline.

---

## Point-spacing-scaled parameters

An important recent change was to derive several mesh parameters from the estimated nearest-neighbour spacing of the registered cloud.

The script estimates:

```text
spacing = typical nearest-neighbour distance
```

and derives parameters using multipliers.

Example experimental values that produced results approximately comparable to earlier manually selected millimetre values were:

```python
alpha_multiplier = 20.0

ball_pivoting_radius_multipliers = (
    10.0,
    15.0,
    20.0,
)

normal_radius_multiplier = 10.0
```

These values are **not final optimized parameters**.

Early generic multipliers such as 1.5–4 were much too small for the tested registered cloud because the estimated nearest-neighbour spacing becomes very small when points from several frames overlap.

Therefore always inspect both:

```text
estimated point spacing
```

and the resulting actual physical values:

```text
derived alpha
derived ball-pivoting radii
derived normal radius
```

rather than interpreting multiplier magnitude alone.

Recommended tuning strategy:

```text
Alpha Shape:
  tune alpha_multiplier first

Ball Pivoting:
  tune the common radius scale first
  then refine relative radii
  then tune normal radius if necessary

Poisson:
  normal quality first
  then depth
  then density trimming
```

---

# 14. Experimental visual hull branch

The visual hull work is **experimental and is currently NOT part of the main reconstruction pipeline**.

Relevant experimental scripts include at least:

```text
reconstruct_visual_hull.py
register_visual_hull_frames.py
debug_visual_hull_frame_alignment.py
```

The original motivation was important:

Sparse SIFT reconstruction did not provide sufficient points on the external edges of low-texture objects.

The hypothesis was:

```text
sparse features → accurate local surface information
silhouettes / visual hull → global object outline
```

---

## Visual hull implementation

The experimental visual hull reconstructs a voxel volume by:

```text
segmenting object silhouettes
creating a 3D voxel grid
projecting voxels into camera images
retaining voxels supported by camera silhouettes
extracting surface voxels
```

The test objects were imaged against a dark background, which makes silhouette segmentation relatively straightforward.

Single-frame visual hulls often show a recognizable object outline when viewed from the correct direction.

However, with only a limited number of camera directions, the hull can be very deep along poorly constrained viewing directions.

This is inherent to silhouette intersection and not necessarily an implementation error.

---

# 15. Visual hull registration experiment

Because `register_sparse_frames.py` already estimates frame transforms, the same transformations were applied to visual hull frames.

The idea was:

```text
visual hull frame 1
visual hull frame 2
...
        ↓
apply frame_transforms.json
        ↓
registered visual hull
```

The merged hull lost much of the recognizable object shape.

A dedicated debug script was created to align only two hull frames and calculate nearest-neighbour distances.

One example produced approximately:

```text
source → target:
mean ≈ 4.05 mm
median ≈ 1.71 mm
p95 ≈ 18.30 mm
max ≈ 34.48 mm

target → source:
mean ≈ 2.44 mm
median ≈ 1.55 mm
p95 ≈ 6.83 mm
max ≈ 18.24 mm

symmetric:
mean ≈ 3.34 mm
median ≈ 1.64 mm
p95 ≈ 13.47 mm
max ≈ 34.48 mm
```

These differences are substantial.

---

## Coordinate-system check

One hypothesis was that sparse points and visual hull voxels might use different coordinate systems.

A same-frame comparison showed that this is **not the obvious problem**:

```text
sparse cloud
```

was located inside the corresponding visual hull and had the same overall orientation and coordinate system.

Therefore the simple hypothesis:

```text
visual hull and sparse cloud use different coordinate frames
```

was not supported by this test.

---

## Current unresolved visual hull question

Before continuing the hull experiments, verify the voxel-grid construction carefully.

Important questions include:

```text
Are bounds_min_xyz and bounds_max_xyz constant between frames?
Is voxel_size constant?
Does every frame use exactly the same reference-camera coordinate system?
Are the voxel-grid dimensions identical?
```

The current experimental code uses explicit visual-hull bounds and voxel size; inspect the current Git implementation before assuming the exact behavior.

---

# 16. Is sparse + visual hull fusion still useful?

This remains unresolved.

The original reason for introducing visual hulls was **not** excessive sparse-cloud outliers.

The original problem was the opposite:

```text
the sparse cloud was too incomplete,
especially near external object boundaries
```

Using the hull only to remove sparse points outside the silhouette-constrained volume therefore does not solve the original problem.

Current conceptual understanding:

```text
Sparse reconstruction:
  good local 3D geometry where texture exists
  poor global coverage on textureless surfaces and silhouettes

Visual hull:
  useful global shape constraint
  poor detailed surface localization
```

The visual hull does not automatically create physically accurate surface points where sparse reconstruction has none.

Potential future uses include:

```text
outlier rejection
mesh clipping / spatial constraints
registration constraints
silhouette-based boundary reconstruction
```

A potentially more direct solution to the original edge problem would be to triangulate silhouette-edge information using epipolar geometry rather than treating the entire visual hull surface as measured geometry.

No final decision has been made.

---

# 17. Experiments already tried on low-texture objects

The following approaches have already been discussed/tested:

```text
matte black background
object masks
largest connected component segmentation
SIFT debug visualization using cv2.drawMatches
manual markings on inexpensive plastic components
increasing number and diversity of object poses
visual hull / silhouette reconstruction
```

Important result:

**Manual texture markings improved the feature-based reconstruction noticeably.**

A key lesson from the debug match images was that visually obvious geometric corners are not necessarily good SIFT correspondences across different viewpoints.

---

# 18. Interpretation of multiple object poses

Multiple frames are expected to add different kinds of information.

Pure translation of the object mainly adds:

```text
redundancy
noise reduction possibilities
more observations of approximately the same surfaces
```

Changing object orientation is more valuable because it exposes:

```text
previously occluded surfaces
different sides
different feature geometry
different silhouette constraints
```

One of the eventual experiments should therefore quantify reconstruction quality as a function of:

```text
number of frames
orientation diversity
possibly translation-only vs translation+rotation datasets
```

Possible frame-count experiments:

```text
1
5
10
20
all available frames
```

---

# 19. Reference measurements and future validation

The eventual goal is not merely to produce visually plausible meshes.

Quantitative comparison should be made against independent reference measurements.

Potential reference data already discussed include:

```text
industrial laser/profile scanner
other 3D scanner
RealityScan
Polycam
possibly Metashape or similar software
```

Useful future metrics may include:

```text
cloud-to-cloud distances
cloud-to-mesh distances
mean / median error
RMSE
95th percentile
maximum deviations
surface coverage
completeness
repeatability across captures
```

CloudCompare is already being used heavily for visual inspection.

---

# 20. Known architectural principle

Keep frame-level and object-level processing separate.

Frame level:

```text
frame_X/
  sparse_point_cloud
  filtered_point_cloud
  frame_info
  debug
```

Object level:

```text
registered/
  registered_sparse_point_cloud
  filtered_registered_point_cloud
  frame_transforms
  registration_report
  final mesh
```

This distinction was introduced after discovering that simply concatenating points from differently posed objects produced meaningless geometry.

Do not reintroduce cross-frame concatenation before registration.

---

# 21. Important debugging lessons

When something looks wrong, debug upstream before adjusting mesh parameters.

Recommended order:

```text
1. Inspect calibration errors.
2. Inspect drawMatches feature correspondences.
3. Restrict reconstruction to one camera pair.
4. Restrict reconstruction to one frame.
5. Inspect sparse point cloud.
6. Inspect filtered point cloud.
7. Inspect RANSAC/ICP registration report.
8. Inspect pairwise registered clouds.
9. Only then evaluate mesh reconstruction.
```

This procedure previously identified:

```text
a severely invalid stereo calibration
incorrect assumptions about camera-pair selection
the mistake of merging differently posed frames before registration
the lack of feature correspondences at object silhouettes
```

---

# 22. Do not assume these previous hypotheses are still valid

Several plausible hypotheses were considered and later weakened or disproved.

Do NOT automatically assume:

```text
Large estimated inter-frame rotation means bad registration.
```

The object genuinely undergoes large rotations.

Do NOT assume:

```text
Sparse + visual hull must automatically improve the final surface.
```

This has not been demonstrated.

Do NOT assume:

```text
The visual-hull problem is caused by a different coordinate system.
```

Same-frame sparse/hull comparison suggests otherwise.

Do NOT assume:

```text
More frames can simply be concatenated.
```

They must first be registered.

---

# 23. Current state of the project

The main feature-based pipeline is now conceptually complete enough to perform:

```text
camera calibration
        ↓
stereo calibration
        ↓
frame-wise sparse reconstruction
        ↓
frame-wise filtering
        ↓
frame registration
        ↓
registered cloud generation
        ↓
registered-cloud filtering
        ↓
object-level mesh generation
```

The next research emphasis should probably shift from adding pipeline stages toward:

```text
validating registration accuracy
optimizing mesh reconstruction
quantifying the value of multiple object poses
comparing against reference 3D measurements
testing texture enhancement approaches
```

The visual-hull branch should remain experimental until its added value is demonstrated.

---

# 24. Instructions for the assisting AI

When helping develop this repository:

1. **Read the current Git code before proposing code-level changes.** This handover describes development history, not necessarily the latest exact function signatures.

2. Preserve the modular script-based architecture unless there is a strong reason to change it.

3. The scripts are intentionally VS Code friendly. Do not unnecessarily convert everything into command-line-only tools.

4. Prefer:
   ```text
   pathlib.Path
   dataclasses
   type hints
   small reusable functions
   concise function docstrings
   project_config.yaml for shared configuration
   ```

5. Avoid duplicating project paths, camera names, and other shared settings in several scripts.

6. When debugging geometry, inspect:
   ```text
   calibration
   feature matches
   reprojection errors
   frame selection
   camera-pair selection
   registration metrics
   ```
   before tuning downstream meshing.

7. Treat all distances according to calibration units. The test pipeline generally uses millimetres.

8. Do not confuse:
   ```text
   sparse reconstruction completeness
   ```
   with:
   ```text
   sparse reconstruction accuracy
   ```

   A sparse cloud can be accurate but incomplete.

9. Keep experimental visual-hull functionality separate from the main production/research pipeline unless evidence shows that it improves reconstruction.

10. Whenever proposing a new algorithm, explain:
    ```text
    what existing problem it addresses,
    what information it uses,
    what output it adds,
    and how its benefit can be experimentally measured.
    ```

---

# 25. Suggested immediate tasks for the new developer

Good next tasks, depending on project priorities, include:

```text
A. Verify the current Git README against the actual pipeline.

B. Run the complete main pipeline on a known dataset and record:
   - points per frame
   - filtering losses
   - registration metrics
   - final registered point count
   - mesh quality

C. Compare object reconstruction using different numbers of frames.

D. Investigate how orientation diversity affects reconstruction completeness.

E. Quantitatively compare the registered reconstruction against a
   profile-scanner or commercial-photogrammetry reference.

F. Continue visual-hull work only as a controlled experimental branch.

G. Consider a denser reconstruction stage after the sparse camera/object
   geometry has been validated.
```

The most important principle is to establish quantitative evidence for each improvement rather than judging only from mesh appearance.