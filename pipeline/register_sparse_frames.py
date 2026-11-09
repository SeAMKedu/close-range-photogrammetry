from __future__ import annotations

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError as exc:
    raise ImportError(
        "This script requires Open3D. Install it with: pip install open3d"
    ) from exc

from pipeline.project_config import load_project_config, make_run_suffix


@dataclass
class CameraCalibration:
    """Intrinsic calibration parameters for one camera."""

    name: str
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


@dataclass
class CameraPose:
    """Rigid transform from the reference camera to one target camera."""

    camera_name: str
    rotation_ref_to_camera: np.ndarray
    translation_ref_to_camera: np.ndarray


@dataclass
class FrameFeatureCloud:
    """Triangulated 3D features tied to SIFT keypoints in one frame."""

    frame_key: str
    reference_image_bgr: np.ndarray
    reference_keypoints: Sequence[cv2.KeyPoint]
    reference_descriptors: np.ndarray
    point_by_reference_keypoint: dict[int, np.ndarray]


@dataclass
class RegistrationSettings:
    """Settings for feature-based registration and ICP refinement."""

    sift_nfeatures: int = 8000
    stereo_lowe_ratio: float = 0.75
    temporal_lowe_ratio: float = 0.70
    max_reprojection_error_px: float = 2.0

    ransac_iterations: int = 2000
    ransac_inlier_threshold: float = 3.0
    ransac_min_inliers: int = 6

    icp_max_correspondence_distance: float = 5.0
    icp_max_iterations: int = 100

    merged_voxel_size: float | None = None

    save_debug_matches: bool = True
    save_debug_clouds: bool = True
    max_debug_matches: int = 100


def natural_sort_key(text: str) -> list[int | str]:
    """Sort strings so that 1, 2, 10 are ordered numerically."""
    parts = re.split(r"([0-9]+)", text)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def default_frame_key_from_filename(path: Path) -> str:
    """Extract the frame key from the filename prefix before the first underscore."""
    return path.stem.split("_", maxsplit=1)[0]


def is_supported_image(
    path: Path,
    image_extensions: Sequence[str] = (
        ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"
    ),
) -> bool:
    """Check whether a path points to a supported image file."""
    allowed = {extension.lower() for extension in image_extensions}
    return path.is_file() and path.suffix.lower() in allowed


def load_camera_calibration(
    path: Path | str,
    camera_name: str,
) -> CameraCalibration:
    """Load one intrinsic calibration JSON file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return CameraCalibration(
        name=camera_name,
        camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
        dist_coeffs=np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
    )


def load_project_calibrations(
    calibration_files_root: Path | str,
    camera_names: Sequence[str],
) -> dict[str, CameraCalibration]:
    """Load intrinsic calibrations for all configured cameras."""
    calibration_files_root = Path(calibration_files_root)

    return {
        camera_name: load_camera_calibration(
            calibration_files_root / f"{camera_name}_calibration.json",
            camera_name,
        )
        for camera_name in camera_names
    }


def load_camera_poses_from_extrinsics(
    extrinsics_json_path: Path | str,
    camera_names: Sequence[str],
    reference_camera_index: int,
) -> dict[str, CameraPose]:
    """Load reference-to-camera transforms from extrinsics_result.json."""
    extrinsics_json_path = Path(extrinsics_json_path)

    with extrinsics_json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    pairwise = data["all_pairwise_transforms"]
    poses: dict[str, CameraPose] = {}

    for camera_index, camera_name in enumerate(camera_names):
        if camera_index == reference_camera_index:
            rotation = np.eye(3, dtype=np.float64)
            translation = np.zeros((3, 1), dtype=np.float64)
        else:
            transform_key = f"cam{reference_camera_index}_to_cam{camera_index}"
            if transform_key not in pairwise:
                raise KeyError(
                    f"Transform {transform_key} not found in {extrinsics_json_path}"
                )

            rotation = np.asarray(pairwise[transform_key]["R"], dtype=np.float64)
            translation = np.asarray(
                pairwise[transform_key]["t"], dtype=np.float64
            ).reshape(3, 1)

        poses[camera_name] = CameraPose(
            camera_name=camera_name,
            rotation_ref_to_camera=rotation,
            translation_ref_to_camera=translation,
        )

    return poses


def select_best_stereo_target_camera_index(
    extrinsics_json_path: Path | str,
    reference_camera_index: int,
    camera_names: Sequence[str],
) -> int:
    """Select the target camera with the smallest stored stereo RMS error."""
    extrinsics_json_path = Path(extrinsics_json_path)

    with extrinsics_json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    pairwise = data["all_pairwise_transforms"]
    candidates: list[tuple[float, int]] = []

    for camera_index in range(len(camera_names)):
        if camera_index == reference_camera_index:
            continue

        key = f"cam{reference_camera_index}_to_cam{camera_index}"
        transform_data = pairwise.get(key)
        if transform_data is None:
            continue

        rms = transform_data.get("rms_reprojection_error")
        if rms is None:
            continue

        try:
            rms_value = float(rms)
        except (TypeError, ValueError):
            continue

        if np.isfinite(rms_value):
            candidates.append((rms_value, camera_index))

    if candidates:
        rms_value, camera_index = min(candidates, key=lambda item: item[0])
        print(
            f"Selected stereo target: {camera_names[camera_index]} "
            f"(index {camera_index}), RMS={rms_value:.4f} px"
        )
        return camera_index

    for camera_index in range(len(camera_names)):
        if camera_index != reference_camera_index:
            print(
                "[WARN] No stereo RMS values found. Falling back to "
                f"{camera_names[camera_index]} (index {camera_index})."
            )
            return camera_index

    raise RuntimeError("No non-reference camera is available.")


def collect_object_frames_by_camera_dirs(
    object_root: Path | str,
    camera_names: Sequence[str],
) -> list[dict[str, Path]]:
    """Collect synchronized frames from the requested camera folders."""
    object_root = Path(object_root)
    files_by_camera_and_key: dict[str, dict[str, Path]] = {}

    for camera_name in camera_names:
        camera_dir = object_root / camera_name
        if not camera_dir.is_dir():
            raise FileNotFoundError(f"Missing camera folder: {camera_dir}")

        files_by_key: dict[str, Path] = {}
        for path in sorted(
            camera_dir.iterdir(),
            key=lambda item: natural_sort_key(item.name),
        ):
            if not is_supported_image(path):
                continue

            frame_key = default_frame_key_from_filename(path)
            if frame_key in files_by_key:
                raise RuntimeError(
                    f"Duplicate frame key '{frame_key}' in {camera_dir}"
                )

            files_by_key[frame_key] = path

        if not files_by_key:
            raise FileNotFoundError(f"No supported images found in {camera_dir}")

        files_by_camera_and_key[camera_name] = files_by_key

    common_frame_keys = set(files_by_camera_and_key[camera_names[0]].keys())
    for camera_name in camera_names[1:]:
        common_frame_keys &= set(files_by_camera_and_key[camera_name].keys())

    frames: list[dict[str, Path]] = []
    for frame_key in sorted(common_frame_keys, key=natural_sort_key):
        frame: dict[str, Path] = {"__frame_key__": Path(frame_key)}
        for camera_name in camera_names:
            frame[camera_name] = files_by_camera_and_key[camera_name][frame_key]
        frames.append(frame)

    return frames


def create_object_mask(
    image_bgr: np.ndarray,
    threshold: int = 25,
    morph_kernel_size: int = 7,
) -> np.ndarray:
    """Create an object mask from a bright object on a dark background."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold).astype(np.uint8) * 255

    kernel = np.ones(
        (morph_kernel_size, morph_kernel_size),
        dtype=np.uint8,
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return mask

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def detect_sift_features(
    image_bgr: np.ndarray,
    nfeatures: int,
) -> tuple[Sequence[cv2.KeyPoint], np.ndarray | None]:
    """Detect masked SIFT features in one image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = create_object_mask(image_bgr)
    detector = cv2.SIFT_create(nfeatures=nfeatures)
    return detector.detectAndCompute(gray, mask)


def ratio_test_matches(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    lowe_ratio: float,
) -> list[cv2.DMatch]:
    """Match SIFT descriptors using BFMatcher and Lowe's ratio test."""
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)

    good_matches: list[cv2.DMatch] = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < lowe_ratio * second.distance:
            good_matches.append(best)

    return sorted(good_matches, key=lambda match: match.distance)


def undistort_points_to_normalized_camera(
    points_px: np.ndarray,
    calibration: CameraCalibration,
) -> np.ndarray:
    """Convert distorted pixels to normalized undistorted coordinates."""
    points = points_px.reshape(-1, 1, 2).astype(np.float64)
    undistorted = cv2.undistortPoints(
        points,
        calibration.camera_matrix,
        calibration.dist_coeffs,
    )
    return undistorted.reshape(-1, 2)


def calculate_pair_reprojection_error_px(
    points_xyz: np.ndarray,
    ref_points_px: np.ndarray,
    target_points_px: np.ndarray,
    reference_calibration: CameraCalibration,
    target_calibration: CameraCalibration,
    target_pose: CameraPose,
) -> np.ndarray:
    """Calculate mean two-camera reprojection error for each 3D point."""
    projected_ref, _ = cv2.projectPoints(
        points_xyz.astype(np.float64),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        reference_calibration.camera_matrix,
        reference_calibration.dist_coeffs,
    )

    target_rvec, _ = cv2.Rodrigues(target_pose.rotation_ref_to_camera)
    projected_target, _ = cv2.projectPoints(
        points_xyz.astype(np.float64),
        target_rvec,
        target_pose.translation_ref_to_camera,
        target_calibration.camera_matrix,
        target_calibration.dist_coeffs,
    )

    projected_ref = projected_ref.reshape(-1, 2)
    projected_target = projected_target.reshape(-1, 2)

    ref_error = np.linalg.norm(projected_ref - ref_points_px, axis=1)
    target_error = np.linalg.norm(projected_target - target_points_px, axis=1)

    return 0.5 * (ref_error + target_error)


def build_frame_feature_cloud(
    frame: dict[str, Path],
    reference_camera_name: str,
    target_camera_name: str,
    reference_calibration: CameraCalibration,
    target_calibration: CameraCalibration,
    target_pose: CameraPose,
    settings: RegistrationSettings,
) -> FrameFeatureCloud:
    """Triangulate SIFT features and retain their reference-image keypoint IDs."""
    frame_key = frame["__frame_key__"].stem

    reference_image = cv2.imread(
        str(frame[reference_camera_name]),
        cv2.IMREAD_COLOR,
    )
    target_image = cv2.imread(
        str(frame[target_camera_name]),
        cv2.IMREAD_COLOR,
    )

    if reference_image is None or target_image is None:
        raise FileNotFoundError(
            f"Could not read stereo images for frame {frame_key}"
        )

    reference_keypoints, reference_descriptors = detect_sift_features(
        reference_image,
        settings.sift_nfeatures,
    )
    target_keypoints, target_descriptors = detect_sift_features(
        target_image,
        settings.sift_nfeatures,
    )

    if reference_descriptors is None or target_descriptors is None:
        raise RuntimeError(f"No SIFT descriptors found for frame {frame_key}")

    stereo_matches = ratio_test_matches(
        reference_descriptors,
        target_descriptors,
        settings.stereo_lowe_ratio,
    )

    if len(stereo_matches) < 8:
        raise RuntimeError(
            f"Too few stereo matches for frame {frame_key}: {len(stereo_matches)}"
        )

    reference_points_px = np.asarray(
        [reference_keypoints[match.queryIdx].pt for match in stereo_matches],
        dtype=np.float64,
    )
    target_points_px = np.asarray(
        [target_keypoints[match.trainIdx].pt for match in stereo_matches],
        dtype=np.float64,
    )

    reference_points_norm = undistort_points_to_normalized_camera(
        reference_points_px,
        reference_calibration,
    )
    target_points_norm = undistort_points_to_normalized_camera(
        target_points_px,
        target_calibration,
    )

    projection_reference = np.hstack(
        [
            np.eye(3, dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
        ]
    )
    projection_target = np.hstack(
        [
            target_pose.rotation_ref_to_camera,
            target_pose.translation_ref_to_camera,
        ]
    ).astype(np.float64)

    points_homogeneous = cv2.triangulatePoints(
        projection_reference,
        projection_target,
        reference_points_norm.T,
        target_points_norm.T,
    )
    points_xyz = (points_homogeneous[:3] / points_homogeneous[3]).T

    valid_mask = np.isfinite(points_xyz).all(axis=1)
    valid_mask &= points_xyz[:, 2] > 0

    points_target = (
        target_pose.rotation_ref_to_camera @ points_xyz.T
        + target_pose.translation_ref_to_camera
    ).T
    valid_mask &= points_target[:, 2] > 0

    reprojection_error = calculate_pair_reprojection_error_px(
        points_xyz=points_xyz,
        ref_points_px=reference_points_px,
        target_points_px=target_points_px,
        reference_calibration=reference_calibration,
        target_calibration=target_calibration,
        target_pose=target_pose,
    )
    valid_mask &= reprojection_error < settings.max_reprojection_error_px

    point_by_reference_keypoint: dict[int, np.ndarray] = {}

    for match, point_xyz, valid in zip(stereo_matches, points_xyz, valid_mask):
        if valid:
            point_by_reference_keypoint[int(match.queryIdx)] = point_xyz.astype(
                np.float64
            )

    print(
        f"frame={frame_key}: stereo matches={len(stereo_matches)}, "
        f"usable 3D feature points={len(point_by_reference_keypoint)}"
    )

    return FrameFeatureCloud(
        frame_key=frame_key,
        reference_image_bgr=reference_image,
        reference_keypoints=reference_keypoints,
        reference_descriptors=reference_descriptors,
        point_by_reference_keypoint=point_by_reference_keypoint,
    )


def build_temporal_3d_correspondences(
    target_frame: FrameFeatureCloud,
    source_frame: FrameFeatureCloud,
    lowe_ratio: float,
) -> tuple[np.ndarray, np.ndarray, list[cv2.DMatch]]:
    """Build source-to-target 3D correspondences from temporal SIFT matches."""
    temporal_matches = ratio_test_matches(
        target_frame.reference_descriptors,
        source_frame.reference_descriptors,
        lowe_ratio,
    )

    source_points: list[np.ndarray] = []
    target_points: list[np.ndarray] = []
    usable_matches: list[cv2.DMatch] = []

    for match in temporal_matches:
        target_index = int(match.queryIdx)
        source_index = int(match.trainIdx)

        if target_index not in target_frame.point_by_reference_keypoint:
            continue
        if source_index not in source_frame.point_by_reference_keypoint:
            continue

        target_points.append(target_frame.point_by_reference_keypoint[target_index])
        source_points.append(source_frame.point_by_reference_keypoint[source_index])
        usable_matches.append(match)

    if not source_points:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            [],
        )

    return (
        np.asarray(source_points, dtype=np.float64),
        np.asarray(target_points, dtype=np.float64),
        usable_matches,
    )


def estimate_rigid_transform_svd(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    """Estimate a rigid 4x4 transform mapping source points to target points."""
    if len(source_points) < 3:
        raise ValueError("At least three 3D correspondences are required.")

    source_centroid = source_points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)

    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T

    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    translation = target_centroid - rotation @ source_centroid

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def transform_points(
    points_xyz: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """Apply a homogeneous 4x4 rigid transform to XYZ points."""
    homogeneous = np.column_stack(
        [points_xyz, np.ones(len(points_xyz), dtype=np.float64)]
    )
    return (transform @ homogeneous.T).T[:, :3]


def estimate_rigid_transform_ransac(
    source_points: np.ndarray,
    target_points: np.ndarray,
    iterations: int,
    inlier_threshold: float,
    minimum_inliers: int,
    random_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate a robust rigid transform from noisy 3D-3D correspondences."""
    if len(source_points) != len(target_points):
        raise ValueError(
            "Source and target correspondence arrays must have equal length."
        )
    if len(source_points) < 3:
        raise RuntimeError(
            f"Too few 3D correspondences for RANSAC: {len(source_points)}"
        )

    rng = np.random.default_rng(random_seed)

    best_inlier_mask: np.ndarray | None = None
    best_inlier_count = 0
    best_rmse = np.inf

    for _ in range(iterations):
        sample_indices = rng.choice(len(source_points), size=3, replace=False)

        try:
            candidate_transform = estimate_rigid_transform_svd(
                source_points[sample_indices],
                target_points[sample_indices],
            )
        except np.linalg.LinAlgError:
            continue

        transformed = transform_points(source_points, candidate_transform)
        residuals = np.linalg.norm(transformed - target_points, axis=1)

        inlier_mask = residuals <= inlier_threshold
        inlier_count = int(inlier_mask.sum())
        if inlier_count < 3:
            continue

        rmse = float(np.sqrt(np.mean(residuals[inlier_mask] ** 2)))

        if (
            inlier_count > best_inlier_count
            or (inlier_count == best_inlier_count and rmse < best_rmse)
        ):
            best_inlier_mask = inlier_mask
            best_inlier_count = inlier_count
            best_rmse = rmse

    if best_inlier_mask is None or best_inlier_count < minimum_inliers:
        raise RuntimeError(
            f"RANSAC failed: best inlier count {best_inlier_count}, "
            f"required {minimum_inliers}"
        )

    refined_transform = estimate_rigid_transform_svd(
        source_points[best_inlier_mask],
        target_points[best_inlier_mask],
    )

    refined_points = transform_points(
        source_points[best_inlier_mask],
        refined_transform,
    )
    refined_rmse = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (refined_points - target_points[best_inlier_mask]) ** 2,
                    axis=1,
                )
            )
        )
    )

    return refined_transform, best_inlier_mask, refined_rmse


def load_open3d_point_cloud(
    path: Path | str,
) -> o3d.geometry.PointCloud:
    """Load points and optional colors from a frame NPZ file."""
    path = Path(path)

    with np.load(path, allow_pickle=True) as data:
        points = np.asarray(data["points_xyz"], dtype=np.float64)

        colors = None
        if "colors_rgb" in data:
            colors = np.asarray(data["colors_rgb"], dtype=np.float64) / 255.0

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    if colors is not None and len(colors) == len(points):
        cloud.colors = o3d.utility.Vector3dVector(colors)

    return cloud


def refine_transform_with_icp(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    initial_transform: np.ndarray,
    settings: RegistrationSettings,
) -> o3d.pipelines.registration.RegistrationResult:
    """Refine a source-to-target transform using point-to-point ICP."""
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=settings.icp_max_iterations
    )

    return o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        settings.icp_max_correspondence_distance,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria,
    )


def save_temporal_match_debug_image(
    target_frame: FrameFeatureCloud,
    source_frame: FrameFeatureCloud,
    matches: Sequence[cv2.DMatch],
    inlier_mask: np.ndarray | None,
    output_path: Path | str,
    max_matches: int,
) -> None:
    """Save temporal SIFT matches used for frame registration."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matches_to_draw = list(matches)
    if inlier_mask is not None and len(inlier_mask) == len(matches_to_draw):
        matches_to_draw = [
            match
            for match, is_inlier in zip(matches_to_draw, inlier_mask)
            if is_inlier
        ]

    matches_to_draw = matches_to_draw[:max_matches]

    visualization = cv2.drawMatches(
        target_frame.reference_image_bgr,
        list(target_frame.reference_keypoints),
        source_frame.reference_image_bgr,
        list(source_frame.reference_keypoints),
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.imwrite(str(output_path), visualization)


def save_cloud_pair_debug(
    target_cloud: o3d.geometry.PointCloud,
    source_cloud: o3d.geometry.PointCloud,
    transform: np.ndarray,
    output_path: Path | str,
) -> None:
    """Save target plus transformed source as one PLY debug cloud."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_copy = o3d.geometry.PointCloud(target_cloud)
    source_copy = o3d.geometry.PointCloud(source_cloud)
    source_copy.transform(transform)

    combined = target_copy + source_copy
    o3d.io.write_point_cloud(str(output_path), combined)


def save_registered_cloud_npz(
    cloud: o3d.geometry.PointCloud,
    output_path: Path | str,
) -> None:
    """Save a registered Open3D point cloud as compressed NPZ."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "points_xyz": np.asarray(cloud.points, dtype=np.float32),
    }

    if cloud.has_colors():
        arrays["colors_rgb"] = np.clip(
            np.asarray(cloud.colors) * 255.0,
            0,
            255,
        ).astype(np.uint8)

    np.savez_compressed(output_path, **arrays)


def register_object_frames(
    object_images_root: Path | str,
    object_results_root: Path | str,
    camera_names: Sequence[str],
    reference_camera_index: int,
    stereo_target_camera_index: int,
    calibrations: dict[str, CameraCalibration],
    poses: dict[str, CameraPose],
    settings: RegistrationSettings,
) -> None:
    """Register all available frame clouds sequentially into the first frame pose."""
    object_images_root = Path(object_images_root)
    object_results_root = Path(object_results_root)

    reference_camera_name = camera_names[reference_camera_index]
    target_camera_name = camera_names[stereo_target_camera_index]

    frames = collect_object_frames_by_camera_dirs(
        object_images_root,
        [reference_camera_name, target_camera_name],
    )

    usable_frames: list[tuple[dict[str, Path], Path]] = []
    for frame in frames:
        frame_key = frame["__frame_key__"].stem
        cloud_path = (
            object_results_root
            / f"frame_{frame_key}"
            / "filtered_point_cloud.npz"
        )
        if cloud_path.exists():
            usable_frames.append((frame, cloud_path))

    if not usable_frames:
        raise FileNotFoundError(
            f"No filtered frame clouds found under {object_results_root}"
        )

    registered_root = object_results_root / "registered"
    debug_root = registered_root / "debug"
    registered_root.mkdir(parents=True, exist_ok=True)

    first_frame, first_cloud_path = usable_frames[0]
    first_frame_key = first_frame["__frame_key__"].stem

    print(f"\nRegistration reference frame: {first_frame_key}")

    previous_feature_cloud = build_frame_feature_cloud(
        frame=first_frame,
        reference_camera_name=reference_camera_name,
        target_camera_name=target_camera_name,
        reference_calibration=calibrations[reference_camera_name],
        target_calibration=calibrations[target_camera_name],
        target_pose=poses[target_camera_name],
        settings=settings,
    )
    previous_native_cloud = load_open3d_point_cloud(first_cloud_path)

    transforms_to_reference: dict[str, np.ndarray] = {
        first_frame_key: np.eye(4, dtype=np.float64)
    }

    registration_report: dict[str, dict] = {
        first_frame_key: {
            "status": "reference",
            "transform_to_reference": transforms_to_reference[
                first_frame_key
            ].tolist(),
        }
    }

    transformed_clouds: list[o3d.geometry.PointCloud] = [
        o3d.geometry.PointCloud(previous_native_cloud)
    ]

    previous_frame_key = first_frame_key

    for frame, source_cloud_path in usable_frames[1:]:
        source_frame_key = frame["__frame_key__"].stem

        print(
            f"\nRegistering frame {source_frame_key} -> {previous_frame_key}"
        )

        try:
            source_feature_cloud = build_frame_feature_cloud(
                frame=frame,
                reference_camera_name=reference_camera_name,
                target_camera_name=target_camera_name,
                reference_calibration=calibrations[reference_camera_name],
                target_calibration=calibrations[target_camera_name],
                target_pose=poses[target_camera_name],
                settings=settings,
            )

            source_points, target_points, temporal_matches = (
                build_temporal_3d_correspondences(
                    target_frame=previous_feature_cloud,
                    source_frame=source_feature_cloud,
                    lowe_ratio=settings.temporal_lowe_ratio,
                )
            )

            print(f"3D-3D correspondences: {len(source_points)}")

            initial_transform, inlier_mask, ransac_rmse = (
                estimate_rigid_transform_ransac(
                    source_points=source_points,
                    target_points=target_points,
                    iterations=settings.ransac_iterations,
                    inlier_threshold=settings.ransac_inlier_threshold,
                    minimum_inliers=settings.ransac_min_inliers,
                )
            )

            source_native_cloud = load_open3d_point_cloud(source_cloud_path)

            icp_result = refine_transform_with_icp(
                source_cloud=source_native_cloud,
                target_cloud=previous_native_cloud,
                initial_transform=initial_transform,
                settings=settings,
            )

            pairwise_transform = icp_result.transformation
            previous_to_reference = transforms_to_reference[previous_frame_key]
            source_to_reference = previous_to_reference @ pairwise_transform

            transforms_to_reference[source_frame_key] = source_to_reference

            transformed_source = o3d.geometry.PointCloud(source_native_cloud)
            transformed_source.transform(source_to_reference)
            transformed_clouds.append(transformed_source)

            registration_report[source_frame_key] = {
                "status": "registered",
                "registered_to_frame": previous_frame_key,
                "temporal_3d_correspondences": int(len(source_points)),
                "ransac_inliers": int(inlier_mask.sum()),
                "ransac_rmse": float(ransac_rmse),
                "icp_fitness": float(icp_result.fitness),
                "icp_inlier_rmse": float(icp_result.inlier_rmse),
                "pairwise_transform": pairwise_transform.tolist(),
                "transform_to_reference": source_to_reference.tolist(),
            }

            if settings.save_debug_matches:
                save_temporal_match_debug_image(
                    target_frame=previous_feature_cloud,
                    source_frame=source_feature_cloud,
                    matches=temporal_matches,
                    inlier_mask=inlier_mask,
                    output_path=(
                        debug_root
                        / "feature_matches"
                        / (
                            f"frame_{previous_frame_key}"
                            f"_to_frame_{source_frame_key}_matches.jpg"
                        )
                    ),
                    max_matches=settings.max_debug_matches,
                )

            if settings.save_debug_clouds:
                save_cloud_pair_debug(
                    target_cloud=previous_native_cloud,
                    source_cloud=source_native_cloud,
                    transform=initial_transform,
                    output_path=(
                        debug_root
                        / "before_icp"
                        / f"frame_{source_frame_key}_initial_alignment.ply"
                    ),
                )
                save_cloud_pair_debug(
                    target_cloud=previous_native_cloud,
                    source_cloud=source_native_cloud,
                    transform=pairwise_transform,
                    output_path=(
                        debug_root
                        / "after_icp"
                        / f"frame_{source_frame_key}_icp_alignment.ply"
                    ),
                )

            previous_feature_cloud = source_feature_cloud
            previous_native_cloud = source_native_cloud
            previous_frame_key = source_frame_key

        except Exception as exc:
            print(
                f"[WARN] Registration failed for frame {source_frame_key}: {exc}"
            )
            registration_report[source_frame_key] = {
                "status": "failed",
                "error": str(exc),
            }

    merged_cloud = o3d.geometry.PointCloud()
    for cloud in transformed_clouds:
        merged_cloud += cloud

    if settings.merged_voxel_size is not None and settings.merged_voxel_size > 0:
        merged_cloud = merged_cloud.voxel_down_sample(settings.merged_voxel_size)

    registered_ply_path = registered_root / "registered_sparse_point_cloud.ply"
    registered_npz_path = registered_root / "registered_sparse_point_cloud.npz"
    transforms_path = registered_root / "frame_transforms.json"
    report_path = registered_root / "registration_report.json"

    o3d.io.write_point_cloud(str(registered_ply_path), merged_cloud)
    save_registered_cloud_npz(merged_cloud, registered_npz_path)

    with transforms_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                frame_key: transform.tolist()
                for frame_key, transform in transforms_to_reference.items()
            },
            file,
            indent=2,
        )

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(registration_report, file, indent=2)

    print(f"\nSaved registered PLY: {registered_ply_path}")
    print(f"Saved registered NPZ: {registered_npz_path}")
    print(f"Saved frame transforms: {transforms_path}")
    print(
        f"Registered frames: {len(transforms_to_reference)} / {len(usable_frames)}"
    )


def register_all_objects(
    project_root: Path | str,
    reconstruction_root: Path | str,
    camera_names: Sequence[str],
    reference_camera_index: int,
    stereo_target_camera_index: int,
    settings: RegistrationSettings,
) -> None:
    """Register frame-specific sparse clouds for every object."""
    project_root = Path(project_root)
    reconstruction_root = Path(reconstruction_root)

    calibration_files_root = project_root / "calibration_files"
    objects_root = project_root / "object"

    calibrations = load_project_calibrations(
        calibration_files_root,
        camera_names,
    )
    poses = load_camera_poses_from_extrinsics(
        calibration_files_root / "extrinsics_result.json",
        camera_names,
        reference_camera_index,
    )

    object_dirs = sorted(
        [path for path in objects_root.iterdir() if path.is_dir()],
        key=lambda path: natural_sort_key(path.name),
    )

    for object_dir in object_dirs:
        object_results_root = reconstruction_root / object_dir.name

        if not object_results_root.is_dir():
            print(
                f"[WARN] Results folder missing for {object_dir.name}: "
                f"{object_results_root}"
            )
            continue

        print(f"\n=== Registering object {object_dir.name} ===")

        register_object_frames(
            object_images_root=object_dir,
            object_results_root=object_results_root,
            camera_names=camera_names,
            reference_camera_index=reference_camera_index,
            stereo_target_camera_index=stereo_target_camera_index,
            calibrations=calibrations,
            poses=poses,
            settings=settings,
        )


def main() -> None:
    """VS Code friendly entry point using project_config.yaml."""
    config = load_project_config(Path("project_config.yaml"))

    project_root = Path(config["img_root"])
    camera_names = config["camera_names"]
    reference_camera_index = config["reference_camera_index"]

    run_suffix = make_run_suffix(config)
    reconstruction_root = (
        project_root
        / (config["folders"]["reconstruction_results"] + run_suffix)
    )

    calibration_files_root = project_root / "calibration_files"
    extrinsics_path = calibration_files_root / "extrinsics_result.json"

    # Optional project_config.yaml override:
    #
    # registration:
    #   stereo_target_camera_index: 2
    #
    # If omitted, the script selects the camera pair with the smallest
    # rms_reprojection_error stored in extrinsics_result.json.
    registration_config = config.get("registration", {})
    configured_target_index = registration_config.get(
        "stereo_target_camera_index"
    )

    if configured_target_index is None:
        stereo_target_camera_index = select_best_stereo_target_camera_index(
            extrinsics_json_path=extrinsics_path,
            reference_camera_index=reference_camera_index,
            camera_names=camera_names,
        )
    else:
        stereo_target_camera_index = int(configured_target_index)

    settings = RegistrationSettings(
        sift_nfeatures=8000,
        stereo_lowe_ratio=0.75,
        temporal_lowe_ratio=0.70,
        max_reprojection_error_px=2.0,

        # Units are the same as the calibration square size.
        ransac_iterations=2000,
        ransac_inlier_threshold=3.0,
        ransac_min_inliers=6,

        icp_max_correspondence_distance=5.0,
        icp_max_iterations=100,

        # Set e.g. 0.5 or 1.0 later if the merged cloud becomes very dense.
        merged_voxel_size=None,

        save_debug_matches=True,
        save_debug_clouds=True,
        max_debug_matches=100,
    )

    register_all_objects(
        project_root=project_root,
        reconstruction_root=reconstruction_root,
        camera_names=camera_names,
        reference_camera_index=reference_camera_index,
        stereo_target_camera_index=stereo_target_camera_index,
        settings=settings,
    )


if __name__ == "__main__":
    main()
