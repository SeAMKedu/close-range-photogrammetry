from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
from project_config import load_project_config, make_run_suffix

import cv2
import numpy as np

@dataclass
class CameraCalibration:
    """Intrinsic calibration and image-size information for one camera."""

    name: str
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int] | None = None


@dataclass
class CameraPose:
    """Rigid transform from the reference camera coordinate system to one camera."""

    camera_name: str
    rotation_ref_to_camera: np.ndarray
    translation_ref_to_camera: np.ndarray


@dataclass
class MatchedImagePair:
    """Matched 2D feature points between two camera images."""

    reference_points_px: np.ndarray
    target_points_px: np.ndarray
    reference_colors_bgr: np.ndarray


@dataclass
class TriangulatedPointCloud:
    """Sparse point cloud reconstructed from one synchronized frame."""

    points_xyz: np.ndarray
    colors_rgb: np.ndarray
    source_object_name: str
    source_frame_key: str
    pair_statistics: list[dict[str, object]] = field(default_factory=list)


def natural_sort_key(text: str) -> list[int | str]:
    """Sort strings so that 1, 2, 10 are ordered numerically, not alphabetically."""
    parts = re.split(r"([0-9]+)", text)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def default_frame_key_from_filename(path: Path) -> str:
    """Extract the frame key from the filename prefix before the first underscore."""
    return path.stem.split("_", maxsplit=1)[0]


def is_supported_image(
    path: Path,
    image_extensions: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
) -> bool:
    """Check whether a path points to a supported image file."""
    allowed = {ext.lower() for ext in image_extensions}
    return path.is_file() and path.suffix.lower() in allowed


def load_camera_calibration(path: Path | str, camera_name: str) -> CameraCalibration:
    """Load one intrinsic calibration JSON file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    image_size = None
    if "image_size" in data:
        image_size = tuple(int(value) for value in data["image_size"])

    return CameraCalibration(
        name=camera_name,
        camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
        dist_coeffs=np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
        image_size=image_size,
    )


def load_camera_poses_from_extrinsics(
    extrinsics_json_path: Path | str,
    camera_names: Sequence[str],
    reference_camera_index: int,
) -> dict[str, CameraPose]:
    """
    Load camera poses from extrinsics_result.json.

    The script expects transforms named cam0_to_cam1, cam0_to_cam2, etc.
    Poses are returned as transforms from the reference camera to each camera.
    """
    extrinsics_json_path = Path(extrinsics_json_path)
    with extrinsics_json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    reference_name = camera_names[reference_camera_index]
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
                    f"Transform {transform_key} not found in {extrinsics_json_path}. "
                    f"Available keys: {list(pairwise.keys())}"
                )
            rotation = np.asarray(pairwise[transform_key]["R"], dtype=np.float64)
            translation = np.asarray(pairwise[transform_key]["t"], dtype=np.float64).reshape(3, 1)

        poses[camera_name] = CameraPose(
            camera_name=camera_name,
            rotation_ref_to_camera=rotation,
            translation_ref_to_camera=translation,
        )

    return poses


def collect_object_frames_by_camera_dirs(
    object_root: Path | str,
    camera_names: Sequence[str],
) -> list[dict[str, Path]]:
    """
    Collect synchronized object image sets from camera-specific folders.

    Expected structure:
        object_root/cam0/1_obj0_xxxx.jpg
        object_root/cam1/1_obj0_yyyy.jpg
        object_root/cam2/1_obj0_zzzz.jpg

    Synchronization is based on the filename prefix before the first underscore.
    """
    object_root = Path(object_root)
    camera_dirs = {camera_name: object_root / camera_name for camera_name in camera_names}

    missing_dirs = [name for name, path in camera_dirs.items() if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(f"Missing camera folders under {object_root}: {missing_dirs}")

    files_by_camera_and_key: dict[str, dict[str, Path]] = {}

    for camera_name, camera_dir in camera_dirs.items():
        files_by_key: dict[str, Path] = {}

        for path in sorted(camera_dir.iterdir(), key=lambda p: natural_sort_key(p.name)):
            if not is_supported_image(path):
                continue

            frame_key = default_frame_key_from_filename(path)
            if frame_key in files_by_key:
                raise RuntimeError(
                    f"Duplicate frame key '{frame_key}' in {camera_dir}: "
                    f"{files_by_key[frame_key].name} and {path.name}"
                )

            files_by_key[frame_key] = path

        if not files_by_key:
            raise FileNotFoundError(f"No supported object images found in {camera_dir}")

        files_by_camera_and_key[camera_name] = files_by_key

    common_frame_keys = set(files_by_camera_and_key[camera_names[0]].keys())
    for camera_name in camera_names[1:]:
        common_frame_keys &= set(files_by_camera_and_key[camera_name].keys())

    if not common_frame_keys:
        raise RuntimeError(
            f"No common frame keys found under {object_root}. "
            "The prefix before the first underscore must match across cameras."
        )

    frames: list[dict[str, Path]] = []
    for frame_key in sorted(common_frame_keys, key=natural_sort_key):
        frame: dict[str, Path] = {"__frame_key__": Path(frame_key)}
        for camera_name in camera_names:
            frame[camera_name] = files_by_camera_and_key[camera_name][frame_key]
        frames.append(frame)

    return frames


def create_feature_detector(method: str = "sift"):
    """Create an OpenCV local feature detector and descriptor extractor."""
    method = method.lower()
    if method == "sift":
        return cv2.SIFT_create(nfeatures=8000)
    if method == "orb":
        return cv2.ORB_create(nfeatures=10000)
    raise ValueError(f"Unsupported feature detector: {method}")

def save_debug_match_image(
    reference_image_bgr: np.ndarray,
    reference_keypoints,
    target_image_bgr: np.ndarray,
    target_keypoints,
    matches,
    output_path: Path | str,
    max_matches: int = 100,
) -> None:
    """Save a side-by-side visualization of accepted feature matches."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    visualization = cv2.drawMatches(
        reference_image_bgr,
        reference_keypoints,
        target_image_bgr,
        target_keypoints,
        list(matches[:max_matches]),
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.imwrite(str(output_path), visualization)

def create_object_mask(
    image_bgr: np.ndarray,
    threshold: int = 25,
    morph_kernel_size: int = 7,
    padding_px: int = 20,
) -> np.ndarray:
    """Create a binary object mask using the largest connected component."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    mask = (gray > threshold).astype(np.uint8) * 255

    kernel = np.ones((morph_kernel_size, morph_kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        return mask

    # Skip label 0 because it is background.
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]
    w = stats[largest_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_label, cv2.CC_STAT_HEIGHT]

    x0 = max(x - padding_px, 0)
    y0 = max(y - padding_px, 0)
    x1 = min(x + w + padding_px, image_bgr.shape[1])
    y1 = min(y + h + padding_px, image_bgr.shape[0])

    bbox_mask = np.zeros_like(mask)
    bbox_mask[y0:y1, x0:x1] = 255

    # Keep only pixels that are bright enough AND inside the object bounding box.
    final_mask = cv2.bitwise_and(mask, bbox_mask)

    return final_mask

def match_features(
    reference_image_bgr: np.ndarray,
    target_image_bgr: np.ndarray,
    method: str = "sift",
    lowe_ratio: float = 0.75,
    debug_match_image_path: Path | str | None = None,
    max_debug_matches: int = 100,
) -> MatchedImagePair | None:
    """Detect and match local features between two images."""
    detector = create_feature_detector(method)

    ref_gray = cv2.cvtColor(reference_image_bgr, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(target_image_bgr, cv2.COLOR_BGR2GRAY)

    ref_mask = create_object_mask(reference_image_bgr)
    tgt_mask = create_object_mask(target_image_bgr)

    ref_keypoints, ref_descriptors = detector.detectAndCompute(ref_gray, ref_mask)
    tgt_keypoints, tgt_descriptors = detector.detectAndCompute(tgt_gray, tgt_mask)

    if ref_descriptors is None or tgt_descriptors is None:
        return None

    if method.lower() == "orb":
        norm_type = cv2.NORM_HAMMING
    else:
        norm_type = cv2.NORM_L2

    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    knn_matches = matcher.knnMatch(ref_descriptors, tgt_descriptors, k=2)

    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) < 2:
            continue
        best, second_best = match_pair
        if best.distance < lowe_ratio * second_best.distance:
            good_matches.append(best)

    if len(good_matches) < 8:
        return None

    ref_points = np.asarray(
        [ref_keypoints[match.queryIdx].pt for match in good_matches],
        dtype=np.float32,
    )
    tgt_points = np.asarray(
        [tgt_keypoints[match.trainIdx].pt for match in good_matches],
        dtype=np.float32,
    )

    colors_bgr = sample_colors(reference_image_bgr, ref_points)

    # For debugging, sort matches by distance so that the best matches are 
    # visualized when max_debug_matches is set.
    good_matches = sorted(good_matches, key=lambda match: match.distance)

    if debug_match_image_path is not None and good_matches:
        save_debug_match_image(
            reference_image_bgr=reference_image_bgr,
            reference_keypoints=ref_keypoints,
            target_image_bgr=target_image_bgr,
            target_keypoints=tgt_keypoints,
            matches=good_matches,
            output_path=debug_match_image_path,
            max_matches=max_debug_matches,
        )

    return MatchedImagePair(
        reference_points_px=ref_points,
        target_points_px=tgt_points,
        reference_colors_bgr=colors_bgr,
    )


def sample_colors(image_bgr: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    """Sample BGR colors from image coordinates using nearest-neighbour indexing."""
    height, width = image_bgr.shape[:2]
    x = np.clip(np.round(points_px[:, 0]).astype(int), 0, width - 1)
    y = np.clip(np.round(points_px[:, 1]).astype(int), 0, height - 1)
    return image_bgr[y, x]


def undistort_points_to_normalized_camera(
    points_px: np.ndarray,
    calibration: CameraCalibration,
) -> np.ndarray:
    """Convert distorted pixel points to normalized undistorted camera coordinates."""
    points = points_px.reshape(-1, 1, 2).astype(np.float64)
    undistorted = cv2.undistortPoints(
        points,
        calibration.camera_matrix,
        calibration.dist_coeffs,
    )
    return undistorted.reshape(-1, 2)


def triangulate_points_between_reference_and_target(
    matched_pair: MatchedImagePair,
    reference_calibration: CameraCalibration,
    target_calibration: CameraCalibration,
    target_pose: CameraPose,
    max_reprojection_error_px: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Triangulate matched points between the reference camera and a target camera.

    Output coordinates are in the reference camera coordinate system.
    The returned boolean mask refers to the original matched feature arrays.
    """
    ref_points_norm = undistort_points_to_normalized_camera(
        matched_pair.reference_points_px,
        reference_calibration,
    )
    tgt_points_norm = undistort_points_to_normalized_camera(
        matched_pair.target_points_px,
        target_calibration,
    )

    projection_ref = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float64)
    projection_tgt = np.hstack(
        [target_pose.rotation_ref_to_camera, target_pose.translation_ref_to_camera]
    ).astype(np.float64)

    points_homogeneous = cv2.triangulatePoints(
        projection_ref,
        projection_tgt,
        ref_points_norm.T,
        tgt_points_norm.T,
    )

    points_xyz = (points_homogeneous[:3] / points_homogeneous[3]).T

    valid_mask = np.isfinite(points_xyz).all(axis=1)
    valid_mask &= points_xyz[:, 2] > 0

    points_in_target = (
        target_pose.rotation_ref_to_camera @ points_xyz.T
        + target_pose.translation_ref_to_camera
    ).T
    valid_mask &= points_in_target[:, 2] > 0

    reprojection_error = calculate_pair_reprojection_error_px(
        points_xyz=points_xyz,
        ref_points_px=matched_pair.reference_points_px,
        target_points_px=matched_pair.target_points_px,
        reference_calibration=reference_calibration,
        target_calibration=target_calibration,
        target_pose=target_pose,
    )
    valid_mask &= reprojection_error < max_reprojection_error_px

    valid_points = points_xyz[valid_mask]
    valid_colors_rgb = matched_pair.reference_colors_bgr[valid_mask][:, ::-1]

    return valid_points, valid_colors_rgb, valid_mask


def calculate_pair_reprojection_error_px(
    points_xyz: np.ndarray,
    ref_points_px: np.ndarray,
    target_points_px: np.ndarray,
    reference_calibration: CameraCalibration,
    target_calibration: CameraCalibration,
    target_pose: CameraPose,
) -> np.ndarray:
    """Calculate mean reprojection error in pixels for each triangulated point."""
    rvec_ref = np.zeros((3, 1), dtype=np.float64)
    tvec_ref = np.zeros((3, 1), dtype=np.float64)
    projected_ref, _ = cv2.projectPoints(
        points_xyz.astype(np.float64),
        rvec_ref,
        tvec_ref,
        reference_calibration.camera_matrix,
        reference_calibration.dist_coeffs,
    )

    rvec_tgt, _ = cv2.Rodrigues(target_pose.rotation_ref_to_camera)
    projected_tgt, _ = cv2.projectPoints(
        points_xyz.astype(np.float64),
        rvec_tgt,
        target_pose.translation_ref_to_camera,
        target_calibration.camera_matrix,
        target_calibration.dist_coeffs,
    )

    projected_ref = projected_ref.reshape(-1, 2)
    projected_tgt = projected_tgt.reshape(-1, 2)

    ref_error = np.linalg.norm(projected_ref - ref_points_px, axis=1)
    tgt_error = np.linalg.norm(projected_tgt - target_points_px, axis=1)

    return 0.5 * (ref_error + tgt_error)



def project_points_to_reference_and_target(
    points_xyz: np.ndarray,
    reference_calibration: CameraCalibration,
    target_calibration: CameraCalibration,
    target_pose: CameraPose,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D points to the reference and target camera image planes."""
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

    return projected_ref.reshape(-1, 2), projected_target.reshape(-1, 2)


def draw_reprojection_debug(
    image_bgr: np.ndarray,
    observed_points_px: np.ndarray,
    reprojected_points_px: np.ndarray,
) -> np.ndarray:
    """Draw observed feature points and reprojected 3D points on an image."""
    visualization = image_bgr.copy()

    for observed, reprojected in zip(observed_points_px, reprojected_points_px):
        ox, oy = np.rint(observed).astype(int)
        rx, ry = np.rint(reprojected).astype(int)

        cv2.circle(visualization, (ox, oy), 4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.drawMarker(
            visualization,
            (rx, ry),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=8,
            thickness=1,
            line_type=cv2.LINE_AA,
        )

    return visualization


def save_debug_reprojection_images(
    reference_image_bgr: np.ndarray,
    target_image_bgr: np.ndarray,
    matched_pair: MatchedImagePair,
    valid_mask: np.ndarray,
    points_xyz: np.ndarray,
    reference_calibration: CameraCalibration,
    target_calibration: CameraCalibration,
    target_pose: CameraPose,
    reference_camera_name: str,
    target_camera_name: str,
    output_dir: Path | str,
) -> None:
    """Save observed-vs-reprojected feature visualizations for one camera pair."""
    if len(points_xyz) == 0:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observed_reference = matched_pair.reference_points_px[valid_mask]
    observed_target = matched_pair.target_points_px[valid_mask]

    projected_reference, projected_target = project_points_to_reference_and_target(
        points_xyz=points_xyz,
        reference_calibration=reference_calibration,
        target_calibration=target_calibration,
        target_pose=target_pose,
    )

    reference_visualization = draw_reprojection_debug(
        reference_image_bgr,
        observed_reference,
        projected_reference,
    )
    target_visualization = draw_reprojection_debug(
        target_image_bgr,
        observed_target,
        projected_target,
    )

    pair_name = f"{reference_camera_name}_to_{target_camera_name}"

    cv2.imwrite(
        str(output_dir / f"{pair_name}_reference_reprojection.png"),
        reference_visualization,
    )
    cv2.imwrite(
        str(output_dir / f"{pair_name}_target_reprojection.png"),
        target_visualization,
    )


def reconstruct_sparse_point_cloud_for_frame(
    frame: dict[str, Path],
    object_name: str,
    camera_names: Sequence[str],
    calibrations: dict[str, CameraCalibration],
    poses: dict[str, CameraPose],
    reference_camera_index: int,
    target_camera_indices: Sequence[int],
    feature_method: str,
    lowe_ratio: float,
    max_reprojection_error_px: float,
    frame_output_dir: Path | str,
    save_debug_images: bool,
) -> TriangulatedPointCloud | None:
    """Reconstruct one sparse point cloud from one synchronized frame."""
    frame_output_dir = Path(frame_output_dir)
    frame_key = frame["__frame_key__"].stem
    reference_camera_name = camera_names[reference_camera_index]

    reference_image = cv2.imread(str(frame[reference_camera_name]), cv2.IMREAD_COLOR)
    if reference_image is None:
        print(f"[WARN] Could not read reference image for frame {frame_key}")
        return None

    frame_points: list[np.ndarray] = []
    frame_colors: list[np.ndarray] = []
    pair_statistics: list[dict[str, object]] = []

    for target_camera_index in target_camera_indices:
        if target_camera_index == reference_camera_index:
            continue

        target_camera_name = camera_names[target_camera_index]
        print(
            f"[DEBUG] Using pair: {reference_camera_name} - {target_camera_name} "
            f"(indices {reference_camera_index}-{target_camera_index}), frame={frame_key}"
        )

        target_image = cv2.imread(str(frame[target_camera_name]), cv2.IMREAD_COLOR)
        if target_image is None:
            print(f"[WARN] Could not read target image {frame[target_camera_name]}")
            continue

        debug_match_image_path = None
        if save_debug_images:
            debug_match_image_path = (
                frame_output_dir
                / "debug"
                / "feature_matches"
                / f"{reference_camera_name}_to_{target_camera_name}_matches.jpg"
            )

        matched_pair = match_features(
            reference_image_bgr=reference_image,
            target_image_bgr=target_image,
            method=feature_method,
            lowe_ratio=lowe_ratio,
            debug_match_image_path=debug_match_image_path,
            max_debug_matches=100,
        )

        if matched_pair is None:
            pair_statistics.append(
                {
                    "target_camera": target_camera_name,
                    "feature_matches": 0,
                    "triangulated_points": 0,
                }
            )
            print(
                f"[WARN] Not enough feature matches: "
                f"frame={frame_key}, target={target_camera_name}"
            )
            continue

        points_xyz, colors_rgb, valid_mask = triangulate_points_between_reference_and_target(
            matched_pair=matched_pair,
            reference_calibration=calibrations[reference_camera_name],
            target_calibration=calibrations[target_camera_name],
            target_pose=poses[target_camera_name],
            max_reprojection_error_px=max_reprojection_error_px,
        )

        print(
            f"frame={frame_key}, pair={reference_camera_name}-{target_camera_name}: "
            f"matches={len(matched_pair.reference_points_px)}, "
            f"triangulated={len(points_xyz)}"
        )

        pair_statistics.append(
            {
                "target_camera": target_camera_name,
                "feature_matches": int(len(matched_pair.reference_points_px)),
                "triangulated_points": int(len(points_xyz)),
            }
        )

        if len(points_xyz) == 0:
            continue

        frame_points.append(points_xyz)
        frame_colors.append(colors_rgb)

        if save_debug_images:
            save_debug_reprojection_images(
                reference_image_bgr=reference_image,
                target_image_bgr=target_image,
                matched_pair=matched_pair,
                valid_mask=valid_mask,
                points_xyz=points_xyz,
                reference_calibration=calibrations[reference_camera_name],
                target_calibration=calibrations[target_camera_name],
                target_pose=poses[target_camera_name],
                reference_camera_name=reference_camera_name,
                target_camera_name=target_camera_name,
                output_dir=frame_output_dir / "debug" / "reprojection",
            )

    if not frame_points:
        print(
            f"[WARN] No triangulated points were produced for "
            f"{object_name}, frame {frame_key}"
        )
        return None

    return TriangulatedPointCloud(
        points_xyz=np.vstack(frame_points).astype(np.float32),
        colors_rgb=np.vstack(frame_colors).astype(np.uint8),
        source_object_name=object_name,
        source_frame_key=frame_key,
        pair_statistics=pair_statistics,
    )

def reconstruct_object_sparse_point_cloud(
    object_root: Path | str,
    object_output_dir: Path | str,
    camera_names: Sequence[str],
    calibrations: dict[str, CameraCalibration],
    poses: dict[str, CameraPose],
    reference_camera_index: int,
    feature_method: str = "sift",
    lowe_ratio: float = 0.75,
    max_reprojection_error_px: float = 5.0,
    target_camera_indices: Sequence[int] | None = None,
    frame_keys_to_use: Sequence[str] | None = None,
    save_debug_images: bool = False,
) -> list[TriangulatedPointCloud]:
    """Reconstruct separate sparse point clouds for all selected frames."""
    object_root = Path(object_root)
    object_output_dir = Path(object_output_dir)
    object_name = object_root.name

    if target_camera_indices is None:
        target_camera_indices = [
            index
            for index in range(len(camera_names))
            if index != reference_camera_index
        ]
    else:
        target_camera_indices = list(target_camera_indices)

    for target_camera_index in target_camera_indices:
        if not (0 <= target_camera_index < len(camera_names)):
            raise ValueError(
                f"Invalid target camera index {target_camera_index}. "
                f"Valid indices are 0 ... {len(camera_names) - 1}."
            )

    frames = collect_object_frames_by_camera_dirs(
        object_root=object_root,
        camera_names=camera_names,
    )

    if frame_keys_to_use is not None:
        allowed_frame_keys = {str(key) for key in frame_keys_to_use}
        frames = [
            frame
            for frame in frames
            if frame["__frame_key__"].stem in allowed_frame_keys
        ]

        if not frames:
            raise RuntimeError(
                f"No frames matched frame_keys_to_use={sorted(allowed_frame_keys)} "
                f"for object {object_root}"
            )

    print(
        f"[DEBUG] {object_name}: using target_camera_indices="
        f"{list(target_camera_indices)}, "
        f"frame_keys={[frame['__frame_key__'].stem for frame in frames]}"
    )

    point_clouds: list[TriangulatedPointCloud] = []

    for frame in frames:
        frame_key = frame["__frame_key__"].stem
        frame_output_dir = object_output_dir / f"frame_{frame_key}"

        point_cloud = reconstruct_sparse_point_cloud_for_frame(
            frame=frame,
            object_name=object_name,
            camera_names=camera_names,
            calibrations=calibrations,
            poses=poses,
            reference_camera_index=reference_camera_index,
            target_camera_indices=target_camera_indices,
            feature_method=feature_method,
            lowe_ratio=lowe_ratio,
            max_reprojection_error_px=max_reprojection_error_px,
            frame_output_dir=frame_output_dir,
            save_debug_images=save_debug_images,
        )

        if point_cloud is not None:
            point_clouds.append(point_cloud)

    if not point_clouds:
        raise RuntimeError(
            f"No triangulated frame point clouds were produced for {object_root}"
        )

    return point_clouds


def save_point_cloud_npz(
    point_cloud: TriangulatedPointCloud,
    output_path: Path | str,
) -> None:
    """Save point cloud and metadata in a NumPy NPZ file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        points_xyz=point_cloud.points_xyz,
        colors_rgb=point_cloud.colors_rgb,
        source_object_name=np.asarray(point_cloud.source_object_name),
        source_frame_key=np.asarray(point_cloud.source_frame_key),
    )


def save_point_cloud_ply(
    point_cloud: TriangulatedPointCloud,
    output_path: Path | str,
) -> None:
    """Save point cloud as an ASCII PLY file for CloudCompare/Open3D viewing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points = point_cloud.points_xyz
    colors = point_cloud.colors_rgb

    with output_path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")

        for point, color in zip(points, colors):
            file.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )



def save_frame_info_json(
    point_cloud: TriangulatedPointCloud,
    output_path: Path | str,
    reference_camera_name: str,
    target_camera_names: Sequence[str],
    processing_time_s: float,
) -> None:
    """Save frame-level reconstruction metadata for later pipeline stages."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "frame_key": point_cloud.source_frame_key,
        "object_name": point_cloud.source_object_name,
        "reference_camera": reference_camera_name,
        "target_cameras": list(target_camera_names),
        "triangulated_points": int(len(point_cloud.points_xyz)),
        "pair_statistics": point_cloud.pair_statistics,
        "filtered_points": None,
        "visual_hull_occupied_voxels": None,
        "visual_hull_surface_voxels": None,
        "processing_time_s": float(processing_time_s),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

def load_project_calibrations(
    calibration_files_root: Path | str,
    camera_names: Sequence[str],
) -> dict[str, CameraCalibration]:
    """Load all camera intrinsic calibration files from calibration_files_root."""
    calibration_files_root = Path(calibration_files_root)

    calibrations = {}
    for camera_name in camera_names:
        calibration_path = calibration_files_root / f"{camera_name}_calibration.json"
        calibrations[camera_name] = load_camera_calibration(calibration_path, camera_name)

    return calibrations


def reconstruct_all_objects_in_project(
    project_root: Path | str,
    camera_names: Sequence[str],
    reference_camera_index: int = 0,
    target_camera_indices: Sequence[int] | None = None,
    frame_keys_to_use: Sequence[str] | None = None,
    output_folder_name: str = "reconstruction_results",
    save_debug_images: bool = False,
) -> None:
    """Reconstruct and save one sparse point cloud per selected frame."""
    project_root = Path(project_root)
    calibration_files_root = project_root / "calibration_files"
    objects_root = project_root / "object"
    output_root = project_root / output_folder_name

    calibrations = load_project_calibrations(
        calibration_files_root=calibration_files_root,
        camera_names=camera_names,
    )

    poses = load_camera_poses_from_extrinsics(
        extrinsics_json_path=calibration_files_root / "extrinsics_result.json",
        camera_names=camera_names,
        reference_camera_index=reference_camera_index,
    )

    object_dirs = sorted(
        [path for path in objects_root.iterdir() if path.is_dir()],
        key=lambda p: natural_sort_key(p.name),
    )

    if not object_dirs:
        raise FileNotFoundError(f"No object folders found under {objects_root}")

    if target_camera_indices is None:
        active_target_indices = [
            index
            for index in range(len(camera_names))
            if index != reference_camera_index
        ]
    else:
        active_target_indices = list(target_camera_indices)

    reference_camera_name = camera_names[reference_camera_index]
    target_camera_names = [
        camera_names[index]
        for index in active_target_indices
        if index != reference_camera_index
    ]

    for object_dir in object_dirs:
        print(f"\nReconstructing object: {object_dir.name}")
        object_output_dir = output_root / object_dir.name

        start_time = time.perf_counter()

        point_clouds = reconstruct_object_sparse_point_cloud(
            object_root=object_dir,
            object_output_dir=object_output_dir,
            camera_names=camera_names,
            calibrations=calibrations,
            poses=poses,
            reference_camera_index=reference_camera_index,
            feature_method="sift",
            lowe_ratio=0.75,
            max_reprojection_error_px=5.0,
            target_camera_indices=active_target_indices,
            frame_keys_to_use=frame_keys_to_use,
            save_debug_images=save_debug_images,
        )

        elapsed_total = time.perf_counter() - start_time
        per_frame_time = elapsed_total / len(point_clouds)

        for point_cloud in point_clouds:
            frame_output_dir = (
                object_output_dir / f"frame_{point_cloud.source_frame_key}"
            )

            npz_path = frame_output_dir / "sparse_point_cloud.npz"
            ply_path = frame_output_dir / "sparse_point_cloud.ply"
            frame_info_path = frame_output_dir / "frame_info.json"

            save_point_cloud_npz(point_cloud, npz_path)
            save_point_cloud_ply(point_cloud, ply_path)
            save_frame_info_json(
                point_cloud=point_cloud,
                output_path=frame_info_path,
                reference_camera_name=reference_camera_name,
                target_camera_names=target_camera_names,
                processing_time_s=per_frame_time,
            )

            print(f"Saved NPZ: {npz_path}")
            print(f"Saved PLY: {ply_path}")
            print(
                f"Frame {point_cloud.source_frame_key}: "
                f"{len(point_cloud.points_xyz)} points"
            )


def main() -> None:
    config = load_project_config(Path("project_config.yaml"))
    project_root = Path(config["img_root"])

    debug = config.get("debug", {})
    debug_enabled = bool(debug.get("enabled", False))
    run_suffix = make_run_suffix(config)

    if debug_enabled:
        target_camera_indices = debug.get("target_camera_indices")
        frame_keys_to_use = debug.get("frame_keys_to_use")
    else:
        target_camera_indices = None
        frame_keys_to_use = None

    camera_names = config["camera_names"]
    output_folder_name = (
        config["folders"]["reconstruction_results"] + run_suffix
    )

    reconstruct_all_objects_in_project(
        project_root=project_root,
        camera_names=camera_names,
        reference_camera_index=config["reference_camera_index"],
        target_camera_indices=target_camera_indices,
        frame_keys_to_use=frame_keys_to_use,
        output_folder_name=output_folder_name,
        save_debug_images=debug_enabled,
    )


if __name__ == "__main__":
    main()
