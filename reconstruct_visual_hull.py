from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from project_config import load_project_config, make_run_suffix


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
class VisualHullSettings:
    """Voxel-grid and silhouette settings for visual-hull reconstruction."""

    bounds_min_xyz: tuple[float, float, float]
    bounds_max_xyz: tuple[float, float, float]
    voxel_size: float
    threshold: int = 25
    morph_kernel_size: int = 5
    mask_dilation_px: int = 3
    keep_largest_component: bool = True
    min_camera_support: int | None = None
    save_debug_masks: bool = True
    save_occupied_voxels: bool = True
    save_surface_voxels: bool = True


@dataclass
class VisualHullResult:
    """Visual-hull voxel reconstruction for one synchronized frame."""

    occupied_voxel_centers: np.ndarray
    surface_voxel_centers: np.ndarray
    voxel_size: float
    object_name: str
    frame_key: str
    camera_names: list[str]


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
    """Load intrinsic calibration files for all configured cameras."""
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


def collect_object_frames_by_camera_dirs(
    object_root: Path | str,
    camera_names: Sequence[str],
) -> list[dict[str, Path]]:
    """Collect synchronized frames from camera-specific object folders."""
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

    if not common_frame_keys:
        raise RuntimeError(
            f"No synchronized frame keys found under {object_root}"
        )

    frames: list[dict[str, Path]] = []
    for frame_key in sorted(common_frame_keys, key=natural_sort_key):
        frame: dict[str, Path] = {"__frame_key__": Path(frame_key)}
        for camera_name in camera_names:
            frame[camera_name] = files_by_camera_and_key[camera_name][frame_key]
        frames.append(frame)

    return frames


def create_silhouette_mask(
    image_bgr: np.ndarray,
    threshold: int,
    morph_kernel_size: int,
    mask_dilation_px: int,
    keep_largest_component: bool,
) -> np.ndarray:
    """Segment a bright object against a dark background."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    if morph_kernel_size > 1:
        kernel = np.ones(
            (morph_kernel_size, morph_kernel_size),
            dtype=np.uint8,
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if keep_largest_component:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    if mask_dilation_px > 0:
        size = 2 * mask_dilation_px + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel)

    return mask


def create_voxel_grid(
    bounds_min_xyz: Sequence[float],
    bounds_max_xyz: Sequence[float],
    voxel_size: float,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Create voxel-center coordinates in the reference-camera frame."""
    bounds_min = np.asarray(bounds_min_xyz, dtype=np.float64)
    bounds_max = np.asarray(bounds_max_xyz, dtype=np.float64)

    if np.any(bounds_max <= bounds_min):
        raise ValueError("Each maximum bound must exceed the corresponding minimum.")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive.")

    x = np.arange(bounds_min[0] + voxel_size / 2, bounds_max[0], voxel_size)
    y = np.arange(bounds_min[1] + voxel_size / 2, bounds_max[1], voxel_size)
    z = np.arange(bounds_min[2] + voxel_size / 2, bounds_max[2], voxel_size)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    centers = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    return centers, (len(x), len(y), len(z))


def project_voxels_to_camera(
    voxel_centers_ref: np.ndarray,
    calibration: CameraCalibration,
    pose: CameraPose,
) -> tuple[np.ndarray, np.ndarray]:
    """Project reference-frame voxel centers into one camera image."""
    points_camera = (
        pose.rotation_ref_to_camera @ voxel_centers_ref.T
        + pose.translation_ref_to_camera
    ).T
    positive_depth = points_camera[:, 2] > 0

    rvec, _ = cv2.Rodrigues(pose.rotation_ref_to_camera)
    pixels, _ = cv2.projectPoints(
        voxel_centers_ref.astype(np.float64),
        rvec,
        pose.translation_ref_to_camera,
        calibration.camera_matrix,
        calibration.dist_coeffs,
    )

    return pixels.reshape(-1, 2), positive_depth


def carve_voxels_with_silhouettes(
    voxel_centers_ref: np.ndarray,
    masks_by_camera: dict[str, np.ndarray],
    calibrations: dict[str, CameraCalibration],
    poses: dict[str, CameraPose],
    selected_camera_names: Sequence[str],
    min_camera_support: int | None,
) -> np.ndarray:
    """Keep voxels whose projections lie inside enough silhouettes."""
    if min_camera_support is None:
        min_camera_support = len(selected_camera_names)

    if not 1 <= min_camera_support <= len(selected_camera_names):
        raise ValueError(
            "min_camera_support must be between 1 and the number of selected cameras."
        )

    support_count = np.zeros(len(voxel_centers_ref), dtype=np.uint16)

    for camera_name in selected_camera_names:
        mask = masks_by_camera[camera_name]
        height, width = mask.shape[:2]

        pixels, positive_depth = project_voxels_to_camera(
            voxel_centers_ref,
            calibrations[camera_name],
            poses[camera_name],
        )

        u = np.rint(pixels[:, 0]).astype(np.int64)
        v = np.rint(pixels[:, 1]).astype(np.int64)

        inside_image = (
            positive_depth
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )

        supported = np.zeros(len(voxel_centers_ref), dtype=bool)
        valid_indices = np.flatnonzero(inside_image)
        supported[valid_indices] = (
            mask[v[valid_indices], u[valid_indices]] > 0
        )

        support_count += supported.astype(np.uint16)

    return support_count >= min_camera_support


def extract_surface_voxel_mask(
    occupied_mask_flat: np.ndarray,
    grid_shape: tuple[int, int, int],
) -> np.ndarray:
    """Mark occupied voxels with at least one empty 6-neighbour."""
    occupied = occupied_mask_flat.reshape(grid_shape)
    padded = np.pad(occupied, 1, mode="constant", constant_values=False)
    interior = padded[1:-1, 1:-1, 1:-1]

    fully_surrounded = (
        padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )

    return (interior & ~fully_surrounded).ravel()


def reconstruct_visual_hull_for_frame(
    object_name: str,
    frame_key: str,
    frame: dict[str, Path],
    camera_names: Sequence[str],
    calibrations: dict[str, CameraCalibration],
    poses: dict[str, CameraPose],
    settings: VisualHullSettings,
    frame_output_dir: Path | str,
) -> VisualHullResult:
    """Reconstruct a voxel visual hull for one synchronized frame."""
    frame_output_dir = Path(frame_output_dir)
    masks_by_camera: dict[str, np.ndarray] = {}

    for camera_name in camera_names:
        image_path = frame[camera_name]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        mask = create_silhouette_mask(
            image_bgr=image,
            threshold=settings.threshold,
            morph_kernel_size=settings.morph_kernel_size,
            mask_dilation_px=settings.mask_dilation_px,
            keep_largest_component=settings.keep_largest_component,
        )
        masks_by_camera[camera_name] = mask

        if settings.save_debug_masks:
            mask_path = (
                frame_output_dir
                / "debug"
                / "visual_hull_masks"
                / f"{camera_name}_mask.png"
            )
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), mask)

    voxel_centers, grid_shape = create_voxel_grid(
        settings.bounds_min_xyz,
        settings.bounds_max_xyz,
        settings.voxel_size,
    )

    print(
        f"frame={frame_key}: testing {len(voxel_centers)} voxels "
        f"with grid shape {grid_shape}"
    )

    occupied_mask = carve_voxels_with_silhouettes(
        voxel_centers_ref=voxel_centers,
        masks_by_camera=masks_by_camera,
        calibrations=calibrations,
        poses=poses,
        selected_camera_names=camera_names,
        min_camera_support=settings.min_camera_support,
    )

    surface_mask = extract_surface_voxel_mask(
        occupied_mask,
        grid_shape,
    )

    return VisualHullResult(
        occupied_voxel_centers=voxel_centers[occupied_mask].astype(np.float32),
        surface_voxel_centers=voxel_centers[surface_mask].astype(np.float32),
        voxel_size=settings.voxel_size,
        object_name=object_name,
        frame_key=frame_key,
        camera_names=list(camera_names),
    )


def save_xyz_point_cloud_ply(
    points_xyz: np.ndarray,
    output_path: Path | str,
    color_rgb: tuple[int, int, int] = (180, 180, 180),
) -> None:
    """Save XYZ points as an ASCII PLY point cloud."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    red, green, blue = color_rgb

    with output_path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points_xyz)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")

        for x, y, z in points_xyz:
            file.write(
                f"{x:.8f} {y:.8f} {z:.8f} "
                f"{red} {green} {blue}\n"
            )


def save_visual_hull_result(
    result: VisualHullResult,
    frame_output_dir: Path | str,
    settings: VisualHullSettings,
) -> None:
    """Save visual-hull voxel data into one frame-specific output folder."""
    frame_output_dir = Path(frame_output_dir)
    frame_output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = frame_output_dir / "visual_hull.npz"

    np.savez_compressed(
        npz_path,
        occupied_voxel_centers=result.occupied_voxel_centers,
        surface_voxel_centers=result.surface_voxel_centers,
        voxel_size=np.asarray(result.voxel_size),
        object_name=np.asarray(result.object_name),
        frame_key=np.asarray(result.frame_key),
        camera_names=np.asarray(result.camera_names),
        bounds_min_xyz=np.asarray(settings.bounds_min_xyz),
        bounds_max_xyz=np.asarray(settings.bounds_max_xyz),
    )

    if settings.save_occupied_voxels:
        save_xyz_point_cloud_ply(
            result.occupied_voxel_centers,
            frame_output_dir / "visual_hull_occupied.ply",
            color_rgb=(180, 180, 180),
        )

    if settings.save_surface_voxels:
        save_xyz_point_cloud_ply(
            result.surface_voxel_centers,
            frame_output_dir / "visual_hull_surface.ply",
            color_rgb=(220, 170, 50),
        )

    print(f"Saved visual hull NPZ: {npz_path}")
    print(
        f"Occupied voxels: {len(result.occupied_voxel_centers)}, "
        f"surface voxels: {len(result.surface_voxel_centers)}"
    )


def update_frame_info_json(
    frame_info_path: Path | str,
    **updates,
) -> None:
    """Update selected fields in an existing frame_info.json file."""
    frame_info_path = Path(frame_info_path)

    if frame_info_path.exists():
        with frame_info_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    else:
        data = {}

    data.update(updates)

    frame_info_path.parent.mkdir(parents=True, exist_ok=True)
    with frame_info_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def reconstruct_all_visual_hulls(
    project_root: Path | str,
    reconstruction_root: Path | str,
    camera_names: Sequence[str],
    reference_camera_index: int,
    settings: VisualHullSettings,
    frame_keys_to_use: Sequence[str] | None = None,
    target_camera_indices: Sequence[int] | None = None,
) -> None:
    """Reconstruct one visual hull per object and synchronized frame."""
    project_root = Path(project_root)
    reconstruction_root = Path(reconstruction_root)

    calibration_files_root = project_root / "calibration_files"
    objects_root = project_root / "object"

    # The reference camera is always included. target_camera_indices uses the
    # original camera indices from project_config.yaml / extrinsics_result.json.
    if target_camera_indices is None:
        selected_camera_indices = list(range(len(camera_names)))
    else:
        selected_camera_indices = [
            reference_camera_index,
            *target_camera_indices,
        ]

    selected_camera_indices = list(dict.fromkeys(selected_camera_indices))

    for camera_index in selected_camera_indices:
        if not 0 <= camera_index < len(camera_names):
            raise ValueError(
                f"Camera index {camera_index} is out of range for "
                f"{len(camera_names)} configured cameras."
            )

    selected_camera_names = [
        camera_names[camera_index]
        for camera_index in selected_camera_indices
    ]

    print(
        "Visual hull cameras: "
        + ", ".join(
            f"{camera_name} (index {camera_index})"
            for camera_index, camera_name in zip(
                selected_camera_indices,
                selected_camera_names,
            )
        )
    )

    calibrations = load_project_calibrations(
        calibration_files_root,
        selected_camera_names,
    )

    # Load poses with the complete original camera list so that transform keys
    # such as cam0_to_cam2 retain their original camera indices.
    all_poses = load_camera_poses_from_extrinsics(
        calibration_files_root / "extrinsics_result.json",
        camera_names,
        reference_camera_index,
    )

    poses = {
        camera_name: all_poses[camera_name]
        for camera_name in selected_camera_names
    }

    object_dirs = sorted(
        [path for path in objects_root.iterdir() if path.is_dir()],
        key=lambda path: natural_sort_key(path.name),
    )

    for object_dir in object_dirs:
        print(f"\nReconstructing visual hull: {object_dir.name}")

        frames = collect_object_frames_by_camera_dirs(
            object_root=object_dir,
            camera_names=selected_camera_names,
        )

        if frame_keys_to_use is not None:
            allowed = {str(key) for key in frame_keys_to_use}
            frames = [
                frame
                for frame in frames
                if frame["__frame_key__"].stem in allowed
            ]

        if not frames:
            print(
                f"[WARN] No selected frames found for "
                f"{object_dir.name}"
            )
            continue

        for frame in frames:
            frame_key = frame["__frame_key__"].stem
            frame_output_dir = (
                reconstruction_root
                / object_dir.name
                / f"frame_{frame_key}"
            )

            result = reconstruct_visual_hull_for_frame(
                object_name=object_dir.name,
                frame_key=frame_key,
                frame=frame,
                camera_names=selected_camera_names,
                calibrations=calibrations,
                poses=poses,
                settings=settings,
                frame_output_dir=frame_output_dir,
            )

            save_visual_hull_result(
                result=result,
                frame_output_dir=frame_output_dir,
                settings=settings,
            )

            update_frame_info_json(
                frame_output_dir / "frame_info.json",
                visual_hull_occupied_voxels=int(
                    len(result.occupied_voxel_centers)
                ),
                visual_hull_surface_voxels=int(
                    len(result.surface_voxel_centers)
                ),
                visual_hull_voxel_size=float(
                    result.voxel_size
                ),
                visual_hull_cameras=list(
                    result.camera_names
                ),
            )


def main() -> None:
    """VS Code friendly entry point using project_config.yaml."""
    config = load_project_config(Path("project_config.yaml"))

    project_root = Path(config["img_root"])
    camera_names = config["camera_names"]
    reference_camera_index = config["reference_camera_index"]

    debug = config.get("debug", {})
    debug_enabled = bool(debug.get("enabled", False))
    run_suffix = make_run_suffix(config)

    if debug_enabled:
        frame_keys_to_use = debug.get("frame_keys_to_use")
        target_camera_indices = debug.get("target_camera_indices")
    else:
        frame_keys_to_use = None
        target_camera_indices = None

    reconstruction_root = (
        project_root
        / (
            config["folders"]["reconstruction_results"]
            + run_suffix
        )
    )

    visual_hull_config = config["visual_hull"]

    settings = VisualHullSettings(
        bounds_min_xyz=tuple(
            visual_hull_config["bounds_min_xyz"]
        ),
        bounds_max_xyz=tuple(
            visual_hull_config["bounds_max_xyz"]
        ),
        voxel_size=float(
            visual_hull_config["voxel_size"]
        ),
        threshold=int(
            visual_hull_config.get("threshold", 25)
        ),
        morph_kernel_size=int(
            visual_hull_config.get(
                "morph_kernel_size",
                5,
            )
        ),
        mask_dilation_px=int(
            visual_hull_config.get(
                "mask_dilation_px",
                3,
            )
        ),
        keep_largest_component=bool(
            visual_hull_config.get(
                "keep_largest_component",
                True,
            )
        ),
        min_camera_support=visual_hull_config.get(
            "min_camera_support"
        ),
        save_debug_masks=bool(
            visual_hull_config.get(
                "save_debug_masks",
                True,
            )
        ),
        save_occupied_voxels=bool(
            visual_hull_config.get(
                "save_occupied_voxels",
                True,
            )
        ),
        save_surface_voxels=bool(
            visual_hull_config.get(
                "save_surface_voxels",
                True,
            )
        ),
    )

    reconstruct_all_visual_hulls(
        project_root=project_root,
        reconstruction_root=reconstruction_root,
        camera_names=camera_names,
        reference_camera_index=reference_camera_index,
        settings=settings,
        frame_keys_to_use=frame_keys_to_use,
        target_camera_indices=target_camera_indices,
    )


if __name__ == "__main__":
    main()
