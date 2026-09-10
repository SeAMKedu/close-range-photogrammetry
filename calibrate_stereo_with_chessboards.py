from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from project_config import load_project_config

import cv2
import numpy as np

from chessboard_utils import (
    create_checkerboard_object_points,
    find_checkerboard_corners,
    save_corner_visualization,
)


@dataclass
class CameraIntrinsics:
    """Intrinsic calibration parameters for one camera."""

    name: str
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int] | None = None  # width, height


@dataclass
class DetectedFrame:
    """Checkerboard detections from one synchronized multi-camera frame."""

    frame_name: str
    image_points_by_camera: dict[int, np.ndarray]


@dataclass
class StereoCalibrationResult:
    """Stereo calibration result between the reference camera and one target camera."""

    reference_camera_index: int
    target_camera_index: int
    rms_error: float
    rotation_reference_to_target: np.ndarray
    translation_reference_to_target: np.ndarray
    essential_matrix: np.ndarray
    fundamental_matrix: np.ndarray
    used_frames: list[str]


@dataclass
class MultiCameraExtrinsicsResult:
    """Extrinsic calibration results for a multi-camera setup."""

    camera_names: list[str]
    reference_camera_index: int
    stereo_results: list[StereoCalibrationResult]
    all_pairwise_transforms: dict[str, dict[str, list]]
    board_size: tuple[int, int]
    square_size: float
    square_size_unit: str

    def to_json_dict(self) -> dict:
        """Convert the multi-camera extrinsics result to a JSON-serializable dict."""
        reference_name = self.camera_names[self.reference_camera_index]

        data = {
            "camera_names": self.camera_names,
            "reference_camera_index": self.reference_camera_index,
            "reference_camera_name": reference_name,
            "checkerboard": {
                "inner_corners_cols": self.board_size[0],
                "inner_corners_rows": self.board_size[1],
                "square_size": self.square_size,
                "square_size_unit": self.square_size_unit,
            },
            "reference_to_camera": {},
            "all_pairwise_transforms": self.all_pairwise_transforms,
        }

        for result in self.stereo_results:
            target_name = self.camera_names[result.target_camera_index]
            data["reference_to_camera"][target_name] = {
                "target_camera_index": result.target_camera_index,
                "rms_reprojection_error": float(result.rms_error),
                "R_reference_to_target": result.rotation_reference_to_target.tolist(),
                "t_reference_to_target": result.translation_reference_to_target.reshape(3).tolist(),
                "translation_unit": self.square_size_unit,
                "used_frames": result.used_frames,
            }

        return data


def load_intrinsics_json(path: Path | str, camera_name: str) -> CameraIntrinsics:
    """Load one OpenCV-style intrinsic calibration JSON file."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)

    image_size = None
    if "image_size" in data:
        image_size = tuple(int(value) for value in data["image_size"])

    return CameraIntrinsics(
        name=camera_name,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
    )


def is_supported_image(
    path: Path,
    image_extensions: Sequence[str],
) -> bool:
    """Check whether a path is a file with a supported image extension."""
    allowed_extensions = {extension.lower() for extension in image_extensions}
    return path.is_file() and path.suffix.lower() in allowed_extensions


def default_frame_key_from_filename(path: Path) -> str:
    """
    Extract a synchronization key from an image filename.

    Example:
        1_extrinsics_XXXX.png  ->  1
        14_extrinsics_YYYY.png ->  14

    The default rule uses the part before the first underscore.
    """
    return path.stem.split("_", maxsplit=1)[0]


def natural_sort_key(text: str) -> list[int | str]:
    """
    Return a sorting key that handles numbers naturally.

    This makes the order 1, 2, 3, ..., 10 instead of 1, 10, 2, 3.
    """
    parts = re.split(r"([0-9]+)", text)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def collect_synchronized_frames_by_camera_dirs(
    images_root: Path | str,
    camera_names: Sequence[str],
    image_extensions: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    frame_key_offsets: dict[str, int] | None = None,
    frame_cycle_length: int | None = None,
) -> list[dict[str, Path]]:
    """
    Collect synchronized image sets from camera-specific folders.

    Expected structure:
        images_root/cam0/1_extrinsics_XXXX.png
        images_root/cam1/1_extrinsics_YYYY.png
        images_root/cam2/1_extrinsics_ZZZ.png

        images_root/cam0/2_extrinsics_XXXX.png
        images_root/cam1/2_extrinsics_YYYY.png
        images_root/cam2/2_extrinsics_ZZZ.png

    Synchronization is based on a frame key extracted from each filename.
    By default, the frame key is the part before the first underscore.
    """
    images_root = Path(images_root)
    frame_key_offsets = frame_key_offsets or {}
    if frame_key_offsets and (frame_cycle_length is None or frame_cycle_length < 1):
        raise ValueError("frame_cycle_length must be positive when frame offsets are used.")
    camera_dirs = {camera_name: images_root / camera_name for camera_name in camera_names}

    missing_dirs = [name for name, path in camera_dirs.items() if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            f"Missing camera folders under {images_root}: {missing_dirs}"
        )

    files_by_camera_and_key: dict[str, dict[str, Path]] = {}

    for camera_name, camera_dir in camera_dirs.items():
        files_by_key: dict[str, Path] = {}

        for path in sorted(camera_dir.iterdir(), key=lambda p: natural_sort_key(p.name)):
            if not is_supported_image(path, image_extensions):
                continue

            source_frame_key = default_frame_key_from_filename(path)
            offset = int(frame_key_offsets.get(camera_name, 0))
            if offset:
                try:
                    source_number = int(source_frame_key)
                except ValueError as exc:
                    raise ValueError(
                        f"Frame offsets require numeric frame keys, got '{source_frame_key}' in {path}."
                    ) from exc
                # An offset describes which source image corresponds to a
                # reference frame: target = reference + offset, cyclically.
                frame_key = str(((source_number - 1 - offset) % frame_cycle_length) + 1)
            else:
                frame_key = source_frame_key

            if frame_key in files_by_key:
                raise RuntimeError(
                    f"Duplicate frame key '{frame_key}' in camera folder {camera_dir}. "
                    f"Files: {files_by_key[frame_key].name} and {path.name}. "
                    "Make the prefix before the first underscore unique per frame."
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
            "No matching frame keys found across camera folders. "
            "The default synchronization rule expects filenames to start with the same "
            "frame number or frame id before the first underscore, for example "
            "1_extrinsics_A.png, 1_extrinsics_B.png, 1_extrinsics_C.png."
        )

    all_frame_keys = set().union(
        *(set(files_by_camera_and_key[camera_name].keys()) for camera_name in camera_names)
    )

    for frame_key in sorted(all_frame_keys, key=natural_sort_key):
        missing_cameras = [
            camera_name
            for camera_name in camera_names
            if frame_key not in files_by_camera_and_key[camera_name]
        ]
        if missing_cameras:
            print(
                f"[WARN] Skipping frame key '{frame_key}': "
                f"missing images for cameras {missing_cameras}"
            )

    frames: list[dict[str, Path]] = []

    for frame_key in sorted(common_frame_keys, key=natural_sort_key):
        frame: dict[str, Path] = {"__frame_name__": Path(frame_key)}
        for camera_name in camera_names:
            frame[camera_name] = files_by_camera_and_key[camera_name][frame_key]
        frames.append(frame)

    return frames


def detect_checkerboards_in_synchronized_frames(
    frames: list[dict[str, Path]],
    camera_names: Sequence[str],
    board_size: tuple[int, int],
    save_corner_images: bool = False,
    corner_output_folder: Path | str | None = None,
) -> tuple[list[DetectedFrame], dict[int, tuple[int, int]]]:
    """Detect checkerboard corners in all cameras for each synchronized frame."""
    detected_frames: list[DetectedFrame] = []
    image_sizes: dict[int, tuple[int, int]] = {}

    if corner_output_folder is not None:
        corner_output_folder = Path(corner_output_folder)

    for frame in frames:
        frame_name = frame["__frame_name__"].stem
        image_points_by_camera: dict[int, np.ndarray] = {}
        all_cameras_detected = True

        for camera_index, camera_name in enumerate(camera_names):
            image_path = frame[camera_name]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

            if image is None:
                print(f"[WARN] Could not read image: {image_path}")
                all_cameras_detected = False
                break

            current_image_size = (image.shape[1], image.shape[0])
            if camera_index not in image_sizes:
                image_sizes[camera_index] = current_image_size
            elif image_sizes[camera_index] != current_image_size:
                raise ValueError(
                    f"Image size mismatch for {camera_name}. "
                    f"Expected {image_sizes[camera_index]}, got {current_image_size} "
                    f"for {image_path}"
                )

            found, corners = find_checkerboard_corners(
                image=image,
                board_size=board_size,
            )

            if not found or corners is None:
                print(f"[WARN] Checkerboard not found: frame={frame_name}, camera={camera_name}")
                all_cameras_detected = False
                break

            image_points_by_camera[camera_index] = corners.reshape(-1, 2).astype(np.float32)

            if save_corner_images and corner_output_folder is not None:
                output_path = corner_output_folder / f"frame_{frame_name}_{camera_name}_{image_path.name}"
                save_corner_visualization(
                    image=image,
                    board_size=board_size,
                    corners=corners,
                    output_path=output_path,
                )

        if all_cameras_detected:
            detected_frames.append(
                DetectedFrame(
                    frame_name=frame_name,
                    image_points_by_camera=image_points_by_camera,
                )
            )

    if not detected_frames:
        raise RuntimeError("No frames had successful checkerboard detection in all cameras.")

    return detected_frames, image_sizes


def calibrate_camera_pair_to_reference(
    reference_camera_index: int,
    target_camera_index: int,
    intrinsics: Sequence[CameraIntrinsics],
    detected_frames: Sequence[DetectedFrame],
    single_object_points: np.ndarray,
    image_size: tuple[int, int],
) -> StereoCalibrationResult:
    """Run stereoCalibrate for one camera pair using fixed intrinsics."""
    object_points = []
    reference_image_points = []
    target_image_points = []
    used_frames = []

    for frame in detected_frames:
        object_points.append(single_object_points.copy())
        reference_image_points.append(frame.image_points_by_camera[reference_camera_index])
        target_image_points.append(frame.image_points_by_camera[target_camera_index])
        used_frames.append(frame.frame_name)

    if len(object_points) < 3:
        raise RuntimeError(
            f"Too few usable shared frames for cameras "
            f"{reference_camera_index} and {target_camera_index}: {len(object_points)}"
        )

    reference_intrinsics = intrinsics[reference_camera_index]
    target_intrinsics = intrinsics[target_camera_index]

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-6,
    )

    rms_error, _, _, _, _, rotation, translation, essential_matrix, fundamental_matrix = cv2.stereoCalibrate(
        objectPoints=object_points,
        imagePoints1=reference_image_points,
        imagePoints2=target_image_points,
        cameraMatrix1=reference_intrinsics.camera_matrix.copy(),
        distCoeffs1=reference_intrinsics.dist_coeffs.copy(),
        cameraMatrix2=target_intrinsics.camera_matrix.copy(),
        distCoeffs2=target_intrinsics.dist_coeffs.copy(),
        imageSize=image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=criteria,
    )

    return StereoCalibrationResult(
        reference_camera_index=reference_camera_index,
        target_camera_index=target_camera_index,
        rms_error=float(rms_error),
        rotation_reference_to_target=rotation,
        translation_reference_to_target=translation.reshape(3, 1),
        essential_matrix=essential_matrix,
        fundamental_matrix=fundamental_matrix,
        used_frames=used_frames,
    )


def invert_transform(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert a rigid transform X_b = R_ab X_a + t_ab."""
    inverse_rotation = rotation.T
    inverse_translation = -inverse_rotation @ translation.reshape(3, 1)
    return inverse_rotation, inverse_translation


def compose_transform(
    rotation_ab: np.ndarray,
    translation_ab: np.ndarray,
    rotation_bc: np.ndarray,
    translation_bc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose transforms a -> b and b -> c into transform a -> c."""
    rotation_ac = rotation_bc @ rotation_ab
    translation_ac = rotation_bc @ translation_ab.reshape(3, 1) + translation_bc.reshape(3, 1)
    return rotation_ac, translation_ac


def build_all_pairwise_transforms(
    num_cameras: int,
    reference_camera_index: int,
    stereo_results: Sequence[StereoCalibrationResult],
) -> dict[str, dict[str, list]]:
    """Build transforms between all camera pairs using the reference camera as hub."""
    reference_to_camera: dict[int, tuple[np.ndarray, np.ndarray]] = {
        reference_camera_index: (
            np.eye(3, dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
        )
    }

    for result in stereo_results:
        reference_to_camera[result.target_camera_index] = (
            result.rotation_reference_to_target,
            result.translation_reference_to_target,
        )

    pairwise: dict[str, dict[str, list]] = {}

    for source_index in range(num_cameras):
        for target_index in range(num_cameras):
            if source_index == target_index:
                rotation = np.eye(3, dtype=np.float64)
                translation = np.zeros((3, 1), dtype=np.float64)
            else:
                rotation_ref_to_source, translation_ref_to_source = reference_to_camera[source_index]
                rotation_source_to_ref, translation_source_to_ref = invert_transform(
                    rotation_ref_to_source,
                    translation_ref_to_source,
                )

                rotation_ref_to_target, translation_ref_to_target = reference_to_camera[target_index]

                rotation, translation = compose_transform(
                    rotation_source_to_ref,
                    translation_source_to_ref,
                    rotation_ref_to_target,
                    translation_ref_to_target,
                )

            pairwise[f"cam{source_index}_to_cam{target_index}"] = {
                "R": rotation.tolist(),
                "t": translation.reshape(3).tolist(),
            }

    return pairwise


def save_extrinsics_json(
    result: MultiCameraExtrinsicsResult,
    output_path: Path | str,
) -> None:
    """Save multi-camera extrinsic calibration result as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result.to_json_dict(), file, indent=2)


def calibrate_multicamera_extrinsics(
    intrinsics_paths: Sequence[Path | str],
    images_root: Path | str,
    camera_names: Sequence[str],
    board_size: tuple[int, int],
    square_size: float,
    reference_camera_index: int = 0,
    square_size_unit: str = "mm",
    save_corner_images: bool = False,
    corner_output_folder: Path | str | None = None,
    output_json_path: Path | str | None = None,
    frame_key_offsets: dict[str, int] | None = None,
    frame_cycle_length: int | None = None,
) -> MultiCameraExtrinsicsResult:
    """Calibrate relative extrinsics for a fixed multi-camera setup."""
    if len(intrinsics_paths) != len(camera_names):
        raise ValueError("intrinsics_paths and camera_names must have the same length.")

    if len(camera_names) < 2:
        raise ValueError("At least two cameras are required.")

    if not (0 <= reference_camera_index < len(camera_names)):
        raise ValueError("reference_camera_index is out of range.")

    intrinsics = [
        load_intrinsics_json(path, camera_name)
        for path, camera_name in zip(intrinsics_paths, camera_names)
    ]

    frames = collect_synchronized_frames_by_camera_dirs(
        images_root=images_root,
        camera_names=camera_names,
        frame_key_offsets=frame_key_offsets,
        frame_cycle_length=frame_cycle_length,
    )

    detected_frames, image_sizes = detect_checkerboards_in_synchronized_frames(
        frames=frames,
        camera_names=camera_names,
        board_size=board_size,
        save_corner_images=save_corner_images,
        corner_output_folder=corner_output_folder,
    )

    single_object_points = create_checkerboard_object_points(
        board_size=board_size,
        square_size=square_size,
    )

    reference_image_size = image_sizes[reference_camera_index]

    stereo_results: list[StereoCalibrationResult] = []

    print(f"Input synchronized frames: {len(frames)}")
    print(f"Frames detected in all cameras: {len(detected_frames)}")

    for target_camera_index in range(len(camera_names)):
        if target_camera_index == reference_camera_index:
            continue

        stereo_result = calibrate_camera_pair_to_reference(
            reference_camera_index=reference_camera_index,
            target_camera_index=target_camera_index,
            intrinsics=intrinsics,
            detected_frames=detected_frames,
            single_object_points=single_object_points,
            image_size=reference_image_size,
        )
        stereo_results.append(stereo_result)

        reference_name = camera_names[reference_camera_index]
        target_name = camera_names[target_camera_index]
        translation = stereo_result.translation_reference_to_target.reshape(3)

        print(
            f"{reference_name} -> {target_name}: "
            f"RMS={stereo_result.rms_error:.6f}, "
            f"t={translation} {square_size_unit}"
        )

    all_pairwise_transforms = build_all_pairwise_transforms(
        num_cameras=len(camera_names),
        reference_camera_index=reference_camera_index,
        stereo_results=stereo_results,
    )

    result = MultiCameraExtrinsicsResult(
        camera_names=list(camera_names),
        reference_camera_index=reference_camera_index,
        stereo_results=stereo_results,
        all_pairwise_transforms=all_pairwise_transforms,
        board_size=board_size,
        square_size=square_size,
        square_size_unit=square_size_unit,
    )

    if output_json_path is not None:
        save_extrinsics_json(result, output_json_path)
        print(f"Saved extrinsics JSON: {output_json_path}")

    return result


def main() -> None:

    config = load_project_config(Path("project_config.yaml"))
    img_root = Path(config["img_root"])
    intrinsics_root_path = img_root / config["folders"]["calibration_files"]
    camera_names = config["camera_names"]

    intrinsics_paths = [
        intrinsics_root_path / f"{camera_name}{config['filenames']['calibration_result_ending']}"
        for camera_name in camera_names
    ]

    chessboard_images_root = img_root / config["folders"]["extrinsic_images"]

    result = calibrate_multicamera_extrinsics(
        intrinsics_paths=intrinsics_paths,
        images_root=chessboard_images_root,
        camera_names=camera_names,
        reference_camera_index=config["reference_camera_index"],
        board_size=tuple(config["checkerboard"]["board_size"]),
        square_size=config["checkerboard"]["square_size"],   
        square_size_unit=config["checkerboard"]["unit"],
        save_corner_images=True,
        corner_output_folder=chessboard_images_root / "extrinsics_found_patterns",
        output_json_path=img_root / config["folders"]["calibration_files"] / config["filenames"]["extrinsics_result"],
        frame_key_offsets=config.get("stereo_sync", {}).get("frame_key_offsets"),
        frame_cycle_length=config.get("stereo_sync", {}).get("frame_cycle_length"),
    )

    print("Calibration finished.")
    print(f"Reference camera: {result.camera_names[result.reference_camera_index]}")


if __name__ == "__main__":
    main()
