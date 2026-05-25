from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from project_config import load_project_config

import cv2
import numpy as np

from chessboard_utils import (
    create_checkerboard_object_points,
    draw_corner_preview,
    find_checkerboard_corners,
    list_image_files,
    save_corner_visualization,
)


@dataclass
class CameraCalibrationResult:
    """Container for single-camera calibration results."""

    rms_error: float
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvecs: tuple[np.ndarray, ...]
    tvecs: tuple[np.ndarray, ...]
    new_camera_matrix: np.ndarray
    roi: tuple[int, int, int, int]
    mean_reprojection_error: float
    used_images: list[Path]
    failed_images: list[Path]
    image_size: tuple[int, int]

    def to_json_dict(self) -> dict:
        """Convert calibration result to a JSON-serializable dictionary."""
        return {
            "rms_error": float(self.rms_error),
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "new_camera_matrix": self.new_camera_matrix.tolist(),
            "roi": list(self.roi),
            "mean_reprojection_error": float(self.mean_reprojection_error),
            "used_images": [str(path) for path in self.used_images],
            "failed_images": [str(path) for path in self.failed_images],
            "image_size": list(self.image_size),
        }


def calculate_mean_reprojection_error(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: tuple[np.ndarray, ...],
    tvecs: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    """Calculate mean reprojection error over all calibration images."""
    errors = []

    for objp, imgp, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected_points, _ = cv2.projectPoints(
            objp,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )

        error = cv2.norm(imgp, projected_points, cv2.NORM_L2) / len(projected_points)
        errors.append(error)

    return float(np.mean(errors))


def save_calibration_json(
    result: CameraCalibrationResult,
    output_path: Path | str,
) -> None:
    """Save camera calibration result as a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result.to_json_dict(), file, indent=2)


def calibrate_camera_from_chessboard_images(
    image_folder: Path | str,
    board_size: tuple[int, int],
    square_size: float,
    show_corners: bool = False,
    save_corner_images: bool = False,
    corner_output_folder: Path | str | None = None,
    save_json_path: Path | str | None = None,
) -> CameraCalibrationResult:
    """Calibrate one camera using checkerboard images from a folder."""
    image_folder = Path(image_folder)

    if corner_output_folder is None:
        corner_output_folder = image_folder / "found_patterns"
    else:
        corner_output_folder = Path(corner_output_folder)

    image_paths = list_image_files(image_folder)

    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in {image_folder}")

    single_object_points = create_checkerboard_object_points(
        board_size=board_size,
        square_size=square_size,
    )

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_images: list[Path] = []
    failed_images: list[Path] = []

    image_size: tuple[int, int] | None = None

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            print(f"[WARN] Could not read image: {image_path}")
            failed_images.append(image_path)
            continue

        current_image_size = (image.shape[1], image.shape[0])

        if image_size is None:
            image_size = current_image_size
        elif image_size != current_image_size:
            raise ValueError(
                f"Image size mismatch. Expected {image_size}, "
                f"got {current_image_size} for image {image_path}"
            )

        found, corners = find_checkerboard_corners(
            image=image,
            board_size=board_size,
        )

        if not found or corners is None:
            print(f"[WARN] Checkerboard corners not found: {image_path}")
            failed_images.append(image_path)
            continue

        object_points.append(single_object_points.copy())
        image_points.append(corners)
        used_images.append(image_path)

        if show_corners:
            preview = draw_corner_preview(
                image=image,
                board_size=board_size,
                corners=corners,
            )
            cv2.imshow("Detected checkerboard corners", preview)
            cv2.waitKey(300)

        if save_corner_images:
            output_path = Path(corner_output_folder) / image_path.name
            save_corner_visualization(
                image=image,
                board_size=board_size,
                corners=corners,
                output_path=output_path,
            )

    if show_corners:
        cv2.destroyAllWindows()

    if image_size is None:
        raise RuntimeError("No readable images were found.")

    if len(object_points) < 5:
        raise RuntimeError(
            f"Only {len(object_points)} usable calibration images found. "
            "Use more successful checkerboard detections for reliable calibration."
        )

    rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        alpha=1,
        newImgSize=image_size,
    )

    mean_reprojection_error = calculate_mean_reprojection_error(
        object_points=object_points,
        image_points=image_points,
        rvecs=tuple(rvecs),
        tvecs=tuple(tvecs),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )

    result = CameraCalibrationResult(
        rms_error=float(rms_error),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=tuple(rvecs),
        tvecs=tuple(tvecs),
        new_camera_matrix=new_camera_matrix,
        roi=tuple(int(value) for value in roi),
        mean_reprojection_error=mean_reprojection_error,
        used_images=used_images,
        failed_images=failed_images,
        image_size=image_size,
    )

    print(f"Used images: {len(used_images)}")
    print(f"Failed images: {len(failed_images)}")
    print(f"OpenCV RMS error: {result.rms_error:.6f}")
    print(f"Mean reprojection error: {result.mean_reprojection_error:.6f} px")

    if save_json_path is not None:
        save_json_path.parents[0].mkdir(parents=True, exist_ok=True)
        save_calibration_json(result, save_json_path)
        print(f"Saved calibration JSON: {save_json_path}")

    return result

def main():
    config = load_project_config(Path("project_config.yaml"))
    img_root = Path(config["img_root"])
    chessboard_path = img_root / config["folders"]["intrinsic_images"]
    camera_names = config["camera_names"]
    for cam_name in camera_names:
        calibrate_camera_from_chessboard_images(
            image_folder=chessboard_path / cam_name,
            board_size=tuple(config["checkerboard"]["board_size"]),
            square_size=config["checkerboard"]["square_size"],    
            show_corners=True,
            save_corner_images=True,
            save_json_path=img_root / config["folders"]["calibration_files"] / f"{cam_name}{config['filenames']['calibration_result_ending']}",
        )

if __name__ == "__main__":
    main()
