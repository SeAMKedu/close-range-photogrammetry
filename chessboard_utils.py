from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def list_image_files(
    image_folder: Path | str,
    extensions: Iterable[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
) -> list[Path]:
    """Return supported image files from a folder."""
    image_folder = Path(image_folder)
    allowed = {ext.lower() for ext in extensions}

    return sorted(
        path
        for path in image_folder.iterdir()
        if path.is_file() and path.suffix.lower() in allowed
    )


def create_checkerboard_object_points(
    board_size: tuple[int, int],
    square_size: float,
) -> np.ndarray:
    """Create 3D checkerboard corner coordinates."""
    cols, rows = board_size

    object_points = np.zeros((cols * rows, 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_points *= float(square_size)

    return object_points


def find_checkerboard_corners(
    image: np.ndarray,
    board_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    """Find and refine checkerboard corners in one image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )

    found, corners = cv2.findChessboardCorners(gray, board_size, flags)

    if not found:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        1e-4,
    )

    refined_corners = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria,
    )

    return True, refined_corners


def save_corner_visualization(
    image: np.ndarray,
    board_size: tuple[int, int],
    corners: np.ndarray,
    output_path: Path | str,
) -> None:
    """Save an image with detected checkerboard corners drawn."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    visualization = image.copy()
    cv2.drawChessboardCorners(visualization, board_size, corners, True)

    cv2.imwrite(str(output_path), visualization)


def draw_corner_preview(
    image: np.ndarray,
    board_size: tuple[int, int],
    corners: np.ndarray,
) -> np.ndarray:
    """Return a copy of image with detected checkerboard corners drawn."""
    preview = image.copy()
    cv2.drawChessboardCorners(preview, board_size, corners, True)
    preview = cv2.resize(preview, (800, 600))
    return preview