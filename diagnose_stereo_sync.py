"""Diagnose frame synchronization by clustering board-derived camera transforms."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from calibrate_stereo_with_chessboards import (
    CameraIntrinsics,
    DetectedFrame,
    calibrate_camera_pair_to_reference,
    load_intrinsics_json,
    natural_sort_key,
)
from chessboard_utils import create_checkerboard_object_points, find_checkerboard_corners
from project_config import load_project_config


@dataclass
class BoardDetection:
    frame_key: str
    points: np.ndarray
    rotations: tuple[np.ndarray, np.ndarray]
    translations: tuple[np.ndarray, np.ndarray]


def detect_camera_frames(
    folder: Path,
    board_size: tuple[int, int],
    object_points: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> list[BoardDetection]:
    detections = []
    for path in sorted(folder.glob("*.jpg"), key=lambda p: natural_sort_key(p.name)):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        found, corners = find_checkerboard_corners(image, board_size)
        if not found or corners is None:
            continue
        points = corners.reshape(-1, 2).astype(np.float32)
        rotations = []
        translations = []
        for candidate in (points, points[::-1].copy()):
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                candidate,
                intrinsics.camera_matrix,
                intrinsics.dist_coeffs,
            )
            if not ok:
                raise RuntimeError(f"solvePnP failed for {path}")
            rotations.append(cv2.Rodrigues(rvec)[0])
            translations.append(tvec.reshape(3))
        detections.append(
            BoardDetection(
                frame_key=path.stem.split("_", maxsplit=1)[0],
                points=points,
                rotations=tuple(rotations),
                translations=tuple(translations),
            )
        )
    return detections


def camera_transform(
    reference: BoardDetection,
    target: BoardDetection,
    target_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = target.rotations[target_order] @ reference.rotations[0].T
    translation = target.translations[target_order] - rotation @ reference.translations[0]
    return rotation, translation


def rotation_difference_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first @ second.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def transform_residual(
    candidate: tuple[np.ndarray, np.ndarray],
    seed: tuple[np.ndarray, np.ndarray],
) -> tuple[float, float, float]:
    rotation_deg = rotation_difference_degrees(candidate[0], seed[0])
    translation_mm = float(np.linalg.norm(candidate[1] - seed[1]))
    return rotation_deg + translation_mm / 5.0, rotation_deg, translation_mm


def find_best_assignment(
    references: list[BoardDetection],
    targets: list[BoardDetection],
) -> tuple[list[dict], dict]:
    transforms = [
        [
            [camera_transform(reference, target, order) for order in (0, 1)]
            for target in targets
        ]
        for reference in references
    ]
    best = None
    for seed_row in transforms:
        for seed_cell in seed_row:
            for seed in seed_cell:
                cost = np.empty((len(references), len(targets)), dtype=np.float64)
                selected_order = np.zeros_like(cost, dtype=np.int8)
                for row in range(len(references)):
                    for col in range(len(targets)):
                        residuals = [transform_residual(value, seed)[0] for value in transforms[row][col]]
                        order = int(np.argmin(residuals))
                        cost[row, col] = residuals[order]
                        selected_order[row, col] = order
                rows, cols = linear_sum_assignment(cost)
                score = float(np.median(cost[rows, cols]))
                if best is None or score < best[0]:
                    best = (score, rows, cols, selected_order, transforms)

    _, rows, cols, selected_order, transforms = best
    assigned = []
    for row, col in zip(rows, cols):
        order = int(selected_order[row, col])
        assigned.append((row, col, order, transforms[row][col][order]))

    medoid = min(
        (item[3] for item in assigned),
        key=lambda seed: np.median([transform_residual(item[3], seed)[0] for item in assigned]),
    )
    mapping = []
    rotation_residuals = []
    translation_residuals = []
    for row, col, order, transform in assigned:
        _, rotation_deg, translation_mm = transform_residual(transform, medoid)
        inlier = rotation_deg < 2.0 and translation_mm < 5.0
        rotation_residuals.append(rotation_deg)
        translation_residuals.append(translation_mm)
        mapping.append(
            {
                "reference_frame": references[row].frame_key,
                "target_frame": targets[col].frame_key,
                "target_corners_reversed": bool(order),
                "rotation_residual_deg": rotation_deg,
                "translation_residual_mm": translation_mm,
                "inlier": inlier,
                "reference_index": int(row),
                "target_index": int(col),
            }
        )
    summary = {
        "assigned_pairs": len(mapping),
        "inlier_pairs": sum(item["inlier"] for item in mapping),
        "median_rotation_residual_deg": float(np.median(rotation_residuals)),
        "median_translation_residual_mm": float(np.median(translation_residuals)),
        "max_translation_residual_mm": float(np.max(translation_residuals)),
    }
    return mapping, summary


def stereo_from_mapping(
    mapping: list[dict],
    references: list[BoardDetection],
    targets: list[BoardDetection],
    intrinsics: list[CameraIntrinsics],
    target_index: int,
    object_points: np.ndarray,
) -> float | None:
    frames = []
    for item in mapping:
        if not item["inlier"]:
            continue
        reference = references[item["reference_index"]]
        target = targets[item["target_index"]]
        target_points = target.points[::-1].copy() if item["target_corners_reversed"] else target.points
        frames.append(
            DetectedFrame(
                frame_name=f"{reference.frame_key}->{target.frame_key}",
                image_points_by_camera={0: reference.points, target_index: target_points},
            )
        )
    if len(frames) < 3:
        return None
    result = calibrate_camera_pair_to_reference(
        0, target_index, intrinsics, frames, object_points, intrinsics[0].image_size
    )
    return result.rms_error


def diagnose_set(set_name: str, root: Path, config: dict) -> dict:
    names = config["camera_names"]
    output = root / config["folders"]["calibration_files"]
    intrinsics = [load_intrinsics_json(output / f"{name}_calibration.json", name) for name in names]
    board_size = tuple(config["checkerboard"]["board_size"])
    object_points = create_checkerboard_object_points(board_size, config["checkerboard"]["square_size"])
    set_root = root / "chessboards" / set_name
    detections = [
        detect_camera_frames(set_root / name, board_size, object_points, calibration)
        for name, calibration in zip(names, intrinsics)
    ]
    report = {"detected_frames": {name: len(items) for name, items in zip(names, detections)}, "pairs": {}}
    for target_index in (1, 2):
        mapping, summary = find_best_assignment(detections[0], detections[target_index])
        rms = stereo_from_mapping(mapping, detections[0], detections[target_index], intrinsics, target_index, object_points)
        report["pairs"][names[target_index]] = {
            **summary,
            "mapped_stereo_rms_px": rms,
            "mapping": mapping,
        }
        print(set_name, names[target_index], summary, "RMS", rms, flush=True)
    return report


def main() -> None:
    config = load_project_config(Path("project_config.yaml"))
    root = Path(config["img_root"])
    report = {
        name: diagnose_set(name, root, config)
        for name in ("extrinsics_img_sets", "extrinsics_img_sets_NEW")
    }
    output_path = root / config["folders"]["calibration_files"] / "stereo_sync_diagnostic.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
