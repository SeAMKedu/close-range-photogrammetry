"""Run and report a configured single-object, single-frame sparse baseline."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from project_config import load_project_config
from reconstruct_sparse_point_cloud import (
    load_camera_poses_from_extrinsics,
    load_project_calibrations,
    reconstruct_object_sparse_point_cloud,
    save_frame_info_json,
    save_point_cloud_npz,
    save_point_cloud_ply,
)


def main() -> None:
    config = load_project_config(Path("project_config.yaml"))
    baseline = config["sparse_baseline"]
    root = Path(config["img_root"])
    calibration_root = root / config["folders"]["calibration_files"]
    camera_names = config["camera_names"]
    reference_index = int(config["reference_camera_index"])
    target_indices = [int(value) for value in baseline["target_camera_indices"]]
    object_name = str(baseline["object_name"])
    frame_key = str(baseline["frame_key"])
    object_root = root / config["folders"]["objects"] / object_name
    output_root = root / baseline["output_folder"] / object_name

    calibrations = load_project_calibrations(calibration_root, camera_names)
    poses = load_camera_poses_from_extrinsics(
        calibration_root / config["filenames"]["extrinsics_result"],
        camera_names,
        reference_index,
    )
    started = time.perf_counter()
    clouds = reconstruct_object_sparse_point_cloud(
        object_root=object_root,
        object_output_dir=output_root,
        camera_names=camera_names,
        calibrations=calibrations,
        poses=poses,
        reference_camera_index=reference_index,
        feature_method=str(baseline["feature_method"]),
        lowe_ratio=float(baseline["lowe_ratio"]),
        max_reprojection_error_px=float(baseline["max_reprojection_error_px"]),
        target_camera_indices=target_indices,
        frame_keys_to_use=[frame_key],
        save_debug_images=True,
    )
    if len(clouds) != 1:
        raise RuntimeError(f"Expected exactly one point cloud, got {len(clouds)}")

    cloud = clouds[0]
    elapsed = time.perf_counter() - started
    frame_root = output_root / f"frame_{frame_key}"
    save_point_cloud_npz(cloud, frame_root / "sparse_point_cloud.npz")
    save_point_cloud_ply(cloud, frame_root / "sparse_point_cloud.ply")
    save_frame_info_json(
        cloud,
        frame_root / "frame_info.json",
        camera_names[reference_index],
        [camera_names[index] for index in target_indices],
        elapsed,
    )

    points = cloud.points_xyz.astype(np.float64)
    finite = np.isfinite(points).all(axis=1)
    finite_points = points[finite]
    report = {
        "accepted_calibration": str(calibration_root / config["filenames"]["extrinsics_result"]),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "object": object_name,
        "frame_key": frame_key,
        "reference_camera": camera_names[reference_index],
        "target_cameras": [camera_names[index] for index in target_indices],
        "feature_method": baseline["feature_method"],
        "lowe_ratio": float(baseline["lowe_ratio"]),
        "max_reprojection_error_px": float(baseline["max_reprojection_error_px"]),
        "triangulated_points": int(len(points)),
        "finite_points": int(finite.sum()),
        "bbox_min_xyz_mm": finite_points.min(axis=0).tolist(),
        "bbox_max_xyz_mm": finite_points.max(axis=0).tolist(),
        "depth_percentiles_mm": {
            str(percentile): float(np.percentile(finite_points[:, 2], percentile))
            for percentile in (1, 5, 50, 95, 99)
        },
        "processing_time_s": elapsed,
    }
    (frame_root / "baseline_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
