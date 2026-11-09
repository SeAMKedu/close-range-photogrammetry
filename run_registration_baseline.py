"""Register the configured Basler sequence pilot and summarize its quality."""
from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from project_config import load_project_config
from register_sparse_frames import (
    RegistrationSettings,
    load_camera_poses_from_extrinsics,
    load_project_calibrations,
    register_object_frames,
)


def rotation_angle_degrees(transform: np.ndarray) -> float:
    """Return the magnitude of a rigid transform's rotation."""
    trace_value = float(np.trace(transform[:3, :3]))
    cosine = np.clip((trace_value - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def run_registration(
    sequence_config_key: str = "sequence_baseline",
    registration_config_key: str = "registration_baseline",
) -> dict:
    config = load_project_config(Path("project_config.yaml"))
    sequence = config[sequence_config_key]
    registration = config[registration_config_key]
    root = Path(config["img_root"])
    camera_names = config["camera_names"]
    reference_index = int(config["reference_camera_index"])
    target_index = int(registration["stereo_target_camera_index"])
    object_name = str(sequence["object_name"])
    object_images_root = root / config["folders"]["objects"] / object_name
    object_results_root = root / sequence["output_folder"] / object_name
    calibration_root = root / config["folders"]["calibration_files"]

    settings = RegistrationSettings(
        sift_nfeatures=int(registration.get("sift_nfeatures", 8000)),
        stereo_lowe_ratio=float(registration.get("stereo_lowe_ratio", 0.75)),
        temporal_lowe_ratio=float(registration.get("temporal_lowe_ratio", 0.70)),
        max_reprojection_error_px=float(
            registration.get("max_reprojection_error_px", 2.0)
        ),
        ransac_iterations=int(registration.get("ransac_iterations", 2000)),
        ransac_inlier_threshold=float(
            registration.get("ransac_inlier_threshold", 3.0)
        ),
        ransac_min_inliers=int(registration.get("ransac_min_inliers", 6)),
        icp_max_correspondence_distance=float(
            registration.get("icp_max_correspondence_distance", 5.0)
        ),
        icp_max_iterations=int(registration.get("icp_max_iterations", 100)),
        merged_voxel_size=registration.get("merged_voxel_size"),
        save_debug_matches=bool(registration.get("save_debug_matches", True)),
        save_debug_clouds=bool(registration.get("save_debug_clouds", True)),
        max_debug_matches=int(registration.get("max_debug_matches", 100)),
    )

    calibrations = load_project_calibrations(calibration_root, camera_names)
    poses = load_camera_poses_from_extrinsics(
        calibration_root / config["filenames"]["extrinsics_result"],
        camera_names,
        reference_index,
    )

    started = time.perf_counter()
    register_object_frames(
        object_images_root=object_images_root,
        object_results_root=object_results_root,
        camera_names=camera_names,
        reference_camera_index=reference_index,
        stereo_target_camera_index=target_index,
        calibrations=calibrations,
        poses=poses,
        settings=settings,
    )
    elapsed = time.perf_counter() - started

    registered_root = object_results_root / "registered"
    registration_report = json.loads(
        (registered_root / "registration_report.json").read_text(encoding="utf-8")
    )
    rows = []
    failed = []
    for frame_key, data in registration_report.items():
        if data["status"] == "failed":
            failed.append(frame_key)
            rows.append({"frame_key": frame_key, "status": "failed", "error": data["error"]})
            continue
        if data["status"] == "reference":
            rows.append({"frame_key": frame_key, "status": "reference"})
            continue

        pairwise = np.asarray(data["pairwise_transform"], dtype=np.float64)
        rows.append(
            {
                "frame_key": frame_key,
                "status": "registered",
                "registered_to_frame": data["registered_to_frame"],
                "temporal_3d_correspondences": data["temporal_3d_correspondences"],
                "ransac_inliers": data["ransac_inliers"],
                "ransac_inlier_percent": (
                    100.0 * data["ransac_inliers"] / data["temporal_3d_correspondences"]
                ),
                "ransac_rmse_mm": data["ransac_rmse"],
                "icp_fitness": data["icp_fitness"],
                "icp_inlier_rmse_mm": data["icp_inlier_rmse"],
                "pairwise_translation_mm": float(np.linalg.norm(pairwise[:3, 3])),
                "pairwise_rotation_deg": rotation_angle_degrees(pairwise),
            }
        )

    registered_rows = [row for row in rows if row["status"] == "registered"]
    review = [
        row["frame_key"]
        for row in registered_rows
        if row["ransac_inliers"] < int(registration.get("review_min_inliers", 10))
        or row["icp_fitness"] < float(registration.get("review_min_icp_fitness", 0.5))
        or row["icp_inlier_rmse_mm"]
        > float(registration.get("review_max_icp_rmse_mm", 2.0))
    ]
    merged_cloud = o3d.io.read_point_cloud(
        str(registered_root / "registered_sparse_point_cloud.ply")
    )
    merged_points = np.asarray(merged_cloud.points, dtype=np.float64)
    transforms = json.loads(
        (registered_root / "frame_transforms.json").read_text(encoding="utf-8")
    )
    aligned_centroids = []
    for frame_key, transform_values in transforms.items():
        with np.load(
            object_results_root / f"frame_{frame_key}" / "filtered_point_cloud.npz"
        ) as data:
            points = np.asarray(data["points_xyz"], dtype=np.float64)
        transform = np.asarray(transform_values, dtype=np.float64)
        aligned = (
            transform
            @ np.column_stack([points, np.ones(len(points), dtype=np.float64)]).T
        ).T[:, :3]
        aligned_centroids.append(aligned.mean(axis=0))

    centroid_span = np.ptp(np.asarray(aligned_centroids), axis=0)
    summary = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "sequence_config": sequence_config_key,
        "registration_config": registration_config_key,
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "open3d": o3d.__version__,
        "object": object_name,
        "reference_camera": camera_names[reference_index],
        "stereo_target_camera": camera_names[target_index],
        "reference_frame": rows[0]["frame_key"],
        "registered_frames": 1 + len(registered_rows),
        "failed_frame_keys": failed,
        "review_recommended_frame_keys": review,
        "merged_points": len(merged_points),
        "merged_bbox_min_xyz_mm": merged_points.min(axis=0).tolist(),
        "merged_bbox_max_xyz_mm": merged_points.max(axis=0).tolist(),
        "merged_bbox_span_xyz_mm": np.ptp(merged_points, axis=0).tolist(),
        "aligned_frame_centroid_span_xyz_mm": centroid_span.tolist(),
        "processing_time_s": elapsed,
        "settings": registration,
        "frames": rows,
    }
    (registered_root / "registration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "frames"}, indent=2))
    return summary


def main() -> None:
    run_registration()


if __name__ == "__main__":
    main()
