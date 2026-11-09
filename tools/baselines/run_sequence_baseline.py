"""Reconstruct, filter and summarize a configured frame sequence."""
from __future__ import annotations

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.paths import PROJECT_ROOT


import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from pipeline.filter_point_cloud import (
    FilteringSettings,
    filter_point_cloud,
    load_point_cloud_npz,
    save_point_cloud_npz as save_filtered_npz,
    save_point_cloud_ply as save_filtered_ply,
    update_frame_info_json,
)
from pipeline.project_config import load_project_config
from pipeline.reconstruct_sparse_point_cloud import (
    load_camera_poses_from_extrinsics,
    load_project_calibrations,
    reconstruct_object_sparse_point_cloud,
    save_frame_info_json,
    save_point_cloud_npz,
    save_point_cloud_ply,
)


def point_summary(points: np.ndarray) -> dict[str, object]:
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) == 0:
        return {"points": int(len(points)), "finite_points": 0}
    return {
        "points": int(len(points)),
        "finite_points": int(len(finite)),
        "depth_min_mm": float(finite[:, 2].min()),
        "depth_median_mm": float(np.median(finite[:, 2])),
        "depth_max_mm": float(finite[:, 2].max()),
        "depth_span_mm": float(np.ptp(finite[:, 2])),
        "bbox_min_xyz_mm": finite.min(axis=0).tolist(),
        "bbox_max_xyz_mm": finite.max(axis=0).tolist(),
    }


def run_sequence(sequence_config_key: str = "sequence_baseline") -> dict:
    config = load_project_config(PROJECT_ROOT / "project_config.yaml")
    sequence = config[sequence_config_key]
    sparse = config["sparse_baseline"]
    filtering = config["filter_baseline"]
    root = Path(config["img_root"])
    calibration_root = root / config["folders"]["calibration_files"]
    object_name = str(sequence["object_name"])
    camera_names = config["camera_names"]
    reference_index = int(config["reference_camera_index"])
    target_indices = [int(value) for value in sequence["target_camera_indices"]]
    output_root = root / sequence["output_folder"] / object_name
    object_root = root / config["folders"]["objects"] / object_name
    configured_frame_keys = sequence["frame_keys"]
    if isinstance(configured_frame_keys, str) and configured_frame_keys.lower() == "all":
        reference_dir = object_root / camera_names[reference_index]
        target_dir = object_root / camera_names[target_indices[0]]
        reference_keys = {path.stem.split("_", 1)[0] for path in reference_dir.iterdir() if path.is_file()}
        target_keys = {path.stem.split("_", 1)[0] for path in target_dir.iterdir() if path.is_file()}
        frame_keys = sorted(reference_keys & target_keys, key=lambda value: int(value))
    else:
        frame_keys = [str(value) for value in configured_frame_keys]

    settings = FilteringSettings(
        remove_non_finite=bool(filtering.get("remove_non_finite", True)),
        min_depth=filtering.get("min_depth"),
        max_depth=filtering.get("max_depth"),
        voxel_size=filtering.get("voxel_size"),
        statistical_nb_neighbors=int(filtering.get("statistical_nb_neighbors", 0)),
        statistical_std_ratio=float(filtering.get("statistical_std_ratio", 2.0)),
        radius_outlier_nb_points=filtering.get("radius_outlier_nb_points"),
        radius_outlier_radius=filtering.get("radius_outlier_radius"),
    )

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
        feature_method=str(sparse["feature_method"]),
        lowe_ratio=float(sparse["lowe_ratio"]),
        max_reprojection_error_px=float(sparse["max_reprojection_error_px"]),
        target_camera_indices=target_indices,
        frame_keys_to_use=frame_keys,
        save_debug_images=bool(sequence.get("save_debug_images", True)),
    )
    reconstruction_elapsed = time.perf_counter() - started

    rows: list[dict[str, object]] = []
    per_frame_reconstruction_time = reconstruction_elapsed / len(clouds)
    for cloud in clouds:
        frame_root = output_root / f"frame_{cloud.source_frame_key}"
        sparse_npz = frame_root / "sparse_point_cloud.npz"
        save_point_cloud_npz(cloud, sparse_npz)
        save_point_cloud_ply(cloud, frame_root / "sparse_point_cloud.ply")
        save_frame_info_json(
            cloud,
            frame_root / "frame_info.json",
            camera_names[reference_index],
            [camera_names[index] for index in target_indices],
            per_frame_reconstruction_time,
        )

        original = load_point_cloud_npz(sparse_npz)
        filter_started = time.perf_counter()
        filtered = filter_point_cloud(original, settings)
        filter_elapsed = time.perf_counter() - filter_started
        save_filtered_npz(filtered, frame_root / "filtered_point_cloud.npz")
        save_filtered_ply(filtered, frame_root / "filtered_point_cloud.ply")

        before = point_summary(original.points_xyz)
        after = point_summary(filtered.points_xyz)
        matches = sum(int(item["feature_matches"]) for item in cloud.pair_statistics)
        row = {
            "frame_key": cloud.source_frame_key,
            "feature_matches": matches,
            "triangulated_points": int(len(original.points_xyz)),
            "filtered_points": int(len(filtered.points_xyz)),
            "removed_points": int(len(original.points_xyz) - len(filtered.points_xyz)),
            "retained_percent": 100.0 * len(filtered.points_xyz) / len(original.points_xyz),
            "depth_min_mm": after.get("depth_min_mm"),
            "depth_median_mm": after.get("depth_median_mm"),
            "depth_max_mm": after.get("depth_max_mm"),
            "depth_span_mm": after.get("depth_span_mm"),
            "filtering_time_s": filter_elapsed,
            "pair_statistics": cloud.pair_statistics,
            "before": before,
            "after": after,
        }
        rows.append(row)
        update_frame_info_json(
            frame_root / "frame_info.json",
            feature_matches=matches,
            filtered_points=int(len(filtered.points_xyz)),
            removed_points=row["removed_points"],
            retained_percent=row["retained_percent"],
            filtered_depth_span_mm=row["depth_span_mm"],
        )

    produced = {str(row["frame_key"]) for row in rows}
    missing = [key for key in frame_keys if key not in produced]
    median_filtered_points = float(
        np.median([int(row["filtered_points"]) for row in rows])
    )
    median_depth_span = float(
        np.median([float(row["depth_span_mm"]) for row in rows])
    )
    for row in rows:
        review_reasons = []
        if int(row["filtered_points"]) < 0.8 * median_filtered_points:
            review_reasons.append("low_point_count")
        if float(row["depth_span_mm"]) > 2.0 * median_depth_span:
            review_reasons.append("large_depth_span")
        row["review_reasons"] = ";".join(review_reasons)

    ranked = sorted(rows, key=lambda row: int(row["filtered_points"]))
    review_keys = [
        str(row["frame_key"])
        for row in rows
        if row["review_reasons"]
    ]
    total_elapsed = time.perf_counter() - started
    report = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=PROJECT_ROOT).strip(),
        "sequence_config": sequence_config_key,
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "open3d": o3d.__version__,
        "object": object_name,
        "reference_camera": camera_names[reference_index],
        "target_cameras": [camera_names[index] for index in target_indices],
        "requested_frame_keys": frame_keys,
        "produced_frames": len(rows),
        "missing_or_failed_frame_keys": missing,
        "review_recommended_frame_keys": review_keys,
        "review_thresholds": {
            "low_point_count_below": 0.8 * median_filtered_points,
            "large_depth_span_above_mm": 2.0 * median_depth_span,
        },
        "feature_method": sparse["feature_method"],
        "lowe_ratio": float(sparse["lowe_ratio"]),
        "max_reprojection_error_px": float(sparse["max_reprojection_error_px"]),
        "filter_settings": {
            "statistical_nb_neighbors": settings.statistical_nb_neighbors,
            "statistical_std_ratio": settings.statistical_std_ratio,
            "radius_filter_enabled": settings.radius_outlier_nb_points is not None,
        },
        "total_feature_matches": sum(int(row["feature_matches"]) for row in rows),
        "total_triangulated_points": sum(int(row["triangulated_points"]) for row in rows),
        "total_filtered_points": sum(int(row["filtered_points"]) for row in rows),
        "weakest_frame_by_filtered_points": ranked[0]["frame_key"] if ranked else None,
        "strongest_frame_by_filtered_points": ranked[-1]["frame_key"] if ranked else None,
        "processing_time_s": total_elapsed,
        "frames": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "sequence_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    csv_fields = [
        "frame_key", "feature_matches", "triangulated_points", "filtered_points",
        "removed_points", "retained_percent", "depth_min_mm", "depth_median_mm",
        "depth_max_mm", "depth_span_mm", "filtering_time_s",
        "review_reasons",
    ]
    with (output_root / "sequence_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({key: value for key, value in report.items() if key != "frames"}, indent=2))
    return report


def main() -> None:
    run_sequence()


if __name__ == "__main__":
    main()
