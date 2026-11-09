"""Filter and report the configured single-frame sparse baseline."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from filter_point_cloud import (
    FilteringSettings,
    filter_point_cloud,
    load_point_cloud_npz,
    save_point_cloud_npz,
    save_point_cloud_ply,
    update_frame_info_json,
)
from project_config import load_project_config


def cloud_summary(points: np.ndarray) -> dict:
    """Return compact geometric statistics for a point array."""
    finite_points = points[np.isfinite(points).all(axis=1)]
    if len(finite_points) == 0:
        return {"points": int(len(points)), "finite_points": 0}

    return {
        "points": int(len(points)),
        "finite_points": int(len(finite_points)),
        "bbox_min_xyz_mm": finite_points.min(axis=0).tolist(),
        "bbox_max_xyz_mm": finite_points.max(axis=0).tolist(),
        "depth_percentiles_mm": {
            str(percentile): float(np.percentile(finite_points[:, 2], percentile))
            for percentile in (1, 5, 50, 95, 99)
        },
    }


def main() -> None:
    config = load_project_config(Path("project_config.yaml"))
    baseline = config["sparse_baseline"]
    filtering = config["filter_baseline"]
    root = Path(config["img_root"])
    object_name = str(baseline["object_name"])
    frame_key = str(baseline["frame_key"])
    frame_root = root / baseline["output_folder"] / object_name / f"frame_{frame_key}"
    input_path = frame_root / "sparse_point_cloud.npz"

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

    original = load_point_cloud_npz(input_path)
    started = time.perf_counter()
    filtered = filter_point_cloud(original, settings)
    elapsed = time.perf_counter() - started

    npz_output = frame_root / "filtered_point_cloud.npz"
    ply_output = frame_root / "filtered_point_cloud.ply"
    save_point_cloud_npz(filtered, npz_output)
    save_point_cloud_ply(filtered, ply_output)
    update_frame_info_json(
        frame_root / "frame_info.json",
        filtered_points=int(len(filtered.points_xyz)),
        filtering_report="filter_report.json",
    )

    report = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "open3d": o3d.__version__,
        "object": object_name,
        "frame_key": frame_key,
        "input": str(input_path),
        "settings": {
            "remove_non_finite": settings.remove_non_finite,
            "min_depth": settings.min_depth,
            "max_depth": settings.max_depth,
            "voxel_size": settings.voxel_size,
            "statistical_nb_neighbors": settings.statistical_nb_neighbors,
            "statistical_std_ratio": settings.statistical_std_ratio,
            "radius_outlier_nb_points": settings.radius_outlier_nb_points,
            "radius_outlier_radius": settings.radius_outlier_radius,
        },
        "before": cloud_summary(original.points_xyz),
        "after": cloud_summary(filtered.points_xyz),
        "removed_points": int(len(original.points_xyz) - len(filtered.points_xyz)),
        "processing_time_s": elapsed,
    }
    (frame_root / "filter_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
