"""Apply and report dominant-plane filtering to a registered sequence cloud."""
from __future__ import annotations

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.paths import PROJECT_ROOT


import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d

from pipeline.filter_registered_point_cloud import (
    filter_by_dominant_plane,
    load_registered_point_cloud_npz,
    remove_non_finite_points,
    save_point_cloud_npz,
    save_point_cloud_ply,
)
from pipeline.project_config import load_project_config


def run_registered_filter(sequence_config_key: str = "sequence_baseline") -> dict:
    config = load_project_config(PROJECT_ROOT / "project_config.yaml")
    sequence = config[sequence_config_key]
    filtering = config["registered_filter"]
    root = Path(config["img_root"])
    object_name = str(sequence["object_name"])
    registered_root = root / sequence["output_folder"] / object_name / "registered"
    input_path = registered_root / "registered_sparse_point_cloud.npz"

    original = load_registered_point_cloud_npz(input_path)
    finite = remove_non_finite_points(original)
    threshold = float(filtering["dominant_plane_distance_mm"])
    filtered, plane, residuals = filter_by_dominant_plane(
        finite,
        distance_threshold=threshold,
        iterations=int(filtering.get("ransac_iterations", 5000)),
    )

    output_stem = "plane_filtered_registered_point_cloud"
    save_point_cloud_npz(filtered, registered_root / f"{output_stem}.npz")
    save_point_cloud_ply(filtered, registered_root / f"{output_stem}.ply")
    kept_residuals = residuals[residuals <= threshold]
    report = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=PROJECT_ROOT).strip(),
        "python": platform.python_version(),
        "open3d": o3d.__version__,
        "sequence_config": sequence_config_key,
        "object": object_name,
        "input": str(input_path),
        "dominant_plane_abcd": plane.tolist(),
        "distance_threshold_mm": threshold,
        "original_points": int(len(original.points_xyz)),
        "finite_points": int(len(finite.points_xyz)),
        "filtered_points": int(len(filtered.points_xyz)),
        "removed_points": int(len(finite.points_xyz) - len(filtered.points_xyz)),
        "retained_percent": 100.0 * len(filtered.points_xyz) / len(finite.points_xyz),
        "residual_mm_before": {
            "median": float(np.median(residuals)),
            "p95": float(np.percentile(residuals, 95)),
            "p99": float(np.percentile(residuals, 99)),
            "max": float(residuals.max()),
        },
        "residual_mm_after": {
            "median": float(np.median(kept_residuals)),
            "p95": float(np.percentile(kept_residuals, 95)),
            "p99": float(np.percentile(kept_residuals, 99)),
            "max": float(kept_residuals.max()),
        },
    }
    (registered_root / "registered_filter_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    run_registered_filter()


if __name__ == "__main__":
    main()
