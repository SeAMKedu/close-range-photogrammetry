"""Align the photogrammetry result to a profile-scanner height-map reference."""
from __future__ import annotations

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import csv
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from pipeline.project_config import load_project_config


def load_profile_scanner_csv(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    """Load the scanner CSV's Part height grid as finite XYZ points."""
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))

    info_index = next(index for index, row in enumerate(rows) if row and row[0] == "Info")
    info = dict(zip(rows[info_index + 1], rows[info_index + 2]))
    part_index = next(index for index, row in enumerate(rows) if row and row[0] == "Part")
    part = dict(zip(rows[part_index + 1], rows[part_index + 2]))
    row_count = int(part["Row Count"])
    column_count = int(part["Column Count"])
    grid_header_index = part_index + 3

    x_values = np.asarray(
        [float(value) for value in rows[grid_header_index][1 : column_count + 1]],
        dtype=np.float64,
    )
    y_values = np.empty(row_count, dtype=np.float64)
    z_values = np.full((row_count, column_count), np.nan, dtype=np.float64)
    for row_index, row in enumerate(
        rows[grid_header_index + 1 : grid_header_index + 1 + row_count]
    ):
        y_values[row_index] = float(row[0])
        for column_index, value in enumerate(row[1 : column_count + 1]):
            if value.strip():
                z_values[row_index, column_index] = float(value)

    yy, xx = np.meshgrid(y_values, x_values, indexing="ij")
    valid = np.isfinite(z_values)
    points = np.column_stack([xx[valid], yy[valid], z_values[valid]])
    metadata = {
        "row_count": row_count,
        "column_count": column_count,
        "valid_points": int(valid.sum()),
        "x_resolution_mm": float(info["XResolution"]),
        "y_resolution_mm": float(info["YResolution"]),
        "z_resolution_mm": float(info["ZResolution"]),
        "x_range_mm": [float(x_values.min()), float(x_values.max())],
        "y_range_mm": [float(y_values.min()), float(y_values.max())],
        "z_range_mm": [float(np.nanmin(z_values)), float(np.nanmax(z_values))],
    }
    return points, metadata


def make_cloud(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return cloud


def plane_basis(normal: np.ndarray) -> np.ndarray:
    """Build a stable right-handed basis whose first axis follows global X."""
    normal = normal.astype(np.float64) / np.linalg.norm(normal)
    axis_u = np.asarray([1.0, 0.0, 0.0])
    axis_u -= normal * float(np.dot(axis_u, normal))
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    axis_v /= np.linalg.norm(axis_v)
    return np.column_stack([axis_u, axis_v, normal])


def bbox_plane_center(points: np.ndarray, basis: np.ndarray) -> np.ndarray:
    local = points @ basis
    center_local = 0.5 * (local.min(axis=0) + local.max(axis=0))
    center_local[2] = np.median(local[:, 2])
    return basis @ center_local


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    return (transform @ homogeneous.T).T[:, :3]


def error_colors(abs_error: np.ndarray, maximum: float) -> np.ndarray:
    """Map 0..maximum error from blue through green/yellow to red."""
    value = np.clip(abs_error / maximum, 0.0, 1.0)
    colors = np.empty((len(value), 3), dtype=np.float64)
    low = value <= 0.5
    colors[low, 0] = 0.0
    colors[low, 1] = 2.0 * value[low]
    colors[low, 2] = 1.0 - 2.0 * value[low]
    colors[~low, 0] = 2.0 * value[~low] - 1.0
    colors[~low, 1] = 2.0 - 2.0 * value[~low]
    colors[~low, 2] = 0.0
    return colors


def points_to_octahedron_mesh(
    points: np.ndarray,
    colors: np.ndarray,
    radius: float,
) -> o3d.geometry.TriangleMesh:
    """Create a small colored octahedron at each point for Blender viewing."""
    offsets = radius * np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    local_triangles = np.asarray(
        [
            [4, 0, 2], [4, 2, 1], [4, 1, 3], [4, 3, 0],
            [5, 2, 0], [5, 1, 2], [5, 3, 1], [5, 0, 3],
        ],
        dtype=np.int32,
    )
    vertices = (points[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    triangles = (
        local_triangles[None, :, :]
        + (6 * np.arange(len(points), dtype=np.int32))[:, None, None]
    ).reshape(-1, 3)
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(triangles),
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.repeat(colors, 6, axis=0)
    )
    mesh.compute_vertex_normals()
    return mesh


def run_profile_comparison(
    sequence_config_key: str = "full_sequence",
    comparison_config_key: str = "profile_comparison",
) -> dict:
    config = load_project_config(Path("project_config.yaml"))
    comparison = config[comparison_config_key]
    sequence = config[sequence_config_key]
    root = Path(config["img_root"])
    object_name = str(sequence["object_name"])
    registered_root = root / sequence["output_folder"] / object_name / "registered"
    output_root = registered_root / "profile_reference_comparison"
    output_root.mkdir(parents=True, exist_ok=True)

    scanner_path = Path(comparison["reference_csv"])
    scanner_points, scanner_metadata = load_profile_scanner_csv(scanner_path)
    photo_path = registered_root / "plane_filtered_registered_point_cloud.npz"
    with np.load(photo_path) as data:
        photo_points = np.asarray(data["points_xyz"], dtype=np.float64)
        photo_colors = (
            np.asarray(data["colors_rgb"], dtype=np.float64) / 255.0
            if "colors_rgb" in data
            else None
        )

    filter_report = json.loads(
        (registered_root / "registered_filter_report.json").read_text(encoding="utf-8")
    )
    photo_normal = np.asarray(filter_report["dominant_plane_abcd"][:3], dtype=np.float64)
    photo_normal /= np.linalg.norm(photo_normal)
    if photo_normal[2] < 0:
        photo_normal *= -1

    o3d.utility.random.seed(0)
    scanner_cloud = make_cloud(scanner_points)
    scanner_plane, scanner_plane_indices = scanner_cloud.segment_plane(
        distance_threshold=float(comparison["scanner_plane_threshold_mm"]),
        ransac_n=3,
        num_iterations=int(comparison["scanner_plane_iterations"]),
    )
    scanner_normal = np.asarray(scanner_plane[:3], dtype=np.float64)
    scanner_normal /= np.linalg.norm(scanner_normal)
    if scanner_normal[2] < 0:
        scanner_normal *= -1

    photo_basis = plane_basis(photo_normal)
    scanner_basis = plane_basis(scanner_normal)
    photo_center = bbox_plane_center(photo_points, photo_basis)
    scanner_center = bbox_plane_center(scanner_points, scanner_basis)
    target_cloud = scanner_cloud.voxel_down_sample(
        float(comparison["scanner_icp_voxel_mm"])
    )
    scanner_tree_3d = cKDTree(scanner_points)
    scanner_tree_xy = cKDTree(scanner_points[:, :2])
    xy_radius = float(comparison["xy_coverage_radius_mm"])
    icp_distance = float(comparison["icp_max_distance_mm"])

    candidates = []
    best = None
    for swap_axes in (False, True):
        for sign_u in (-1, 1):
            for sign_v in (-1, 1):
                if swap_axes:
                    mapping_2d = np.asarray(
                        [[0.0, sign_u], [sign_v, 0.0]], dtype=np.float64
                    )
                else:
                    mapping_2d = np.asarray(
                        [[sign_u, 0.0], [0.0, sign_v]], dtype=np.float64
                    )

                local_rotation = np.eye(3, dtype=np.float64)
                local_rotation[:2, :2] = mapping_2d
                # Preserve a proper 3D rotation even for a mirrored 2D convention.
                local_rotation[2, 2] = float(np.linalg.det(mapping_2d))
                rotation = scanner_basis @ local_rotation @ photo_basis.T
                initial = np.eye(4, dtype=np.float64)
                initial[:3, :3] = rotation
                initial[:3, 3] = scanner_center - rotation @ photo_center

                source_cloud = make_cloud(photo_points)
                icp = o3d.pipelines.registration.registration_icp(
                    source_cloud,
                    target_cloud,
                    icp_distance,
                    initial,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
                )
                aligned = transform_points(photo_points, icp.transformation)
                nearest_3d, _ = scanner_tree_3d.query(aligned, k=1)
                nearest_xy, scanner_indices = scanner_tree_xy.query(aligned[:, :2], k=1)
                covered = nearest_xy <= xy_radius
                signed_z = aligned[:, 2] - scanner_points[scanner_indices, 2]
                score = float(
                    covered.mean() - 0.02 * np.percentile(nearest_3d, 95)
                )
                candidate = {
                    "swap_axes": swap_axes,
                    "sign_u": sign_u,
                    "sign_v": sign_v,
                    "score": score,
                    "icp_fitness": float(icp.fitness),
                    "icp_inlier_rmse_mm": float(icp.inlier_rmse),
                    "xy_coverage_percent": 100.0 * float(covered.mean()),
                    "nearest_3d_median_mm": float(np.median(nearest_3d)),
                    "nearest_3d_p95_mm": float(np.percentile(nearest_3d, 95)),
                    "abs_z_median_mm": float(np.median(np.abs(signed_z[covered]))),
                    "abs_z_p95_mm": float(np.percentile(np.abs(signed_z[covered]), 95)),
                    "transform_photo_to_scanner": icp.transformation.tolist(),
                }
                candidates.append(candidate)
                if best is None or candidate["score"] > best["candidate"]["score"]:
                    best = {
                        "candidate": candidate,
                        "aligned": aligned,
                        "covered": covered,
                        "nearest_3d": nearest_3d,
                        "signed_z": signed_z,
                    }

    assert best is not None
    candidates.sort(key=lambda item: item["score"], reverse=True)
    aligned = best["aligned"]
    covered = best["covered"]
    signed_z = best["signed_z"][covered]
    abs_z = np.abs(signed_z)
    nearest_3d = best["nearest_3d"]

    scanner_colors = np.tile([0.15, 0.55, 1.0], (len(scanner_points), 1))
    aligned_colors = np.tile([1.0, 0.25, 0.0], (len(aligned), 1))
    overlay = make_cloud(
        np.vstack([scanner_points, aligned]),
        np.vstack([scanner_colors, aligned_colors]),
    )
    o3d.io.write_point_cloud(str(output_root / "scanner_reference.ply"), make_cloud(scanner_points))
    o3d.io.write_point_cloud(
        str(output_root / "aligned_photogrammetry.ply"),
        make_cloud(aligned, photo_colors),
    )
    o3d.io.write_point_cloud(str(output_root / "alignment_overlay.ply"), overlay)

    all_signed_z = best["signed_z"]
    map_colors = error_colors(
        np.abs(all_signed_z), float(comparison["error_map_max_mm"])
    )
    map_colors[~covered] = [1.0, 0.0, 1.0]
    o3d.io.write_point_cloud(
        str(output_root / "photogrammetry_error_map.ply"),
        make_cloud(aligned, map_colors),
    )
    blender_mesh = points_to_octahedron_mesh(
        aligned,
        map_colors,
        radius=float(comparison["blender_point_radius_mm"]),
    )
    o3d.io.write_triangle_mesh(
        str(output_root / "photogrammetry_error_map_blender.ply"),
        blender_mesh,
        write_vertex_colors=True,
    )
    blender_overlay = points_to_octahedron_mesh(
        np.vstack([scanner_points, aligned]),
        np.vstack([scanner_colors, aligned_colors]),
        radius=float(comparison["blender_overlay_point_radius_mm"]),
    )
    o3d.io.write_triangle_mesh(
        str(output_root / "alignment_overlay_blender.ply"),
        blender_overlay,
        write_vertex_colors=True,
    )

    report = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "open3d": o3d.__version__,
        "object": object_name,
        "reference_csv": str(scanner_path),
        "scanner_metadata": scanner_metadata,
        "scanner_dominant_plane_abcd": [float(value) for value in scanner_plane],
        "scanner_plane_inliers": len(scanner_plane_indices),
        "photogrammetry_points": len(photo_points),
        "selected_candidate": best["candidate"],
        "candidate_score_margin": candidates[0]["score"] - candidates[1]["score"],
        "xy_coverage_percent": 100.0 * float(covered.mean()),
        "covered_points": int(covered.sum()),
        "nearest_3d_mm": {
            "median": float(np.median(nearest_3d)),
            "rmse": float(math.sqrt(np.mean(nearest_3d**2))),
            "p95": float(np.percentile(nearest_3d, 95)),
            "p99": float(np.percentile(nearest_3d, 99)),
        },
        "signed_z_error_mm_on_covered_area": {
            "mean": float(np.mean(signed_z)),
            "median": float(np.median(signed_z)),
            "standard_deviation": float(np.std(signed_z)),
            "rmse": float(math.sqrt(np.mean(signed_z**2))),
            "mae": float(np.mean(abs_z)),
            "p95_absolute": float(np.percentile(abs_z, 95)),
            "p99_absolute": float(np.percentile(abs_z, 99)),
            "max_absolute": float(abs_z.max()),
        },
        "absolute_z_error_distribution_percent": {
            f"at_or_below_{threshold:g}_mm": 100.0 * float(np.mean(abs_z <= threshold))
            for threshold in (0.25, 0.5, 1.0, 1.5)
        },
        "points_above_1_5_mm": int(np.sum(abs_z > 1.5)),
        "candidates": candidates,
        "visual_alignment_review": {
            "accepted": bool(comparison.get("alignment_accepted", False)),
            "reviewed_items": [
                "object outline and footprint",
                "cut-outs and distinctive regions",
                "systematic lateral shift or rotation",
                "spatial distribution of error-map colors",
            ],
        },
        "status": (
            "accepted_post_alignment_surface_comparison"
            if comparison.get("alignment_accepted", False)
            else "preliminary_pending_visual_alignment_review"
        ),
    }
    (output_root / "profile_comparison_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))
    return report


def main() -> None:
    run_profile_comparison()


if __name__ == "__main__":
    main()
