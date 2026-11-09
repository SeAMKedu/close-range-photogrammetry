from __future__ import annotations

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.paths import PROJECT_ROOT


import json
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:
    raise ImportError(
        "This script requires Open3D. Install it with: pip install open3d"
    ) from exc

from pipeline.project_config import load_project_config, make_run_suffix


def load_frame_transforms(
    path: Path | str,
) -> dict[str, np.ndarray]:
    """Load frame-to-reference transforms from frame_transforms.json."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    transforms: dict[str, np.ndarray] = {}

    for frame_key, matrix in data.items():
        transform = np.asarray(matrix, dtype=np.float64)

        if transform.shape != (4, 4):
            raise ValueError(
                f"Transform for frame {frame_key} has shape "
                f"{transform.shape}, expected (4, 4)."
            )

        transforms[str(frame_key)] = transform

    return transforms


def load_visual_hull_points(
    path: Path | str,
    use_surface_voxels: bool = True,
) -> np.ndarray:
    """Load visual-hull surface or occupied voxel centers from NPZ."""
    path = Path(path)

    with np.load(path, allow_pickle=True) as data:
        key = (
            "surface_voxel_centers"
            if use_surface_voxels
            else "occupied_voxel_centers"
        )

        if key not in data:
            raise KeyError(
                f"{key} not found in {path}. "
                f"Available arrays: {list(data.keys())}"
            )

        points = np.asarray(data[key], dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Unexpected point array shape in {path}: {points.shape}"
        )

    return points


def transform_points(
    points_xyz: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """Apply a homogeneous 4x4 transform to XYZ points."""
    if len(points_xyz) == 0:
        return points_xyz.copy()

    homogeneous = np.column_stack(
        [
            points_xyz,
            np.ones(len(points_xyz), dtype=np.float64),
        ]
    )

    return (transform @ homogeneous.T).T[:, :3]


def calculate_source_to_target_transform(
    source_to_reference: np.ndarray,
    target_to_reference: np.ndarray,
) -> np.ndarray:
    """
    Calculate a transform from source-frame coordinates to target-frame coordinates.

    frame_transforms.json stores:
        p_reference = T_frame_to_reference @ p_frame

    Therefore:
        p_target =
            inv(T_target_to_reference)
            @ T_source_to_reference
            @ p_source
    """
    return (
        np.linalg.inv(target_to_reference)
        @ source_to_reference
    )


def points_to_open3d_cloud(
    points_xyz: np.ndarray,
    color_rgb: tuple[float, float, float] | None = None,
) -> o3d.geometry.PointCloud:
    """Convert XYZ points to an Open3D point cloud."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        points_xyz.astype(np.float64)
    )

    if color_rgb is not None:
        colors = np.tile(
            np.asarray(color_rgb, dtype=np.float64),
            (len(points_xyz), 1),
        )
        cloud.colors = o3d.utility.Vector3dVector(colors)

    return cloud



def nearest_neighbor_distances(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    """Calculate nearest-neighbour distances from source points to target points."""
    if len(source_points) == 0 or len(target_points) == 0:
        return np.empty(0, dtype=np.float64)

    target_cloud = points_to_open3d_cloud(target_points)
    kdtree = o3d.geometry.KDTreeFlann(target_cloud)

    distances = np.empty(len(source_points), dtype=np.float64)

    for index, point in enumerate(source_points):
        _, _, squared_distances = kdtree.search_knn_vector_3d(
            point.astype(np.float64),
            1,
        )

        if squared_distances:
            distances[index] = np.sqrt(squared_distances[0])
        else:
            distances[index] = np.nan

    return distances[np.isfinite(distances)]


def summarize_distances(
    distances: np.ndarray,
) -> dict[str, float | int | None]:
    """Return basic statistics for a nearest-neighbour distance array."""
    if len(distances) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "percentile_95": None,
            "max": None,
        }

    return {
        "count": int(len(distances)),
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "percentile_95": float(np.percentile(distances, 95)),
        "max": float(np.max(distances)),
    }


def calculate_symmetric_distance_report(
    target_points: np.ndarray,
    transformed_source_points: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    """
    Calculate nearest-neighbour distances in both directions.

    This measures how close each point set lies to the other after registration.
    """
    source_to_target = nearest_neighbor_distances(
        transformed_source_points,
        target_points,
    )
    target_to_source = nearest_neighbor_distances(
        target_points,
        transformed_source_points,
    )

    combined = np.concatenate([source_to_target, target_to_source])

    return {
        "source_to_target": summarize_distances(source_to_target),
        "target_to_source": summarize_distances(target_to_source),
        "symmetric_combined": summarize_distances(combined),
    }


def print_distance_report(
    report: dict[str, dict[str, float | int | None]],
) -> None:
    """Print nearest-neighbour distance statistics."""
    print("\nNearest-neighbour distance statistics:")

    for label, stats in report.items():
        print(f"\n  {label}")
        print(f"    count: {stats['count']}")

        if stats["mean"] is None:
            print("    no valid distances")
            continue

        print(f"    mean:   {stats['mean']:.4f}")
        print(f"    median: {stats['median']:.4f}")
        print(f"    p95:    {stats['percentile_95']:.4f}")
        print(f"    max:    {stats['max']:.4f}")


def save_distance_report_json(
    report: dict[str, dict[str, float | int | None]],
    output_path: Path | str,
    source_frame_key: str,
    target_frame_key: str,
) -> None:
    """Save nearest-neighbour distance statistics as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "source_frame_key": source_frame_key,
        "target_frame_key": target_frame_key,
        "distance_units": (
            "same units as calibration/extrinsics "
            "(e.g. millimetres if checkerboard square size was in mm)"
        ),
        "nearest_neighbour_statistics": report,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

def save_debug_comparison(
    target_points: np.ndarray,
    transformed_source_points: np.ndarray,
    output_dir: Path | str,
    source_frame_key: str,
    target_frame_key: str,
) -> None:
    """Save transformed source hull and a two-color source/target comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_cloud = points_to_open3d_cloud(
        target_points,
        color_rgb=(0.80, 0.80, 0.80),
    )
    source_cloud = points_to_open3d_cloud(
        transformed_source_points,
        color_rgb=(1.00, 0.55, 0.00),
    )

    transformed_source_path = (
        output_dir
        / (
            f"frame_{source_frame_key}_transformed_"
            f"to_frame_{target_frame_key}.ply"
        )
    )

    comparison_path = (
        output_dir
        / (
            f"frame_{source_frame_key}_vs_"
            f"frame_{target_frame_key}.ply"
        )
    )

    o3d.io.write_point_cloud(
        str(transformed_source_path),
        source_cloud,
    )

    comparison_cloud = target_cloud + source_cloud

    o3d.io.write_point_cloud(
        str(comparison_path),
        comparison_cloud,
    )

    print(
        f"Saved transformed source hull: "
        f"{transformed_source_path}"
    )
    print(
        f"Saved two-color comparison: "
        f"{comparison_path}"
    )


def compare_visual_hull_frames(
    object_results_root: Path | str,
    source_frame_key: str,
    target_frame_key: str,
    use_surface_voxels: bool = True,
) -> None:
    """
    Transform one frame's visual hull into another frame's coordinate system.

    The target frame stays unchanged. The source frame is transformed into
    the target frame using frame_transforms.json.
    """
    object_results_root = Path(object_results_root)

    registered_root = object_results_root / "registered"
    transforms_path = (
        registered_root / "frame_transforms.json"
    )

    if not transforms_path.exists():
        raise FileNotFoundError(
            f"Missing {transforms_path}. "
            "Run pipeline/register_sparse_frames.py first."
        )

    transforms = load_frame_transforms(
        transforms_path
    )

    if source_frame_key not in transforms:
        raise KeyError(
            f"Source frame {source_frame_key} has no transform in "
            f"{transforms_path}"
        )

    if target_frame_key not in transforms:
        raise KeyError(
            f"Target frame {target_frame_key} has no transform in "
            f"{transforms_path}"
        )

    source_hull_path = (
        object_results_root
        / f"frame_{source_frame_key}"
        / "visual_hull.npz"
    )
    target_hull_path = (
        object_results_root
        / f"frame_{target_frame_key}"
        / "visual_hull.npz"
    )

    if not source_hull_path.exists():
        raise FileNotFoundError(
            f"Missing source visual hull: {source_hull_path}"
        )

    if not target_hull_path.exists():
        raise FileNotFoundError(
            f"Missing target visual hull: {target_hull_path}"
        )

    source_points = load_visual_hull_points(
        source_hull_path,
        use_surface_voxels=use_surface_voxels,
    )
    target_points = load_visual_hull_points(
        target_hull_path,
        use_surface_voxels=use_surface_voxels,
    )

    source_to_target = (
        calculate_source_to_target_transform(
            source_to_reference=(
                transforms[source_frame_key]
            ),
            target_to_reference=(
                transforms[target_frame_key]
            ),
        )
    )

    transformed_source_points = transform_points(
        source_points,
        source_to_target,
    )

    distance_report = calculate_symmetric_distance_report(
        target_points=target_points,
        transformed_source_points=transformed_source_points,
    )

    print(
        f"Source frame {source_frame_key}: "
        f"{len(source_points)} hull points"
    )
    print(
        f"Target frame {target_frame_key}: "
        f"{len(target_points)} hull points"
    )

    print(
        "\nSource-to-target transform:"
    )
    print(source_to_target)

    output_dir = (
        registered_root
        / "debug"
        / "visual_hull_pair_alignment"
        / (
            f"frame_{source_frame_key}"
            f"_to_frame_{target_frame_key}"
        )
    )

    print_distance_report(distance_report)

    save_distance_report_json(
        report=distance_report,
        output_path=output_dir / "distance_report.json",
        source_frame_key=source_frame_key,
        target_frame_key=target_frame_key,
    )

    save_debug_comparison(
        target_points=target_points,
        transformed_source_points=(
            transformed_source_points
        ),
        output_dir=output_dir,
        source_frame_key=source_frame_key,
        target_frame_key=target_frame_key,
    )


def main() -> None:
    """VS Code friendly debug entry point."""
    config = load_project_config(
        PROJECT_ROOT / "project_config.yaml"
    )

    project_root = Path(
        config["img_root"]
    )

    run_suffix = make_run_suffix(config)

    reconstruction_root = (
        project_root
        / (
            config["folders"][
                "reconstruction_results"
            ]
            + run_suffix
        )
    )

    # ------------------------------------------------------------------
    # Debug settings: edit only these values when comparing frame pairs.
    # ------------------------------------------------------------------
    object_name = "terminal_green"
    source_frame_key = "10"
    target_frame_key = "9"

    # True = compare visual_hull_surface
    # False = compare all occupied visual-hull voxels.
    use_surface_voxels = True
    # ------------------------------------------------------------------

    object_results_root = (
        reconstruction_root / object_name
    )

    compare_visual_hull_frames(
        object_results_root=object_results_root,
        source_frame_key=source_frame_key,
        target_frame_key=target_frame_key,
        use_surface_voxels=use_surface_voxels,
    )


if __name__ == "__main__":
    main()
