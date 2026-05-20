from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from project_config import load_project_config

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:
    raise ImportError(
        "This script requires Open3D. Install it with: pip install open3d"
    ) from exc


@dataclass
class PointCloudData:
    """Point cloud data loaded from or saved to NPZ."""

    points_xyz: np.ndarray
    colors_rgb: np.ndarray | None
    source_object_name: str
    source_frame_keys: list[str]


@dataclass
class FilteringSettings:
    """Settings for sparse point-cloud filtering."""

    remove_non_finite: bool = True
    min_depth: float | None = 0.0
    max_depth: float | None = None
    voxel_size: float | None = None
    statistical_nb_neighbors: int = 20
    statistical_std_ratio: float = 2.0
    radius_outlier_nb_points: int | None = None
    radius_outlier_radius: float | None = None


def natural_sort_key(text: str) -> list[int | str]:
    """Sort strings so that 1, 2, 10 are ordered numerically."""
    parts = re.split(r"([0-9]+)", text)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def load_point_cloud_npz(path: Path | str) -> PointCloudData:
    """Load a point cloud saved by reconstruct_sparse_point_cloud.py."""
    path = Path(path)

    with np.load(path, allow_pickle=True) as data:
        points_xyz = np.asarray(data["points_xyz"], dtype=np.float64)

        colors_rgb = None
        if "colors_rgb" in data:
            colors_rgb = np.asarray(data["colors_rgb"], dtype=np.uint8)

        source_object_name = path.parent.name
        if "source_object_name" in data:
            source_object_name = str(data["source_object_name"])

        source_frame_keys: list[str] = []
        if "source_frame_keys" in data:
            source_frame_keys = [str(value) for value in data["source_frame_keys"]]

    return PointCloudData(
        points_xyz=points_xyz,
        colors_rgb=colors_rgb,
        source_object_name=source_object_name,
        source_frame_keys=source_frame_keys,
    )


def save_point_cloud_npz(point_cloud: PointCloudData, output_path: Path | str) -> None:
    """Save a filtered point cloud as compressed NPZ."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "points_xyz": point_cloud.points_xyz.astype(np.float32),
        "source_object_name": np.asarray(point_cloud.source_object_name),
        "source_frame_keys": np.asarray(point_cloud.source_frame_keys),
    }

    if point_cloud.colors_rgb is not None:
        arrays["colors_rgb"] = point_cloud.colors_rgb.astype(np.uint8)

    np.savez_compressed(output_path, **arrays)


def point_cloud_data_to_open3d(point_cloud: PointCloudData) -> o3d.geometry.PointCloud:
    """Convert PointCloudData to an Open3D point cloud."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_cloud.points_xyz.astype(np.float64))

    if point_cloud.colors_rgb is not None and len(point_cloud.colors_rgb) == len(point_cloud.points_xyz):
        cloud.colors = o3d.utility.Vector3dVector(point_cloud.colors_rgb.astype(np.float64) / 255.0)

    return cloud


def open3d_to_point_cloud_data(
    cloud: o3d.geometry.PointCloud,
    source_object_name: str,
    source_frame_keys: Sequence[str],
) -> PointCloudData:
    """Convert an Open3D point cloud back to PointCloudData."""
    points_xyz = np.asarray(cloud.points, dtype=np.float64)

    colors_rgb = None
    if cloud.has_colors():
        colors_rgb = np.clip(np.asarray(cloud.colors) * 255.0, 0, 255).astype(np.uint8)

    return PointCloudData(
        points_xyz=points_xyz,
        colors_rgb=colors_rgb,
        source_object_name=source_object_name,
        source_frame_keys=list(source_frame_keys),
    )


def remove_non_finite_points(point_cloud: PointCloudData) -> PointCloudData:
    """Remove points containing NaN or infinite coordinates."""
    mask = np.isfinite(point_cloud.points_xyz).all(axis=1)

    colors = None
    if point_cloud.colors_rgb is not None:
        colors = point_cloud.colors_rgb[mask]

    return PointCloudData(
        points_xyz=point_cloud.points_xyz[mask],
        colors_rgb=colors,
        source_object_name=point_cloud.source_object_name,
        source_frame_keys=point_cloud.source_frame_keys,
    )


def filter_by_depth(
    point_cloud: PointCloudData,
    min_depth: float | None,
    max_depth: float | None,
) -> PointCloudData:
    """
    Filter points by Z coordinate in the reference camera coordinate system.

    This is useful for removing points behind the reference camera or very far
    triangulation outliers.
    """
    z = point_cloud.points_xyz[:, 2]
    mask = np.ones(len(z), dtype=bool)

    if min_depth is not None:
        mask &= z >= min_depth
    if max_depth is not None:
        mask &= z <= max_depth

    colors = None
    if point_cloud.colors_rgb is not None:
        colors = point_cloud.colors_rgb[mask]

    return PointCloudData(
        points_xyz=point_cloud.points_xyz[mask],
        colors_rgb=colors,
        source_object_name=point_cloud.source_object_name,
        source_frame_keys=point_cloud.source_frame_keys,
    )


def voxel_downsample_with_open3d(
    point_cloud: PointCloudData,
    voxel_size: float,
) -> PointCloudData:
    """Voxel-downsample the point cloud using Open3D."""
    cloud = point_cloud_data_to_open3d(point_cloud)
    cloud = cloud.voxel_down_sample(voxel_size=voxel_size)

    return open3d_to_point_cloud_data(
        cloud,
        source_object_name=point_cloud.source_object_name,
        source_frame_keys=point_cloud.source_frame_keys,
    )


def remove_statistical_outliers(
    point_cloud: PointCloudData,
    nb_neighbors: int,
    std_ratio: float,
) -> PointCloudData:
    """Remove sparse outliers based on neighbour-distance statistics."""
    if len(point_cloud.points_xyz) < nb_neighbors:
        print(
            f"[WARN] Too few points for statistical outlier removal: "
            f"{len(point_cloud.points_xyz)} < {nb_neighbors}"
        )
        return point_cloud

    cloud = point_cloud_data_to_open3d(point_cloud)
    filtered_cloud, _ = cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )

    return open3d_to_point_cloud_data(
        filtered_cloud,
        source_object_name=point_cloud.source_object_name,
        source_frame_keys=point_cloud.source_frame_keys,
    )


def remove_radius_outliers(
    point_cloud: PointCloudData,
    nb_points: int,
    radius: float,
) -> PointCloudData:
    """Remove points that do not have enough neighbours within a radius."""
    if len(point_cloud.points_xyz) < nb_points:
        print(
            f"[WARN] Too few points for radius outlier removal: "
            f"{len(point_cloud.points_xyz)} < {nb_points}"
        )
        return point_cloud

    cloud = point_cloud_data_to_open3d(point_cloud)
    filtered_cloud, _ = cloud.remove_radius_outlier(
        nb_points=nb_points,
        radius=radius,
    )

    return open3d_to_point_cloud_data(
        filtered_cloud,
        source_object_name=point_cloud.source_object_name,
        source_frame_keys=point_cloud.source_frame_keys,
    )


def estimate_normals(
    cloud: o3d.geometry.PointCloud,
    radius: float,
    max_nn: int = 30,
) -> o3d.geometry.PointCloud:
    """Estimate point normals for visualization and later meshing."""
    if len(cloud.points) == 0:
        return cloud

    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn,
        )
    )
    cloud.orient_normals_consistent_tangent_plane(k=max_nn)
    return cloud


def save_point_cloud_ply(point_cloud: PointCloudData, output_path: Path | str) -> None:
    """Save point cloud as PLY using Open3D."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cloud = point_cloud_data_to_open3d(point_cloud)
    o3d.io.write_point_cloud(str(output_path), cloud)


def print_point_cloud_summary(label: str, point_cloud: PointCloudData) -> None:
    """Print basic point-cloud statistics."""
    points = point_cloud.points_xyz

    print(f"\n{label}")
    print(f"  points: {len(points)}")

    if len(points) == 0:
        return

    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    xyz_span = xyz_max - xyz_min

    print(f"  x range: {xyz_min[0]:.3f} ... {xyz_max[0]:.3f}")
    print(f"  y range: {xyz_min[1]:.3f} ... {xyz_max[1]:.3f}")
    print(f"  z range: {xyz_min[2]:.3f} ... {xyz_max[2]:.3f}")
    print(f"  span:    {xyz_span[0]:.3f}, {xyz_span[1]:.3f}, {xyz_span[2]:.3f}")


def filter_point_cloud(
    point_cloud: PointCloudData,
    settings: FilteringSettings,
) -> PointCloudData:
    """Apply the full filtering pipeline to a point cloud."""
    filtered = point_cloud

    print_point_cloud_summary("Original point cloud", filtered)

    if settings.remove_non_finite:
        filtered = remove_non_finite_points(filtered)
        print_point_cloud_summary("After removing non-finite points", filtered)

    if settings.min_depth is not None or settings.max_depth is not None:
        filtered = filter_by_depth(filtered, settings.min_depth, settings.max_depth)
        print_point_cloud_summary("After depth filtering", filtered)

    if settings.voxel_size is not None and settings.voxel_size > 0:
        filtered = voxel_downsample_with_open3d(filtered, settings.voxel_size)
        print_point_cloud_summary("After voxel downsampling", filtered)

    if settings.statistical_nb_neighbors > 0:
        filtered = remove_statistical_outliers(
            filtered,
            nb_neighbors=settings.statistical_nb_neighbors,
            std_ratio=settings.statistical_std_ratio,
        )
        print_point_cloud_summary("After statistical outlier removal", filtered)

    if (
        settings.radius_outlier_nb_points is not None
        and settings.radius_outlier_radius is not None
        and settings.radius_outlier_nb_points > 0
        and settings.radius_outlier_radius > 0
    ):
        filtered = remove_radius_outliers(
            filtered,
            nb_points=settings.radius_outlier_nb_points,
            radius=settings.radius_outlier_radius,
        )
        print_point_cloud_summary("After radius outlier removal", filtered)

    return filtered


def visualize_point_clouds(
    original: PointCloudData,
    filtered: PointCloudData,
    estimate_filtered_normals: bool = False,
    normal_radius: float = 5.0,
) -> None:
    """Visualize original and filtered point clouds in separate Open3D windows."""
    original_cloud = point_cloud_data_to_open3d(original)
    filtered_cloud = point_cloud_data_to_open3d(filtered)

    if estimate_filtered_normals:
        filtered_cloud = estimate_normals(filtered_cloud, radius=normal_radius)

    print("Opening original point cloud window...")
    o3d.visualization.draw_geometries(
        [original_cloud],
        window_name="Original sparse point cloud",
    )

    print("Opening filtered point cloud window...")
    o3d.visualization.draw_geometries(
        [filtered_cloud],
        window_name="Filtered point cloud",
    )


def find_input_point_clouds(input_root: Path | str) -> list[Path]:
    """Find sparse point-cloud NPZ files under the reconstruction output folder."""
    input_root = Path(input_root)
    paths = sorted(
        input_root.glob("*/**/*_sparse_point_cloud.npz"),
        key=lambda p: natural_sort_key(str(p)),
    )

    if not paths:
        paths = sorted(
            input_root.glob("**/*.npz"),
            key=lambda p: natural_sort_key(str(p)),
        )

    return paths


def filter_all_point_clouds(
    input_root: Path | str,
    output_root: Path | str,
    settings: FilteringSettings,
    visualize_each: bool = False,
) -> None:
    """Filter every NPZ point cloud found under input_root."""
    input_root = Path(input_root)
    output_root = Path(output_root)

    input_paths = find_input_point_clouds(input_root)
    if not input_paths:
        raise FileNotFoundError(f"No NPZ point clouds found under {input_root}")

    for input_path in input_paths:
        print(f"\nProcessing: {input_path}")

        original = load_point_cloud_npz(input_path)
        filtered = filter_point_cloud(original, settings)

        object_name = original.source_object_name
        object_output_root = output_root / object_name

        npz_output_path = object_output_root / f"{object_name}_filtered_point_cloud.npz"
        ply_output_path = object_output_root / f"{object_name}_filtered_point_cloud.ply"

        save_point_cloud_npz(filtered, npz_output_path)
        save_point_cloud_ply(filtered, ply_output_path)

        print(f"Saved filtered NPZ: {npz_output_path}")
        print(f"Saved filtered PLY: {ply_output_path}")

        if visualize_each:
            visualize_point_clouds(
                original=original,
                filtered=filtered,
                estimate_filtered_normals=False,
            )


def main() -> None:
    config = load_project_config(Path("project_config.yaml"))
    project_root = Path(config["img_root"])

    input_root = project_root / config["folders"]["reconstruction_results"]
    output_root = project_root / config["folders"]["filtered_point_clouds"]

    # Coordinate units are the same as your extrinsic calibration square size.
    # If your checkerboard square size was in millimetres, these values are in millimetres.
    settings = FilteringSettings(
        remove_non_finite=True,
        min_depth=0.0,
        max_depth=None,

        # Set this after looking at point spacing. For a sparse cloud, start with None.
        # Example for millimetre-scale data: voxel_size=0.5 or 1.0
        voxel_size=None,

        # Good initial values for sparse feature-based triangulation.
        statistical_nb_neighbors=20,
        statistical_std_ratio=2.0,

        # Optional stricter local-density filter. Enable only after testing.
        radius_outlier_nb_points=6,
        radius_outlier_radius=2.0,
    )

    filter_all_point_clouds(
        input_root=input_root,
        output_root=output_root,
        settings=settings,
        visualize_each=True,
    )


if __name__ == "__main__":
    main()
