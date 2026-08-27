from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:
    raise ImportError(
        "This script requires Open3D. Install it with: pip install open3d"
    ) from exc

from project_config import load_project_config, make_run_suffix


@dataclass
class PointCloudData:
    """Registered object-level point cloud data."""

    points_xyz: np.ndarray
    colors_rgb: np.ndarray | None
    source_object_name: str


@dataclass
class FilteringSettings:
    """Settings for filtering a registered object-level point cloud."""

    remove_non_finite: bool = True
    min_depth: float | None = None
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


def load_registered_point_cloud_npz(path: Path | str) -> PointCloudData:
    """Load a registered object-level sparse point cloud from NPZ."""
    path = Path(path)

    with np.load(path, allow_pickle=True) as data:
        points_xyz = np.asarray(data["points_xyz"], dtype=np.float64)

        colors_rgb = None
        if "colors_rgb" in data:
            colors_rgb = np.asarray(data["colors_rgb"], dtype=np.uint8)

    return PointCloudData(
        points_xyz=points_xyz,
        colors_rgb=colors_rgb,
        source_object_name=path.parent.parent.name,
    )


def save_point_cloud_npz(
    point_cloud: PointCloudData,
    output_path: Path | str,
) -> None:
    """Save a filtered registered point cloud as compressed NPZ."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "points_xyz": point_cloud.points_xyz.astype(np.float32),
        "source_object_name": np.asarray(point_cloud.source_object_name),
    }

    if point_cloud.colors_rgb is not None:
        arrays["colors_rgb"] = point_cloud.colors_rgb.astype(np.uint8)

    np.savez_compressed(output_path, **arrays)


def point_cloud_data_to_open3d(
    point_cloud: PointCloudData,
) -> o3d.geometry.PointCloud:
    """Convert PointCloudData to an Open3D point cloud."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        point_cloud.points_xyz.astype(np.float64)
    )

    if (
        point_cloud.colors_rgb is not None
        and len(point_cloud.colors_rgb) == len(point_cloud.points_xyz)
    ):
        cloud.colors = o3d.utility.Vector3dVector(
            point_cloud.colors_rgb.astype(np.float64) / 255.0
        )

    return cloud


def open3d_to_point_cloud_data(
    cloud: o3d.geometry.PointCloud,
    source_object_name: str,
) -> PointCloudData:
    """Convert an Open3D point cloud back to PointCloudData."""
    points_xyz = np.asarray(cloud.points, dtype=np.float64)

    colors_rgb = None
    if cloud.has_colors():
        colors_rgb = np.clip(
            np.asarray(cloud.colors) * 255.0,
            0,
            255,
        ).astype(np.uint8)

    return PointCloudData(
        points_xyz=points_xyz,
        colors_rgb=colors_rgb,
        source_object_name=source_object_name,
    )


def remove_non_finite_points(
    point_cloud: PointCloudData,
) -> PointCloudData:
    """Remove points containing NaN or infinite coordinates."""
    mask = np.isfinite(point_cloud.points_xyz).all(axis=1)

    colors = None
    if point_cloud.colors_rgb is not None:
        colors = point_cloud.colors_rgb[mask]

    return PointCloudData(
        points_xyz=point_cloud.points_xyz[mask],
        colors_rgb=colors,
        source_object_name=point_cloud.source_object_name,
    )


def filter_by_depth(
    point_cloud: PointCloudData,
    min_depth: float | None,
    max_depth: float | None,
) -> PointCloudData:
    """Filter points by Z coordinate."""
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
    )


def voxel_downsample_with_open3d(
    point_cloud: PointCloudData,
    voxel_size: float,
) -> PointCloudData:
    """Voxel-downsample the registered point cloud."""
    cloud = point_cloud_data_to_open3d(point_cloud)
    cloud = cloud.voxel_down_sample(voxel_size=voxel_size)

    return open3d_to_point_cloud_data(
        cloud,
        source_object_name=point_cloud.source_object_name,
    )


def remove_statistical_outliers(
    point_cloud: PointCloudData,
    nb_neighbors: int,
    std_ratio: float,
) -> PointCloudData:
    """Remove sparse outliers using neighbour-distance statistics."""
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
    )


def remove_radius_outliers(
    point_cloud: PointCloudData,
    nb_points: int,
    radius: float,
) -> PointCloudData:
    """Remove points without enough neighbours inside a radius."""
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
    )


def save_point_cloud_ply(
    point_cloud: PointCloudData,
    output_path: Path | str,
) -> None:
    """Save point cloud as PLY using Open3D."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cloud = point_cloud_data_to_open3d(point_cloud)
    o3d.io.write_point_cloud(str(output_path), cloud)


def print_point_cloud_summary(
    label: str,
    point_cloud: PointCloudData,
) -> None:
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
    print(
        f"  span:    {xyz_span[0]:.3f}, "
        f"{xyz_span[1]:.3f}, {xyz_span[2]:.3f}"
    )


def filter_registered_point_cloud(
    point_cloud: PointCloudData,
    settings: FilteringSettings,
) -> PointCloudData:
    """Apply the filtering pipeline to a registered object cloud."""
    filtered = point_cloud
    print_point_cloud_summary("Original registered point cloud", filtered)

    if settings.remove_non_finite:
        filtered = remove_non_finite_points(filtered)
        print_point_cloud_summary(
            "After removing non-finite points",
            filtered,
        )

    if settings.min_depth is not None or settings.max_depth is not None:
        filtered = filter_by_depth(
            filtered,
            settings.min_depth,
            settings.max_depth,
        )
        print_point_cloud_summary("After depth filtering", filtered)

    if settings.voxel_size is not None and settings.voxel_size > 0:
        filtered = voxel_downsample_with_open3d(
            filtered,
            settings.voxel_size,
        )
        print_point_cloud_summary("After voxel downsampling", filtered)

    if settings.statistical_nb_neighbors > 0:
        filtered = remove_statistical_outliers(
            filtered,
            nb_neighbors=settings.statistical_nb_neighbors,
            std_ratio=settings.statistical_std_ratio,
        )
        print_point_cloud_summary(
            "After statistical outlier removal",
            filtered,
        )

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
        print_point_cloud_summary(
            "After radius outlier removal",
            filtered,
        )

    return filtered


def visualize_point_clouds(
    original: PointCloudData,
    filtered: PointCloudData,
) -> None:
    """Visualize original and filtered registered clouds."""
    original_cloud = point_cloud_data_to_open3d(original)
    filtered_cloud = point_cloud_data_to_open3d(filtered)

    o3d.visualization.draw_geometries(
        [original_cloud],
        window_name="Original registered sparse point cloud",
    )

    o3d.visualization.draw_geometries(
        [filtered_cloud],
        window_name="Filtered registered point cloud",
    )


def update_registration_report(
    report_path: Path | str,
    original_point_count: int,
    filtered_point_count: int,
) -> None:
    """Append registered-cloud filtering statistics to the report."""
    report_path = Path(report_path)

    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    else:
        data = {}

    data["_registered_cloud_filtering"] = {
        "original_points": int(original_point_count),
        "filtered_points": int(filtered_point_count),
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def find_registered_point_clouds(
    reconstruction_root: Path | str,
) -> list[Path]:
    """Find registered sparse NPZ point clouds for all objects."""
    reconstruction_root = Path(reconstruction_root)

    return sorted(
        reconstruction_root.glob(
            "*/registered/registered_sparse_point_cloud.npz"
        ),
        key=lambda path: natural_sort_key(str(path)),
    )


def filter_all_registered_point_clouds(
    reconstruction_root: Path | str,
    settings: FilteringSettings,
    visualize_each: bool = True,
) -> None:
    """Filter every registered object-level sparse point cloud."""
    reconstruction_root = Path(reconstruction_root)

    input_paths = find_registered_point_clouds(reconstruction_root)

    if not input_paths:
        raise FileNotFoundError(
            f"No registered_sparse_point_cloud.npz files found under "
            f"{reconstruction_root}"
        )

    for input_path in input_paths:
        print(f"\nProcessing registered cloud: {input_path}")

        original = load_registered_point_cloud_npz(input_path)
        filtered = filter_registered_point_cloud(original, settings)

        registered_root = input_path.parent

        npz_output_path = (
            registered_root / "filtered_registered_point_cloud.npz"
        )
        ply_output_path = (
            registered_root / "filtered_registered_point_cloud.ply"
        )

        save_point_cloud_npz(filtered, npz_output_path)
        save_point_cloud_ply(filtered, ply_output_path)

        update_registration_report(
            registered_root / "registration_report.json",
            original_point_count=len(original.points_xyz),
            filtered_point_count=len(filtered.points_xyz),
        )

        print(f"Saved filtered registered NPZ: {npz_output_path}")
        print(f"Saved filtered registered PLY: {ply_output_path}")

        if visualize_each:
            visualize_point_clouds(
                original=original,
                filtered=filtered,
            )


def main() -> None:
    """VS Code friendly entry point using project_config.yaml."""
    config = load_project_config(Path("project_config.yaml"))
    project_root = Path(config["img_root"])

    run_suffix = make_run_suffix(config)

    reconstruction_root = (
        project_root
        / (
            config["folders"]["reconstruction_results"]
            + run_suffix
        )
    )

    # Units are the same as in calibration. If checkerboard square size
    # was given in millimetres, voxel and radius values are also millimetres.
    settings = FilteringSettings(
        remove_non_finite=True,
        min_depth=None,
        max_depth=None,

        # The registered cloud contains overlapping observations from
        # multiple frames. Start with None; later try e.g. 0.5 or 1.0 mm.
        voxel_size=None,

        statistical_nb_neighbors=20,
        statistical_std_ratio=2.0,

        # Registered clouds are denser than single-frame clouds, so this can
        # usually be somewhat stricter than the frame-level radius filter.
        radius_outlier_nb_points=8,
        radius_outlier_radius=2.0,
    )

    filter_all_registered_point_clouds(
        reconstruction_root=reconstruction_root,
        settings=settings,
        visualize_each=True,
    )


if __name__ == "__main__":
    main()
