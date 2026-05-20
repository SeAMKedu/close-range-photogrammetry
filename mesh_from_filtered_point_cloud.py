from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence
from project_config import load_project_config

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:
    raise ImportError(
        "This script requires Open3D. Install it with: pip install open3d"
    ) from exc


MeshMethod = Literal["alpha", "ball_pivoting", "poisson"]


@dataclass
class PointCloudData:
    """Point cloud data loaded from an NPZ file."""

    points_xyz: np.ndarray
    colors_rgb: np.ndarray | None
    source_object_name: str
    source_frame_keys: list[str]


@dataclass
class MeshSettings:
    """Settings for point-cloud meshing."""

    method: MeshMethod = "alpha"

    # Alpha shape. Smaller alpha gives a tighter surface, larger alpha fills more gaps.
    alpha: float = 5.0

    # Ball pivoting. Radii should be related to point spacing.
    ball_pivoting_radii: tuple[float, ...] = (1.0, 2.0, 4.0)

    # Poisson. Useful only if normals are good and the cloud is reasonably dense.
    poisson_depth: int = 8
    poisson_density_quantile: float = 0.02

    # Normal estimation.
    normal_radius: float = 5.0
    normal_max_nn: int = 30

    # Optional pre-downsampling before meshing. Keep None initially.
    voxel_size: float | None = None

    # Mesh cleanup.
    remove_degenerate_triangles: bool = True
    remove_duplicated_triangles: bool = True
    remove_duplicated_vertices: bool = True
    remove_non_manifold_edges: bool = True
    keep_largest_component: bool = False


def natural_sort_key(text: str) -> list[int | str]:
    """Sort strings so that 1, 2, 10 are ordered numerically."""
    parts = re.split(r"([0-9]+)", text)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def load_point_cloud_npz(path: Path | str) -> PointCloudData:
    """Load a filtered point cloud from an NPZ file."""
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


def point_cloud_data_to_open3d(point_cloud: PointCloudData) -> o3d.geometry.PointCloud:
    """Convert PointCloudData to an Open3D point cloud."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_cloud.points_xyz.astype(np.float64))

    if point_cloud.colors_rgb is not None and len(point_cloud.colors_rgb) == len(point_cloud.points_xyz):
        cloud.colors = o3d.utility.Vector3dVector(point_cloud.colors_rgb.astype(np.float64) / 255.0)

    return cloud


def estimate_point_cloud_spacing(
    cloud: o3d.geometry.PointCloud,
    sample_size: int = 5000,
) -> float:
    """Estimate typical nearest-neighbour spacing of a point cloud."""
    points = np.asarray(cloud.points)
    if len(points) < 2:
        return 1.0

    if len(points) > sample_size:
        indices = np.linspace(0, len(points) - 1, sample_size).astype(int)
        sampled = cloud.select_by_index(indices.tolist())
    else:
        sampled = cloud

    distances = np.asarray(sampled.compute_nearest_neighbor_distance())
    distances = distances[np.isfinite(distances)]
    distances = distances[distances > 0]

    if len(distances) == 0:
        return 1.0

    return float(np.median(distances))


def prepare_point_cloud_for_meshing(
    point_cloud: PointCloudData,
    settings: MeshSettings,
) -> o3d.geometry.PointCloud:
    """Convert, optionally downsample, and estimate normals for meshing."""
    cloud = point_cloud_data_to_open3d(point_cloud)

    if settings.voxel_size is not None and settings.voxel_size > 0:
        cloud = cloud.voxel_down_sample(settings.voxel_size)

    if len(cloud.points) == 0:
        raise RuntimeError("Point cloud has no points after optional downsampling.")

    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=settings.normal_radius,
            max_nn=settings.normal_max_nn,
        )
    )

    if len(cloud.points) > settings.normal_max_nn:
        cloud.orient_normals_consistent_tangent_plane(k=settings.normal_max_nn)

    return cloud


def create_alpha_shape_mesh(
    cloud: o3d.geometry.PointCloud,
    alpha: float,
) -> o3d.geometry.TriangleMesh:
    """Create a mesh using alpha shapes."""
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        cloud,
        alpha=alpha,
    )
    mesh.compute_vertex_normals()
    return mesh


def create_ball_pivoting_mesh(
    cloud: o3d.geometry.PointCloud,
    radii: Sequence[float],
) -> o3d.geometry.TriangleMesh:
    """Create a mesh using the ball-pivoting algorithm."""
    radii_vector = o3d.utility.DoubleVector([float(radius) for radius in radii])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        cloud,
        radii_vector,
    )
    mesh.compute_vertex_normals()
    return mesh


def create_poisson_mesh(
    cloud: o3d.geometry.PointCloud,
    depth: int,
    density_quantile: float,
) -> o3d.geometry.TriangleMesh:
    """Create a mesh using Poisson surface reconstruction and trim low-density vertices."""
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud,
        depth=depth,
    )

    densities_np = np.asarray(densities)
    density_threshold = np.quantile(densities_np, density_quantile)
    vertices_to_remove = densities_np < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    return mesh


def keep_largest_triangle_component(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Keep only the largest connected triangle component in a mesh."""
    if len(mesh.triangles) == 0:
        return mesh

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    largest_cluster_index = int(np.argmax(cluster_n_triangles))
    triangles_to_remove = triangle_clusters != largest_cluster_index
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()

    return mesh


def clean_mesh(
    mesh: o3d.geometry.TriangleMesh,
    settings: MeshSettings,
) -> o3d.geometry.TriangleMesh:
    """Apply basic triangle-mesh cleanup operations."""
    if settings.remove_degenerate_triangles:
        mesh.remove_degenerate_triangles()
    if settings.remove_duplicated_triangles:
        mesh.remove_duplicated_triangles()
    if settings.remove_duplicated_vertices:
        mesh.remove_duplicated_vertices()
    if settings.remove_non_manifold_edges:
        mesh.remove_non_manifold_edges()
    if settings.keep_largest_component:
        mesh = keep_largest_triangle_component(mesh)

    mesh.remove_unreferenced_vertices()

    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()

    return mesh


def create_mesh_from_point_cloud(
    point_cloud: PointCloudData,
    settings: MeshSettings,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
    """Create a triangle mesh from a filtered point cloud."""
    cloud = prepare_point_cloud_for_meshing(point_cloud, settings)

    estimated_spacing = estimate_point_cloud_spacing(cloud)
    print(f"Estimated point spacing: {estimated_spacing:.3f}")

    if settings.method == "alpha":
        mesh = create_alpha_shape_mesh(cloud, settings.alpha)
    elif settings.method == "ball_pivoting":
        mesh = create_ball_pivoting_mesh(cloud, settings.ball_pivoting_radii)
    elif settings.method == "poisson":
        mesh = create_poisson_mesh(
            cloud,
            depth=settings.poisson_depth,
            density_quantile=settings.poisson_density_quantile,
        )
    else:
        raise ValueError(f"Unsupported mesh method: {settings.method}")

    mesh = clean_mesh(mesh, settings)
    return cloud, mesh


def print_mesh_summary(mesh: o3d.geometry.TriangleMesh, label: str = "Mesh") -> None:
    """Print basic mesh statistics."""
    print(f"\n{label}")
    print(f"  vertices:  {len(mesh.vertices)}")
    print(f"  triangles: {len(mesh.triangles)}")

    if len(mesh.vertices) > 0:
        vertices = np.asarray(mesh.vertices)
        xyz_min = vertices.min(axis=0)
        xyz_max = vertices.max(axis=0)
        span = xyz_max - xyz_min
        print(f"  x range: {xyz_min[0]:.3f} ... {xyz_max[0]:.3f}")
        print(f"  y range: {xyz_min[1]:.3f} ... {xyz_max[1]:.3f}")
        print(f"  z range: {xyz_min[2]:.3f} ... {xyz_max[2]:.3f}")
        print(f"  span:    {span[0]:.3f}, {span[1]:.3f}, {span[2]:.3f}")


def save_mesh(mesh: o3d.geometry.TriangleMesh, output_path: Path | str) -> None:
    """Save a triangle mesh to PLY, OBJ, or another Open3D-supported format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_path), mesh)


def save_prepared_point_cloud(cloud: o3d.geometry.PointCloud, output_path: Path | str) -> None:
    """Save the normal-estimated point cloud used for meshing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), cloud)


def visualize_cloud_and_mesh(
    cloud: o3d.geometry.PointCloud,
    mesh: o3d.geometry.TriangleMesh,
    object_name: str,
) -> None:
    """Visualize the prepared point cloud and generated mesh in Open3D."""
    print("Opening point cloud window...")
    o3d.visualization.draw_geometries(
        [cloud],
        window_name=f"Prepared point cloud: {object_name}",
        point_show_normal=False,
    )

    print("Opening mesh window...")
    o3d.visualization.draw_geometries(
        [mesh],
        window_name=f"Mesh: {object_name}",
        mesh_show_back_face=True,
    )


def find_filtered_point_clouds(input_root: Path | str) -> list[Path]:
    """Find filtered point-cloud NPZ files under input_root."""
    input_root = Path(input_root)

    paths = sorted(
        input_root.glob("*/**/*_filtered_point_cloud.npz"),
        key=lambda p: natural_sort_key(str(p)),
    )

    if not paths:
        paths = sorted(
            input_root.glob("**/*.npz"),
            key=lambda p: natural_sort_key(str(p)),
        )

    return paths


def mesh_all_filtered_point_clouds(
    input_root: Path | str,
    output_root: Path | str,
    settings: MeshSettings,
    visualize_each: bool = True,
) -> None:
    """Create meshes for all filtered point clouds found under input_root."""
    input_paths = find_filtered_point_clouds(input_root)
    output_root = Path(output_root)

    if not input_paths:
        raise FileNotFoundError(f"No filtered NPZ point clouds found under {input_root}")

    for input_path in input_paths:
        print(f"\nProcessing: {input_path}")

        point_cloud = load_point_cloud_npz(input_path)
        cloud, mesh = create_mesh_from_point_cloud(point_cloud, settings)
        print_mesh_summary(mesh, label=f"Mesh for {point_cloud.source_object_name}")

        object_output_dir = output_root / point_cloud.source_object_name
        method_name = settings.method

        prepared_cloud_path = object_output_dir / f"{point_cloud.source_object_name}_prepared_for_mesh.ply"
        mesh_ply_path = object_output_dir / f"{point_cloud.source_object_name}_{method_name}_mesh.ply"
        mesh_obj_path = object_output_dir / f"{point_cloud.source_object_name}_{method_name}_mesh.obj"

        save_prepared_point_cloud(cloud, prepared_cloud_path)
        save_mesh(mesh, mesh_ply_path)
        save_mesh(mesh, mesh_obj_path)

        print(f"Saved prepared point cloud: {prepared_cloud_path}")
        print(f"Saved mesh PLY: {mesh_ply_path}")
        print(f"Saved mesh OBJ: {mesh_obj_path}")

        if visualize_each:
            visualize_cloud_and_mesh(
                cloud=cloud,
                mesh=mesh,
                object_name=point_cloud.source_object_name,
            )


def main() -> None:
    config = load_project_config(Path("project_config.yaml"))
    project_root = Path(config["img_root"])

    input_root = project_root / config["folders"]["filtered_point_clouds"]
    output_root = project_root / config["folders"]["mesh_results"]

    # Coordinate units are the same as in calibration. If square size was mm,
    # alpha, radii, normal_radius, and voxel_size are also in mm.
    settings = MeshSettings(
        # Start with "alpha". Then try "ball_pivoting".
        # Use "poisson" only if the filtered cloud is dense enough.
        method="alpha",

        # For sparse clouds, tune this interactively.
        # Smaller: tighter but more holes. Larger: fills gaps but may create false surfaces.
        alpha=5.0,

        # Try radii related to estimated point spacing.
        # If estimated spacing is about 1 mm, use e.g. (1.0, 2.0, 4.0).
        ball_pivoting_radii=(1.0, 2.0, 4.0),

        poisson_depth=8,
        poisson_density_quantile=0.02,

        normal_radius=5.0,
        normal_max_nn=30,

        voxel_size=None,
        keep_largest_component=False,
    )

    mesh_all_filtered_point_clouds(
        input_root=input_root,
        output_root=output_root,
        settings=settings,
        visualize_each=True,
    )


if __name__ == "__main__":
    main()
