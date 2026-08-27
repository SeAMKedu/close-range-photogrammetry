from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:
    raise ImportError(
        "This script requires Open3D. Install it with: pip install open3d"
    ) from exc

from project_config import load_project_config, make_run_suffix


MeshMethod = Literal["alpha", "ball_pivoting", "poisson"]


@dataclass
class PointCloudData:
    """Filtered registered object-level point cloud data."""

    points_xyz: np.ndarray
    colors_rgb: np.ndarray | None
    source_object_name: str


@dataclass
class MeshSettings:
    """Settings for meshing a registered object-level point cloud."""

    method: MeshMethod = "alpha"

    # Parameters below are expressed relative to the estimated
    # nearest-neighbour point spacing of the prepared cloud.
    alpha_multiplier: float = 3.0

    ball_pivoting_radius_multipliers: tuple[float, ...] = (
        1.5,
        2.5,
        4.0,
    )

    normal_radius_multiplier: float = 5.0

    # Poisson reconstruction parameters are not spacing multipliers.
    poisson_depth: int = 8
    poisson_density_quantile: float = 0.02

    # Maximum number of neighbours used in normal estimation.
    normal_max_nn: int = 30

    # Optional pre-downsampling before meshing.
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
    return [
        int(part) if part.isdigit() else part.lower()
        for part in parts
    ]


def load_filtered_registered_point_cloud_npz(
    path: Path | str,
) -> PointCloudData:
    """Load a filtered registered object-level point cloud from NPZ."""
    path = Path(path)

    with np.load(path, allow_pickle=True) as data:
        points_xyz = np.asarray(
            data["points_xyz"],
            dtype=np.float64,
        )

        colors_rgb = None
        if "colors_rgb" in data:
            colors_rgb = np.asarray(
                data["colors_rgb"],
                dtype=np.uint8,
            )

        source_object_name = path.parent.parent.name
        if "source_object_name" in data:
            source_object_name = str(
                data["source_object_name"]
            )

    return PointCloudData(
        points_xyz=points_xyz,
        colors_rgb=colors_rgb,
        source_object_name=source_object_name,
    )


def point_cloud_data_to_open3d(
    point_cloud: PointCloudData,
) -> o3d.geometry.PointCloud:
    """Convert PointCloudData to an Open3D point cloud."""
    cloud = o3d.geometry.PointCloud()

    cloud.points = o3d.utility.Vector3dVector(
        point_cloud.points_xyz.astype(
            np.float64
        )
    )

    if (
        point_cloud.colors_rgb is not None
        and len(point_cloud.colors_rgb)
        == len(point_cloud.points_xyz)
    ):
        cloud.colors = o3d.utility.Vector3dVector(
            point_cloud.colors_rgb.astype(
                np.float64
            )
            / 255.0
        )

    return cloud


def estimate_point_cloud_spacing(
    cloud: o3d.geometry.PointCloud,
    sample_size: int = 5000,
) -> float:
    """Estimate typical nearest-neighbour spacing."""
    points = np.asarray(cloud.points)

    if len(points) < 2:
        return 1.0

    if len(points) > sample_size:
        indices = np.linspace(
            0,
            len(points) - 1,
            sample_size,
        ).astype(int)

        sampled = cloud.select_by_index(
            indices.tolist()
        )
    else:
        sampled = cloud

    distances = np.asarray(
        sampled.compute_nearest_neighbor_distance()
    )

    distances = distances[
        np.isfinite(distances)
    ]
    distances = distances[
        distances > 0
    ]

    if len(distances) == 0:
        return 1.0

    return float(
        np.median(distances)
    )


def prepare_point_cloud_for_meshing(
    point_cloud: PointCloudData,
    settings: MeshSettings,
) -> o3d.geometry.PointCloud:
    """Convert and optionally downsample a point cloud before meshing."""
    cloud = point_cloud_data_to_open3d(
        point_cloud
    )

    if (
        settings.voxel_size is not None
        and settings.voxel_size > 0
    ):
        cloud = cloud.voxel_down_sample(
            settings.voxel_size
        )

    if len(cloud.points) == 0:
        raise RuntimeError(
            "Point cloud has no points after optional downsampling."
        )

    return cloud


def estimate_normals_for_meshing(
    cloud: o3d.geometry.PointCloud,
    normal_radius: float,
    normal_max_nn: int,
) -> None:
    """Estimate and orient normals using a spacing-scaled search radius."""
    cloud.estimate_normals(
        search_param=(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius,
                max_nn=normal_max_nn,
            )
        )
    )

    if len(cloud.points) > normal_max_nn:
        cloud.orient_normals_consistent_tangent_plane(
            k=normal_max_nn
        )


def create_alpha_shape_mesh(
    cloud: o3d.geometry.PointCloud,
    alpha: float,
) -> o3d.geometry.TriangleMesh:
    """Create a mesh using alpha shapes."""
    mesh = (
        o3d.geometry.TriangleMesh
        .create_from_point_cloud_alpha_shape(
            cloud,
            alpha=alpha,
        )
    )

    mesh.compute_vertex_normals()
    return mesh


def create_ball_pivoting_mesh(
    cloud: o3d.geometry.PointCloud,
    radii: Sequence[float],
) -> o3d.geometry.TriangleMesh:
    """Create a mesh using ball pivoting."""
    radii_vector = (
        o3d.utility.DoubleVector(
            [
                float(radius)
                for radius in radii
            ]
        )
    )

    mesh = (
        o3d.geometry.TriangleMesh
        .create_from_point_cloud_ball_pivoting(
            cloud,
            radii_vector,
        )
    )

    mesh.compute_vertex_normals()
    return mesh


def create_poisson_mesh(
    cloud: o3d.geometry.PointCloud,
    depth: int,
    density_quantile: float,
) -> o3d.geometry.TriangleMesh:
    """Create a Poisson mesh and trim low-density vertices."""
    mesh, densities = (
        o3d.geometry.TriangleMesh
        .create_from_point_cloud_poisson(
            cloud,
            depth=depth,
        )
    )

    densities_np = np.asarray(
        densities
    )

    density_threshold = np.quantile(
        densities_np,
        density_quantile,
    )

    vertices_to_remove = (
        densities_np
        < density_threshold
    )

    mesh.remove_vertices_by_mask(
        vertices_to_remove
    )

    mesh.compute_vertex_normals()
    return mesh


def keep_largest_triangle_component(
    mesh: o3d.geometry.TriangleMesh,
) -> o3d.geometry.TriangleMesh:
    """Keep only the largest connected triangle component."""
    if len(mesh.triangles) == 0:
        return mesh

    (
        triangle_clusters,
        cluster_n_triangles,
        _,
    ) = mesh.cluster_connected_triangles()

    triangle_clusters = np.asarray(
        triangle_clusters
    )
    cluster_n_triangles = np.asarray(
        cluster_n_triangles
    )

    largest_cluster_index = int(
        np.argmax(cluster_n_triangles)
    )

    triangles_to_remove = (
        triangle_clusters
        != largest_cluster_index
    )

    mesh.remove_triangles_by_mask(
        triangles_to_remove
    )
    mesh.remove_unreferenced_vertices()

    return mesh


def clean_mesh(
    mesh: o3d.geometry.TriangleMesh,
    settings: MeshSettings,
) -> o3d.geometry.TriangleMesh:
    """Apply basic mesh cleanup operations."""
    if settings.remove_degenerate_triangles:
        mesh.remove_degenerate_triangles()

    if settings.remove_duplicated_triangles:
        mesh.remove_duplicated_triangles()

    if settings.remove_duplicated_vertices:
        mesh.remove_duplicated_vertices()

    if settings.remove_non_manifold_edges:
        mesh.remove_non_manifold_edges()

    if settings.keep_largest_component:
        mesh = keep_largest_triangle_component(
            mesh
        )

    mesh.remove_unreferenced_vertices()

    if (
        len(mesh.vertices) > 0
        and len(mesh.triangles) > 0
    ):
        mesh.compute_vertex_normals()

    return mesh


def create_mesh_from_registered_point_cloud(
    point_cloud: PointCloudData,
    settings: MeshSettings,
) -> tuple[
    o3d.geometry.PointCloud,
    o3d.geometry.TriangleMesh,
]:
    """Create a triangle mesh using parameters scaled by point spacing."""
    cloud = prepare_point_cloud_for_meshing(
        point_cloud,
        settings,
    )

    estimated_spacing = estimate_point_cloud_spacing(
        cloud
    )

    if not np.isfinite(estimated_spacing) or estimated_spacing <= 0:
        raise RuntimeError(
            f"Invalid estimated point spacing: {estimated_spacing}"
        )

    alpha = (
        settings.alpha_multiplier
        * estimated_spacing
    )

    ball_pivoting_radii = tuple(
        multiplier * estimated_spacing
        for multiplier
        in settings.ball_pivoting_radius_multipliers
    )

    normal_radius = (
        settings.normal_radius_multiplier
        * estimated_spacing
    )

    print(
        f"Estimated point spacing: "
        f"{estimated_spacing:.3f}"
    )
    print(
        f"Derived alpha: "
        f"{alpha:.3f}"
    )
    print(
        "Derived ball pivoting radii: "
        + ", ".join(
            f"{radius:.3f}"
            for radius in ball_pivoting_radii
        )
    )
    print(
        f"Derived normal radius: "
        f"{normal_radius:.3f}"
    )

    estimate_normals_for_meshing(
        cloud=cloud,
        normal_radius=normal_radius,
        normal_max_nn=settings.normal_max_nn,
    )

    if settings.method == "alpha":
        mesh = create_alpha_shape_mesh(
            cloud,
            alpha,
        )

    elif settings.method == "ball_pivoting":
        mesh = create_ball_pivoting_mesh(
            cloud,
            ball_pivoting_radii,
        )

    elif settings.method == "poisson":
        mesh = create_poisson_mesh(
            cloud,
            depth=settings.poisson_depth,
            density_quantile=(
                settings.poisson_density_quantile
            ),
        )

    else:
        raise ValueError(
            f"Unsupported mesh method: "
            f"{settings.method}"
        )

    mesh = clean_mesh(
        mesh,
        settings,
    )

    return cloud, mesh


def print_mesh_summary(
    mesh: o3d.geometry.TriangleMesh,
    label: str = "Mesh",
) -> None:
    """Print basic mesh statistics."""
    print(f"\n{label}")
    print(
        f"  vertices:  "
        f"{len(mesh.vertices)}"
    )
    print(
        f"  triangles: "
        f"{len(mesh.triangles)}"
    )

    if len(mesh.vertices) == 0:
        return

    vertices = np.asarray(
        mesh.vertices
    )

    xyz_min = vertices.min(axis=0)
    xyz_max = vertices.max(axis=0)
    span = xyz_max - xyz_min

    print(
        f"  x range: "
        f"{xyz_min[0]:.3f} ... "
        f"{xyz_max[0]:.3f}"
    )
    print(
        f"  y range: "
        f"{xyz_min[1]:.3f} ... "
        f"{xyz_max[1]:.3f}"
    )
    print(
        f"  z range: "
        f"{xyz_min[2]:.3f} ... "
        f"{xyz_max[2]:.3f}"
    )
    print(
        f"  span:    "
        f"{span[0]:.3f}, "
        f"{span[1]:.3f}, "
        f"{span[2]:.3f}"
    )


def save_mesh(
    mesh: o3d.geometry.TriangleMesh,
    output_path: Path | str,
) -> None:
    """Save a triangle mesh."""
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    o3d.io.write_triangle_mesh(
        str(output_path),
        mesh,
    )


def save_prepared_point_cloud(
    cloud: o3d.geometry.PointCloud,
    output_path: Path | str,
) -> None:
    """Save the normal-estimated cloud used for meshing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    o3d.io.write_point_cloud(
        str(output_path),
        cloud,
    )


def visualize_cloud_and_mesh(
    cloud: o3d.geometry.PointCloud,
    mesh: o3d.geometry.TriangleMesh,
    object_name: str,
) -> None:
    """Visualize the prepared registered cloud and generated mesh."""
    print(
        "Opening prepared registered "
        "point cloud window..."
    )

    o3d.visualization.draw_geometries(
        [cloud],
        window_name=(
            f"Prepared registered cloud: "
            f"{object_name}"
        ),
        point_show_normal=False,
    )

    print(
        "Opening registered mesh window..."
    )

    o3d.visualization.draw_geometries(
        [mesh],
        window_name=(
            f"Registered mesh: "
            f"{object_name}"
        ),
        mesh_show_back_face=True,
    )


def find_filtered_registered_point_clouds(
    reconstruction_root: Path | str,
) -> list[Path]:
    """Find filtered registered NPZ point clouds for all objects."""
    reconstruction_root = Path(
        reconstruction_root
    )

    return sorted(
        reconstruction_root.glob(
            "*/registered/"
            "filtered_registered_point_cloud.npz"
        ),
        key=lambda path: natural_sort_key(
            str(path)
        ),
    )


def mesh_all_registered_point_clouds(
    reconstruction_root: Path | str,
    settings: MeshSettings,
    visualize_each: bool = True,
) -> None:
    """Create object-level meshes for all filtered registered clouds."""
    reconstruction_root = Path(
        reconstruction_root
    )

    input_paths = (
        find_filtered_registered_point_clouds(
            reconstruction_root
        )
    )

    if not input_paths:
        raise FileNotFoundError(
            "No filtered_registered_point_cloud.npz "
            f"files found under "
            f"{reconstruction_root}"
        )

    for input_path in input_paths:
        print(
            f"\nProcessing registered cloud: "
            f"{input_path}"
        )

        point_cloud = (
            load_filtered_registered_point_cloud_npz(
                input_path
            )
        )

        cloud, mesh = (
            create_mesh_from_registered_point_cloud(
                point_cloud,
                settings,
            )
        )

        print_mesh_summary(
            mesh,
            label=(
                f"Registered mesh for "
                f"{point_cloud.source_object_name}"
            ),
        )

        registered_root = (
            input_path.parent
        )

        method_name = (
            settings.method
        )

        prepared_cloud_path = (
            registered_root
            / "prepared_registered_point_cloud.ply"
        )

        mesh_ply_path = (
            registered_root
            / f"{method_name}_registered_mesh.ply"
        )

        mesh_obj_path = (
            registered_root
            / f"{method_name}_registered_mesh.obj"
        )

        save_prepared_point_cloud(
            cloud,
            prepared_cloud_path,
        )

        save_mesh(
            mesh,
            mesh_ply_path,
        )

        save_mesh(
            mesh,
            mesh_obj_path,
        )

        print(
            f"Saved prepared registered cloud: "
            f"{prepared_cloud_path}"
        )
        print(
            f"Saved registered mesh PLY: "
            f"{mesh_ply_path}"
        )
        print(
            f"Saved registered mesh OBJ: "
            f"{mesh_obj_path}"
        )

        if visualize_each:
            visualize_cloud_and_mesh(
                cloud=cloud,
                mesh=mesh,
                object_name=(
                    point_cloud.source_object_name
                ),
            )


def main() -> None:
    """VS Code friendly entry point using project_config.yaml."""
    config = load_project_config(
        Path("project_config.yaml")
    )

    project_root = Path(
        config["img_root"]
    )

    run_suffix = make_run_suffix(
        config
    )

    reconstruction_root = (
        project_root
        / (
            config["folders"][
                "reconstruction_results"
            ]
            + run_suffix
        )
    )

    # Coordinate units are the same as in calibration.
    # If checkerboard square size was in millimetres,
    # alpha, radii, normal_radius and voxel_size are also millimetres.
    settings = MeshSettings(
        # Start with alpha.
        # Then compare against ball_pivoting.
        # Poisson is most useful when the registered cloud is dense enough.
        method="alpha",

        # Multipliers of automatically estimated nearest-neighbour spacing.
        alpha_multiplier=20.0,

        ball_pivoting_radius_multipliers=(
            1.5,
            2.5,
            4.0,
        ),

        normal_radius_multiplier=10.0,

        poisson_depth=8,
        poisson_density_quantile=0.02,

        normal_max_nn=30,

        # Usually leave None initially because filtering already has its
        # own optional voxel downsampling step.
        voxel_size=None,

        keep_largest_component=False,
    )

    mesh_all_registered_point_clouds(
        reconstruction_root=(
            reconstruction_root
        ),
        settings=settings,
        visualize_each=True,
    )


if __name__ == "__main__":
    main()
