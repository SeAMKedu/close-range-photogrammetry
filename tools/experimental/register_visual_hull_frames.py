from __future__ import annotations

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.paths import PROJECT_ROOT


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

from pipeline.project_config import load_project_config, make_run_suffix


@dataclass
class VisualHullRegistrationSettings:
    """Settings for merging registered visual-hull frame surfaces."""

    use_surface_voxels: bool = True
    voxel_downsample_size: float | None = None
    save_individual_registered_frames: bool = False


def natural_sort_key(text: str) -> list[int | str]:
    """Sort strings so that 1, 2, 10 are ordered numerically."""
    parts = re.split(r"([0-9]+)", text)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def load_frame_transforms(
    path: Path | str,
) -> dict[str, np.ndarray]:
    """
    Load frame-to-reference transforms from frame_transforms.json.

    The transform for each frame maps points from that frame's native
    reference-camera coordinate system into the common registration frame.
    """
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
    use_surface_voxels: bool,
) -> np.ndarray:
    """Load surface or occupied visual-hull voxel centers from an NPZ file."""
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

    transformed = (
        transform @ homogeneous.T
    ).T[:, :3]

    return transformed


def numpy_points_to_open3d(
    points_xyz: np.ndarray,
) -> o3d.geometry.PointCloud:
    """Convert XYZ points to an Open3D point cloud."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        points_xyz.astype(np.float64)
    )
    return cloud


def save_xyz_npz(
    points_xyz: np.ndarray,
    output_path: Path | str,
) -> None:
    """Save XYZ points as compressed NPZ."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        points_xyz=points_xyz.astype(np.float32),
    )


def save_xyz_ply(
    points_xyz: np.ndarray,
    output_path: Path | str,
) -> None:
    """Save XYZ points as PLY using Open3D."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cloud = numpy_points_to_open3d(points_xyz)
    o3d.io.write_point_cloud(str(output_path), cloud)


def register_visual_hull_frames_for_object(
    object_results_root: Path | str,
    settings: VisualHullRegistrationSettings,
) -> None:
    """
    Transform and merge frame-specific visual hulls into a common frame.

    Uses the transforms previously estimated by register_sparse_frames.py.
    """
    object_results_root = Path(object_results_root)
    registered_root = object_results_root / "registered"

    transforms_path = (
        registered_root / "frame_transforms.json"
    )

    if not transforms_path.exists():
        raise FileNotFoundError(
            f"Missing frame transforms: {transforms_path}. "
            "Run pipeline/register_sparse_frames.py first."
        )

    transforms = load_frame_transforms(transforms_path)

    frame_dirs = sorted(
        [
            path
            for path in object_results_root.iterdir()
            if path.is_dir()
            and path.name.startswith("frame_")
        ],
        key=lambda path: natural_sort_key(path.name),
    )

    if not frame_dirs:
        raise FileNotFoundError(
            f"No frame_* directories found under "
            f"{object_results_root}"
        )

    transformed_frame_points: list[np.ndarray] = []
    used_frame_keys: list[str] = []
    skipped_frames: dict[str, str] = {}

    for frame_dir in frame_dirs:
        frame_key = frame_dir.name.removeprefix("frame_")

        if frame_key not in transforms:
            skipped_frames[frame_key] = (
                "No transform in frame_transforms.json"
            )
            print(
                f"[WARN] Skipping frame {frame_key}: "
                "no registration transform."
            )
            continue

        visual_hull_path = (
            frame_dir / "visual_hull.npz"
        )

        if not visual_hull_path.exists():
            skipped_frames[frame_key] = (
                f"Missing {visual_hull_path.name}"
            )
            print(
                f"[WARN] Skipping frame {frame_key}: "
                f"{visual_hull_path} not found."
            )
            continue

        points = load_visual_hull_points(
            visual_hull_path,
            use_surface_voxels=(
                settings.use_surface_voxels
            ),
        )

        if len(points) == 0:
            skipped_frames[frame_key] = (
                "Visual hull contains no points"
            )
            print(
                f"[WARN] Skipping frame {frame_key}: "
                "visual hull is empty."
            )
            continue

        transformed = transform_points(
            points,
            transforms[frame_key],
        )

        transformed_frame_points.append(
            transformed
        )
        used_frame_keys.append(frame_key)

        print(
            f"frame={frame_key}: "
            f"{len(points)} visual-hull points transformed"
        )

        if settings.save_individual_registered_frames:
            individual_dir = (
                registered_root
                / "visual_hull_frames"
                / f"frame_{frame_key}"
            )

            save_xyz_ply(
                transformed,
                individual_dir
                / "registered_visual_hull.ply",
            )
            save_xyz_npz(
                transformed,
                individual_dir
                / "registered_visual_hull.npz",
            )

    if not transformed_frame_points:
        raise RuntimeError(
            f"No visual hull frames could be registered for "
            f"{object_results_root.name}"
        )

    merged_points = np.vstack(
        transformed_frame_points
    ).astype(np.float64)

    points_before_downsampling = len(
        merged_points
    )

    if (
        settings.voxel_downsample_size is not None
        and settings.voxel_downsample_size > 0
    ):
        merged_cloud = numpy_points_to_open3d(
            merged_points
        )

        merged_cloud = (
            merged_cloud.voxel_down_sample(
                settings.voxel_downsample_size
            )
        )

        merged_points = np.asarray(
            merged_cloud.points,
            dtype=np.float64,
        )

    registered_ply_path = (
        registered_root
        / "registered_visual_hull_surface.ply"
    )
    registered_npz_path = (
        registered_root
        / "registered_visual_hull_surface.npz"
    )
    report_path = (
        registered_root
        / "visual_hull_registration_report.json"
    )

    save_xyz_ply(
        merged_points,
        registered_ply_path,
    )
    save_xyz_npz(
        merged_points,
        registered_npz_path,
    )

    report = {
        "object_name": object_results_root.name,
        "source": (
            "surface_voxel_centers"
            if settings.use_surface_voxels
            else "occupied_voxel_centers"
        ),
        "used_frame_keys": used_frame_keys,
        "used_frame_count": len(
            used_frame_keys
        ),
        "skipped_frames": skipped_frames,
        "points_before_downsampling": int(
            points_before_downsampling
        ),
        "points_after_downsampling": int(
            len(merged_points)
        ),
        "voxel_downsample_size": (
            None
            if settings.voxel_downsample_size is None
            else float(
                settings.voxel_downsample_size
            )
        ),
        "frame_transforms_file": (
            str(transforms_path)
        ),
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print(
        f"Saved registered visual hull PLY: "
        f"{registered_ply_path}"
    )
    print(
        f"Saved registered visual hull NPZ: "
        f"{registered_npz_path}"
    )
    print(
        f"Used frames: {len(used_frame_keys)}"
    )
    print(
        f"Points: {points_before_downsampling} "
        f"-> {len(merged_points)}"
    )


def register_all_visual_hulls(
    reconstruction_root: Path | str,
    settings: VisualHullRegistrationSettings,
) -> None:
    """Register visual hull frames for every object under reconstruction_root."""
    reconstruction_root = Path(
        reconstruction_root
    )

    object_dirs = sorted(
        [
            path
            for path in reconstruction_root.iterdir()
            if path.is_dir()
            and not path.name.startswith(".")
        ],
        key=lambda path: natural_sort_key(
            path.name
        ),
    )

    if not object_dirs:
        raise FileNotFoundError(
            f"No object result directories found under "
            f"{reconstruction_root}"
        )

    for object_dir in object_dirs:
        print(
            f"\n=== Registering visual hulls for "
            f"{object_dir.name} ==="
        )

        try:
            register_visual_hull_frames_for_object(
                object_results_root=object_dir,
                settings=settings,
            )
        except Exception as exc:
            print(
                f"[WARN] Visual hull registration failed for "
                f"{object_dir.name}: {exc}"
            )


def main() -> None:
    """VS Code friendly entry point using project_config.yaml."""
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

    # Optional project_config.yaml block:
    #
    # visual_hull_registration:
    #   use_surface_voxels: true
    #   voxel_downsample_size: 1.0
    #   save_individual_registered_frames: false
    #
    # All settings also have defaults, so the block is optional.
    registration_config = config.get(
        "visual_hull_registration",
        {},
    )

    settings = VisualHullRegistrationSettings(
        use_surface_voxels=bool(
            registration_config.get(
                "use_surface_voxels",
                True,
            )
        ),
        voxel_downsample_size=(
            None
            if registration_config.get(
                "voxel_downsample_size"
            )
            is None
            else float(
                registration_config[
                    "voxel_downsample_size"
                ]
            )
        ),
        save_individual_registered_frames=bool(
            registration_config.get(
                "save_individual_registered_frames",
                False,
            )
        ),
    )

    register_all_visual_hulls(
        reconstruction_root=(
            reconstruction_root
        ),
        settings=settings,
    )


if __name__ == "__main__":
    main()
