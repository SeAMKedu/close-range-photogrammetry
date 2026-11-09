"""Reconstruct timestamped laser captures, preserving every exposure and pair."""
from __future__ import annotations

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.paths import PROJECT_ROOT


import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from pipeline import reconstruct_sparse_point_cloud as sparse
from pipeline.compare_profile_reference import points_to_octahedron_mesh

CAMERAS = {'21578796': 'Basler 96', '21578807': 'Basler 07', '02F21752586': 'Hikrobot'}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def export_cloud(cloud, folder, stem):
    """Export actual points plus visible point glyphs and a local surface preview."""
    folder.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(folder / f'{stem}.ply'), cloud)
    points, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
    if len(points) == 0:
        return {'points': 0}
    glyphs = points_to_octahedron_mesh(points, colors, radius=0.12)
    o3d.io.write_triangle_mesh(str(folder / f'{stem}_blender.ply'), glyphs)
    o3d.io.write_triangle_mesh(str(folder / f'{stem}_blender.obj'), glyphs, write_vertex_colors=True)
    result = {'points': len(points), 'bbox_mm': np.ptp(points, axis=0).tolist()}
    if len(points) >= 30:
        clean = cloud.voxel_down_sample(0.10)
        distances = np.asarray(clean.compute_nearest_neighbor_distance())
        spacing = float(np.median(distances[distances > 0]))
        clean.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(1.0, 4*spacing), max_nn=30))
        clean.orient_normals_towards_camera_location(np.zeros(3))
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            clean, o3d.utility.DoubleVector([1.5*spacing, 2.5*spacing]))
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        if len(mesh.triangles):
            o3d.io.write_triangle_mesh(str(folder / f'{stem}_surface_preview.ply'), mesh)
            o3d.io.write_triangle_mesh(str(folder / f'{stem}_surface_preview.obj'), mesh)
        result.update(surface_triangles=len(mesh.triangles), spacing_mm=spacing)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT / '2026-10-29')
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / 'laser_results'
    calibration_root = PROJECT_ROOT / 'calibration_files'
    names = list(CAMERAS.values())
    calibrations = sparse.load_project_calibrations(calibration_root, names)
    poses = sparse.load_camera_poses_from_extrinsics(calibration_root / 'extrinsics_result.json', names, 0)
    cv2.setRNGSeed(42)
    o3d.utility.random.seed(42)
    report = {'calibration': str(calibration_root), 'calibration_geometry_confirmed': False,
              'calibration_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in calibration_root.glob('*.json')},
              'settings': {'feature': 'SIFT', 'lowe_ratio': 0.75, 'max_reprojection_error_px': 2.0,
                           'statistical_neighbors': 10, 'statistical_std_ratio': 2.0},
              'notes': ['Calibration reuse assumes unchanged camera geometry and focus.',
                        'Point glyph meshes are visualization only, not reconstructed surfaces.',
                        'Surface previews are sparse ball-pivoting approximations, not complete solids.',
                        'Multi-pose ICP is experimental; projected dots are not material-fixed features.',
                        'No dominant-plane filter: objects have real depth.'], 'datasets': []}
    for dataset in sorted(root.iterdir()):
        if not dataset.is_dir() or not list(dataset.glob('*.jpg')):
            continue
        by_camera = {name: [] for name in names}
        for path in sorted(dataset.glob('*.jpg')):
            serial = path.stem.split('_')[2]
            if serial not in CAMERAS:
                raise ValueError(f'Unknown camera: {path}')
            timestamp = datetime.strptime('_'.join(path.stem.split('_')[-2:]), '%Y-%m-%d_%H-%M-%S-%f')
            by_camera[CAMERAS[serial]].append((timestamp, path))
        counts = [len(paths) for paths in by_camera.values()]
        if len(set(counts)) != 1:
            raise ValueError(f'Unbalanced camera images: {dataset}: {counts}')
        entry = {'name': dataset.name, 'frames': [], 'registration': []}
        clouds = []
        for index in range(counts[0]):
            frame = {'__frame_key__': Path(str(index+1))}
            timestamps = []
            for name in names:
                timestamp, path = by_camera[name][index]
                frame[name] = path
                timestamps.append(timestamp)
                image = cv2.imread(str(path))
                if image is None or (image.shape[1], image.shape[0]) != calibrations[name].image_size:
                    raise ValueError(f'Calibration image size mismatch: {path}')
            delta = (max(timestamps)-min(timestamps)).total_seconds()
            if delta > 0.5:
                raise ValueError(f'Unsynchronized capture: {dataset}, frame {index+1}, {delta}s')
            folder = output / dataset.name / f'frame_{index+1}'
            row = {'frame': index+1, 'timestamp_spread_s': delta,
                   'source_images': {name: str(frame[name].relative_to(root)) for name in names}, 'pairs': []}
            combined = o3d.geometry.PointCloud()
            for target in [1, 2]:
                pair_name = f'96_to_{"07" if target == 1 else "hikrobot"}'
                pair_folder = folder / pair_name
                reconstruction = sparse.reconstruct_sparse_point_cloud_for_frame(
                    frame, dataset.name, names, calibrations, poses, 0, [target],
                    'sift', 0.75, 2.0, pair_folder, True)
                if reconstruction is None:
                    row['pairs'].append({'pair': pair_name, 'points': 0})
                    continue
                sparse.save_point_cloud_npz(reconstruction, pair_folder / 'sparse_point_cloud.npz')
                sparse.save_point_cloud_ply(reconstruction, pair_folder / 'sparse_point_cloud.ply')
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(reconstruction.points_xyz.astype(float))
                cloud.colors = o3d.utility.Vector3dVector(reconstruction.colors_rgb.astype(float)/255)
                if len(cloud.points) > 10:
                    cloud, _ = cloud.remove_statistical_outlier(10, 2.0)
                stats = export_cloud(cloud, pair_folder, 'filtered_points')
                stats.update(pair=pair_name, matching=reconstruction.pair_statistics)
                row['pairs'].append(stats)
                combined += cloud
            row['combined'] = export_cloud(combined, folder, 'combined_points')
            write_json(folder / 'frame_report.json', row)
            entry['frames'].append(row)
            clouds.append(combined)
            print(f'{dataset.name} frame {index+1}: {len(combined.points)} filtered points', flush=True)
        # Preserve independent singles; only merge explicitly named three-pose captures.
        if 'threepiece' in dataset.name and all(len(c.points) >= 30 for c in clouds):
            target = clouds[0]
            merged = o3d.geometry.PointCloud(target)
            accepted = 1
            for index, source in enumerate(clouds[1:], 2):
                initial = np.eye(4)
                initial[:3, 3] = np.median(np.asarray(target.points), axis=0)-np.median(np.asarray(source.points), axis=0)
                icp = o3d.pipelines.registration.registration_icp(source, target, 3.0, initial,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
                reverse = o3d.pipelines.registration.evaluate_registration(target, source, 3.0, np.linalg.inv(icp.transformation))
                ok = icp.fitness >= 0.65 and reverse.fitness >= 0.65 and icp.inlier_rmse <= 1.5
                entry['registration'].append({'frame': index, 'fitness': icp.fitness,
                    'reverse_fitness': reverse.fitness, 'rmse_mm': icp.inlier_rmse,
                    'accepted_numerically': ok, 'transform_to_frame_1': icp.transformation.tolist()})
                if ok:
                    aligned = o3d.geometry.PointCloud(source).transform(icp.transformation)
                    merged += aligned
                    accepted += 1
            entry['registered_frames'] = accepted
            if accepted > 1:
                entry['experimental_merged'] = export_cloud(merged, output / dataset.name / 'registered', 'experimental_registered_points')
        report['datasets'].append(entry)
        write_json(output / 'run_report.json', report)
    print(f'Completed: {output}', flush=True)


if __name__ == '__main__':
    main()
