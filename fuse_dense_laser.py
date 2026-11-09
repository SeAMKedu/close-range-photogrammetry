"""Refine moving-object poses with dense geometry and integrate depth maps."""
from pathlib import Path
import argparse
import json
import time

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from dense_laser_pipeline import Settings, cloud_of, dump, mesh_cleanup, quantiles, save_mesh, tsdf_mesh
from merge_laser_threepiece import object_mask, boundary_in_plane, fit_outline
from reconstruct_sparse_point_cloud import load_project_calibrations

ROOT = Path(__file__).resolve().parent / '2026-10-29'
OUTPUT = ROOT / 'dense_results'


def dense_outline_initial(target_data, source_data, target_stats, source_stats):
    normal = np.asarray(json.loads((ROOT / 'laser_results/leveling_report.json').read_text())['normal_camera'])
    axis_x = np.array([1., 0., 0.]) - normal[0]*normal
    axis_x /= np.linalg.norm(axis_x)
    basis = np.column_stack([axis_x, np.cross(normal, axis_x), normal])
    cal = load_project_calibrations(ROOT.parent / 'calibration_files', ['Basler 96'])['Basler 96']
    boundaries, offsets = [], []
    for data, stats in [(target_data, target_stats), (source_data, source_stats)]:
        image = cv2.imread(str(ROOT / stats['source_images']['Basler 96']))
        contour = object_mask(image)[2]
        offset = float(np.median(data['points'] @ normal))
        offsets.append(offset)
        boundaries.append(boundary_in_plane(contour, cal, normal, offset, basis))
    rotation_2d, translation_2d, diagnostics = fit_outline(boundaries[1], boundaries[0])
    local_rotation = np.eye(3)
    local_rotation[:2, :2] = rotation_2d
    transform = np.eye(4)
    transform[:3, :3] = basis @ local_rotation @ basis.T
    transform[:3, 3] = basis @ np.r_[translation_2d, offsets[0]-offsets[1]]
    return transform, diagnostics


def icp_cloud(data):
    cloud = cloud_of(data['points'], data['colors']).voxel_down_sample(0.4)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=40))
    cloud.orient_normals_towards_camera_location(np.zeros(3))
    return cloud


def metrics(source, target, transform):
    forward = o3d.pipelines.registration.evaluate_registration(source, target, 1., transform)
    reverse = o3d.pipelines.registration.evaluate_registration(target, source, 1., np.linalg.inv(transform))
    return {'forward_fitness': forward.fitness, 'reverse_fitness': reverse.fitness,
            'rmse_mm': forward.inlier_rmse}


def refine(source, target, initial, anchored_features):
    before = metrics(source, target, initial)
    candidate = initial.copy()
    estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane(
        o3d.pipelines.registration.TukeyLoss(k=0.5))
    for distance in [2., 1., 0.5]:
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance, candidate, estimator,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
        candidate = result.transformation
    after = metrics(source, target, candidate)
    relative = candidate[:3, :3] @ initial[:3, :3].T
    rotation = float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1))))
    center = np.r_[np.asarray(source.points).mean(axis=0), 1.]
    displacement = float(np.linalg.norm((candidate @ center-initial @ center)[:3]))
    # A low ICP residual on a plane alone does not establish the tangential pose.
    limit = 0.8 if anchored_features else 2.
    accepted = (displacement < limit and rotation < 2. and
                min(after['forward_fitness'], after['reverse_fitness']) >= 0.55 and
                after['rmse_mm'] <= 0.65 and after['rmse_mm'] <= before['rmse_mm']+0.03 and
                min(after['forward_fitness'], after['reverse_fitness']) >=
                min(before['forward_fitness'], before['reverse_fitness'])-0.02)
    selected = candidate if accepted else initial
    quality = after if accepted else before
    return selected, {'initial': before, 'candidate': after, 'refinement_accepted': bool(accepted),
                       'center_correction_mm': displacement, 'rotation_correction_deg': rotation,
                       'selected': quality, 'transform_to_frame_1': selected.tolist(),
                       'review_required': bool(min(quality['forward_fitness'], quality['reverse_fitness']) < 0.65
                                               or quality['rmse_mm'] > 0.5)}


def compare_meshes(cloud, tsdf, folder):
    points = cloud.voxel_down_sample(0.3)
    points.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=40))
    points.orient_normals_towards_camera_location(np.zeros(3))
    spacing = float(np.median(points.compute_nearest_neighbor_distance()))
    tree = cKDTree(np.asarray(cloud.points))
    methods = {'tsdf': tsdf}
    methods['ball_pivoting'] = mesh_cleanup(o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        points, o3d.utility.DoubleVector([1.5*spacing, 2.5*spacing])))
    poisson, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(points, depth=8, n_threads=4)
    unsupported = tree.query(np.asarray(poisson.vertices))[0] > 0.6
    low_density = np.asarray(density) < np.percentile(density, 2)
    poisson.remove_vertices_by_mask(unsupported | low_density)
    methods['poisson_trimmed'] = mesh_cleanup(poisson)
    report = {'note': 'Nearest-observation distance measures support, not independent dimensional accuracy.', 'methods': {}}
    for name, mesh in methods.items():
        if not len(mesh.triangles):
            report['methods'][name] = {'triangles': 0}
            continue
        stats = save_mesh(mesh, folder, name)
        stats['vertex_to_observation_mm'] = quantiles(tree.query(np.asarray(mesh.vertices))[0])
        _, counts, areas = mesh.cluster_connected_triangles()
        stats['connected_components'] = len(counts)
        stats['area_mm2'] = float(sum(areas))
        report['methods'][name] = stats
    dump(folder / 'mesh_comparison.json', report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='*')
    parser.add_argument('--compare-meshes', action='store_true')
    args = parser.parse_args()
    o3d.utility.random.seed(42)
    cfg = Settings()
    dense = json.loads((OUTPUT / 'dense_report.json').read_text(encoding='utf-8'))
    sparse_fusion = json.loads((ROOT / 'laser_results/fusion_report.json').read_text(encoding='utf-8'))
    poses = {entry['name']: entry for entry in sparse_fusion['datasets']}
    # Keep unaffected models when a single dataset is rerun.
    models_by_name = {}
    for path in sorted((OUTPUT / 'models').glob('*/model_report.json')):
        existing = json.loads(path.read_text(encoding='utf-8'))
        models_by_name[existing['name']] = existing
    for dataset in dense['datasets']:
        name = dataset['name']
        if args.datasets and name not in args.datasets:
            continue
        is_sequence = name.endswith('_threepiece')
        groups = [dataset['frames']] if is_sequence else [[frame] for frame in dataset['frames']]
        for group in groups:
            started = time.perf_counter()
            model_name = name if is_sequence or len(dataset['frames']) == 1 else f"{name}_frame_{group[0]['frame']}"
            folder = OUTPUT / 'models' / model_name
            folder.mkdir(parents=True, exist_ok=True)
            frames = []
            for stats in group:
                with np.load(OUTPUT / name / f"frame_{stats['frame']}" / 'dense_frame.npz') as data:
                    frames.append({key: data[key] for key in ['depth', 'color', 'points', 'colors', 'P1', 'R1']})
            transforms = [np.eye(4)]
            registration = [{'frame': group[0]['frame'], 'method': 'reference', 'transform_to_frame_1': np.eye(4).tolist()}]
            if is_sequence:
                target = icp_cloud(frames[0])
                for index, frame in enumerate(frames[1:], 1):
                    prior = next(f for f in poses[name]['frames'] if f['frame'] == group[index]['frame'])
                    initial = np.asarray(prior['transform_to_frame_1'])
                    source = icp_cloud(frame)
                    sparse_initial_metrics = metrics(source, target, initial)
                    initialization = {'method': prior['method'], 'sparse_prior_metrics': sparse_initial_metrics}
                    if min(sparse_initial_metrics['forward_fitness'], sparse_initial_metrics['reverse_fitness']) < 0.5:
                        alternative, outline_stats = dense_outline_initial(frames[0], frame, group[0], group[index])
                        alternative_metrics = metrics(source, target, alternative)
                        if (outline_stats['outline_median_mm'] < 2. and
                            min(alternative_metrics['forward_fitness'], alternative_metrics['reverse_fitness']) >
                            min(sparse_initial_metrics['forward_fitness'], sparse_initial_metrics['reverse_fitness'])+0.1):
                            initial = alternative
                            initialization.update(method='outline_with_dense_depth_prior', outline=outline_stats,
                                                  dense_prior_metrics=alternative_metrics)
                    transform, result = refine(source, target, initial, 'temporal_3d' in prior['method'])
                    result.update(frame=group[index]['frame'], initialization=initialization)
                    if min(result['selected']['forward_fitness'], result['selected']['reverse_fitness']) < 0.4:
                        raise RuntimeError(f'Cannot safely merge frame {group[index]["frame"]} of {name}: insufficient overlap')
                    transforms.append(transform)
                    registration.append(result)
            merged = o3d.geometry.PointCloud()
            for index, (frame, transform) in enumerate(zip(frames, transforms)):
                part = cloud_of(frame['points'], frame['colors']).transform(transform)
                merged += part
                o3d.io.write_point_cloud(str(folder / f'frame_{index+1}_aligned.ply'), part.voxel_down_sample(cfg.cloud_voxel_mm))
            merged = merged.voxel_down_sample(cfg.cloud_voxel_mm)
            o3d.io.write_point_cloud(str(folder / 'dense_fused_points.ply'), merged)
            surface = tsdf_mesh(frames, transforms, cfg)
            surface_stats = save_mesh(surface, folder, 'dense_fused_surface')
            # Surface support diagnostic, evaluated against the accepted observations.
            support = cKDTree(np.asarray(merged.points)).query(np.asarray(surface.vertices))[0]
            model = {'name': model_name, 'dataset': name, 'frames_used': [stats['frame'] for stats in group],
                     'input_depth_points': sum(stats['points'] for stats in group), 'fused_points': len(merged.points),
                     'coverage_percent_by_frame': [stats['coverage_percent'] for stats in group],
                     'surface': surface_stats, 'surface_to_observation_mm': quantiles(support),
                     'registration': registration, 'elapsed_s': time.perf_counter()-started,
                     'review_required': bool(any(item.get('review_required', False) for item in registration) or
                                             any(stats['calibration_check']['review_required'] for stats in group)),
                     'point_file': str((folder / 'dense_fused_points.ply').relative_to(OUTPUT)),
                     'surface_file': str((folder / 'dense_fused_surface.ply').relative_to(OUTPUT))}
            dump(folder / 'model_report.json', model)
            models_by_name[model_name] = model
            dump(OUTPUT / 'models_report.json', {'settings': asdict_config(cfg),
                 'models': [models_by_name[key] for key in sorted(models_by_name)]})
            print(model_name, 'frames', len(group), 'points', len(merged.points), 'triangles', len(surface.triangles),
                  'review', model['review_required'], flush=True)
            if args.compare_meshes and name == 'Sini_single':
                comparison_folder = folder / 'mesh_comparison'
                comparison_folder.mkdir(exist_ok=True)
                compare_meshes(merged, surface, comparison_folder)


def asdict_config(settings):
    from dataclasses import asdict
    return asdict(settings)


if __name__ == '__main__':
    main()
