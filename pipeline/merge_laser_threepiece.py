"""Fuse three moving-object captures using silhouettes and calibrated 3D clouds.

Projected dots are stereo features, not temporal landmarks. Use object outlines
to initialize motion in the observed support plane, then combine measured 3D
points before generating a single surface. Raw reconstructions remain intact.
"""

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.paths import PROJECT_ROOT

from pathlib import Path
import json
import cv2
import numpy as np
import open3d as o3d
from pipeline import register_sparse_frames as registration
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from pipeline.reconstruct_sparse_point_cloud import load_project_calibrations, load_camera_poses_from_extrinsics
from pipeline.run_laser_validation import export_cloud, write_json

ROOT = PROJECT_ROOT / '2026-10-29'
RESULTS = ROOT / 'laser_results'
NAMES = ['Basler 96', 'Basler 07', 'Hikrobot']


def object_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(35., min(60., 0.50 * otsu))
    mask = np.uint8(blur > threshold) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    mask[:] = 0
    cv2.drawContours(mask, [contour], -1, 255, -1)
    # Small border allowance retains actual edge observations.
    padded = cv2.dilate(mask, np.ones((17, 17), np.uint8))
    return mask, padded, contour[:, 0, :].astype(float), threshold


def inside(points, mask, calibration, pose):
    rvec, _ = cv2.Rodrigues(pose.rotation_ref_to_camera)
    projected, _ = cv2.projectPoints(points, rvec, pose.translation_ref_to_camera,
                                    calibration.camera_matrix, calibration.dist_coeffs)
    px = np.round(projected[:, 0, :]).astype(int)
    valid = (px[:, 0] >= 0) & (px[:, 0] < mask.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < mask.shape[0])
    result = np.zeros(len(points), dtype=bool)
    result[valid] = mask[px[valid, 1], px[valid, 0]] > 0
    return result


def boundary_in_plane(contour, calibration, normal, plane_offset, basis):
    norm = cv2.undistortPoints(contour.reshape(-1, 1, 2), calibration.camera_matrix,
                                calibration.dist_coeffs)[:, 0, :]
    rays = np.column_stack([norm, np.ones(len(norm))])
    points = rays * (plane_offset / (rays @ normal))[:, None]
    boundary = points @ basis[:, :2]
    # Uniform arc-length sampling avoids dependence on contour pixel density.
    loop = np.vstack([boundary, boundary[:1]])
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))]
    samples = np.linspace(0, lengths[-1], 400, endpoint=False)
    return np.column_stack([np.interp(samples, lengths, loop[:, i]) for i in range(2)])


def fit_outline(source, target):
    src_center, tgt_center = source.mean(axis=0), target.mean(axis=0)
    src, tgt = source-src_center, target-tgt_center
    target_tree = cKDTree(tgt)

    def transformed(params):
        angle, tx, ty = params
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return src @ rot.T + [tx, ty], rot

    def residual(params):
        moved, _ = transformed(params)
        return np.r_[target_tree.query(moved)[0], cKDTree(moved).query(tgt)[0]]

    candidates = []
    for angle in np.deg2rad([-25, -12, 0, 12, 25]):
        fit = least_squares(residual, [angle, 0, 0], loss='soft_l1', f_scale=0.4,
                            bounds=([-np.pi/4, -10, -10], [np.pi/4, 10, 10]), max_nfev=100)
        distances = residual(fit.x)
        score = np.mean(np.minimum(distances, 3.)**2)
        candidates.append((score, fit.x, distances))
    _, params, distances = min(candidates, key=lambda item: (item[0], abs(item[1][0])))
    moved, rotation = transformed(params)
    translation = tgt_center + params[1:] - rotation @ src_center
    return rotation, translation, {'rotation_in_plane_deg': float(np.degrees(params[0])),
                                    'outline_median_mm': float(np.median(distances)),
                                    'outline_p95_mm': float(np.percentile(distances, 95))}


def feature_motion(target, source, target_mask, source_mask, calibration, pose):
    source_points, target_points, _ = registration.build_temporal_3d_correspondences(target, source, 0.7)
    if len(source_points) < 12:
        return None
    valid = inside(source_points, source_mask, calibration, pose)
    valid &= inside(target_points, target_mask, calibration, pose)
    # Stationary background / stationary projected dots are not object landmarks.
    valid &= np.linalg.norm(source_points-target_points, axis=1) > 10.
    source_points, target_points = source_points[valid], target_points[valid]
    if len(source_points) < 12:
        return None
    try:
        transform, inliers, rmse = registration.estimate_rigid_transform_ransac(
            source_points, target_points, 3000, 1.0, 12, random_seed=42)
    except RuntimeError:
        return None
    ratio = float(np.mean(inliers))
    if ratio < 0.6 or rmse > 0.7:
        return None
    # Correspondences must cover an area, rather than a tiny repeated dot cluster.
    spread = np.linalg.svd(source_points[inliers]-source_points[inliers].mean(axis=0), compute_uv=False)
    if spread[1] / np.sqrt(inliers.sum()) < 2.:
        return None
    return transform, {'temporal_correspondences': len(source_points), 'temporal_inliers': int(inliers.sum()),
                       'temporal_inlier_fraction': ratio, 'temporal_rmse_mm': rmse}


def main(root=ROOT):
    root = Path(root).resolve()
    results = root / 'laser_results'
    cv2.setRNGSeed(42)
    o3d.utility.random.seed(42)
    report = json.loads((results / 'run_report.json').read_text(encoding='utf-8'))
    leveling = json.loads((results / 'leveling_report.json').read_text())
    normal = np.array(leveling['normal_camera'])
    axis_x = np.array([1., 0., 0.])
    axis_x -= normal * (normal @ axis_x)
    axis_x /= np.linalg.norm(axis_x)
    basis = np.column_stack([axis_x, np.cross(normal, axis_x), normal])
    calibration_root = PROJECT_ROOT / 'calibration_files'
    calibrations = load_project_calibrations(calibration_root, NAMES)
    poses = load_camera_poses_from_extrinsics(calibration_root / 'extrinsics_result.json', NAMES, 0)
    overall = {'method': 'Foreground temporal 3D feature registration with outline validation; outline-assisted fallback; fusion then one meshing pass',
               'assumption': 'Mostly in-plane object motion; small depth translations estimated from foreground 3D points',
               'limitations': ['Outline registration is approximate for non-planar objects and changing occlusion.',
                              'Rotationally symmetric or sparse objects have uncertain orientation.',
                              'This is rigid registration and fusion, not joint bundle adjustment or dense MVS.'],
               'datasets': []}
    for dataset in report['datasets']:
        if 'threepiece' not in dataset['name']:
            continue
        folder = results / dataset['name'] / 'fused'
        folder.mkdir(exist_ok=True)
        clouds, contours, offsets, frames, feature_clouds, reference_masks = [], [], [], [], [], []
        for row in dataset['frames']:
            masks = {}
            for name in NAMES[:2]:
                image = cv2.imread(str(root / row['source_images'][name]))
                mask, padded, contour, threshold = object_mask(image)
                masks[name] = padded
                cv2.imwrite(str(folder / f"frame_{row['frame']}_{name}_mask.png"), mask)
                if name == NAMES[0]:
                    contours.append(contour)
            cloud = o3d.geometry.PointCloud()
            reference_masks.append(masks[NAMES[0]])
            frame_images = {name: root / row['source_images'][name] for name in NAMES}
            frame_images['__frame_key__'] = Path(str(row['frame']))
            try:
                feature_clouds.append(registration.build_frame_feature_cloud(
                    frame_images, NAMES[0], NAMES[1], calibrations[NAMES[0]], calibrations[NAMES[1]],
                    poses[NAMES[1]], registration.RegistrationSettings()))
            except RuntimeError:
                feature_clouds.append(None)
            before = 0
            for pair, target in [('96_to_07', 'Basler 07'), ('96_to_hikrobot', 'Hikrobot')]:
                path = results / dataset['name'] / f"frame_{row['frame']}" / pair / 'filtered_points.ply'
                if not path.exists():
                    continue
                part = o3d.io.read_point_cloud(str(path))
                points = np.asarray(part.points)
                before += len(points)
                valid = inside(points, masks[NAMES[0]], calibrations[NAMES[0]], poses[NAMES[0]])
                if target in masks:
                    valid &= inside(points, masks[target], calibrations[target], poses[target])
                cloud += part.select_by_index(np.flatnonzero(valid).tolist())
            if len(cloud.points) < 6:
                raise RuntimeError(f"Too few foreground points: {dataset['name']} frame {row['frame']}")
            # Remove extreme depth mismatches without forcing the object onto a plane.
            distances = np.asarray(cloud.points) @ normal
            center = np.median(distances)
            mad = np.median(np.abs(distances-center))
            valid = np.abs(distances-center) < max(5., 6*1.4826*mad)
            cloud = cloud.select_by_index(np.flatnonzero(valid).tolist())
            offsets.append(float(np.median(np.asarray(cloud.points) @ normal)))
            clouds.append(cloud)
            frames.append({'frame': row['frame'], 'before_points': before, 'foreground_points': len(cloud.points)})
        boundaries = [boundary_in_plane(c, calibrations[NAMES[0]], normal, offset, basis)
                      for c, offset in zip(contours, offsets)]
        merged = o3d.geometry.PointCloud()
        aligned_frames = []
        for index, cloud in enumerate(clouds):
            transform = np.eye(4)
            metrics = {}
            method = 'reference'
            if index:
                rotation_2d, translation_2d, metrics = fit_outline(boundaries[index], boundaries[0])
                local_rotation = np.eye(3)
                local_rotation[:2, :2] = rotation_2d
                transform[:3, :3] = basis @ local_rotation @ basis.T
                transform[:3, 3] = basis @ np.r_[translation_2d, offsets[0]-offsets[index]]
                method = 'outline_assisted_rigid'
                if feature_clouds[0] is not None and feature_clouds[index] is not None:
                    candidate = feature_motion(feature_clouds[0], feature_clouds[index], reference_masks[0],
                                               reference_masks[index], calibrations[NAMES[0]], poses[NAMES[0]])
                    if candidate is not None:
                        feature_transform, feature_metrics = candidate
                        boundary_3d = boundaries[index] @ basis[:, :2].T + offsets[index]*normal
                        moved = boundary_3d @ feature_transform[:3, :3].T + feature_transform[:3, 3]
                        moved_2d = moved @ basis[:, :2]
                        distances = np.r_[cKDTree(boundaries[0]).query(moved_2d)[0],
                                          cKDTree(moved_2d).query(boundaries[0])[0]]
                        median = float(np.median(distances))
                        if median <= max(0.8, metrics['outline_median_mm']+0.25):
                            transform = feature_transform
                            metrics.update(feature_metrics, outline_median_mm=median,
                                           outline_p95_mm=float(np.percentile(distances, 95)))
                            method = 'temporal_3d_ransac_outline_validated'
                if metrics['outline_median_mm'] > 2.0:
                    raise RuntimeError(f"Outline registration unreliable: {dataset['name']} frame {index+1}: {metrics}")
            aligned = o3d.geometry.PointCloud(cloud).transform(transform)
            aligned_frames.append(aligned)
            # Color by source pose, preserving provenance in the combined cloud.
            aligned.paint_uniform_color([(0.2, 0.65, 1.), (1., 0.5, 0.15), (0.25, 0.9, 0.4)][index])
            merged += aligned
            frames[index].update(transform_to_frame_1=transform.tolist(), method=method, **metrics)
            o3d.io.write_point_cloud(str(folder / f'frame_{index+1}_aligned.ply'), aligned)
        # Fuse close repeated observations; generate the surface only after merging.
        merged = merged.voxel_down_sample(0.2)
        stats = export_cloud(merged, folder, 'fused_points')
        entry = {'name': dataset['name'], 'frames_used': len(clouds), 'frames': frames, **stats,
                 'review_required': dataset['name'].startswith('Musti') or any(
                     f.get('outline_p95_mm', 0) > 2 and f['method'] == 'outline_assisted_rigid' for f in frames)}
        if not stats.get('surface_triangles', 0):
            raise RuntimeError(f"No surface produced: {dataset['name']}")
        # Verify fusion uses all three captures and no scale or non-rigid deformation.
        for frame in frames:
            matrix = np.array(frame['transform_to_frame_1'])[:3, :3]
            assert np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-8)
            assert np.isclose(np.linalg.det(matrix), 1.)
        assert stats['points'] <= sum(f['foreground_points'] for f in frames)
        write_json(folder / 'fusion_report.json', entry)
        overall['datasets'].append(entry)
        print(json.dumps(entry, indent=2), flush=True)
        write_json(results / 'fusion_report.json', overall)
    report['threepiece_fusion_report'] = 'fusion_report.json'
    write_json(results / 'run_report.json', report)


if __name__ == '__main__':
    main()
