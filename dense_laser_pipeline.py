"""Calibrated dense stereo, confidence filtering, rigid fusion and meshing.

Coordinates and depth are in millimetres throughout, including Open3D TSDF.
Raw images and earlier sparse results are never overwritten.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from merge_laser_threepiece import object_mask, inside
from reconstruct_sparse_point_cloud import load_project_calibrations, load_camera_poses_from_extrinsics

PROJECT = Path(__file__).resolve().parent
NAMES = ['Basler 96', 'Basler 07', 'Hikrobot']


@dataclass
class Settings:
    block_size: int = 7
    uniqueness_ratio: int = 10
    lr_max_error_px: float = 1.0
    ncc_min: float = 0.50
    ncc_window: int = 11
    texture_std_min: float = 3.0
    tsdf_voxel_mm: float = 0.25
    tsdf_truncation_mm: float = 1.0
    cloud_voxel_mm: float = 0.2


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def cloud_of(points, colors=None):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=float)))
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float))
    return cloud


def quantiles(values):
    values = np.asarray(values)
    return {str(p): float(np.percentile(values, p)) for p in [5, 50, 95]} if len(values) else {}


class Stereo:
    def __init__(self, settings):
        self.settings = settings
        folder = PROJECT / 'calibration_files'
        self.calibrations = load_project_calibrations(folder, NAMES)
        self.poses = load_camera_poses_from_extrinsics(folder / 'extrinsics_result.json', NAMES, 0)
        a, b = [self.calibrations[n] for n in NAMES[:2]]
        pose = self.poses[NAMES[1]]
        self.size = a.image_size
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            a.camera_matrix, a.dist_coeffs, b.camera_matrix, b.dist_coeffs, self.size,
            pose.rotation_ref_to_camera, pose.translation_ref_to_camera, flags=0, alpha=1)
        self.maps = [cv2.initUndistortRectifyMap(c.camera_matrix, c.dist_coeffs, r, p,
                                               self.size, cv2.CV_32FC1)
                     for c, r, p in [(a, self.R1, self.P1), (b, self.R2, self.P2)]]

    def rectify(self, image, index, nearest=False):
        return cv2.remap(image, *self.maps[index], cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)

    def diagnostic(self, left, right, masks, disparity_bounds):
        sift = cv2.SIFT_create(nfeatures=12000)
        kl, dl = sift.detectAndCompute(left, masks[0])
        kr, dr = sift.detectAndCompute(right, masks[1])
        if dl is None or dr is None:
            raise RuntimeError('No object features for epipolar calibration check')
        matcher = cv2.BFMatcher()
        forward = matcher.knnMatch(dl, dr, k=2)
        reverse = matcher.knnMatch(dr, dl, k=2)
        reverse_good = {p[0].queryIdx: p[0].trainIdx for p in reverse
                        if len(p) == 2 and p[0].distance < 0.7*p[1].distance}
        matches = [p[0] for p in forward if len(p) == 2 and p[0].distance < 0.7*p[1].distance
                   and reverse_good.get(p[0].trainIdx) == p[0].queryIdx]
        a = np.array([kl[m.queryIdx].pt for m in matches], dtype=float)
        b = np.array([kr[m.trainIdx].pt for m in matches], dtype=float)
        if len(a) < 8:
            return {'matches': len(a), 'review_required': True}
        disparity = a[:, 0]-b[:, 0]
        plausible = (disparity >= disparity_bounds[0]) & (disparity < disparity_bounds[1])
        errors = np.abs(a[plausible, 1]-b[plausible, 1])
        # Descriptor outliers remain visible in the ungated statistics.
        inliers = errors[errors < 2.]
        if len(inliers) < 8:
            return {'matches': len(matches), 'plausible_disparity_matches': len(errors),
                    'inliers_below_2px': len(inliers), 'vertical_error_px_all': quantiles(errors),
                    'review_required': True, 'reason': 'Too few foreground features for an independent per-frame calibration check'}
        return {'matches': len(matches), 'plausible_disparity_matches': len(errors),
                'vertical_error_px_all': quantiles(errors), 'inliers_below_2px': len(inliers),
                'vertical_error_px_inliers': quantiles(inliers),
                'review_required': bool(np.median(inliers) > 0.5 or len(inliers) < 20)}

    def process(self, root, dataset, frame, output):
        cfg = self.settings
        output.mkdir(parents=True, exist_ok=True)
        paths = [root / frame['source_images'][name] for name in NAMES[:2]]
        images = [cv2.imread(str(p)) for p in paths]
        if any(im is None or im.shape[1::-1] != self.size for im in images):
            raise ValueError('Unreadable image or resolution differs from calibration')
        raw_masks = [object_mask(im)[0] for im in images]
        masks = [self.rectify(m, i, True) for i, m in enumerate(raw_masks)]
        rectified = [self.rectify(im, i) for i, im in enumerate(images)]
        gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in rectified]
        sparse_path = root / 'laser_results' / dataset / f"frame_{frame['frame']}" / '96_to_07' / 'filtered_points.ply'
        sparse = np.asarray(o3d.io.read_point_cloud(str(sparse_path)).points)
        foreground = inside(sparse, raw_masks[0], self.calibrations[NAMES[0]], self.poses[NAMES[0]])
        sparse = sparse[foreground]
        prior_source = 'Basler pair sparse foreground'
        if len(sparse) < 6:
            sparse_path = sparse_path.parent.parent / 'combined_points.ply'
            sparse = np.asarray(o3d.io.read_point_cloud(str(sparse_path)).points)
            foreground = inside(sparse, raw_masks[0], self.calibrations[NAMES[0]], self.poses[NAMES[0]])
            sparse = sparse[foreground]
            prior_source = 'Both calibrated camera pairs sparse foreground (Basler pair alone insufficient)'
            if len(sparse) < 6:
                raise RuntimeError(f'Insufficient sparse foreground depth prior: {dataset}')
        med = np.median(sparse[:, 2])
        mad = np.median(np.abs(sparse[:, 2]-med))
        sparse = sparse[np.abs(sparse[:, 2]-med) < max(20., 6*1.4826*mad)]
        rect_sparse = sparse @ self.R1.T
        d_prior = (self.P1[0, 2]-self.P2[0, 2]) - self.P2[0, 3]/rect_sparse[:, 2]
        minimum = int(np.floor((np.percentile(d_prior, 2)-24)/16)*16)
        maximum = int(np.ceil((np.percentile(d_prior, 98)+24)/16)*16)
        number = max(64, maximum-minimum)
        if number > 384:
            raise RuntimeError('Disparity prior too broad; inspect sparse calibration/foreground')
        diagnostic = self.diagnostic(gray[0], gray[1], masks, (minimum, minimum+number))
        rows = np.arange(0, self.size[1], 80)
        overlay = np.hstack(rectified)
        for y in rows:
            cv2.line(overlay, (0, int(y)), (overlay.shape[1]-1, int(y)), (0, 200, 0), 1)
        cv2.imwrite(str(output / 'rectification_check.jpg'), overlay)
        matcher = cv2.StereoSGBM_create(minDisparity=minimum, numDisparities=number,
            blockSize=cfg.block_size, P1=8*cfg.block_size**2, P2=32*cfg.block_size**2,
            disp12MaxDiff=1, uniquenessRatio=cfg.uniqueness_ratio, speckleWindowSize=80,
            speckleRange=2, preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        right_matcher = cv2.ximgproc.createRightMatcher(matcher)
        disparity = matcher.compute(*gray).astype(np.float32)/16.
        right_disparity = right_matcher.compute(gray[1], gray[0]).astype(np.float32)/16.
        yy, xx = np.indices(disparity.shape, dtype=np.float32)
        xr = xx-disparity
        right_d = cv2.remap(right_disparity, xr, yy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=float(right_matcher.getMinDisparity()-1))
        right_mask = cv2.remap(masks[1], xr, yy, cv2.INTER_NEAREST)
        in_images = (xr >= 0) & (xr < self.size[0]-1)
        valid_d = (disparity > minimum-1) & (right_d > right_matcher.getMinDisparity()-1)
        lr_error = np.abs(disparity+right_d)
        warped = cv2.remap(gray[1].astype(np.float32), xr, yy, cv2.INTER_LINEAR)
        a = gray[0].astype(np.float32)
        kernel = (cfg.ncc_window, cfg.ncc_window)
        mean = lambda im: cv2.boxFilter(im, -1, kernel, normalize=True)
        ma, mb = mean(a), mean(warped)
        va = np.maximum(mean(a*a)-ma*ma, 0.)
        vb = np.maximum(mean(warped*warped)-mb*mb, 0.)
        ncc = (mean(a*warped)-ma*mb)/np.sqrt(np.maximum(va*vb, 1e-6))
        ncc = np.clip(ncc, -1., 1.)
        stages = {}
        valid = (masks[0] > 0) & (right_mask > 0) & in_images & valid_d
        stages['object_and_valid_disparity'] = int(valid.sum())
        valid &= lr_error <= cfg.lr_max_error_px
        stages['left_right_consistent'] = int(valid.sum())
        valid &= (ncc >= cfg.ncc_min) & (va >= cfg.texture_std_min**2) & (vb >= cfg.texture_std_min**2)
        stages['photometrically_supported'] = int(valid.sum())
        xyz_rect = cv2.reprojectImageTo3D(disparity, self.Q)
        depth_min, depth_max = np.percentile(rect_sparse[:, 2], [1, 99]) + [-15., 15.]
        valid &= np.isfinite(xyz_rect).all(axis=2) & (xyz_rect[:, :, 2] >= depth_min) & (xyz_rect[:, :, 2] <= depth_max)
        # Remove tiny disconnected disparity islands without filling missing data.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(valid), connectivity=8)
        keep = np.zeros(count, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 20
        valid &= keep[labels]
        stages['final_valid_depth'] = int(valid.sum())
        if valid.sum() < 100:
            raise RuntimeError(f'Too few reliable dense pixels: {dataset} frame {frame["frame"]}: {stages}')
        depth = np.where(valid, xyz_rect[:, :, 2], 0).astype(np.float32)
        points = xyz_rect[valid] @ self.R1  # rectified camera -> original reference camera
        colors = rectified[0][valid][:, ::-1].astype(float)/255.
        cloud = cloud_of(points, colors)
        o3d.io.write_point_cloud(str(output / 'dense_points.ply'), cloud)
        np.savez_compressed(output / 'dense_frame.npz', depth=depth, color=rectified[0][:, :, ::-1],
                            points=points.astype(np.float32), colors=colors.astype(np.float32),
                            disparity=disparity, valid=valid, object_mask=masks[0], P1=self.P1, R1=self.R1)
        visualization = cv2.applyColorMap(np.uint8(np.clip((disparity-minimum)/number, 0, 1)*255), cv2.COLORMAP_TURBO)
        visualization[~valid] = 0
        cv2.imwrite(str(output / 'disparity_valid.png'), visualization)
        cv2.imwrite(str(output / 'valid_mask.png'), np.uint8(valid)*255)
        cv2.imwrite(str(output / 'object_mask.png'), masks[0])
        confidence = rectified[0].copy()
        confidence[(masks[0] > 0) & ~valid] = (0, 0, 160)
        cv2.imwrite(str(output / 'coverage_overlay.jpg'), confidence)
        # Agreement is a diagnostic, not an independent accuracy measurement.
        nearest = cKDTree(points).query(sparse)[0]
        report = {'dataset': dataset, 'frame': frame['frame'], 'source_images': frame['source_images'],
                  'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  'settings': asdict(cfg), 'calibration_check': diagnostic, 'stages': stages,
                  'disparity_search': [minimum, minimum+number], 'rectified_depth_range_mm': [depth_min, depth_max],
                  'points': len(points), 'object_pixels': int(np.count_nonzero(masks[0])),
                  'coverage_percent': float(100*valid.sum()/np.count_nonzero(masks[0])),
                  'left_right_error_px': quantiles(lr_error[valid]), 'ncc': quantiles(ncc[valid]),
                  'sparse_to_dense_nearest_mm': quantiles(nearest), 'sparse_depth_prior_points': len(sparse),
                  'sparse_depth_prior_source': prior_source}
        dump(output / 'frame_report.json', report)
        return report


def mesh_cleanup(mesh):
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def tsdf_mesh(frames, transforms, settings):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=settings.tsdf_voxel_mm, sdf_trunc=settings.tsdf_truncation_mm,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8, depth_sampling_stride=2)
    for frame, transform in zip(frames, transforms):
        h, w = frame['depth'].shape
        p = frame['P1']
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, p[0, 0], p[1, 1], p[0, 2], p[1, 2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(frame['color'])),
            o3d.geometry.Image(np.ascontiguousarray(frame['depth'])), depth_scale=1., depth_trunc=1000.,
            convert_rgb_to_intensity=False)
        rectification = np.eye(4)
        rectification[:3, :3] = frame['R1']
        extrinsic = rectification @ np.linalg.inv(transform)
        volume.integrate(rgbd, intrinsic, extrinsic)
    return mesh_cleanup(volume.extract_triangle_mesh())


def save_mesh(mesh, folder, stem):
    if not len(mesh.triangles):
        raise RuntimeError(f'Empty mesh: {folder / stem}')
    assert np.isfinite(np.asarray(mesh.vertices)).all()
    o3d.io.write_triangle_mesh(str(folder / f'{stem}.ply'), mesh)
    o3d.io.write_triangle_mesh(str(folder / f'{stem}.obj'), mesh, write_vertex_colors=True)
    return {'vertices': len(mesh.vertices), 'triangles': len(mesh.triangles)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='*', default=['Sini_single'])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = Settings()
    cv2.setNumThreads(4)
    cv2.setRNGSeed(42)
    o3d.utility.random.seed(42)
    root = PROJECT / '2026-10-29'
    output = root / 'dense_results'
    output.mkdir(exist_ok=True)
    old = json.loads((root / 'laser_results' / 'run_report.json').read_text(encoding='utf-8'))
    stereo = Stereo(cfg)
    calibration = {'R1': stereo.R1.tolist(), 'R2': stereo.R2.tolist(),
                   'P1': stereo.P1.tolist(), 'P2': stereo.P2.tolist(), 'Q': stereo.Q.tolist(),
                   'image_size': list(stereo.size), 'unit': 'mm'}
    if args.resume and (output / 'calibration.json').exists():
        cached_calibration = json.loads((output / 'calibration.json').read_text())
        for key in ['R1', 'R2', 'P1', 'P2', 'Q']:
            if not np.allclose(calibration[key], cached_calibration[key], rtol=0, atol=1e-10):
                raise RuntimeError('Calibration changed; rerun without --resume')
    calibration['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (PROJECT / 'calibration_files').glob('*.json')}
    dump(output / 'calibration.json', calibration)
    overall = {'settings': asdict(cfg), 'datasets': []}
    registry = {}
    if (output / 'dense_report.json').exists():
        registry = {d['name']: d for d in json.loads((output / 'dense_report.json').read_text(encoding='utf-8'))['datasets']}
    for dataset in old['datasets']:
        if not args.all and dataset['name'] not in args.datasets:
            continue
        started = time.perf_counter()
        name = dataset['name']
        entry = {'name': name, 'frames': []}
        for frame in dataset['frames']:
            folder = output / name / f"frame_{frame['frame']}"
            report_path = folder / 'frame_report.json'
            if args.resume and report_path.exists():
                stats = json.loads(report_path.read_text(encoding='utf-8'))
                if stats['settings'] != asdict(cfg):
                    raise RuntimeError('Resume settings differ; rerun without --resume')
                for camera in NAMES[:2]:
                    path = root / frame['source_images'][camera]
                    if hashlib.sha256(path.read_bytes()).hexdigest() != stats['source_sha256'][path.name]:
                        raise RuntimeError(f'Source image changed; rerun without --resume: {path}')
            else:
                stats = stereo.process(root, name, frame, folder)
            entry['frames'].append(stats)
            print(name, 'frame', frame['frame'], stats['points'], 'points, coverage', round(stats['coverage_percent'], 1), flush=True)
            # A single-frame surface is useful for assessing stereo independently of registration.
            if not (args.resume and (folder / 'dense_surface.ply').exists() and 'mesh' in stats):
                with np.load(folder / 'dense_frame.npz') as data:
                    frame_data = {k: data[k] for k in data.files}
                mesh = tsdf_mesh([frame_data], [np.eye(4)], cfg)
                stats['mesh'] = save_mesh(mesh, folder, 'dense_surface')
            dump(report_path, stats)
        entry['elapsed_s'] = time.perf_counter()-started
        registry[name] = entry
        overall['datasets'] = [registry[key] for key in sorted(registry)]
        dump(output / 'dense_report.json', overall)


if __name__ == '__main__':
    main()
