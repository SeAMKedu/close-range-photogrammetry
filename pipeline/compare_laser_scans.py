"""Rigidly compare October photogrammetry to imported Gocator reference scans.

No scale fitting, no reference-based removal of evaluation points, no changes
to source files. Robust/subsampled clouds are used only to estimate alignment.
"""
from __future__ import annotations

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from pipeline.laser_reference_io import load_scan
from pipeline.laser_comparison_plots import plot_comparison
from pipeline.paths import PROJECT_ROOT
from pipeline.source_fingerprints import fingerprints_match

ROOT = PROJECT_ROOT
SCAN_ROOT = ROOT / 'LaserScans'
MODEL_ROOT = ROOT / '2026-10-29/dense_results/models'
OUT = SCAN_ROOT / 'comparison_results'
SCAN_OBJECTS = {0: 'Musti', 1: 'Sini', 2: 'Metal', 3: 'Harmu', 4: 'Harmu', 5: 'Piiri'}
LEVELING_PATH = ROOT/'2026-10-29/laser_results/leveling_report.json'
LEGACY_METHOD_SHA256 = '706735f9d8cbf993f9ef744c7d8b391517b2e6f4aa426324a24ce21fff8422e4'


def cached_method_matches(registration, model_name):
    current = method_fingerprint()
    saved = registration.get('method_fingerprint', {})
    # The original geometric method remains unchanged for every non-Sini model.
    legacy = () if model_name.startswith('Sini_') else (LEGACY_METHOD_SHA256,)
    return fingerprints_match(saved, current, 'pipeline/compare_laser_scans.py',
                              hash_key='method_sha256', legacy_hashes=legacy)


def sini_coordinate_conversion():
    """Fixed feature-supported parity conversion, not a fitted reflection.

    Express a leveled horizontal-axis reversal in native photo coordinates.
    Heights and physical scale are preserved. The acquisition/export origin of
    the convention mismatch is unresolved; this conversion is Sini-specific.
    """
    base = np.asarray(json.loads(LEVELING_PATH.read_text())['rotation_after_axis_flip']) @ np.diag([1.,-1.,-1.])
    conversion = np.eye(4)
    conversion[:3,:3] = base.T @ np.diag([1.,-1.,1.]) @ base
    return conversion


def method_fingerprint():
    return {'method_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'leveling_sha256':hashlib.sha256(LEVELING_PATH.read_bytes()).hexdigest()}


def cloud(points, color=None):
    result = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if color is not None:
        if np.asarray(color).ndim == 1:
            result.paint_uniform_color(color)
        else:
            result.colors = o3d.utility.Vector3dVector(color)
    return result


def apply_transform(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def rigid_fit(source, target):
    a, b = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source-a).T @ (target-b))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = b-rotation @ a
    return transform


def icp_rigid(source, target, initial, gates=(10., 4., 2.), iterations=35):
    """Trimmed point-to-point ICP; trimming affects fitting only."""
    tree = cKDTree(target)
    transform = initial.copy()
    for gate in gates:
        for _ in range(iterations):
            current = apply_transform(source, transform)
            distances, indices = tree.query(current, workers=1)
            keep = distances <= min(gate, float(np.quantile(distances, .90)))
            if keep.sum() < 30:
                break
            delta = rigid_fit(current[keep], target[indices[keep]])
            transform = delta @ transform
            displacement = np.linalg.norm(apply_transform(current[keep], delta)-current[keep],axis=1)
            if float(np.mean(displacement)) < 0.0005:
                break
    return transform


def stats(values):
    values = np.asarray(values)
    if not len(values):
        return {'count': 0}
    return {
        'count': int(len(values)), 'mean_mm': float(np.mean(values)),
        'median_mm': float(np.median(values)),
        'rmse_mm': float(np.sqrt(np.mean(values**2))),
        'p95_mm': float(np.percentile(values,95)),
        'p99_mm': float(np.percentile(values,99)), 'max_mm': float(np.max(values)),
        'within_percent': {str(t): float(100*np.mean(values<=t)) for t in [.25,.5,1.,2.]},
    }


def score_alignment(source, target, transform):
    transformed = apply_transform(source, transform)
    forward = cKDTree(target).query(transformed, workers=1)[0]
    reverse = cKDTree(transformed).query(target, workers=1)[0]
    # Bounded errors prevent a few existing outliers dominating pose estimation.
    # Every point is retained again for final metrics.
    score = .65*np.mean(np.minimum(forward,3.)**2)+.35*np.mean(np.minimum(reverse,3.)**2)
    return {'selection_score':float(score),
            'forward_within_1mm_percent':float(100*np.mean(forward<=1.)),
            'reverse_within_1mm_percent':float(100*np.mean(reverse<=1.)),
            'forward_median_mm':float(np.median(forward)),
            'reverse_median_mm':float(np.median(reverse))}


def register(source, reference):
    level = json.loads(LEVELING_PATH.read_text())
    base_rotation = np.asarray(level['rotation_after_axis_flip']) @ np.diag([1.,-1.,-1.])
    def orientation_tilt(transform):
        up = transform[:3,:3] @ base_rotation.T @ np.array([0.,0.,1.])
        return float(np.degrees(np.arccos(np.clip(up[2],-1,1))))
    source_fit = np.asarray(cloud(source).voxel_down_sample(.8).points)
    target_fit = np.asarray(cloud(reference).voxel_down_sample(.8).points)
    local = source_fit @ base_rotation.T
    # Robust initial centers only, no removal from fitting/evaluation outputs.
    source_center = (np.percentile(local,1,axis=0)+np.percentile(local,99,axis=0))/2
    target_center = (np.percentile(target_fit,1,axis=0)+np.percentile(target_fit,99,axis=0))/2
    source_center[2] = np.median(local[:,2])
    target_center[2] = np.median(target_fit[:,2])
    candidates = []
    for degrees in range(0,360,30):
        theta = np.deg2rad(degrees)
        rz = np.array([[np.cos(theta),-np.sin(theta),0.],
                       [np.sin(theta), np.cos(theta),0.],[0.,0.,1.]])
        initial = np.eye(4)
        initial[:3,:3] = rz @ base_rotation
        initial[:3,3] = target_center-rz @ source_center
        fitted = icp_rigid(source_fit,target_fit,initial)
        # Do not silently rotate a top surface onto the reverse face.
        tilt = orientation_tilt(fitted)
        result = score_alignment(source_fit,target_fit,fitted)
        candidates.append(dict(result,initial_yaw_deg=degrees,top_normal_tilt_deg=tilt,
                               transform=fitted.tolist(),plausible_top_orientation=tilt<25.))
    candidates.sort(key=lambda c:c['selection_score'] if c['plausible_top_orientation'] else float('inf'))
    if not candidates[0]['plausible_top_orientation']:
        raise RuntimeError('No physically plausible top-surface alignment')
    best = np.asarray(candidates[0]['transform'])
    fine_source = np.asarray(cloud(source).voxel_down_sample(.3).points)
    fine_target = np.asarray(cloud(reference).voxel_down_sample(.3).points)
    fine_trial = icp_rigid(fine_source,fine_target,best,gates=(2.,1.),iterations=50)
    fine_accepted = orientation_tilt(fine_trial)<25.
    if fine_accepted:
        best = fine_trial
    # Try local point-to-plane refinement; keep it only if the balanced geometric
    # comparison improves on the same coarse cloud used for candidate selection.
    target_cloud = cloud(fine_target)
    target_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.,max_nn=40))
    estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane(
        o3d.pipelines.registration.TukeyLoss(k=.75))
    trial = o3d.pipelines.registration.registration_icp(
        cloud(fine_source),target_cloud,1.5,best,estimator,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40)).transformation
    before = score_alignment(source_fit,target_fit,best)
    after = score_alignment(source_fit,target_fit,trial)
    refinement_accepted = (orientation_tilt(trial)<25. and
                           after['selection_score'] < before['selection_score'])
    if refinement_accepted:
        best = trial
    assert np.allclose(best[:3,:3].T @ best[:3,:3],np.eye(3),atol=1e-7)
    assert np.isclose(np.linalg.det(best[:3,:3]),1.,atol=1e-7)
    assert orientation_tilt(best)<25.
    # Record distinct rotation alternatives, including symmetric configurations.
    distinct = []
    for c in candidates:
        rotation = np.asarray(c['transform'])[:3,:3] @ best[:3,:3].T
        angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1))))
        if angle > 15 and c['plausible_top_orientation']:
            distinct.append(dict(c,rotation_difference_deg=angle))
    runner = min(distinct,key=lambda c:c['selection_score']) if distinct else None
    best_score = score_alignment(source_fit,target_fit,best)['selection_score']
    symmetry = bool(runner and runner['selection_score'] <= best_score*1.08+.01)
    return best, {'method':'12 yaw initializations; trimmed rigid point-to-point ICP; optional robust point-to-plane refinement',
                  'initialization_voxel_mm':.8,'refinement_voxel_mm':.3,
                  'fitting_trim_fraction':.10,'selection_distance_cap_mm':3.,
                  'scale_fitted':False,'reflection_allowed':False,
                  'fine_point_to_point_accepted':bool(fine_accepted),
                  'final_top_normal_tilt_deg':orientation_tilt(best),
                  'point_to_plane_refinement_accepted':bool(refinement_accepted),
                  'coarse_candidates':candidates,'distinct_rotation_runner_up':runner,
                  'near_equal_rotation_alternative':symmetry,
                  'transform_photo_to_scanner':best.tolist(),
                  'rotation_determinant':float(np.linalg.det(best[:3,:3])),
                  'final_selection_diagnostics':score_alignment(source_fit,target_fit,best)}


def compare(model_path,scan_number,scan,force=False,plots=True):
    name = model_path.parent.name
    key = f'{name}__scan_{scan_number:04d}'
    destination = OUT / key
    destination.mkdir(parents=True,exist_ok=True)
    transform_path = destination/'registration.json'
    started=time.perf_counter()
    original = o3d.io.read_point_cloud(str(model_path))
    source = np.asarray(original.points)
    if not np.isfinite(source).all():
        raise ValueError(f'Nonfinite source points in {model_path}')
    reference=scan['points']
    model_hash=hashlib.sha256(model_path.read_bytes()).hexdigest()
    reference_hash=hashlib.sha256(next(SCAN_ROOT.glob(f'*Frame-{scan_number:04d}*.pcd')).read_bytes()).hexdigest()
    if transform_path.exists() and not force:
        registration=json.loads(transform_path.read_text())
        if registration['model_sha256']!=model_hash or registration['reference_sha256']!=reference_hash:
            raise RuntimeError('Source changed; rerun with --force-registration')
        if not cached_method_matches(registration,name):
            raise RuntimeError('Registration implementation or leveling changed; rerun with --force-registration')
        transform=np.asarray(registration['transform_photo_to_scanner'])
    else:
        if name.startswith('Sini_'):
            conversion = sini_coordinate_conversion()
            rigid,registration = register(apply_transform(source,conversion),reference)
            transform = rigid @ conversion
            registration.update(
                source_coordinate_conversion=conversion.tolist(),
                rigid_transform_after_coordinate_conversion=rigid.tolist(),
                coordinate_conversion_determinant=-1.,
                composite_transform_determinant=float(np.linalg.det(transform[:3,:3])),
                coordinate_conversion_basis='Sini long/narrow and short/wide recess correspondence; fixed horizontal parity reversal in leveled photo coordinates',
                coordinate_conversion_scope='Sini only; acquisition/export origin not established',
                coordinate_conversion_version=1,
                transform_photo_to_scanner=transform.tolist())
        else:
            transform,registration=register(source,reference)
        registration.update(model_sha256=model_hash,reference_sha256=reference_hash,
                            method_fingerprint=method_fingerprint())
        transform_path.write_text(json.dumps(registration,indent=2),encoding='utf-8')
    aligned=apply_transform(source,transform)
    forward=cKDTree(reference).query(aligned,workers=4)[0]
    reverse=cKDTree(aligned).query(reference,workers=4)[0]
    xy_distance,xy_index=cKDTree(reference[:,:2]).query(aligned[:,:2],workers=4)
    signed_z=aligned[:,2]-reference[xy_index,2]
    height={}
    for radius in [.15,.30,.50]:
        supported=xy_distance<=radius
        dz=signed_z[supported]
        height[str(radius)]={'supported_points':int(supported.sum()),
                             'support_percent':float(100*supported.mean()),
                             'excluded_points':int((~supported).sum()),
                             'signed_mean_mm':float(np.mean(dz)) if len(dz) else None,
                             'signed_median_mm':float(np.median(dz)) if len(dz) else None,
                             'absolute_height':stats(np.abs(dz))}
    model_report=json.loads((model_path.parent/'model_report.json').read_text())
    flags=[]
    if 'source_coordinate_conversion' in registration:
        flags.append('Sini coordinate handedness corrected using asymmetric recess correspondence before rigid alignment; raw geometry preserved')
    if model_report.get('review_required'):
        flags.append('Photogrammetry reconstruction already flagged for review')
    if registration['near_equal_rotation_alternative']:
        flags.append('Near-equal orientation alternative; object symmetry limits pose identifiability')
    if name.startswith('Harmu'):
        flags.append('Two Harmu scans retained separately; surface-face identity needs interpretation')
    if height['0.3']['supported_points']<30:
        flags.append('Insufficient XY support for a meaningful height comparison')
    summary={'comparison':key,'object':SCAN_OBJECTS[scan_number], 'model':name,'scan_frame':scan_number,
             'units':'mm','status':'provisional_post_alignment_surface_comparison',
             'pairing_basis':'object shape inspection; Harmu duplicate confirmed by user',
             'source_cloud':str(model_path.relative_to(ROOT)),
             'reference_pcd':str(next(SCAN_ROOT.glob(f'*Frame-{scan_number:04d}*.pcd')).relative_to(ROOT)),
             'photogrammetry_points':len(source),'reference_points':len(reference),
             'evaluated_points_removed':0,
             'forward_photo_to_scan':stats(forward),'reverse_scan_to_photo':stats(reverse),
             'nearest_xy_height_sensitivity':height,'primary_xy_support_radius_mm':.30,
             'flags':flags,'registration':registration,'scanner_metadata':scan['metadata'],
             'interpretation':'All-point nearest-3D distances after rigid surface-based alignment. Reverse distances describe sampled coverage as well as disagreement. These are not calibrated uncertainty bounds or repeatability.',
             'elapsed_seconds':time.perf_counter()-started}
    if 'source_coordinate_conversion' in registration:
        summary['interpretation'] = 'All-point nearest-3D distances after a fixed, feature-supported Sini handedness conversion followed by rigid alignment. No scale fitting or point removal. Acquisition/export origin of the convention mismatch remains unresolved. Not calibrated uncertainty bounds.'
    np.savez_compressed(destination/'distances.npz',photo_aligned=aligned,
                        photo_to_scan_mm=forward,scan_to_photo_mm=reverse,
                        nearest_xy_distance_mm=xy_distance,signed_height_mm=signed_z)
    original.transform(transform)
    o3d.io.write_point_cloud(str(destination/'aligned_photogrammetry.ply'),original)
    import matplotlib
    colors=matplotlib.colormaps['viridis'](np.clip(forward/2.,0,1))[:,:3]
    o3d.io.write_point_cloud(str(destination/'photogrammetry_error_map.ply'),cloud(aligned,colors))
    # Compact true point cloud overlay, not glyph geometry or a smoothed surface.
    overlay=cloud(reference,[.18,.43,.67])+cloud(aligned,[.92,.40,.06])
    o3d.io.write_point_cloud(str(destination/'alignment_overlay.ply'),overlay)
    (destination/'comparison_report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if plots:
        plot_comparison(key,aligned,reference,forward,reverse,destination,summary)
    print(f"{key}: N={len(source)} median={np.median(forward):.3f} RMSE={np.sqrt(np.mean(forward**2)):.3f} P95={np.percentile(forward,95):.3f} mm",flush=True)
    return summary


def write_summary():
    reports=[json.loads(p.read_text()) for p in sorted(OUT.glob('*/comparison_report.json'))]
    (OUT/'all_comparisons.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    columns=['model','scan_frame','photogrammetry_points','reference_points','median_mm','rmse_mm','p95_mm','p99_mm','max_mm','within_1mm_percent','reverse_p95_mm','xy_support_percent','height_rmse_mm','status']
    with (OUT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in reports:
            d=r['forward_photo_to_scan']; h=r['nearest_xy_height_sensitivity']['0.3']
            writer.writerow({**{k:r[k] for k in columns[:4]},**{k:d[k] for k in columns[4:9]},
                             'within_1mm_percent':d['within_percent']['1.0'],
                             'reverse_p95_mm':r['reverse_scan_to_photo']['p95_mm'],
                             'xy_support_percent':h['support_percent'],
                             'height_rmse_mm':h['absolute_height'].get('rmse_mm'),'status':r['status']})
    return reports


def self_check():
    rng=np.random.default_rng(13)
    source=rng.normal(size=(600,3))*[7,4,2]
    angle=.08
    known=np.eye(4);known[:3,:3]=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]
    known[:3,3]=[.4,-.2,.3]
    target=apply_transform(source,known)
    recovered=icp_rigid(source,target,np.eye(4),gates=(3.,1.),iterations=80)
    assert np.max(np.abs(recovered-known))<1e-8
    assert np.isclose(np.linalg.det(recovered[:3,:3]),1.)
    assert np.isclose(stats(np.array([0.,3.,4.]))['rmse_mm'],np.sqrt(25/3))
    print('Known rigid transform and metric checks passed.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-single',action='store_true')
    parser.add_argument('--objects',nargs='*')
    parser.add_argument('--scan-frames',nargs='*',type=int)
    parser.add_argument('--force-registration',action='store_true')
    parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:
        self_check();return
    OUT.mkdir(parents=True,exist_ok=True)
    for number,obj in SCAN_OBJECTS.items():
        if args.objects and obj not in args.objects:continue
        if args.scan_frames and number not in args.scan_frames:continue
        scan=load_scan(next(SCAN_ROOT.glob(f'*Frame-{number:04d}*.pcd')))
        paths=sorted(MODEL_ROOT.glob(f'{obj}_*/dense_fused_points.ply'))
        for p in paths:
            if not args.include_single and not p.parent.name.endswith('_threepiece'):continue
            compare(p,number,scan,args.force_registration,not args.no_plots)
            write_summary()


if __name__=='__main__':main()
