"""Estimate one display-only rigid leveling rotation from the flat metal sample."""
from pathlib import Path
import json
import numpy as np
import open3d as o3d

root = Path(__file__).resolve().parent / '2026-10-29' / 'laser_results'
o3d.utility.random.seed(42)
fits = []
for path in sorted(root.glob('Metal_*/frame_*/96_to_07/filtered_points.ply')):
    cloud = o3d.io.read_point_cloud(str(path))
    plane, inliers = cloud.segment_plane(1.5, 3, 5000)
    normal = np.array(plane[:3])
    if normal[2] < 0:
        normal = -normal
    if len(inliers)/len(cloud.points) < 0.7:
        continue
    fits.append({'source': str(path.relative_to(root)), 'normal_camera': normal.tolist(),
                 'inlier_fraction': len(inliers)/len(cloud.points)})
assert len(fits) >= 3, 'Insufficient reliable planar captures'
normal = np.median([f['normal_camera'] for f in fits], axis=0)
normal /= np.linalg.norm(normal)
# Camera-to-Blender axis convention used in the original export.
axis_flip = np.diag([1., -1., -1.])
source = -(axis_flip @ normal)
target = np.array([0., 0., 1.])
cross = np.cross(source, target)
cosine = float(source @ target)
skew = np.array([[0., -cross[2], cross[1]], [cross[2], 0., -cross[0]], [-cross[1], cross[0], 0.]])
rotation = np.eye(3) + skew + skew @ skew / (1. + cosine)
assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
assert np.isclose(np.linalg.det(rotation), 1.)
assert np.allclose(rotation @ source, target, atol=1e-12)
report = {'method': 'Common rigid rotation from median RANSAC normals of metal sample captures',
          'reference_is_estimated_not_surveyed_table': True,
          'normal_camera': normal.tolist(), 'rotation_after_axis_flip': rotation.tolist(),
          'angle_deg': float(np.degrees(np.arccos(cosine))), 'reference_fits': fits,
          'raw_reconstructions_modified': False,
          'verification': {'determinant': float(np.linalg.det(rotation)),
                           'orthogonality_max_error': float(np.max(np.abs(rotation.T @ rotation-np.eye(3))))}}
for fit in fits:
    transformed = rotation @ (-(axis_flip @ np.array(fit['normal_camera'])))
    fit['remaining_tilt_deg'] = float(np.degrees(np.arccos(np.clip(transformed[2], -1, 1))))
(root / 'leveling_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
