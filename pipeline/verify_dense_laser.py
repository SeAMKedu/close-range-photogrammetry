"""Geometry and artifact checks for calibrated dense stereo, using synthetic ground truth."""

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
from pipeline.dense_laser_pipeline import Stereo, Settings, tsdf_mesh, dump

root = PROJECT_ROOT / '2026-10-29/dense_results'
stereo = Stereo(Settings())
# Independent geometry roundtrip, including negative disparities / unequal principal points.
points = np.array([[x, y, z] for x in [-30., 0., 30.] for y in [-20., 0., 20.] for z in [350., 380., 410.]])
rect = points @ stereo.R1.T
homogeneous = np.column_stack([rect, np.ones(len(rect))])
left, right = homogeneous @ stereo.P1.T, homogeneous @ stereo.P2.T
left, right = left[:, :2]/left[:, 2:], right[:, :2]/right[:, 2:]
q_input = np.column_stack([left, left[:, 0]-right[:, 0], np.ones(len(left))])
back = q_input @ stereo.Q.T
recovered = (back[:, :3]/back[:, 3:]) @ stereo.R1
roundtrip_error = float(np.max(np.linalg.norm(points-recovered, axis=1)))
assert roundtrip_error < 1e-8

# Two synthetic depth cameras observe the same known plane after object-pose transforms.
w, h, focal = 64, 48, 600.
p = np.array([[focal, 0, w/2, 0], [0, focal, h/2, 0], [0, 0, 1, 0.]])
rect_rotation = cv2.Rodrigues(np.array([0.04, 0.08, 0.]))[0]
yy, xx = np.indices((h, w))
rays = np.stack([(xx-w/2)/focal, (yy-h/2)/focal, np.ones((h, w))], axis=-1)
frames, transforms = [], []
for translation in [[0., 0., 0.], [2., 4., 1.]]:
    transform = np.eye(4)
    transform[:3, 3] = translation
    rectification = np.eye(4)
    rectification[:3, :3] = rect_rotation
    world_from_camera = np.linalg.inv(rectification @ np.linalg.inv(transform))
    direction = rays @ world_from_camera[:3, :3].T
    depth = ((380.-world_from_camera[2, 3])/direction[:, :, 2]).astype(np.float32)
    frames.append({'depth': depth, 'color': np.full((h, w, 3), 150, np.uint8), 'P1': p, 'R1': rect_rotation})
    transforms.append(transform)
mesh = tsdf_mesh(frames, transforms, Settings(tsdf_voxel_mm=0.5))
assert len(mesh.triangles) > 100
plane_error = float(np.percentile(np.abs(np.asarray(mesh.vertices)[:, 2]-380.), 95))
assert plane_error < 0.3, plane_error

dense = json.loads((root / 'dense_report.json').read_text())
models = json.loads((root / 'models_report.json').read_text())
frame_count, observations = 0, 0
for dataset in dense['datasets']:
    for frame in dataset['frames']:
        folder = root / dataset['name'] / f"frame_{frame['frame']}"
        with np.load(folder / 'dense_frame.npz') as data:
            assert int(np.count_nonzero(data['depth'])) == frame['points']
            assert np.count_nonzero(data['valid']) == frame['points']
            assert np.isfinite(data['points']).all()
            assert (data['points'][:, 2] > 0).all()
        assert frame['stages']['final_valid_depth'] <= frame['stages']['left_right_consistent']
        frame_count += 1
        observations += frame['points']
assert frame_count == 21
assert len(models['models']) == 11
for model in models['models']:
    cloud = o3d.io.read_point_cloud(str(root / model['point_file']))
    surface = o3d.io.read_triangle_mesh(str(root / model['surface_file']))
    assert len(cloud.points) == model['fused_points']
    assert len(surface.triangles) == model['surface']['triangles'] > 0
    assert np.isfinite(np.asarray(surface.vertices)).all()
    triangles = np.sort(np.asarray(surface.triangles), axis=1)
    assert np.all(np.diff(triangles, axis=1) > 0), model['name']
    assert len(np.unique(triangles, axis=0)) == len(triangles), model['name']
    if model['dataset'].endswith('_threepiece'):
        assert model['frames_used'] == [1, 2, 3]
    for frame in model['registration']:
        matrix = np.asarray(frame['transform_to_frame_1'])
        assert np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-7)
        assert np.isclose(np.linalg.det(matrix[:3, :3]), 1.)
        if 'selected' in frame:
            assert min(frame['selected']['forward_fitness'], frame['selected']['reverse_fitness']) >= 0.4

result = {'geometry_roundtrip_max_error_mm': roundtrip_error, 'synthetic_tsdf_plane_p95_error_mm': plane_error,
          'verified_frames': frame_count, 'accepted_depth_observations': observations, 'verified_models': len(models['models'])}
dump(root / 'verification.json', result)
print(json.dumps(result, indent=2))
