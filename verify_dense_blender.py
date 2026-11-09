"""Verify delivered .blend files and the eye-icon point-cloud controls."""
from pathlib import Path
import json
import bpy

root = Path(__file__).resolve().parent / '2026-10-29/dense_results'
models = json.loads((root / 'models_report.json').read_text())['models']
checks = []
for model in models:
    path = root / 'blender' / (model['name']+'.blend')
    bpy.ops.wm.open_mainfile(filepath=str(path))
    scene = bpy.context.scene
    visible = [obj for obj in scene.objects if obj.type == 'MESH' and obj.visible_get()]
    assert len(visible) == 1
    assert len(visible[0].data.polygons) == model['surface']['triangles'], (
        model['name'], visible[0].name, len(visible[0].data.polygons), model['surface']['triangles'])
    assert visible[0]['source_frames'] == len(model['frames_used'])
    points = next(obj for obj in scene.objects if obj.type == 'MESH' and obj.get('point_count'))
    assert len(points.data.vertices) == model['fused_points']
    assert not points.hide_viewport and points.hide_get()
    points.hide_set(False)
    assert points.visible_get()
    assert any(modifier.type == 'NODES' for modifier in points.modifiers)
    checks.append({'file': path.name, 'surface_triangles': len(visible[0].data.polygons),
                   'points': len(points.data.vertices), 'point_cloud_eye_toggle_verified': True})
for filename, count in [('dense_threepiece.blend', 5), ('dense_single.blend', 6),
                        ('Sini_ennen_jalkeen.blend', 2), ('Sini_mesh_vertailu.blend', 3)]:
    bpy.ops.wm.open_mainfile(filepath=str(root / 'blender' / filename))
    visible = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH' and obj.visible_get()]
    assert len(visible) == count
(root / 'blender/verification.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
print('DENSE_BLENDER_VERIFIED: 11 individual models, 4 comparison/overview files, eye-icon point toggles.')
