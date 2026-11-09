"""Verify saved Blender files contain one visible fused model per object."""

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.paths import PROJECT_ROOT

from pathlib import Path
import json
import bpy

root = PROJECT_ROOT / '2026-10-29' / 'laser_results'
report = json.loads((root / 'fusion_report.json').read_text(encoding='utf-8'))
verified = []
for dataset in report['datasets']:
    path = root / 'blender_yhdistetty' / f"{dataset['name']}.blend"
    bpy.ops.wm.open_mainfile(filepath=str(path))
    scene = bpy.context.scene
    assert scene.name == dataset['name']
    visible_meshes = [obj for obj in scene.objects if obj.type == 'MESH' and obj.visible_get()]
    assert len(visible_meshes) == 1, (scene.name, [obj.name for obj in visible_meshes])
    mesh = visible_meshes[0]
    assert mesh['source_frames'] == 3
    assert len(mesh.data.polygons) == dataset['surface_triangles']
    verified.append({'file': path.name, 'visible_models': 1, 'source_frames': 3,
                     'triangles': len(mesh.data.polygons)})
for filename in ['laser_pinnat.blend', 'laser_kaikki.blend']:
    bpy.ops.wm.open_mainfile(filepath=str(root / 'blender_yhdistetty' / filename))
    visible = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH' and obj.visible_get()]
    assert len(visible) == 5, (filename, len(visible))
(root / 'blender_yhdistetty' / 'verification.json').write_text(json.dumps(verified, indent=2), encoding='utf-8')
print('VERIFIED: five files, one visible surface each, all three captures used; both overviews have five models.')
