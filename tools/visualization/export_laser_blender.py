"""Run using Blender --background --python tools/visualization/export_laser_blender.py."""

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.paths import PROJECT_ROOT

from pathlib import Path
import json
import math
import sys
import bpy
from mathutils import Vector, Quaternion, Matrix

ROOT = PROJECT_ROOT / '2026-10-29' / 'laser_results'
FUSED = '--fused' in sys.argv
DEST = ROOT / ('blender_yhdistetty' if FUSED else 'blender_oikaistu')
DEST.mkdir(exist_ok=True)
LEVELING_REPORT = json.loads((ROOT / 'leveling_report.json').read_text(encoding='utf-8'))
LEVELING = Matrix(LEVELING_REPORT['rotation_after_axis_flip'])


def setup(scene):
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 0.001
    scene.unit_settings.length_unit = 'MILLIMETERS'
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.light = 'FLAT'
    scene.display.shading.color_type = 'MATERIAL'
    scene.display.shading.background_type = 'WORLD'
    scene.world = bpy.data.worlds.new(scene.name + '_world')
    scene.world.color = (0.025, 0.025, 0.025)
    scene.render.resolution_x = 1800
    scene.render.resolution_y = 1300
    scene.render.resolution_percentage = 100
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                space.shading.type = 'SOLID'
                space.shading.color_type = 'MATERIAL'
                space.clip_end = 10000
                space.region_3d.view_rotation = Quaternion((1, 0, 0, 0))
                space.region_3d.view_perspective = 'ORTHO'


def material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*color, 1)
    shader.inputs['Roughness'].default_value = 0.7
    return mat


def load(path, name, collection, mat, center):
    bpy.ops.wm.ply_import(filepath=str(path))
    obj = bpy.context.object
    obj.name = name
    for parent in list(obj.users_collection):
        parent.objects.unlink(obj)
    collection.objects.link(obj)
    for vertex in obj.data.vertices:
        p = vertex.co - center
        vertex.co = LEVELING @ Vector((p.x, -p.y, -p.z))
    if path.name.endswith('_blender.ply'):
        # Enlarge only display glyphs, leaving their measured centers unchanged.
        for start in range(0, len(obj.data.vertices), 6):
            group = [obj.data.vertices[start+i] for i in range(6)]
            midpoint = sum((v.co for v in group), Vector()) / 6
            for vertex in group:
                vertex.co = midpoint + 2.5 * (vertex.co-midpoint)
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return obj


def frame_view(scene):
    bpy.context.view_layer.update()
    visible = [o for o in scene.objects if o.type == 'MESH' and not o.hide_viewport]
    coords = [o.matrix_world @ Vector(p) for o in visible for p in o.bound_box]
    lo = Vector(tuple(min(p[i] for p in coords) for i in range(3)))
    hi = Vector(tuple(max(p[i] for p in coords) for i in range(3)))
    center = (lo+hi)/2
    distance = max(hi.x-lo.x, hi.y-lo.y)*1.4
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.region_3d.view_location = center
                area.spaces.active.region_3d.view_distance = distance
    return center, distance


bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
overview = bpy.context.scene
overview.name = 'Kaikki esineet'
setup(overview)
surface_overview = bpy.data.scenes.new('Pinnat - kaikki')
setup(surface_overview)
palette = [(0.2,0.65,1.0),(1.0,0.5,0.15),(0.25,0.9,0.4)]
mats = [material(f'Asento {i+1}', color) for i,color in enumerate(palette)]
surface_mat = material('Pintaesikatselu', (0.65,0.7,0.8))
report = json.loads((ROOT / 'run_report.json').read_text(encoding='utf-8'))
fusion_by_name = {}
if FUSED:
    fusion = json.loads((ROOT / 'fusion_report.json').read_text(encoding='utf-8'))
    fusion_by_name = {entry['name']: entry for entry in fusion['datasets']}
    report['datasets'] = [dataset for dataset in report['datasets'] if dataset['name'] in fusion_by_name]
manifest = []
for index, dataset in enumerate(report['datasets']):
    name = dataset['name']
    scene = bpy.data.scenes.new(name)
    bpy.context.window.scene = scene
    setup(scene)
    scene['leveling_angle_deg'] = LEVELING_REPORT['angle_deg']
    scene['leveling_method'] = LEVELING_REPORT['method']
    points_collection = bpy.data.collections.new('Yhdistetty pistepilvi' if FUSED else 'Pisteet - asennot erikseen')
    scene.collection.children.link(points_collection)
    surfaces = bpy.data.collections.new('Pintaesikatselut - kokeellinen')
    scene.collection.children.link(surfaces)
    surfaces.hide_viewport = not FUSED
    surfaces.hide_render = not FUSED
    points_collection.hide_viewport = FUSED
    points_collection.hide_render = FUSED
    center = None
    loaded = []
    rows = [{'frame': 1, 'combined': fusion_by_name[name]}] if FUSED else dataset['frames']
    for row in rows:
        folder = ROOT / name / 'fused' if FUSED else ROOT / name / f"frame_{row['frame']}"
        stem = 'fused_points' if FUSED else 'combined_points'
        path = folder / f'{stem}_blender.ply'
        if not path.exists():
            continue
        if center is None:
            bounds = row['combined']
            # Read the first PLY once to obtain a shared camera-space center.
            bpy.ops.wm.ply_import(filepath=str(path))
            temp = bpy.context.object
            center = sum((v.co for v in temp.data.vertices), Vector()) / len(temp.data.vertices)
            bpy.data.objects.remove(temp, do_unlink=True)
        obj = load(path, f"{name} - yhdistetyt pisteet" if FUSED else f"{name} - asento {row['frame']} - pisteet", points_collection, mats[(row['frame']-1)%3], center)
        obj['point_count'] = row['combined']['points']
        obj['description'] = 'Each octahedron represents one measured sparse point; display radius 0.30 mm.'
        # Show each independent pose separately. Registered candidate is available in another collection.
        loaded.append(obj)
        if FUSED:
            obj['source_frames'] = 3
            scene['source_frames'] = 3
            scene['fusion_review_required'] = fusion_by_name[name]['review_required']
        surface_path = folder / f'{stem}_surface_preview.ply'
        if surface_path.exists():
            surface_obj = load(surface_path, f"{name} - yksi yhdistetty pinta" if FUSED else f"{name} - asento {row['frame']} - pinta", surfaces, surface_mat, center)
            if FUSED:
                surface_obj['source_frames'] = 3
            surface_copy = surface_obj.copy()
            surface_overview.collection.objects.link(surface_copy)
            surface_copy.location = ((index % 2)*160, -(index//2)*115, 0)
    registered_path = ROOT / name / 'registered' / 'experimental_registered_points_blender.ply'
    if registered_path.exists() and not FUSED:
        registered = bpy.data.collections.new('Yhdistetty - kokeellinen ICP')
        scene.collection.children.link(registered)
        registered.hide_viewport = True
        registered.hide_render = True
        load(registered_path, f'{name} - yhdistetty ICP', registered, mats[0], center)
    if not loaded:
        continue
    frame_view(scene)
    # Save just this dataset in its own file, then restore scenes for the overview.
    filepath = DEST / f'{name}.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(filepath))
    for obj in loaded:
        copy = obj.copy()
        copy.data = obj.data
        overview.collection.objects.link(copy)
        copy.location = ((index % 2)*160, -(index//2)*115, 0)
    label_data = bpy.data.curves.new(name + '_label', 'FONT')
    label_data.body = name
    label_data.size = 6
    label = bpy.data.objects.new(name + '_label', label_data)
    overview.collection.objects.link(label)
    label.location = ((index % 2)*160 - 60, -(index//2)*115 - 43, 20)
    surface_overview.collection.objects.link(label.copy())
    manifest.append({'name': name, 'file': filepath.name, 'objects': len(loaded)})

bpy.context.window.scene = overview
center, distance = frame_view(overview)
camera_data = bpy.data.cameras.new('Overview camera')
camera = bpy.data.objects.new('Overview camera', camera_data)
overview.collection.objects.link(camera)
camera.location = center + Vector((0,0,1000))
camera_data.type = 'ORTHO'
camera_data.ortho_scale = distance*1.12
overview.camera = camera
bpy.context.window.scene = surface_overview
surface_center, surface_distance = frame_view(surface_overview)
surface_camera = camera.copy()
surface_camera.data = camera.data.copy()
surface_camera.location = surface_center + Vector((0,0,1000))
surface_camera.data.ortho_scale = surface_distance*1.12
surface_overview.collection.objects.link(surface_camera)
surface_overview.camera = surface_camera
bpy.ops.wm.save_as_mainfile(filepath=str(DEST / 'laser_pinnat.blend'))
if FUSED:
    surface_overview.render.filepath = str(DEST / 'surface_overview.png')
    bpy.ops.render.render(write_still=True)
bpy.context.window.scene = overview
frame_view(overview)
bpy.ops.wm.save_as_mainfile(filepath=str(DEST / 'laser_kaikki.blend'))
overview.render.filepath = str(DEST / 'overview.png')
bpy.ops.render.render(write_still=True)
(DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('BLENDER_EXPORT_COMPLETE', len(manifest))
