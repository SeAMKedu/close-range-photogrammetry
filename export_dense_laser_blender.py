"""Blender background export of dense reconstructed surfaces and point clouds."""
import json
import math
from pathlib import Path
import warnings

import bpy
import numpy as np
from mathutils import Vector, Quaternion

ROOT = Path(__file__).resolve().parent / '2026-10-29'
RESULTS = ROOT / 'dense_results'
DEST = RESULTS / 'blender'
DEST.mkdir(exist_ok=True)
LEVELING = np.array(json.loads((ROOT / 'laser_results/leveling_report.json').read_text())['rotation_after_axis_flip'])
AXES = LEVELING @ np.diag([1., -1., -1.])
REPORT = json.loads((RESULTS / 'models_report.json').read_text(encoding='utf-8'))
warnings.filterwarnings('ignore', category=DeprecationWarning)


def setup(name):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.save_version = 0
    scene = bpy.context.scene
    scene.name = name
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 0.001
    scene.unit_settings.length_unit = 'MILLIMETERS'
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.light = 'STUDIO'
    scene.display.shading.studiolight_rotate_z = 0.3
    scene.display.shading.color_type = 'MATERIAL'
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = 'BOTH'
    scene.world = bpy.data.worlds.new('World')
    scene.world.color = (0.035, 0.035, 0.035)
    scene.display.shading.background_type = 'WORLD'
    scene.render.resolution_x = 1800
    scene.render.resolution_y = 1400
    scene.render.resolution_percentage = 100
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'OUTLINER':
                area.spaces.active.show_restrict_column_viewport = True
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                space.clip_end = 10000
                space.shading.type = 'SOLID'
                space.shading.color_type = 'MATERIAL'
                space.shading.show_cavity = True
                space.region_3d.view_rotation = Quaternion((1, 0, 0, 0))
                space.region_3d.view_perspective = 'ORTHO'
    return scene


def material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.)
    return mat


def import_geometry(path, label, center=None):
    bpy.ops.wm.ply_import(filepath=str(path), merge_verts=False, use_scene_unit=False,
                          global_scale=1.0, forward_axis='Y', up_axis='Z')
    obj = bpy.context.object
    obj.name = label
    coords = np.empty(len(obj.data.vertices)*3, dtype=np.float32)
    obj.data.vertices.foreach_get('co', coords)
    coords = coords.reshape(-1, 3)
    if center is None:
        center = np.median(coords, axis=0)
    moved = (coords-center) @ AXES.T
    obj.data.vertices.foreach_set('co', moved.astype(np.float32).ravel())
    obj.data.update()
    if len(obj.data.polygons):
        obj.data.polygons.foreach_set('use_smooth', np.ones(len(obj.data.polygons), dtype=bool))
    return obj, center


def point_display(obj, mat):
    """Instance small glyphs without replacing the actual stored point vertices."""
    group = bpy.data.node_groups.new('Pisteiden naytto - sade 0.10 mm', 'GeometryNodeTree')
    group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    source = group.nodes.new('NodeGroupInput')
    output = group.nodes.new('NodeGroupOutput')
    sphere = group.nodes.new('GeometryNodeMeshIcoSphere')
    sphere.inputs['Radius'].default_value = 0.10
    sphere.inputs['Subdivision' if 'Subdivision' in sphere.inputs else 'Subdivisions'].default_value = 1
    assign = group.nodes.new('GeometryNodeSetMaterial')
    assign.inputs['Material'].default_value = mat
    instance = group.nodes.new('GeometryNodeInstanceOnPoints')
    group.links.new(source.outputs['Geometry'], instance.inputs['Points'])
    group.links.new(sphere.outputs['Mesh'], assign.inputs['Geometry'])
    group.links.new(assign.outputs['Geometry'], instance.inputs['Instance'])
    group.links.new(instance.outputs['Instances'], output.inputs['Geometry'])
    modifier = obj.modifiers.new('Nayta pisteet', 'NODES')
    modifier.node_group = group
    obj['point_display_radius_mm'] = 0.10


def frame_view(objects):
    bpy.context.view_layer.update()
    coords = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    low = Vector(tuple(min(c[i] for c in coords) for i in range(3)))
    high = Vector(tuple(max(c[i] for c in coords) for i in range(3)))
    center = (low+high)/2
    size = max(high.x-low.x, high.y-low.y, high.z-low.z)*1.3
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.region_3d.view_location = center
                area.spaces.active.region_3d.view_distance = size
    return center, size


def render(scene, objects, path, tilted=False):
    center, size = frame_view(objects)
    data = bpy.data.cameras.new('Esikatselukamera')
    camera = bpy.data.objects.new('Esikatselukamera', data)
    scene.collection.objects.link(camera)
    camera.location = center + Vector((0, -500 if tilted else 0, 1000))
    camera.rotation_euler = (center-camera.location).to_track_quat('-Z', 'Y').to_euler()
    data.type = 'ORTHO'
    data.ortho_scale = size*1.2
    data.clip_end = 5000
    scene.camera = camera
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(camera, do_unlink=True)


def add_model(model, offset=(0, 0, 0), with_points=True):
    surface, center = import_geometry(RESULTS / model['surface_file'], model['name']+' - pinta')
    surface.location = offset
    surface.data.materials.append(material('Pinta', (0.63, 0.69, 0.74)))
    surface['source_frames'] = len(model['frames_used'])
    surface['model_name'] = model['name']
    surface['review_required'] = model['review_required']
    surface['method'] = 'Confidence-filtered stereo depth, rigid pose registration, TSDF surface'
    if with_points:
        points, _ = import_geometry(RESULTS / model['point_file'], model['name']+' - pistepilvi (silma: nayta) ', center)
        points.location = offset
        points['source_frames'] = len(model['frames_used'])
        points['point_count'] = model['fused_points']
        point_display(points, material('Pisteet', (0.15, 0.65, 0.95)))
        points.hide_set(True)  # Eye icon only: collection remains enabled.
        points.hide_render = True
    return surface


def add_label(name, location):
    data = bpy.data.curves.new(name, 'FONT')
    data.body = name
    data.size = 5
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location


manifest = []
for model in REPORT['models']:
    scene = setup(model['name'])
    surface = add_model(model)
    frame_view([surface])
    bpy.ops.object.select_all(action='DESELECT')
    surface.select_set(True)
    bpy.context.view_layer.objects.active = surface
    bpy.ops.wm.save_as_mainfile(filepath=str(DEST / (model['name']+'.blend')))
    manifest.append({'name': model['name'], 'frames_used': model['frames_used'],
                     'triangles': model['surface']['triangles'], 'fused_points': model['fused_points']})

for suffix, sequence in [('threepiece', True), ('single', False)]:
    scene = setup('Tiheat pinnat - '+suffix)
    models = [m for m in REPORT['models'] if (len(m['frames_used']) == 3) == sequence]
    surfaces = []
    for index, model in enumerate(models):
        x, y = (index % 2)*160, -(index//2)*110
        surfaces.append(add_model(model, (x, y, 0)))
        add_label(model['name'], (x-60, y-40, 10))
    frame_view(surfaces)
    bpy.ops.wm.save_as_mainfile(filepath=str(DEST / f'dense_{suffix}.blend'))
    render(scene, surfaces, DEST / f'dense_{suffix}.png')
    if sequence:
        render(scene, surfaces, DEST / 'dense_threepiece_oblique.png', tilted=True)

# A direct sparse/dense comparison of the same three-pose dataset.
scene = setup('Sini - harva ja tihea')
model = next(m for m in REPORT['models'] if m['name'] == 'Sini_threepiece')
old_path = ROOT / 'laser_results/Sini_threepiece/fused/fused_points_surface_preview.ply'
old, _ = import_geometry(old_path, 'Aiempi harva pinta')
old.data.materials.append(material('Harva', (0.75, 0.55, 0.3)))
new = add_model(model, (160, 0, 0), with_points=False)
add_label('Aiempi: 468 pistetta', (-60, -40, 10))
add_label(f'Tihea: {model["fused_points"]} pistetta', (100, -40, 10))
frame_view([old, new])
bpy.ops.wm.save_as_mainfile(filepath=str(DEST / 'Sini_ennen_jalkeen.blend'))
render(scene, [old, new], DEST / 'Sini_ennen_jalkeen.png')

comparison_folder = RESULTS / 'models/Sini_single/mesh_comparison'
if (comparison_folder / 'mesh_comparison.json').exists():
    scene = setup('Sini - mesh-menetelmat')
    surfaces = []
    for index, method in enumerate(['tsdf', 'ball_pivoting', 'poisson_trimmed']):
        obj, _ = import_geometry(comparison_folder / f'{method}.ply', method)
        obj.location.x = index*155
        obj.data.materials.append(material(method, (0.63, 0.69, 0.74)))
        surfaces.append(obj)
        add_label(method, (index*155-55, -40, 10))
    frame_view(surfaces)
    bpy.ops.wm.save_as_mainfile(filepath=str(DEST / 'Sini_mesh_vertailu.blend'))
    render(scene, surfaces, DEST / 'Sini_mesh_vertailu.png')

(DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('DENSE_BLENDER_EXPORT_COMPLETE', len(manifest))
