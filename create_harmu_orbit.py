"""Render a reproducible 12-second Harmu orbit, preserving measured mesh data.

Run with Blender --background --threads 4 --python create_harmu_orbit.py.
"""
from pathlib import Path
import json
import math
import sys

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import export_laser_comparison_blender as visual

BASE = ROOT / 'LaserScans/comparison_results/blender'
OUT = BASE / 'animations'
OUT.mkdir(exist_ok=True)
STEM = OUT / 'Harmu_360_poikkeamat_4K'
bpy.ops.wm.open_mainfile(filepath=str(BASE / 'Harmu_threepiece_virhevertailu.blend'))
bpy.context.preferences.filepaths.save_version = 0
scene = bpy.data.scenes['01 Poikkeamat']
bpy.context.window.scene = scene
for other in list(bpy.data.scenes):
    if other != scene:
        bpy.data.scenes.remove(other)
for obj in list(bpy.data.objects):
    if obj not in list(scene.objects) or obj.get('role') not in ('photo_error', 'laser_reference', 'photo_overlay'):
        bpy.data.objects.remove(obj, do_unlink=True)

clouds = list(scene.objects)
photo = next(o for o in clouds if o.get('role') == 'photo_error')
laser = next(o for o in clouds if o.get('role') == 'laser_reference')
def vertices(obj):
    arr = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get('co', arr)
    return arr.reshape(-1, 3).astype(np.float64)

# A physical reference sets the centre; all photo outliers still affect framing.
ref = vertices(laser)
centre = (ref.min(axis=0) + ref.max(axis=0)) / 2
points = np.vstack([vertices(photo), ref]) - centre
radius = float(np.linalg.norm(points, axis=1).max()) + .1
for obj in clouds:
    obj.location = -Vector(centre)
    obj['display_offset_mm'] = (-centre).tolist()
    visible = obj == photo
    obj.hide_set(not visible)
    obj.hide_render = not visible

scene.name = 'Harmu | 360 asteen poikkeamavertailu'
scene.render.resolution_x = 3840
scene.render.resolution_y = 2160
scene.render.resolution_percentage = 100
scene.render.fps = 30
scene.frame_start = 1
scene.frame_end = 360
scene.world.color = (.15, .17, .19)
aspect = scene.render.resolution_x / scene.render.resolution_y
# 31% larger on screen than the first preview. Per-frame checks below
# include every point and keep the cloud clear of the legend and footer.
height = 2 * radius * .98
width = height * aspect
distance = radius * 4
rig = bpy.data.objects.new('Kameran 360 asteen rata', None)
scene.collection.objects.link(rig)
rig.rotation_euler.z = 0
rig.keyframe_insert(data_path='rotation_euler', index=2, frame=1)
rig.rotation_euler.z = 2 * math.pi
rig.keyframe_insert(data_path='rotation_euler', index=2, frame=361)
for layer in rig.animation_data.action.layers:
    for strip in layer.strips:
        for bag in strip.channelbags:
            for curve in bag.fcurves:
                for key in curve.keyframe_points:
                    key.interpolation = 'LINEAR'

camera = bpy.data.objects.new('Orbitkamera | 40 astetta', bpy.data.cameras.new('Orbitkamera'))
scene.collection.objects.link(camera)
camera.parent = rig
angle = math.radians(40)
camera.location = (0, -distance * math.cos(angle), distance * math.sin(angle))
camera.rotation_euler = (-camera.location).to_track_quat('-Z', 'Y').to_euler()
camera.data.type = 'ORTHO'
camera.data.ortho_scale = width
camera.data.clip_start = .01
camera.data.clip_end = distance + radius * 3
scene.camera = camera

hud = visual.collection(scene, 'Videon kiinteat selitteet')
white = visual.mat('Videoteksti', tuple(visual.linear('F3F5F7')))
def label(body, x, y, size):
    obj = visual.text(hud, body, (x * height, y * height, -1), size * height, white)
    obj.parent = camera
    return obj

label('HARMU', -.84, .43, .034)
label('Fotogrammetria / laserreferenssi 0003', -.84, .385, .018)
label('Poikkeama (mm)', -.84, .27, .021)
for index, (name, colour) in enumerate(zip(visual.BAND_LABELS, visual.COLORS)):
    yy = .215 - index * .052
    swatch = visual.rect(hud, -.84 * height, yy * height, .027 * height, .025 * height, -1,
                         visual.mat('Videon variluokka ' + str(index), tuple(colour)))
    swatch.parent = camera
    label(name, -.795, yy + .002, .018)
label('Mediaani  0,106 mm\nRMSE       0,407 mm\nP95           0,397 mm', -.84, -.23, .021)
label('360 asteen kierros  |  katselukulma 40 astetta', -.84, -.43, .019)
label('Etäisyys lähimpään laserpisteeseen; ei kalibroitu ±mittausepävarmuus.', -.84, -.473, .016)

# Verify each rendered frame against actual point coordinates plus glyph radius.
margin = 1.0
for frame in range(1, 362):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    matrix = np.array(camera.matrix_world.inverted())
    local = points @ matrix[:3, :3].T + matrix[:3, 3]
    mx = float(np.max(np.abs(local[:, 0]))) + .1
    my = float(np.max(np.abs(local[:, 1]))) + .1
    assert mx < width / 2 and my < height / 2, (frame, mx, my)
    assert np.all(-local[:, 2] > camera.data.clip_start)
    assert np.all(-local[:, 2] < camera.data.clip_end)
    margin = min(margin, .5 - mx / width, .5 - my / height)
    # Left HUD ends at x=-0.53*height; point cloud remains to its right.
    assert local[:, 0].min() - .1 > -.53 * height
    assert local[:, 1].min() - .1 > -.40 * height
    position = camera.matrix_world.translation
    elevation = math.degrees(math.atan2(position.z, math.hypot(position.x, position.y)))
    assert abs(elevation - 40) < .001
    forward = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    assert forward.normalized().dot((-position).normalized()) > .999999

scene['orbit_elevation_degrees'] = 40
scene['orbit_duration_seconds'] = 12
scene['orbit_centre_scanner_mm'] = centre.tolist()
scene['orbit_framing_includes_all_points'] = True
scene['orbit_minimum_frame_edge_margin_fraction'] = margin
scene.frame_set(1)
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces.active.region_3d.view_perspective = 'CAMERA'
scene.render.image_settings.media_type = 'VIDEO'
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'HIGH'
scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
scene.render.filepath = str(STEM.with_suffix('.mp4'))
bpy.ops.wm.save_as_mainfile(filepath=str(STEM.with_suffix('.blend')))
bpy.ops.render.render(animation=True)
scene.render.image_settings.media_type = 'IMAGE'
scene.render.image_settings.file_format = 'PNG'
scene.frame_set(46)
scene.render.filepath = str(STEM.with_suffix('.png'))
bpy.ops.render.render(write_still=True)
report = {
    'status': 'passed', 'frames': 360, 'fps': 30, 'duration_seconds': 12,
    'resolution': [3840, 2160], 'elevation_degrees': 40,
    'linear_screen_size_relative_to_first_preview': 1.28 / .98,
    'centre_scanner_mm': centre.tolist(), 'bounding_radius_mm': radius,
    'minimum_frame_edge_margin_fraction': margin,
    'all_points_and_glyphs_inside_frame': True,
    'camera_always_faces_origin': True, 'linear_rotation_degrees': 360,
    'source': 'Harmu_threepiece_virhevertailu.blend', 'scan_frame': 3,
    'source_mesh_coordinates_unchanged': True,
}
STEM.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('ORBIT_COMPLETE', str(STEM))
