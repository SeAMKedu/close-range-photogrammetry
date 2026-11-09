"""4K turntable videos of existing dense surfaces and laser-error point clouds.

Blender --background --threads 4 --python create_model_orbits.py -- --kind surface
Blender --background --threads 4 --python create_model_orbits.py -- --kind error
Existing completed outputs are reused only when source/script hashes match.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import export_laser_comparison_blender as visual

OUT = ROOT / 'Animaatiot_4K'
COMPARISONS = ROOT / 'LaserScans/comparison_results/blender'
MODELS = json.loads((COMPARISONS / 'data/manifest.json').read_text(encoding='utf-8'))['models']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coordinates(obj):
    values = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get('co', values)
    return values.reshape(-1, 3).astype(np.float64)


def world_coordinates(obj):
    matrix = np.array(obj.matrix_world)
    return coordinates(obj) @ matrix[:3, :3].T + matrix[:3, 3]


def build(model, kind, preview_only=False):
    name = model['model']
    is_error = kind == 'error'
    folder = OUT / ('poikkeamat' if is_error else 'pinnat')
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / (name + ('_poikkeamat_4K' if is_error else '_pinta_4K'))
    video = ROOT / 'videot' / (stem.name + '.mp4')
    video.parent.mkdir(exist_ok=True)
    source = (COMPARISONS / (name + '_virhevertailu.blend') if is_error else
              ROOT / '2026-10-29/dense_results/blender' / (name + '.blend'))
    fingerprint = {'source': sha(source), 'script': sha(Path(__file__)),
                   'manifest': sha(COMPARISONS / 'data/manifest.json')}
    if not preview_only and stem.with_suffix('.json').exists():
        previous = json.loads(stem.with_suffix('.json').read_text(encoding='utf-8'))
        if (previous.get('fingerprint') == fingerprint and previous.get('render_complete')
                and video.exists() and all(stem.with_suffix(ext).exists() for ext in ('.blend', '.png'))):
            print('ORBIT_REUSED', name, kind, flush=True)
            return
    print('ORBIT_START', name, kind, flush=True)
    bpy.ops.wm.open_mainfile(filepath=str(source))
    bpy.context.preferences.filepaths.save_version = 0
    scene = bpy.data.scenes['01 Poikkeamat'] if is_error else bpy.context.scene
    bpy.context.window.scene = scene
    for other in list(bpy.data.scenes):
        if other != scene:
            bpy.data.scenes.remove(other)
    keep = [o for o in scene.objects if
            (o.get('role') in ('photo_error', 'laser_reference', 'photo_overlay') if is_error
             else o.type == 'MESH')]
    for obj in list(bpy.data.objects):
        if obj not in keep:
            bpy.data.objects.remove(obj, do_unlink=True)
    subject = (next(o for o in keep if o.get('role') == 'photo_error') if is_error else
               next(o for o in keep if len(o.data.polygons)))
    mesh_hashes = {o.name: hashlib.sha256(coordinates(o).tobytes()).hexdigest() for o in keep}
    bpy.context.view_layer.update()
    if is_error:
        reference = next(o for o in keep if o.get('role') == 'laser_reference')
        ref = world_coordinates(reference)
        centre = (ref.min(axis=0) + ref.max(axis=0)) / 2
        points = np.vstack([world_coordinates(subject), ref]) - centre
    else:
        xyz = world_coordinates(subject)
        centre = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
        points = xyz - centre
    for obj in keep:
        obj.location -= Vector(centre)
        obj['animation_translation_mm'] = (-centre).tolist()
        obj.hide_set(obj != subject)
        obj.hide_render = obj != subject
    glyph = .1 if is_error else 0.
    angle = math.radians(40)
    radial = np.linalg.norm(points[:, :2], axis=1)
    max_x = float(radial.max()) + glyph
    max_y = float((radial * math.sin(angle) + np.abs(points[:, 2]) * math.cos(angle)).max()) + glyph
    # Analytic envelope for every orbit angle, not only sampled frames.
    # Keep body right of the legend and above the footer, with a little space.
    height = max(max_x / .51, max_y / .385) * 1.015
    width = height * 16 / 9
    radius = float(np.linalg.norm(points, axis=1).max()) + glyph
    distance = radius * 4
    scene.name = name + (' | Poikkeamat 360' if is_error else ' | Pinta 360')
    scene.render.resolution_x = 3840
    scene.render.resolution_y = 2160
    scene.render.resolution_percentage = 100
    scene.render.fps = 30
    scene.frame_start, scene.frame_end = 1, 360
    scene.world.color = (.15, .17, .19)
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    rig = bpy.data.objects.new('Kameran 360 asteen rata', None)
    scene.collection.objects.link(rig)
    rig.rotation_euler.z = 0
    rig.keyframe_insert(data_path='rotation_euler', index=2, frame=1)
    rig.rotation_euler.z = math.tau
    rig.keyframe_insert(data_path='rotation_euler', index=2, frame=361)
    for layer in rig.animation_data.action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    for key in curve.keyframe_points:
                        key.interpolation = 'LINEAR'
    camera = bpy.data.objects.new('Orbitkamera 40 astetta', bpy.data.cameras.new('Orbitkamera'))
    scene.collection.objects.link(camera)
    camera.parent = rig
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

    variant = name.split('_', 1)[1].replace('_', ' ')
    label(model['object'].upper() + ' | ' + variant, -.84, .43, .034)
    if is_error:
        label('Fotogrammetria / laserreferenssi ' + f"{model['scan_frame']:04d}", -.84, .385, .018)
        if model.get('source_coordinate_conversion'):
            label('Koordinaatiston suunta korjattu', -.84, .345, .016)
        label('Poikkeama (mm)', -.84, .27, .021)
        for index, (text, colour) in enumerate(zip(visual.BAND_LABELS, visual.COLORS)):
            yy = .215 - index * .052
            swatch = visual.rect(hud, -.84 * height, yy * height, .027 * height, .025 * height, -1,
                                  visual.mat('Videon variluokka ' + str(index), tuple(colour)))
            swatch.parent = camera
            label(text, -.795, yy + .002, .018)
        stats = model['forward_photo_to_scan']
        lines = [f'{title}  {visual.number(stats[key])} mm' for title, key in
                 [('Mediaani', 'median_mm'), ('RMSE', 'rmse_mm'), ('P95', 'p95_mm')]]
        label('\n'.join(lines), -.84, -.23, .021)
        label('Etäisyys lähimpään laserpisteeseen; ei kalibroitu ±mittausepävarmuus.', -.84, -.473, .016)
    else:
        label('Fotogrammetrinen 3D-pintamalli', -.84, .385, .018)
        label('Dense-rekonstruktio\nTSDF-pinta', -.84, .25, .022)
        label(f'{len(subject.data.vertices):,} verteksiä\n{len(subject.data.polygons):,} kolmiota'.replace(',', ' '),
              -.84, .13, .018)
        label('Alkuperäinen rekonstruoitu geometria; ei uutta suodatusta tai tasoitusta.', -.84, -.473, .016)
    label('360 asteen kierros  |  katselukulma 40 astetta', -.84, -.43, .019)
    margin = 1.
    # Project the unmodified measurements at every animation frame.
    for frame in range(1, 362):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        matrix = np.array(camera.matrix_world.inverted())
        local = points @ matrix[:3, :3].T + matrix[:3, 3]
        mx = float(np.max(np.abs(local[:, 0]))) + glyph
        my = float(np.max(np.abs(local[:, 1]))) + glyph
        assert mx < width / 2 and my < height / 2, (name, frame)
        assert local[:, 0].min() - glyph > -.53 * height
        assert local[:, 1].min() - glyph > -.40 * height
        assert local[:, 1].max() + glyph < .40 * height
        assert np.all(-local[:, 2] > camera.data.clip_start)
        assert np.all(-local[:, 2] < camera.data.clip_end)
        margin = min(margin, .5 - mx / width, .5 - my / height)
        pos = camera.matrix_world.translation
        elevation = math.degrees(math.atan2(pos.z, math.hypot(pos.x, pos.y)))
        assert abs(elevation - 40) < .001
        forward = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
        assert forward.normalized().dot((-pos).normalized()) > .999999
    for obj in keep:
        assert hashlib.sha256(coordinates(obj).tobytes()).hexdigest() == mesh_hashes[obj.name]
    scene['orbit_elevation_degrees'] = 40
    scene['orbit_duration_seconds'] = 12
    scene['orbit_source_centre_mm'] = centre.tolist()
    scene['orbit_framing_includes_all_points'] = True
    scene['orbit_minimum_frame_edge_margin_fraction'] = margin
    scene.frame_set(1)
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.region_3d.view_perspective = 'CAMERA'
    # Render previews first so layout can be reviewed during the long batch.
    scene.render.image_settings.media_type = 'IMAGE'
    scene.render.image_settings.file_format = 'PNG'
    scene.frame_set(46)
    scene.render.filepath = str(stem.with_suffix('.png'))
    bpy.ops.render.render(write_still=True)
    scene.frame_set(1)
    scene.render.image_settings.media_type = 'VIDEO'
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'HIGH'
    scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
    scene.render.filepath = str(video)
    bpy.ops.wm.save_as_mainfile(filepath=str(stem.with_suffix('.blend')))
    if not preview_only:
        bpy.ops.render.render(animation=True)
    assert sha(source) == fingerprint['source']
    report = {
        'model': name, 'kind': kind, 'fingerprint': fingerprint,
        'render_complete': not preview_only, 'framing_verification': 'passed',
        'frames': 360, 'fps': 30, 'duration_seconds': 12, 'resolution': [3840, 2160],
        'elevation_degrees': 40, 'source_centre_mm': centre.tolist(),
        'minimum_frame_edge_margin_fraction': margin, 'all_geometry_inside_frame': True,
        'camera_always_faces_origin': True, 'linear_rotation_degrees': 360,
        'source': source.relative_to(ROOT).as_posix(), 'source_geometry_unchanged': True,
        'reference_scan': model['scan_frame'] if is_error else None,
        'mesh_coordinate_sha256': mesh_hashes,
        'statistics': model['forward_photo_to_scan'] if is_error else None,
        'video': '../videot/' + video.name,
        'blend': stem.with_suffix('.blend').relative_to(OUT).as_posix(),
    }
    stem.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('ORBIT_COMPLETE', name, kind, flush=True)


parser = argparse.ArgumentParser()
parser.add_argument('--kind', choices=['surface', 'error'], required=True)
parser.add_argument('--models', nargs='+')
parser.add_argument('--preview-only', action='store_true')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
for model in MODELS:
    if not args.models or model['model'] in args.models:
        build(model, args.kind, args.preview_only)
