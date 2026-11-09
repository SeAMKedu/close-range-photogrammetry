"""Refresh a standalone collection of the current deliverable 3D models.

Pipeline sources stay in place because reconstruction, validation and animation
scripts refer to them. Copies retain their exact bytes and are hash-checked.
"""

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.paths import PROJECT_ROOT

from pathlib import Path
import hashlib
import json
import shutil

ROOT = PROJECT_ROOT
DEST = ROOT / '3D_models'
DENSE = ROOT / '2026-10-29/dense_results'
COMPARISONS = ROOT / 'LaserScans/comparison_results/blender'
models = json.loads((COMPARISONS / 'data/manifest.json').read_text(encoding='utf-8'))['models']
records = []
rows = []


def collect(source, relative):
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copy2(source, target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == source_hash
    records.append({'file': relative.as_posix(), 'source': source.relative_to(ROOT).as_posix(),
                    'sha256': source_hash, 'bytes': target.stat().st_size})
    return relative.as_posix()


for model in models:
    name = model['model']
    variant = 'threepiece' if name.endswith('_threepiece') else 'single'
    surface_dir = Path('surfaces') / variant
    error_dir = Path('error_comparisons') / variant
    blend = collect(DENSE / 'blender' / (name + '.blend'), surface_dir / (name + '.blend'))
    obj = collect(DENSE / 'models' / name / 'dense_fused_surface.obj', surface_dir / (name + '.obj'))
    ply = collect(DENSE / 'models' / name / 'dense_fused_surface.ply', surface_dir / (name + '.ply'))
    collect(DENSE / 'models' / name / 'dense_fused_points.ply', surface_dir / (name + '_points.ply'))
    error = collect(COMPARISONS / (name + '_virhevertailu.blend'), error_dir / (name + '_errors.blend'))
    collect(ROOT / model['files']['ply'], error_dir / (name + '_errors.ply'))
    rows.append(f'| {name} | [Blender]({blend}) · [OBJ]({obj}) · [PLY]({ply}) | [Blender]({error}) |')

overview = collect(COMPARISONS / 'Kaikki_kappaleet_virhevertailu.blend',
                   Path('error_comparisons/All_objects_errors.blend'))
readme = [
    '# Ready-to-use 3D models', '',
    'Current dense reconstructions and laser-error comparisons, grouped into `single` and `threepiece` folders. '
    'Both Piiri single frames are included. Sini error models contain the corrected coordinate-handedness alignment.', '',
    '- `surfaces`: `.blend` for Blender, `.obj` and `.ply` for surface geometry, and `_points.ply` for the original dense point cloud.',
    '- `error_comparisons`: `.blend` with error-colour and overlay scenes; `_errors.ply` contains aligned points and numerical error attributes. Generic PLY viewers may not display custom attributes automatically.',
    f'- [All five fused error models together]({overview}).', '',
    '| Model | Surface model | Error comparison |',
    '| --- | --- | --- |', *rows, '',
    'Coordinates are in millimetres. Surface Blender scenes are centered/leveled for viewing; source OBJ/PLY files retain their original reconstruction coordinates. '
    'Error clouds use scanner coordinates after alignment (including the documented Sini coordinate conversion). '
    'Do not directly overlay differently framed files without their saved transforms.', '',
    'These are byte-identical delivery copies. Pipeline source files remain in their original locations so existing scripts and provenance checks keep working. '
    'The collection is a snapshot; refresh it after rebuilding models with `.venv\\Scripts\\python.exe tools/visualization/collect_3d_models.py`.', '',
    'Videos remain in [videot](../videot/). Animated Blender scenes remain in [Animaatiot_4K](../Animaatiot_4K/). '
    'This folder contains static model files for viewing and sharing.', '',
]
(DEST / 'README.md').write_text('\n'.join(readme), encoding='utf-8')
(DEST / 'manifest.json').write_text(json.dumps({'models':len(models), 'files':records}, indent=2), encoding='utf-8')
print(f'Collected and hash-verified {len(records)} files for {len(models)} model versions in {DEST}')
