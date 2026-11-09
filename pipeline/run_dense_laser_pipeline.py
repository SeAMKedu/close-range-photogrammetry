"""Run and verify the complete dense experiment from the existing sparse baseline."""

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.paths import PROJECT_ROOT

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT = PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recompute', action='store_true', help='Recompute stereo instead of validating and reusing cached frames')
    parser.add_argument('--blender', type=Path,
                        default=Path('C:/Program Files/Blender Foundation/Blender 5.0/blender.exe'))
    args = parser.parse_args()
    if not args.blender.is_file():
        raise FileNotFoundError('Blender executable not found; pass --blender PATH')
    results = PROJECT / '2026-10-29/dense_results'
    logs = results / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    steps = [
        ('stereo', [sys.executable, '-u', str(PROJECT / 'pipeline' / 'dense_laser_pipeline.py'), '--all'] + ([] if args.recompute else ['--resume'])),
        ('fusion', [sys.executable, '-u', str(PROJECT / 'pipeline' / 'fuse_dense_laser.py'), '--compare-meshes']),
        ('geometry_verification', [sys.executable, str(PROJECT / 'pipeline' / 'verify_dense_laser.py')]),
        ('blender_export', [str(args.blender), '--background', '--python-exit-code', '1', '--python', str(PROJECT / 'pipeline' / 'export_dense_laser_blender.py')]),
        ('blender_verification', [str(args.blender), '--background', '--python-exit-code', '1', '--python', str(PROJECT / 'pipeline' / 'verify_dense_blender.py')]),
    ]
    summary = []
    for name, command in steps:
        started = time.perf_counter()
        print(f'Running {name}...', flush=True)
        with (logs / f'{name}.log').open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT)
        summary.append({'step': name, 'exit_code': result.returncode, 'elapsed_s': time.perf_counter()-started})
        (results / 'pipeline_execution.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {logs / (name+".log")}')
    print(f'Complete: {results / "blender/dense_threepiece.blend"}', flush=True)


if __name__ == '__main__':
    main()
