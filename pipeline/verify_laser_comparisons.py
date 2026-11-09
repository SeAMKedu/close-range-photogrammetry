"""Independently verify exported laser comparisons without rerunning registration.

Checks original hashes, proper rigid transforms, every exported point/distance,
both directions of the reported metrics, XY-support sensitivity and summary
tables. This verifies computation/provenance, not metrological accuracy or pose
identifiability. --partial skips missing/stale reports during an active run.
"""
from __future__ import annotations

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from pipeline.laser_reference_io import load_scan
from pipeline.paths import PROJECT_ROOT
from pipeline.source_fingerprints import fingerprints_match

ROOT = PROJECT_ROOT
OUT = ROOT / 'LaserScans/comparison_results'
MODELS = ROOT / '2026-10-29/dense_results/models'
LEVEL = ROOT / '2026-10-29/laser_results/leveling_report.json'
SCAN_OBJECTS = {0: 'Musti', 1: 'Sini', 2: 'Metal', 3: 'Harmu', 4: 'Harmu', 5: 'Piiri'}
ATOL_MM = 1e-8


def fingerprint_matches(registration, model, fingerprint):
    saved = registration.get('method_fingerprint')
    legacy = (() if model.startswith('Sini_') else
              ('706735f9d8cbf993f9ef744c7d8b391517b2e6f4aa426324a24ce21fff8422e4',))
    return fingerprints_match(saved, fingerprint, 'pipeline/compare_laser_scans.py',
                              hash_key='method_sha256', legacy_hashes=legacy)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, label, atol=ATOL_MM):
    if expected is None:
        require(actual is None, f'{label}: expected null, got {actual}')
    else:
        require(np.allclose(actual, expected, rtol=1e-9, atol=atol),
                f'{label}: {actual} != {expected}')


def recompute_stats(values):
    """Independent evaluation, deliberately not importing the comparison code."""
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {'count': 0}
    return {
        'count': int(a.size), 'mean_mm': float(a.sum()/a.size),
        'median_mm': float(np.quantile(a, .5)),
        'rmse_mm': float(np.linalg.norm(a)/np.sqrt(a.size)),
        'p95_mm': float(np.quantile(a, .95)),
        'p99_mm': float(np.quantile(a, .99)), 'max_mm': float(a.max()),
        'within_percent': {str(t): float(np.count_nonzero(a <= t)*100/a.size)
                           for t in (.25, .5, 1., 2.)},
    }


def check_stats(saved, calculated, label):
    require(set(saved) == set(calculated), f'{label}: statistic fields differ')
    for key, value in calculated.items():
        if key == 'within_percent':
            require(set(saved[key]) == set(value), f'{label}: threshold fields differ')
            for threshold, percent in value.items():
                close(saved[key][threshold], percent, f'{label}/{threshold}')
        else:
            close(saved[key], value, f'{label}/{key}')


def verify_one(path, report, scan, fingerprint):
    registration = report['registration']
    disk_registration = json.loads((path.parent/'registration.json').read_text())
    require(registration == disk_registration, f'{path.parent.name}: registration copies differ')
    require(fingerprint_matches(registration, report['model'], fingerprint), 'Stale method/leveling fingerprint')
    source_path = ROOT / report['source_cloud']
    reference_path = ROOT / report['reference_pcd']
    require(digest(source_path) == registration['model_sha256'], 'Photogrammetry source changed')
    require(digest(reference_path) == registration['reference_sha256'], 'Reference PCD changed')
    require(report['scanner_metadata']['source_sha256'] == scan['metadata']['source_sha256'],
            'Reference PCD/CSV/SUR/SRF hashes changed')
    require(report['units'] == scan['metadata']['unit'] == 'mm', 'Units are not millimetres')
    require(registration['scale_fitted'] is False and registration['reflection_allowed'] is False,
            'Registration unexpectedly permits scaling/reflection')
    require(report['evaluated_points_removed'] == 0, 'Evaluation points were removed')

    source = np.asarray(o3d.io.read_point_cloud(str(source_path)).points)
    aligned = np.asarray(o3d.io.read_point_cloud(str(path.parent/'aligned_photogrammetry.ply')).points)
    reference = scan['points']
    require(source.shape == aligned.shape and len(source) == report['photogrammetry_points'],
            'Source/export/report point counts differ')
    require(len(reference) == report['reference_points'], 'Reference point count differs')
    require(np.isfinite(source).all() and np.isfinite(aligned).all(), 'Nonfinite photo coordinates')
    transform = np.asarray(registration['transform_photo_to_scanner'])
    require(transform.shape == (4,4) and np.isfinite(transform).all(), 'Invalid transform shape/values')
    close(transform[3], [0,0,0,1], 'Transform bottom row')
    rotation = transform[:3,:3]
    close(rotation.T @ rotation, np.eye(3), 'Rotation orthogonality')
    conversion = np.eye(4)
    if report['object'] == 'Sini':
        require(registration.get('coordinate_conversion_version') == 1, 'Sini parity conversion missing')
        conversion = np.asarray(registration['source_coordinate_conversion'])
        base = np.asarray(json.loads(LEVEL.read_text())['rotation_after_axis_flip']) @ np.diag([1.,-1.,-1.])
        expected_conversion = np.eye(4)
        expected_conversion[:3,:3] = base.T @ np.diag([1.,-1.,1.]) @ base
        close(conversion, expected_conversion, 'Fixed Sini coordinate conversion')
        rigid = np.asarray(registration['rigid_transform_after_coordinate_conversion'])
        close(rigid @ conversion, transform, 'Composition of conversion and rigid fit')
        close(np.linalg.det(rigid[:3,:3]), 1., 'Fitted rigid determinant')
        close(np.linalg.det(rotation), -1., 'Declared composite parity')
    else:
        require('source_coordinate_conversion' not in registration, 'Unexpected conversion on another object')
        close(np.linalg.det(rotation), 1., 'Rotation determinant')
    close(np.linalg.svd(rotation)[1], np.ones(3), 'Unit singular values: no scale')
    reconstructed = (rotation @ source.T).T + transform[:3,3]
    require(np.max(np.abs(reconstructed-aligned)) <= ATOL_MM,
            'Exported points disagree with original points under saved transform')
    level = np.asarray(json.loads(LEVEL.read_text())['rotation_after_axis_flip']) @ np.diag([1.,-1.,-1.])
    up = (rotation @ conversion[:3,:3].T) @ level.T @ np.array([0.,0.,1.])
    tilt = float(np.rad2deg(np.arccos(np.clip(up[2], -1, 1))))
    require(tilt < 25., f'Final transform violates top-surface orientation: {tilt} degrees')
    close(registration['final_top_normal_tilt_deg'], tilt, 'Final tilt')

    # Distances are recomputed from the saved PLY, not the in-memory fitting cloud.
    forward = cKDTree(reference).query(aligned, workers=2)[0]
    reverse = cKDTree(aligned).query(reference, workers=2)[0]
    xy, indices = cKDTree(reference[:,:2]).query(aligned[:,:2], workers=2)
    dz = aligned[:,2] - reference[indices,2]
    forward_stats, reverse_stats = recompute_stats(forward), recompute_stats(reverse)
    check_stats(report['forward_photo_to_scan'], forward_stats, 'Forward statistics')
    check_stats(report['reverse_scan_to_photo'], reverse_stats, 'Reverse statistics')
    with np.load(path.parent/'distances.npz') as archive:
        expected = {'photo_aligned': aligned, 'photo_to_scan_mm': forward,
                    'scan_to_photo_mm': reverse, 'nearest_xy_distance_mm': xy,
                    'signed_height_mm': dz}
        for name, values in expected.items():
            require(archive[name].shape == values.shape, f'NPZ shape mismatch: {name}')
            require(np.max(np.abs(archive[name]-values)) <= ATOL_MM,
                    f'NPZ values disagree with recomputed {name}')
    height = {}
    for radius in (.15, .30, .50):
        key = str(radius)
        supported = xy <= radius
        values = dz[supported]
        saved = report['nearest_xy_height_sensitivity'][key]
        require(saved['supported_points'] == int(supported.sum()), f'{radius}: support count')
        require(saved['excluded_points'] == int((~supported).sum()), f'{radius}: excluded count')
        close(saved['support_percent'], 100*np.count_nonzero(supported)/len(source), 'Support percent')
        close(saved['signed_mean_mm'], float(values.mean()) if len(values) else None, 'Signed mean')
        close(saved['signed_median_mm'], float(np.median(values)) if len(values) else None, 'Signed median')
        check_stats(saved['absolute_height'], recompute_stats(np.abs(values)), f'Height/{radius}')
        height[key] = saved['supported_points']
    require(list(height.values()) == sorted(height.values()), 'XY support must increase with radius')
    return {
        'comparison': report['comparison'], 'passed': True,
        'original_points': len(source), 'aligned_points': len(aligned),
        'reference_points': len(reference), 'final_top_normal_tilt_deg': tilt,
        'maximum_transform_export_difference_mm': float(np.max(np.abs(reconstructed-aligned))),
        'forward_recomputed': forward_stats, 'reverse_recomputed': reverse_stats,
        'xy_supported_points': height,
    }


def verify_tables(reports, partial):
    if not reports:
        return {'checked_rows': 0, 'pending_rows': []}
    with (OUT/'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    table = {(row['model'], int(row['scan_frame'])): row for row in rows}
    require(len(table) == len(rows), 'Duplicate CSV comparison rows')
    aggregate = {r['comparison']: r for r in json.loads((OUT/'all_comparisons.json').read_text())}
    pending, checked = [], 0
    for report in reports:
        key = (report['model'], report['scan_frame'])
        if key not in table or aggregate.get(report['comparison']) != report:
            if partial:
                pending.append(report['comparison'])
                continue
            raise AssertionError(f'Summary missing or stale: {report["comparison"]}')
        row = table[key]
        forward = report['forward_photo_to_scan']
        height = report['nearest_xy_height_sensitivity']['0.3']
        numbers = {k: forward[k] for k in ('median_mm','rmse_mm','p95_mm','p99_mm','max_mm')}
        numbers.update(photogrammetry_points=report['photogrammetry_points'],
                       reference_points=report['reference_points'],
                       within_1mm_percent=forward['within_percent']['1.0'],
                       reverse_p95_mm=report['reverse_scan_to_photo']['p95_mm'],
                       xy_support_percent=height['support_percent'],
                       height_rmse_mm=height['absolute_height'].get('rmse_mm'))
        for name, value in numbers.items():
            close(float(row[name]) if row[name] else None, value, f'CSV {key}/{name}')
        require(row['status'] == report['status'], f'CSV status differs for {key}')
        checked += 1
    if not partial:
        require(len(table) == len(aggregate) == len(reports), 'Unexpected summary rows')
    return {'checked_rows': checked, 'pending_rows': pending}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial', action='store_true', help='Skip missing/stale reports during an active run')
    args = parser.parse_args()
    fingerprint = {'method_sha256': digest(ROOT/'pipeline/compare_laser_scans.py'),
                   'leveling_sha256': digest(LEVEL)}
    expected = []
    for number, obj in SCAN_OBJECTS.items():
        expected.extend((number, f'{p.parent.name}__scan_{number:04d}')
                        for p in sorted(MODELS.glob(f'{obj}_*/dense_fused_points.ply')))
    require(len(expected) == 13, f'Expected 13 model/scan combinations, found {len(expected)}')
    baseline = json.loads((ROOT/'LaserScans/reference_import/import_audit.json').read_text())
    audit_records = {record['frame_index']: record for record in baseline['records']}
    scans, raw_hashes = {}, {}
    for number in SCAN_OBJECTS:
        path = next((ROOT/'LaserScans').glob(f'*Frame-{number:04d}*.pcd'))
        scans[number] = load_scan(path)
        current = scans[number]['metadata']['source_sha256']
        require(current == audit_records[number]['source_sha256'], f'Original scan {number} changed since import audit')
        raw_hashes.update(current)
    checks, checked_reports, pending, failures = [], [], [], []
    for number, key in expected:
        path = OUT/key/'comparison_report.json'
        try:
            if not path.exists():
                pending.append({'comparison': key, 'reason': 'report not written'})
                continue
            report = json.loads(path.read_text())
            if not fingerprint_matches(report['registration'], report['model'], fingerprint):
                pending.append({'comparison': key, 'reason': 'stale implementation/leveling fingerprint'})
                continue
            checks.append(verify_one(path, report, scans[number], fingerprint))
            checked_reports.append(report)
            print(f'PASS {key}', flush=True)
        except Exception as error:
            failures.append({'comparison': key, 'error': f'{type(error).__name__}: {error}'})
            print(f'FAIL {key}: {error}', flush=True)
    table_check = None
    try:
        table_check = verify_tables(checked_reports, args.partial)
    except Exception as error:
        failures.append({'comparison': 'summary tables', 'error': f'{type(error).__name__}: {error}'})
    if pending and not args.partial:
        failures.append({'comparison': 'completeness', 'error': f'{len(pending)} reports missing/stale; use --partial during a run'})
    face_check = None
    if not args.partial:
        try:
            policy_path = ROOT/'LaserScans/reference_pairings.json'
            policy = json.loads(policy_path.read_text())
            require(policy['harmu_opposite_faces_confirmed_by_user'] is True, 'Missing user face confirmation')
            opposite = int(policy['harmu_opposite_face_frame'])
            eligible, excluded = [], []
            for r in checked_reports:
                expected_eligible = not (r['object']=='Harmu' and r['scan_frame']==opposite)
                require(r['accuracy_comparison_eligible'] is expected_eligible, 'Incorrect physical-face eligibility')
                require(r['reference_pairing_policy_sha256']==digest(policy_path), 'Stale face-selection policy')
                (eligible if expected_eligible else excluded).append(r['comparison'])
                if not expected_eligible:
                    require(r['status']=='excluded_opposite_face_not_an_accuracy_measurement', 'Opposite face lacks exclusion label')
                    require(r['reference_role']=='opposite_face_diagnostic_only', 'Opposite face has wrong role')
            with (OUT/'accuracy_summary.csv').open(encoding='utf-8-sig',newline='') as stream:
                selected=list(csv.DictReader(stream))
            selected_keys={f"{r['model']}__scan_{int(r['scan_frame']):04d}" for r in selected}
            require(selected_keys==set(eligible), 'Accuracy summary includes wrong faces or omits eligible rows')
            require(all(r['accuracy_comparison_eligible']=='True' for r in selected), 'Accuracy summary flag mismatch')
            require(len(eligible)==11 and len(excluded)==2, 'Unexpected eligible/diagnostic counts')
            with (OUT/'summary.csv').open(encoding='utf-8-sig',newline='') as stream:
                all_rows=list(csv.DictReader(stream))
            all_keys={(r['model'],r['scan_frame']):r for r in all_rows}
            require(all(all_keys[(r['model'],r['scan_frame'])]==r for r in selected), 'Filtered summary changed numerical values')
            face_check={'eligible_comparisons':len(eligible),'excluded_opposite_face_diagnostics':excluded,
                        'matching_harmu_frame':policy['harmu_matching_face_frame'],
                        'basis':'User opposite-face confirmation and photographed rounded features; not residual ranking'}
        except Exception as error:
            failures.append({'comparison':'physical-face selection','error':f'{type(error).__name__}: {error}'})
    output = {
        'status': 'failed' if failures else ('partial_pass' if pending or args.partial else 'complete_pass'),
        'expected_comparisons': len(expected), 'verified_comparisons': len(checks),
        'method_fingerprint': fingerprint, 'coordinate_tolerance_mm': ATOL_MM,
        'raw_original_files_verified': len(raw_hashes), 'raw_original_sha256': raw_hashes,
        'tables': table_check, 'pending': pending, 'failures': failures, 'checks': checks,
        'physical_face_selection':face_check,
        'scope': 'Independent computational/provenance verification; not proof of absolute accuracy, reference uncertainty or unique physical registration',
    }
    (OUT/'verification.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(f"{output['status']}: {len(checks)}/{len(expected)} comparisons; {len(raw_hashes)} original scan files verified", flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
