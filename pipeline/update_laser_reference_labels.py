"""Apply physical-face eligibility to saved numeric comparison results.

This is the interpretation stage, after compare_laser_scans.py. It preserves
all measured distances, transforms and original data, including the deliberately
excluded opposite-face diagnostic. The pipeline wrapper always runs this stage.
"""

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path
import csv
import hashlib
import json

from pipeline.compare_laser_scans import OUT, ROOT, write_summary


def main():
    policy_path = ROOT/'LaserScans/reference_pairings.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    assert policy['harmu_opposite_faces_confirmed_by_user'] is True
    matching = int(policy['harmu_matching_face_frame'])
    opposite = int(policy['harmu_opposite_face_frame'])
    assert matching != opposite
    preserved = {}
    for path in sorted(OUT.glob('*/comparison_report.json')):
        report = json.loads(path.read_text(encoding='utf-8'))
        numerical_files = [path.parent/name for name in ('registration.json','distances.npz','aligned_photogrammetry.ply','photogrammetry_error_map.ply','alignment_overlay.ply')]
        before = {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in numerical_files}
        report.setdefault('original_numerical_status', report['status'])
        report['accuracy_comparison_eligible'] = True
        report['reference_role'] = 'corresponding_object_surface'
        report['exclusion_reason'] = ''
        report['reference_pairing_policy_sha256'] = hashlib.sha256(policy_path.read_bytes()).hexdigest()
        if report['object'] == 'Harmu':
            report['flags'] = [f for f in report['flags'] if not f.startswith(('Two Harmu scans retained', 'Opposite physical face', 'Matching photographed face'))]
            report['face_selection_evidence'] = policy['evidence']
            report['pairing_basis'] = 'User confirmed opposite faces; rounded protrusions identify scan 0003 as the photographed face'
            if report['scan_frame'] == matching:
                report['reference_role'] = 'matching_photographed_face'
                report['status'] = 'provisional_post_alignment_surface_comparison'
                report['flags'].append('Matching photographed face identified from rounded protrusions in recessed pockets')
            elif report['scan_frame'] == opposite:
                report['accuracy_comparison_eligible'] = False
                report['reference_role'] = 'opposite_face_diagnostic_only'
                report['status'] = 'excluded_opposite_face_not_an_accuracy_measurement'
                report['exclusion_reason'] = 'Laser scan represents the opposite physical face, without the rounded protrusions visible in the photogrammetry images'
                report['flags'].append('Opposite physical face: numerical residuals retained only as a diagnostic, excluded from accuracy results')
                report['interpretation'] = 'Different physical faces were aligned. These numerical shape distances must not be interpreted as photogrammetry measurement errors or included in accuracy summaries.'
            else:
                raise ValueError(f"Unclassified Harmu reference: {report['scan_frame']}")
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        after = {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in numerical_files}
        assert before == after
        preserved[report['comparison']] = after

    reports = write_summary()
    by_key = {(r['model'],r['scan_frame']):r for r in reports}
    with (OUT/'summary.csv').open(encoding='utf-8-sig',newline='') as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames)+['accuracy_comparison_eligible','reference_role','exclusion_reason']
        rows = list(reader)
    for row in rows:
        report = by_key[(row['model'],int(row['scan_frame']))]
        row.update({key:report[key] for key in columns[-3:]})
    for filename, selected in [('summary.csv',rows),('accuracy_summary.csv',[row for row in rows if row['accuracy_comparison_eligible']])]:
        with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns)
            writer.writeheader();writer.writerows(selected)
    log={'policy':policy,'policy_sha256':hashlib.sha256(policy_path.read_bytes()).hexdigest(),
         'eligible_comparisons':sum(r['accuracy_comparison_eligible'] for r in reports),
         'excluded_diagnostics':[r['comparison'] for r in reports if not r['accuracy_comparison_eligible']],
         'registration_recomputed':False,'numerical_artifacts_unchanged_sha256':preserved}
    (OUT/'face_selection_review.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
    print(f"Face labels applied: {log['eligible_comparisons']} eligible comparisons; {len(log['excluded_diagnostics'])} opposite-face diagnostics excluded.")


if __name__=='__main__':main()
