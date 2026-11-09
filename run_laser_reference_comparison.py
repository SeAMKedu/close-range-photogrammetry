"""Run numerical comparison followed by required physical-face eligibility labels.

Accepts the same options as compare_laser_scans.py. Existing registrations remain
cached unless --force-registration is requested. Opposite-face diagnostics are
retained, but excluded from accuracy_summary.csv and visibly labelled in figures.
"""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    arguments=sys.argv[1:]
    subprocess.run([sys.executable,str(ROOT/'compare_laser_scans.py'),*arguments],cwd=ROOT,check=True)
    if '--self-check' in arguments:
        return
    subprocess.run([sys.executable,str(ROOT/'update_laser_reference_labels.py')],cwd=ROOT,check=True)
    if '--no-plots' not in arguments:
        subprocess.run([sys.executable,str(ROOT/'render_laser_comparison_figures.py')],cwd=ROOT,check=True)


if __name__=='__main__':main()
