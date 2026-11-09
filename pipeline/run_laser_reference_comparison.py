"""Run numerical comparison followed by required physical-face eligibility labels.

Accepts the same options as compare_laser_scans.py. Existing registrations remain
cached unless --force-registration is requested. Opposite-face diagnostics are
retained, but excluded from accuracy_summary.csv and visibly labelled in figures.
"""

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.paths import PROJECT_ROOT

from pathlib import Path
import subprocess
import sys

ROOT=PROJECT_ROOT


def main():
    arguments=sys.argv[1:]
    subprocess.run([sys.executable,str(ROOT/'pipeline'/'compare_laser_scans.py'),*arguments],cwd=ROOT,check=True)
    if '--self-check' in arguments:
        return
    subprocess.run([sys.executable,str(ROOT/'pipeline'/'update_laser_reference_labels.py')],cwd=ROOT,check=True)
    if '--no-plots' not in arguments:
        subprocess.run([sys.executable,str(ROOT/'pipeline'/'render_laser_comparison_figures.py')],cwd=ROOT,check=True)


if __name__=='__main__':main()
