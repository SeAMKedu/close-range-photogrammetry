"""Rerender laser-comparison figures from saved measurements, without registration.

Usage from the project root:
    .venv/Scripts/python.exe pipeline/render_laser_comparison_figures.py
    .venv/Scripts/python.exe pipeline/render_laser_comparison_figures.py --comparison Sini_threepiece__scan_0001
"""
from __future__ import annotations

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.paths import PROJECT_ROOT


import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pipeline.laser_comparison_plots import plot_comparison
from pipeline.laser_reference_io import load_scan


ROOT = PROJECT_ROOT
DEFAULT_RESULTS = ROOT / "LaserScans/comparison_results"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--comparison", action="append", help="Optional comparison directory name; may repeat")
    args = parser.parse_args()
    results = args.results.resolve()
    reports = sorted(results.glob("*/comparison_report.json"))
    if args.comparison:
        requested = set(args.comparison)
        reports = [path for path in reports if path.parent.name in requested]
        missing = requested - {path.parent.name for path in reports}
        if missing:
            raise FileNotFoundError(f"Missing comparisons: {sorted(missing)}")
    if not reports:
        raise FileNotFoundError(f"No saved comparison reports found under {results}")
    references = {}
    render_records = []
    for report_path in reports:
        folder = report_path.parent
        measurements = [report_path, folder / "distances.npz", folder / "registration.json"]
        before = {path.name: digest(path) for path in measurements}
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reference_path = (ROOT / report["reference_pcd"]).resolve()
        if reference_path not in references:
            references[reference_path] = load_scan(reference_path)
        scan = references[reference_path]
        expected_hashes = report["scanner_metadata"]["source_sha256"]
        if expected_hashes != scan["metadata"]["source_sha256"]:
            raise RuntimeError(f"Reference source changed since measurement: {reference_path}")
        with np.load(folder / "distances.npz") as data:
            forward, reverse = data["photo_to_scan_mm"], data["scan_to_photo_mm"]
            for distances, key in ((forward, "forward_photo_to_scan"),
                                   (reverse, "reverse_scan_to_photo")):
                recorded = report[key]
                if len(distances) != recorded["count"]:
                    raise ValueError(f"Distance count mismatch: {folder.name} {key}")
                for actual, expected in (
                    (np.median(distances), recorded["median_mm"]),
                    (np.sqrt(np.mean(distances ** 2)), recorded["rmse_mm"]),
                    (np.percentile(distances, 95), recorded["p95_mm"]),
                ):
                    if not np.isclose(actual, expected, rtol=1e-10, atol=1e-10):
                        raise ValueError(f"Distance statistics mismatch: {folder.name} {key}")
            paths = plot_comparison(report["comparison"], data["photo_aligned"], scan["points"],
                                    forward, reverse, folder, report)
        after = {path.name: digest(path) for path in measurements}
        if before != after:
            raise RuntimeError(f"Measurement artifacts unexpectedly changed: {folder}")
        render_records.append({"comparison": report["comparison"], "paths": paths,
                               "unchanged_measurement_sha256": after})
        print(f"Rendered {report['comparison']}; measurements unchanged", flush=True)
    log = {"rendered_comparisons": len(render_records),
           "renderer_sha256": digest(ROOT / "pipeline" / "laser_comparison_plots.py"),
           "registration_recomputed": False, "records": render_records}
    (results / "figures_render_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered {len(render_records)} comparison figure sets (PNG/PDF/text).", flush=True)


if __name__ == "__main__":
    main()
