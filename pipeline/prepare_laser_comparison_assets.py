"""Export existing eligible laser comparisons as precise per-point assets.

No registration or distance fitting is performed here. Saved measurements retain
their original order and float64 precision. The attributed PLY and gzip CSV make
the same measurements available independently of the Blender presentation.
"""
from __future__ import annotations

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.paths import PROJECT_ROOT


import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from pipeline.laser_reference_io import load_scan


ROOT = PROJECT_ROOT
COMPARISONS = ROOT / "LaserScans" / "comparison_results"
DEFAULT_OUTPUT = COMPARISONS / "blender" / "data"
XY_SUPPORT_RADIUS_MM = 0.3
CSV_COLUMNS = (
    "x_mm", "y_mm", "z_mm", "nearest_laser_distance_mm",
    "xy_support_distance_mm", "signed_height_mm", "xy_supported",
)
PLY_DTYPE = np.dtype([
    ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
    ("nearest_laser_distance_mm", "<f8"),
    ("xy_support_distance_mm", "<f8"), ("signed_height_mm", "<f8"),
    ("xy_supported", "u1"),
])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_attributed_ply(path: Path, arrays: dict[str, np.ndarray]) -> None:
    records = np.empty(len(arrays["points_mm"]), dtype=PLY_DTYPE)
    for index, axis in enumerate("xyz"):
        records[axis] = arrays["points_mm"][:, index]
    records["nearest_laser_distance_mm"] = arrays["error_mm"]
    records["xy_support_distance_mm"] = arrays["xy_support_distance_mm"]
    records["signed_height_mm"] = arrays["signed_height_mm"]
    records["xy_supported"] = arrays["xy_supported"]
    header = [
        "ply", "format binary_little_endian 1.0",
        "comment coordinates and distances in millimetres",
        "comment all finite photogrammetry points retained in source order",
        "comment no scale fitting; existing rigid registration preserved",
        "comment nearest_laser_distance_mm is unsigned sampled 3D separation",
        "comment signed_height_mm is photo Z minus nearest XY laser point Z",
        "comment xy_supported is 1 when XY distance is at most 0.3 mm",
        f"element vertex {len(records)}",
    ]
    header.extend(f"property {'uchar' if name == 'xy_supported' else 'double'} {name}"
                  for name in PLY_DTYPE.names)
    header.append("end_header")
    with path.open("wb") as stream:
        stream.write(("\n".join(header) + "\n").encode("ascii"))
        records.tofile(stream)


def write_point_csv(path: Path, arrays: dict[str, np.ndarray]) -> None:
    # A fixed timestamp and empty embedded filename make repeated exports stable.
    # 17 significant digits preserve every binary64 measurement on readback.
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(CSV_COLUMNS)
                rows = zip(arrays["points_mm"], arrays["error_mm"],
                           arrays["xy_support_distance_mm"], arrays["signed_height_mm"],
                           arrays["xy_supported"])
                for point, distance, xy_distance, signed_height, supported in rows:
                    writer.writerow([*(format(value, ".17g") for value in point),
                                     format(distance, ".17g"), format(xy_distance, ".17g"),
                                     format(signed_height, ".17g"), int(supported)])


def verify_exports(files: dict[str, Path], arrays: dict[str, np.ndarray]) -> dict:
    """Check full-precision round trips, including CSV rows and PLY attributes."""
    with np.load(files["npz"], allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise ValueError("NPZ array keys changed during export")
        for key, expected in arrays.items():
            if saved[key].dtype != expected.dtype or not np.array_equal(saved[key], expected):
                raise ValueError(f"NPZ round trip changed {key}")
    with files["ply"].open("rb") as stream:
        while True:
            line = stream.readline()
            if not line or stream.tell() > 4096:
                raise ValueError("Malformed exported PLY header")
            if line.strip() == b"end_header":
                break
        records = np.frombuffer(stream.read(), dtype=PLY_DTYPE)
    if len(records) != len(arrays["points_mm"]):
        raise ValueError("PLY point count changed")
    for index, axis in enumerate("xyz"):
        if not np.array_equal(records[axis], arrays["points_mm"][:, index]):
            raise ValueError(f"PLY {axis} coordinates changed")
    for target, source in (("nearest_laser_distance_mm", "error_mm"),
                           ("xy_support_distance_mm", "xy_support_distance_mm"),
                           ("signed_height_mm", "signed_height_mm"),
                           ("xy_supported", "xy_supported")):
        if not np.array_equal(records[target], arrays[source]):
            raise ValueError(f"PLY {target} attribute changed")
    with gzip.open(files["csv_gz"], "rt", encoding="utf-8", newline="") as stream:
        rows = csv.reader(stream)
        if tuple(next(rows)) != CSV_COLUMNS:
            raise ValueError("CSV columns changed")
        # Parse all values using Python's correctly rounded float parser.
        values = np.asarray([[float(value) for value in row] for row in rows], dtype=np.float64)
    expected = np.column_stack((arrays["points_mm"], arrays["error_mm"],
                                arrays["xy_support_distance_mm"], arrays["signed_height_mm"],
                                arrays["xy_supported"]))
    if not np.array_equal(values, expected):
        raise ValueError("CSV round trip changed point measurements")
    return {"npz_exact_round_trip": True, "ply_exact_round_trip": True,
            "csv_exact_round_trip": True, "checked_photogrammetry_points": len(values)}


def build_assets(output_dir: Path = DEFAULT_OUTPUT) -> dict:
    output_dir = output_dir.resolve()
    # Keep linked exports inside this workspace, including custom destinations.
    output_dir.relative_to(ROOT)
    policy_path = ROOT / "LaserScans" / "reference_pairings.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_hash = sha256(policy_path)
    matching_harmu_frame = int(policy["harmu_matching_face_frame"])
    selected = []
    excluded = []
    for report_path in sorted(COMPARISONS.glob("*/comparison_report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("accuracy_comparison_eligible") is not True:
            excluded.append(report["comparison"])
            continue
        if report["reference_pairing_policy_sha256"] != policy_hash:
            raise ValueError(f"Stale physical-face policy: {report_path}")
        if report["object"] == "Harmu" and int(report["scan_frame"]) != matching_harmu_frame:
            raise ValueError("Eligible Harmu report uses the opposite face")
        if report["units"] != "mm" or report["evaluated_points_removed"] != 0:
            raise ValueError(f"Expected unfiltered millimetre measurements: {report_path}")
        if report["registration"]["scale_fitted"] or report["registration"]["reflection_allowed"]:
            raise ValueError(f"Registration must preserve metric scale and handedness: {report_path}")
        selected.append((report_path, report))
    if not selected:
        raise ValueError("No eligible saved comparisons found; run the reference comparison first")
    models = [report["model"] for _, report in selected]
    if len(models) != len(set(models)):
        raise ValueError("A model has multiple eligible reference scans")
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_cache = {}
    entries = []
    for report_path, report in selected:
        folder = report_path.parent
        source_cloud = ROOT / report["source_cloud"]
        reference_pcd = ROOT / report["reference_pcd"]
        if reference_pcd not in scan_cache:
            scan_cache[reference_pcd] = load_scan(reference_pcd)
        scan = scan_cache[reference_pcd]
        if scan["metadata"]["source_sha256"] != report["scanner_metadata"]["source_sha256"]:
            raise ValueError(f"Original scan files changed: {reference_pcd}")
        if sha256(source_cloud) != report["registration"]["model_sha256"]:
            raise ValueError(f"Original photogrammetry cloud changed: {source_cloud}")
        if sha256(reference_pcd) != report["registration"]["reference_sha256"]:
            raise ValueError(f"Reference geometry changed: {reference_pcd}")
        registration_path = folder / "registration.json"
        if json.loads(registration_path.read_text(encoding="utf-8")) != report["registration"]:
            raise ValueError(f"Registration and comparison report disagree: {folder}")
        distance_path = folder / "distances.npz"
        with np.load(distance_path, allow_pickle=False) as saved:
            arrays = {
                "points_mm": saved["photo_aligned"],
                "error_mm": saved["photo_to_scan_mm"],
                "reference_points_mm": scan["points"],
                "reference_error_mm": saved["scan_to_photo_mm"],
                "xy_support_distance_mm": saved["nearest_xy_distance_mm"],
                "signed_height_mm": saved["signed_height_mm"],
            }
        photo_count, reference_count = report["photogrammetry_points"], report["reference_points"]
        expected_shapes = {"points_mm": (photo_count, 3), "reference_points_mm": (reference_count, 3),
                           "reference_error_mm": (reference_count,)}
        for key, values in arrays.items():
            if values.shape != expected_shapes.get(key, (photo_count,)):
                raise ValueError(f"Unexpected array shape for {report['model']}: {key}")
            if values.dtype != np.dtype("float64") or not np.isfinite(values).all():
                raise ValueError(f"Expected finite float64 measurements: {report['model']}/{key}")
        for key in ("error_mm", "reference_error_mm", "xy_support_distance_mm"):
            if np.any(arrays[key] < 0):
                raise ValueError(f"Unsigned distance has negative values: {key}")
        arrays["xy_supported"] = arrays["xy_support_distance_mm"] <= XY_SUPPORT_RADIUS_MM
        if int(arrays["xy_supported"].sum()) != report["nearest_xy_height_sensitivity"]["0.3"]["supported_points"]:
            raise ValueError("XY support count disagrees with saved report")
        for key, metrics in (("error_mm", report["forward_photo_to_scan"]),
                             ("reference_error_mm", report["reverse_scan_to_photo"])):
            values = arrays[key]
            actual = {"mean_mm": np.mean(values), "median_mm": np.median(values),
                      "rmse_mm": np.sqrt(np.mean(values ** 2)),
                      "p95_mm": np.percentile(values, 95), "max_mm": np.max(values)}
            if len(values) != metrics["count"] or any(
                    not np.isclose(value, metrics[name], rtol=1e-12, atol=1e-12)
                    for name, value in actual.items()):
                raise ValueError(f"Distances disagree with report metrics: {report['model']}/{key}")
        model = report["model"]
        files = {"npz": output_dir / f"{model}.npz",
                 "csv_gz": output_dir / f"{model}_pistekohtaiset_poikkeamat.csv.gz",
                 "ply": output_dir / f"{model}_pistekohtaiset_poikkeamat.ply"}
        source_files = {"comparison_report": report_path, "distances": distance_path,
                        "registration": registration_path, "photogrammetry_cloud": source_cloud,
                        "reference_pcd": reference_pcd, "reference_pairing_policy": policy_path}
        before_hashes = {key: sha256(path) for key, path in source_files.items()}
        np.savez_compressed(files["npz"], **arrays)
        write_point_csv(files["csv_gz"], arrays)
        write_attributed_ply(files["ply"], arrays)
        validation = verify_exports(files, arrays)
        if before_hashes != {key: sha256(path) for key, path in source_files.items()}:
            raise ValueError("A measurement source changed during asset preparation")
        entries.append({
            **{key: report[key] for key in (
                "model", "object", "comparison", "scan_frame", "accuracy_comparison_eligible",
                "reference_role", "status", "photogrammetry_points", "reference_points",
                "forward_photo_to_scan", "reverse_scan_to_photo", "flags", "interpretation")},
            "report_path": relative_path(report_path),
            "reference_pcd": relative_path(reference_pcd),
            "files": {key: relative_path(path) for key, path in files.items()},
            "source_sha256": before_hashes,
            "original_scan_sha256": scan["metadata"]["source_sha256"],
            "asset_sha256": {key: sha256(path) for key, path in files.items()},
            "asset_bytes": {key: path.stat().st_size for key, path in files.items()},
            "transform_photo_to_scanner": report["registration"]["transform_photo_to_scanner"],
            "source_coordinate_conversion": report['registration'].get('source_coordinate_conversion'),
            "coordinate_conversion_basis": report['registration'].get('coordinate_conversion_basis'),
            "all_finite_points_preserved": True,
            "evaluated_points_removed": 0,
            "scale_fitted": False,
            "xy_supported_points": int(arrays["xy_supported"].sum()),
            "validation": validation,
        })
        print(f"{model}: {photo_count:,} photo + {reference_count:,} laser points; exact NPZ/CSV/PLY checks passed", flush=True)
    manifest = {
        "schema_version": 1, "units": "mm", "coordinate_frame": "native scanner coordinates after saved rigid registration",
        "registration_recomputed": False, "distance_measurements_recomputed": False,
        "all_finite_points_preserved": True, "xy_support_radius_mm": XY_SUPPORT_RADIUS_MM,
        "precision": "float64 coordinates and measurements; boolean XY support mask",
        "error_definition": "Unsigned nearest 3D scanner sample distance after rigid alignment; all points retained",
        "signed_height_definition": "Photogrammetry Z minus the laser sample Z nearest in XY; interpret only where xy_supported is true",
        "interpretation": "Sampled surface deviations, not calibrated measurement uncertainty or confidence intervals",
        "scanner_reference_uncertainty": "not provided",
        "colors": "Stored separately by visualization; raw distances are never clipped",
        "reference_pairing_policy_sha256": policy_hash,
        "excluded_comparisons": excluded, "model_count": len(entries), "models": entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {relative_path(manifest_path)} with {len(entries)} eligible comparisons", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_assets(args.output_dir)


if __name__ == "__main__":
    main()
