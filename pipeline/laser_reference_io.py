"""Validated import of this project's Gocator organized point-cloud exports.

PCD geometry is used without filtering finite points or filling missing cells.
The companion CSV supplies acquisition metadata and independently checks every
coordinate. The companion Digital Surf header establishes millimetre units.
Only the exact binary XYZ float32 PCD layout present in this dataset is accepted.
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
import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np


ROOT = PROJECT_ROOT
INFERRED_OBJECTS = {
    0: "Musti", 1: "Sini", 2: "Metal",
    3: "Harmu_scan3", 4: "Harmu_scan4", 5: "Piiri",
}
CSV_COORDINATE_TOLERANCE_MM = 0.0001


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv_section(rows: list[list[str]], name: str) -> tuple[int, dict[str, str]]:
    indices = [i for i, row in enumerate(rows) if row and row[0] == name]
    if len(indices) != 1:
        raise ValueError(f"Expected one CSV {name!r} block; found {len(indices)}")
    i = indices[0]
    if i + 2 >= len(rows) or len(rows[i + 1]) != len(rows[i + 2]):
        raise ValueError(f"Malformed CSV {name!r} metadata")
    return i, dict(zip(rows[i + 1], rows[i + 2]))


def _load_csv_grid(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    _, info = _csv_section(rows, "Info")
    _, device = _csv_section(rows, "DeviceInfo")
    part_index, part = _csv_section(rows, "Part")
    if info.get("CSV Version") != "2" or info.get("Sensor Count") != "1":
        raise ValueError("Only the current single-sensor Gocator CSV version 2 is supported")
    height, width = int(part["Row Count"]), int(part["Column Count"])
    header = rows[part_index + 3]
    if header[0] != "Y\\X" or len(header) < width + 1:
        raise ValueError("Missing organized CSV Y\\X grid header")
    x = np.asarray(header[1:width + 1], dtype=np.float64)
    y = np.empty(height, dtype=np.float64)
    z = np.full((height, width), np.nan, dtype=np.float64)
    for index, row in enumerate(rows[part_index + 4:part_index + 4 + height]):
        if len(row) < width + 1:
            raise ValueError(f"CSV grid row {index} has too few columns")
        y[index] = float(row[0])
        for j, value in enumerate(row[1:width + 1]):
            if value.strip():
                z[index, j] = float(value)
    if len(rows[part_index + 4:part_index + 4 + height]) != height:
        raise ValueError("CSV grid has too few rows")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Nonfinite CSV grid coordinates")
    valid = np.isfinite(z)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    grid = np.stack([xx, yy, z], axis=-1)
    grid[~valid] = np.nan
    return grid, valid, {"info": info, "device": device, "part": part}


def _load_binary_pcd(path: Path) -> tuple[np.ndarray, dict[str, str]]:
    with path.open("rb") as stream:
        fields = {}
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("PCD header has no DATA line")
            line = raw.decode("ascii").strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition(" ")
            if key in fields:
                raise ValueError(f"Duplicate PCD header key {key}")
            fields[key] = value.strip()
            if key == "DATA":
                break
        required = {
            "VERSION": "0.7", "FIELDS": "x y z", "SIZE": "4 4 4",
            "TYPE": "F F F", "COUNT": "1 1 1", "DATA": "binary",
            "VIEWPOINT": "0 0 0 1 0 0 0",
        }
        for key, value in required.items():
            if fields.get(key) != value:
                raise ValueError(f"Unsupported PCD {key}={fields.get(key)!r}; expected {value!r}")
        width, height, count = (int(fields[k]) for k in ("WIDTH", "HEIGHT", "POINTS"))
        if width <= 0 or height <= 0 or width * height != count:
            raise ValueError("Inconsistent organized PCD dimensions")
        payload = stream.read()
    if len(payload) != count * 12:
        raise ValueError(f"PCD binary length mismatch: {len(payload)} != {count * 12}")
    grid = np.frombuffer(payload, dtype="<f4").reshape(height, width, 3).astype(np.float64)
    invalid = ~np.isfinite(grid).all(axis=2)
    if invalid.any() and not np.isnan(grid[invalid]).all():
        raise ValueError("Expected missing cells to contain three NaNs, not partial or infinite XYZ")
    return grid, fields


def _confirm_sur_units(path: Path, width: int, height: int) -> dict:
    header = path.read_bytes()[:512]
    if len(header) != 512 or not header.startswith(b"DIGITAL SURF\x00"):
        raise ValueError("Companion SUR lacks the expected Digital Surf header")
    dimensions = struct.unpack_from("<III", header, 108)
    if dimensions != (width, height, width * height):
        raise ValueError("SUR and PCD grid dimensions disagree")
    units = [header[i:i + 16].split(b"\x00", 1)[0].decode("ascii")
             for i in (180, 196, 212, 228, 244, 260)]
    if units != ["mm"] * 6:
        raise ValueError(f"Unexpected SUR axis/length units: {units}; conversion is not implicit")
    return {"source": str(path.resolve()), "signature": "DIGITAL SURF",
            "axis_and_length_units": units, "header_offsets": [180, 196, 212, 228, 244, 260]}


def load_scan(pcd_path: str | Path) -> dict:
    """Return points, organized grid_xyz, valid mask and JSON-safe metadata.

    Paired .csv and .sur files are required. No registration, scale fitting,
    denoising, background deletion or interpolation is performed by this loader.
    """
    path = Path(pcd_path).resolve()
    if path.suffix.lower() != ".pcd":
        raise ValueError("load_scan expects a .pcd path with companion .csv and .sur exports")
    csv_path, sur_path = path.with_suffix(".csv"), path.with_suffix(".sur")
    for required_path in (path, csv_path, sur_path):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)
    grid, pcd_header = _load_binary_pcd(path)
    valid = np.isfinite(grid).all(axis=2)
    csv_grid, csv_valid, csv_metadata = _load_csv_grid(csv_path)
    if grid.shape != csv_grid.shape or not np.array_equal(valid, csv_valid):
        raise ValueError("CSV and PCD organized geometry or invalid masks disagree")
    if not valid.any():
        raise ValueError("Scan has no finite samples")
    max_diff = np.max(np.abs(grid[valid] - csv_grid[valid]), axis=0)
    if np.max(max_diff) > CSV_COORDINATE_TOLERANCE_MM:
        raise ValueError(f"PCD/CSV mismatch exceeds {CSV_COORDINATE_TOLERANCE_MM} mm: {max_diff}")
    height, width = valid.shape
    unit_evidence = _confirm_sur_units(sur_path, width, height)
    info, device = csv_metadata["info"], csv_metadata["device"]
    frame_match = re.search(r"Frame-(\d+)", path.stem)
    frame_index = int(frame_match.group(1)) if frame_match else None
    points = grid[valid]
    sampling = {axis: float(info[f"{axis.upper()}Resolution"]) for axis in ("x", "y", "z")}
    # The organized PCD retains more Y-coordinate precision than the CSV text.
    finite_row_indices, finite_col_indices = np.nonzero(valid)
    x_fit = np.polyfit(finite_col_indices.astype(float), points[:, 0], 1)
    y_fit = np.polyfit(finite_row_indices.astype(float), points[:, 1], 1)
    source_files = [path, csv_path, sur_path]
    if path.with_suffix(".srf").exists():
        source_files.append(path.with_suffix(".srf"))
    metadata = {
        "source_pcd": str(path), "unit": "mm", "unit_evidence": unit_evidence,
        "frame_index": frame_index, "inferred_object": INFERRED_OBJECTS.get(frame_index),
        "mapping_status": "inferred_from_shape; Harmu alternatives remain separate",
        "mapping_evidence": "Scanner height shapes compared with project object photographs; user reports two Harmu scans",
        "scanner_model": f"Gocator {device['Model']}", "scanner_device_id": device["ID"],
        "scanner_firmware": device["Version"], "width": width, "height": height,
        "grid_cells": width * height, "valid_points": len(points),
        "invalid_cells": int((~valid).sum()), "invalid_representation": "NaN XYZ triples",
        "csv_resolution_mm": sampling,
        "pcd_grid_spacing_fit_mm": {"x": float(x_fit[0]), "y": float(y_fit[0])},
        "resolution_note": "X/Y are sampling spacing and Z is quantization, not calibrated uncertainty",
        "coordinate_bounds_mm": {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()},
        "pcd_header": pcd_header, "csv_metadata": csv_metadata,
        "csv_crosscheck": {"valid_mask_identical": True, "max_abs_xyz_difference_mm": max_diff.tolist(),
                           "tolerance_mm": CSV_COORDINATE_TOLERANCE_MM},
        "source_sha256": {p.name: _sha256(p) for p in source_files},
        "import_processing": "Remove nonfinite cells only; retain all finite measurements and native holes",
    }
    return {"points": points, "grid_xyz": grid, "valid": valid, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", required=True,
                        help="Validate all paired exports and write converted references and metadata")
    parser.add_argument("--scan-dir", type=Path, default=ROOT / "LaserScans")
    parser.add_argument("--output", type=Path, default=ROOT / "LaserScans" / "reference_import")
    args = parser.parse_args()
    paths = sorted(args.scan_dir.glob("*.pcd"))
    if not paths:
        parser.error(f"No PCD files found in {args.scan_dir}")
    import open3d as o3d
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in paths:
        scan = load_scan(path)
        metadata = scan["metadata"]
        name = f"frame_{metadata['frame_index']:04d}_{metadata['inferred_object'] or 'unidentified'}"
        ply_path, npz_path = args.output / f"{name}.ply", args.output / f"{name}.npz"
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan["points"]))
        if not o3d.io.write_point_cloud(str(ply_path), cloud, write_ascii=False):
            raise RuntimeError(f"Failed writing {ply_path}")
        np.savez_compressed(npz_path, points=scan["points"], grid_xyz=scan["grid_xyz"], valid=scan["valid"])
        metadata["converted_files"] = {"ply": str(ply_path.resolve()), "npz": str(npz_path.resolve())}
        (args.output / f"{name}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        reports.append(metadata)
        print(f"{name}: {metadata['valid_points']:,} finite samples; CSV check passed", flush=True)
    summary = {"format": "Gocator binary organized XYZ float32", "units": "mm",
               "scanner_reference_uncertainty": "not provided",
               "mapping_status": "inferred from shapes; Harmu faces require separate comparison",
               "records": reports}
    (args.output / "import_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
