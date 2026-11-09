"""Independently verify the saved laser-reference comparison Blender files.

Run with Blender in background mode.  The verification reads the original PCD
files and the existing numerical comparison arrays; it does not import the
exporter's implementation or trust its prepared display arrays.  Scene changes
made while exercising eye-icon controls are never saved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "LaserScans" / "comparison_results"
DEST = RESULTS / "blender"
EDGES = np.asarray([0.1, 0.25, 0.5, 1.0, 2.0, 5.0], dtype=np.float64)
PALETTE_HEX = ["440154", "443983", "31688E", "21918C", "35B779", "90D743", "FDE725"]
ROLES = {"photo_error", "laser_reference", "photo_overlay"}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_reference_pcd(path):
    """Read the exact native little-endian binary XYZ float32 layout."""
    with path.open("rb") as stream:
        header = {}
        for _ in range(100):
            raw = stream.readline()
            require(raw, f"Missing PCD DATA header: {path}")
            line = raw.decode("ascii").strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition(" ")
            require(key not in header, f"Duplicate PCD header key: {key}")
            header[key] = value.strip()
            if key == "DATA":
                break
        else:
            raise AssertionError(f"PCD header too long: {path}")
        for key, value in {"FIELDS": "x y z", "SIZE": "4 4 4", "TYPE": "F F F",
                           "COUNT": "1 1 1", "DATA": "binary"}.items():
            require(header.get(key) == value, f"Unsupported PCD {key}: {path}")
        count = int(header["POINTS"])
        require(count == int(header["WIDTH"]) * int(header["HEIGHT"]), "PCD dimensions")
        payload = stream.read()
    require(len(payload) == count * 12, f"PCD payload size: {path}")
    xyz = np.frombuffer(payload, dtype="<f4").reshape(count, 3)
    finite = np.isfinite(xyz).all(axis=1)
    require(np.isnan(xyz[~finite]).all(), "Native missing points must be NaN triples")
    return xyz[finite].copy()


def palette_linear():
    srgb = np.asarray([[int(color[k:k + 2], 16) / 255.0 for k in (0, 2, 4)]
                       for color in PALETTE_HEX], dtype=np.float64)
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return np.column_stack([linear, np.ones(len(linear))]).astype(np.float32)


def get_float_attribute(mesh, name):
    attribute = mesh.attributes.get(name)
    require(attribute is not None, f"Missing {name} in {mesh.name}")
    require(attribute.domain == "POINT" and attribute.data_type == "FLOAT",
            f"{name} must be a per-point float attribute")
    values = np.empty(len(mesh.vertices), dtype=np.float32)
    attribute.data.foreach_get("value", values)
    return values


def get_color_attribute(mesh, name):
    attribute = mesh.color_attributes.get(name)
    require(attribute is not None, f"Missing {name} in {mesh.name}")
    require(attribute.domain == "POINT" and attribute.data_type == "FLOAT_COLOR",
            f"{name} must be a per-point float color attribute")
    values = np.empty((len(mesh.vertices), 4), dtype=np.float32)
    attribute.data.foreach_get("color", values.ravel())
    return values


def load_expectations(models):
    with (RESULTS / "accuracy_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 11, f"Expected 11 eligible comparisons, received {len(rows)}")
    require(all(row["accuracy_comparison_eligible"].lower() == "true" for row in rows),
            "Accuracy summary contains excluded comparisons")
    policy = json.loads((ROOT / "LaserScans" / "reference_pairings.json").read_text(encoding="utf-8"))
    require(policy["harmu_matching_face_frame"] == 3 and policy["harmu_opposite_face_frame"] == 4,
            "Unexpected Harmu face policy")
    if models:
        available = {row["model"] for row in rows}
        require(set(models) <= available, f"Unknown models: {set(models) - available}")
        rows = [row for row in rows if row["model"] in models]
    result = {}
    reference_cache = {}
    for row in rows:
        frame = int(row["scan_frame"])
        require(frame != 4, "Opposite-face scan 0004 cannot appear in accuracy deliverables")
        model = row["model"]
        comparison = f"{model}__scan_{frame:04d}"
        report_path = RESULTS / comparison / "comparison_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        require(report["accuracy_comparison_eligible"] is True, f"Ineligible report: {comparison}")
        require(report["units"] == "mm", f"Unexpected report units: {comparison}")
        require(report["evaluated_points_removed"] == 0, "Numerical comparison removed evaluated points")
        if model.startswith("Harmu"):
            require(frame == 3 and report["reference_role"] == "matching_photographed_face",
                    "Harmu must use the photographed face, scan 0003")
        arrays_path = RESULTS / comparison / "distances.npz"
        with np.load(arrays_path) as saved:
            points = saved["photo_aligned"].copy()
            errors = saved["photo_to_scan_mm"].copy()
            reverse_errors = saved["scan_to_photo_mm"].copy()
        if frame not in reference_cache:
            reference_cache[frame] = read_reference_pcd(ROOT / report["reference_pcd"])
        ref = reference_cache[frame]
        require(points.shape == (report["photogrammetry_points"], 3), "Photo count")
        require(ref.shape == (report["reference_points"], 3), "Reference count")
        require(errors.shape == (len(points),) and reverse_errors.shape == (len(ref),), "Error counts")
        require(np.isfinite(points).all() and np.isfinite(errors).all() and np.isfinite(reverse_errors).all(),
                "Nonfinite comparison values")
        require((errors >= 0).all() and (reverse_errors >= 0).all(), "Negative unsigned distance")
        bands = np.searchsorted(EDGES, errors, side="right")
        result[model] = {"report": report, "points": points.astype(np.float32),
                         "errors": errors.astype(np.float32), "bands": bands.astype(np.float32),
                         "colors": palette_linear()[bands], "reference": ref,
                         "reverse_errors": reverse_errors.astype(np.float32),
                         "distances_sha256": sha256(arrays_path), "report_sha256": sha256(report_path)}
    return result


def verify_display_modifier(obj):
    modifiers = [modifier for modifier in obj.modifiers if modifier.type == "NODES"]
    require(len(modifiers) == 1, f"Expected one point display modifier: {obj.name}")
    modifier = modifiers[0]
    require(modifier.show_viewport and modifier.show_render, "Point glyph modifier disabled")
    group = modifier.node_group
    require(group is not None, "Missing point display node tree")
    instances = [node for node in group.nodes if node.bl_idname == "GeometryNodeInstanceOnPoints"]
    require(instances, "No point glyph instances in display node tree")
    require(any(node.bl_idname == "NodeGroupInput" for node in group.nodes), "Point geometry input absent")
    output = next((node for node in group.nodes if node.bl_idname == "NodeGroupOutput"), None)
    require(output is not None and output.inputs["Geometry"].is_linked, "Point geometry output absent")
    join = output.inputs["Geometry"].links[0].from_node
    require(join.bl_idname == "GeometryNodeJoinGeometry", "Glyph output does not join point instances")
    require({link.from_node for link in join.inputs["Geometry"].links} == set(instances),
            "Some point glyph bands are absent from the display output")
    for node in instances:
        require(node.inputs["Points"].is_linked and node.inputs["Instance"].is_linked,
                "Disconnected point glyph instancing")
        require(node.inputs["Points"].links[0].from_node.bl_idname == "NodeGroupInput",
                "Display geometry must use the original measured vertex positions directly")
        require(not node.inputs["Scale"].is_linked and np.allclose(node.inputs["Scale"].default_value, 1),
                "Point glyph scale must not change by error magnitude")
        require(not node.inputs["Rotation"].is_linked and np.allclose(node.inputs["Rotation"].default_value, 0),
                "Unexpected point glyph rotation")
    radius = float(obj["point_display_radius_mm"])
    spheres = [node for node in group.nodes if node.bl_idname == "GeometryNodeMeshIcoSphere"]
    require(spheres, "Missing point glyph geometry")
    require(all(np.isclose(node.inputs["Radius"].default_value, radius, rtol=0, atol=1e-7)
                for node in spheres), "Glyph radius metadata disagrees with node tree")
    if obj["role"] == "photo_error":
        require(len(instances) == 7, "Error point display must contain seven fixed color bands")
        named = [node for node in group.nodes if node.bl_idname == "GeometryNodeInputNamedAttribute"]
        require(any(node.inputs["Name"].default_value == "error_band" for node in named),
                "Error glyph color selection does not read error_band")
        compares = [node for node in group.nodes if node.bl_idname == "ShaderNodeMath"
                    and node.operation == "COMPARE"]
        require(len(compares) == 7, "Error display lacks seven band comparisons")
        compared = []
        for node in compares:
            linked = [slot for slot in node.inputs[:2] if slot.is_linked]
            unlinked = [slot for slot in node.inputs[:2] if not slot.is_linked]
            require(len(linked) == 1 and len(unlinked) == 1, "Unexpected band comparison wiring")
            source = linked[0].links[0].from_node
            require(source.bl_idname == "GeometryNodeInputNamedAttribute"
                    and source.inputs["Name"].default_value == "error_band", "Band source is not error_band")
            compared.append(float(unlinked[0].default_value))
            require(0 < node.inputs[2].default_value < 0.5, "Band comparison epsilon overlaps neighboring bins")
        require(sorted(compared) == list(range(7)), "Band comparisons are not 0 through 6")
        for node in instances:
            require(node.inputs["Selection"].is_linked, "Error band selection is disconnected")
            compare = node.inputs["Selection"].links[0].from_node
            require(compare in compares, "Unexpected selection source")
            band = int(next(slot.default_value for slot in compare.inputs[:2] if not slot.is_linked))
            assign = node.inputs["Instance"].links[0].from_node
            require(assign.bl_idname == "GeometryNodeSetMaterial", "Glyph instances lack band material assignment")
            material = assign.inputs["Material"].default_value
            require(material is not None, "Missing band material")
            require(np.allclose(np.asarray(material.diffuse_color), palette_linear()[band], rtol=0, atol=2e-7),
                    f"Displayed material does not match color band {band}")
            require(assign.inputs["Geometry"].is_linked
                    and assign.inputs["Geometry"].links[0].from_node in spheres,
                    "Unexpected glyph geometry transform before instancing")


def verify_object(obj, expected, overview):
    report = expected["report"]
    role = obj["role"]
    require(obj.type == "MESH", f"Comparison geometry is not mesh vertices: {obj.name}")
    require(obj.get("comparison") == report["comparison"], f"Incorrect comparison metadata: {obj.name}")
    require(int(obj.get("scan_frame", -1)) == report["scan_frame"] != 4, "Wrong scan frame metadata")
    require(len(obj.data.edges) == 0 and len(obj.data.polygons) == 0,
            "Stored comparison geometry must remain points, without artificial surface geometry")
    reference = role == "laser_reference"
    coordinates = expected["reference" if reference else "points"]
    require(len(obj.data.vertices) == len(coordinates), f"Dropped or duplicated points in {obj.name}")
    require(int(obj["point_count"]) == len(coordinates), "Point count metadata differs")
    actual = np.empty_like(coordinates)
    obj.data.vertices.foreach_get("co", actual.ravel())
    require(np.array_equal(actual, coordinates), f"Scanner coordinate values changed in {obj.name}")
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    require(np.array_equal(matrix[:3, :3], np.eye(3)), f"Unexpected rotation, reflection, or scale: {obj.name}")
    require(np.array_equal(matrix[3], [0, 0, 0, 1]), "Non-affine display transform")
    if overview:
        offset = np.asarray(obj.get("display_offset_mm"), dtype=np.float64)
        require(offset.shape == (3,) and np.allclose(matrix[:3, 3], offset, rtol=0, atol=2e-5),
                f"Unrecorded overview display translation: {obj.name}")
    else:
        require(np.array_equal(matrix, np.eye(4)), f"Individual model is not in scanner coordinates: {obj.name}")
    if role in {"photo_error", "laser_reference"}:
        errors = get_float_attribute(obj.data, "error_mm")
        require(np.array_equal(errors, expected["reverse_errors" if reference else "errors"]),
                f"Raw error_mm values changed, capped, or reordered: {obj.name}")
    if role == "photo_error":
        require(np.array_equal(get_float_attribute(obj.data, "error_band"), expected["bands"]),
                f"Wrong bin mapping: {obj.name}")
        require(np.allclose(get_color_attribute(obj.data, "error_color"), expected["colors"], rtol=0, atol=2e-7),
                f"Wrong linear RGBA palette mapping: {obj.name}")
    verify_display_modifier(obj)
    require(not obj.hide_viewport, f"Monitor switch disables eye toggling: {obj.name}")
    return matrix[:3, 3].tolist()


def verify_scene(scene, expected_models, overview, error_scene):
    bpy.context.window.scene = scene
    bpy.context.view_layer.update()
    units = scene.unit_settings
    require(units.system == "METRIC" and units.length_unit == "MILLIMETERS"
            and abs(units.scale_length - 0.001) < 1e-9, "Blender scene units must preserve mm")
    objects = [obj for obj in scene.objects if obj.get("role") in ROLES]
    require(not any(obj.get("scan_frame") == 4 for obj in scene.objects), "Excluded scan 0004 in scene")
    require(len(objects) == 3 * len(expected_models), "Missing or duplicate comparison layers")
    require({obj.get("model_name") for obj in objects} == set(expected_models), "Unexpected scene model")
    texts = [obj.data.body for obj in scene.objects if obj.type == "FONT"]
    require(texts, "Saved comparison has no readable annotations")
    if error_scene:
        swatches = [obj for obj in scene.objects if obj.get("role") == "legend_swatch"]
        require(len(swatches) == 7, "Deviation legend lacks seven color bins")
        # Swatch vertices are authored directly at the legend positions.
        swatches.sort(key=lambda obj: float(min(vertex.co.x for vertex in obj.data.vertices)) if overview
                      else -float(min(vertex.co.y for vertex in obj.data.vertices)))
        for index, obj in enumerate(swatches):
            require(len(obj.data.materials) == 1
                    and np.allclose(obj.data.materials[0].diffuse_color, palette_linear()[index], rtol=0, atol=2e-7),
                    "Legend color order disagrees with error bands")
        labels = ["< 0,10 mm", "0,10–0,25", "0,25–0,50", "0,50–1,00",
                  "1,00–2,00", "2,00–5,00", "≥ 5,00 mm"]
        require(all(label in texts for label in labels), "Readable legend boundaries differ from point bins")
    for model, expected in expected_models.items():
        model_objects = [obj for obj in objects if obj.get("model_name") == model]
        require({obj.get("role") for obj in model_objects} == ROLES, f"Incomplete layers for {model}")
        offsets = []
        for obj in model_objects:
            offsets.append(verify_object(obj, expected, overview))
            role = obj["role"]
            should_show = role == "photo_error" if error_scene else role in {"laser_reference", "photo_overlay"}
            require(obj.visible_get() == should_show, f"Wrong initial visibility in {scene.name}: {obj.name}")
            require(obj.hide_get() == (not should_show), f"Eye state mismatch: {obj.name}")
            obj.hide_set(not should_show)
            require(obj.visible_get() == should_show, "Setting expected eye state failed")
            obj.hide_set(should_show)
            require(obj.visible_get() == (not should_show), f"Eye icon does not reveal/hide {obj.name}")
            obj.hide_set(not should_show)
        require(np.array_equal(offsets, [offsets[0]] * 3), f"Photo and reference offsets differ: {model}")
        stats = expected["report"]["forward_photo_to_scan"]
        text_stats = [f"{stats[key]:.3f}".replace(".", ",") + " mm"
                      for key in (("rmse_mm", "p95_mm") if overview else ("median_mm", "rmse_mm", "p95_mm"))]
        require(any(all(value in body for value in text_stats) for body in texts),
                f"Visible statistic labels differ from measured results for {model}")
    if not overview:
        expected = next(iter(expected_models.values()))
        report = expected["report"]
        require(scene.get("comparison") == report["comparison"], "Scene comparison metadata")
        require(scene.get("model_name") == report["model"], "Scene model metadata")
        require(int(scene.get("scan_frame", -1)) == report["scan_frame"], "Scene reference metadata")
        stats = report["forward_photo_to_scan"]
        for key in ("median_mm", "rmse_mm", "p95_mm"):
            require(np.isclose(float(scene[key]), stats[key], rtol=0, atol=1e-12), f"Scene {key} differs")
        require(np.array_equal(np.asarray(scene["color_bin_upper_edges_mm"]), EDGES), "Legend boundaries differ")
    return {"scene": scene.name, "mode": "deviations" if error_scene else "overlay",
            "models": list(expected_models), "comparison_objects": len(objects),
            "eye_icon_controls_verified": True}


def verify_file(path, expected_models, overview=False):
    require(path.is_file(), f"Missing deliverable: {path}")
    original_hash = sha256(path)
    bpy.ops.wm.open_mainfile(filepath=str(path))
    scenes = list(bpy.data.scenes)
    require(len(scenes) == 2, f"Expected deviation and overlay scenes in {path.name}")
    deviation_scenes = [scene for scene in scenes if scene.name.startswith("01")]
    overlay_scenes = [scene for scene in scenes if scene.name.startswith("02")]
    require(len(deviation_scenes) == len(overlay_scenes) == 1, "Scene names must identify two viewing modes")
    require(bpy.context.scene == deviation_scenes[0], "File must open in deviation view")
    checks = [verify_scene(deviation_scenes[0], expected_models, overview, True),
              verify_scene(overlay_scenes[0], expected_models, overview, False)]
    require(sha256(path) == original_hash, "Validation unexpectedly modified the saved file")
    return {"file": path.name, "sha256": original_hash, "scenes": checks,
            "saved_file_reopened": True, "scanner_coordinates_exact_float32": True,
            "uncapped_error_attributes_exact_float32": True, "display_band_colors_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[], help="Verify one selected model; repeatable")
    parser.add_argument("--skip-overview", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    expected = load_expectations(args.model)
    checks = [verify_file(DEST / f"{model}_virhevertailu.blend", {model: data})
              for model, data in expected.items()]
    if not args.skip_overview and not args.model:
        fused = {name: data for name, data in expected.items() if name.endswith("_threepiece")}
        require(len(fused) == 5, "Expected five fused overview models")
        checks.append(verify_file(DEST / "Kaikki_kappaleet_virhevertailu.blend", fused, overview=True))
    baseline = json.loads((RESULTS / "verification.json").read_text(encoding="utf-8"))
    raw = baseline["raw_original_sha256"]
    for name, original_hash in raw.items():
        require(sha256(ROOT / "LaserScans" / name) == original_hash, f"Original scanner file changed: {name}")
    report = {"status": "complete_pass" if len(expected) == 11 and len(checks) == 12 else "selected_files_pass",
              "verified_files": len(checks), "verified_individual_models": len(expected),
              "raw_scanner_files_unchanged": len(raw), "excluded_opposite_face_scan": 4,
              "harmu_matching_scan": 3, "verification_source": "independent raw PCD and saved comparison arrays",
              "storage_note": "Blender stores geometry and attributes at float32 precision; no physical rescaling or error clipping.",
              "comparisons": {model: {"distances_sha256": data["distances_sha256"],
                                       "report_sha256": data["report_sha256"]} for model, data in expected.items()},
              "files": checks}
    path = DEST / ("verification.json" if report["status"] == "complete_pass" else "verification_selected.json")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("LASER_COMPARISON_BLENDER_VERIFIED", json.dumps({"files": len(checks), "models": len(expected),
                                                         "raw_scans_unchanged": len(raw)}), flush=True)


if __name__ == "__main__":
    main()
