"""Scientific-import checks: round trips and refusal of inconsistent exports."""

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.paths import PROJECT_ROOT

from pathlib import Path
import csv
import shutil
import sys
import tempfile
import unittest

import numpy as np
import open3d as o3d

ROOT = PROJECT_ROOT
from pipeline.laser_reference_io import load_scan


class ReferenceImportTests(unittest.TestCase):
    def setUp(self):
        self.scan_paths = sorted((ROOT / "LaserScans").glob("*.pcd"))
        self.assertEqual(
            len(self.scan_paths), 6,
            "Expected all six scanner PCD exports under LaserScans; restore the external scan data first.",
        )

    def test_all_converted_references_preserve_samples_and_holes(self):
        for pcd in self.scan_paths:
            scan = load_scan(pcd)
            meta = scan["metadata"]
            name = f"frame_{meta['frame_index']:04d}_{meta['inferred_object']}"
            imported = ROOT / "LaserScans" / "reference_import"
            with np.load(imported / f"{name}.npz") as saved:
                np.testing.assert_array_equal(saved["valid"], scan["valid"])
                np.testing.assert_array_equal(saved["grid_xyz"], scan["grid_xyz"])
                np.testing.assert_array_equal(saved["points"], scan["points"])
            cloud = o3d.io.read_point_cloud(str(imported / f"{name}.ply"))
            np.testing.assert_array_equal(np.asarray(cloud.points), scan["points"])
            self.assertEqual(meta["unit"], "mm")

    def test_refuses_corrupt_geometry_units_and_csv_disagreement(self):
        parent = (ROOT / "LaserScans" / "_inspection").resolve()
        source = self.scan_paths[0]
        with tempfile.TemporaryDirectory(prefix="import_test_", dir=parent) as temp:
            destination = Path(temp).resolve()
            self.assertTrue(destination.is_relative_to(parent))
            for suffix in (".pcd", ".csv", ".sur"):
                shutil.copy2(source.with_suffix(suffix), destination / source.with_suffix(suffix).name)
            pcd = destination / source.name
            original_pcd = pcd.read_bytes()
            pcd.write_bytes(original_pcd + b"x")
            with self.assertRaisesRegex(ValueError, "binary length mismatch"):
                load_scan(pcd)
            pcd.write_bytes(original_pcd)
            sur = pcd.with_suffix(".sur")
            original_sur = sur.read_bytes()
            altered = bytearray(original_sur)
            altered[180:182] = b"um"
            sur.write_bytes(altered)
            with self.assertRaisesRegex(ValueError, "units"):
                load_scan(pcd)
            sur.write_bytes(original_sur)
            csv_file = pcd.with_suffix(".csv")
            with csv_file.open(newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.reader(stream))
            part = next(i for i, row in enumerate(rows) if row and row[0] == "Part")
            changed = False
            for row in rows[part + 4:]:
                for i in range(1, len(row)):
                    if row[i].strip():
                        row[i] = str(float(row[i]) + 0.1)
                        changed = True
                        break
                if changed:
                    break
            with csv_file.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerows(rows)
            with self.assertRaisesRegex(ValueError, "mismatch exceeds"):
                load_scan(pcd)


if __name__ == "__main__":
    unittest.main()
