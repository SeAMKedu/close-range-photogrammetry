"""Cache compatibility must stop at the audited relocation, not future edits."""
import hashlib
import json
import unittest
from unittest.mock import patch

from pipeline.paths import PROJECT_ROOT
from pipeline.source_fingerprints import (
    RELOCATION_MANIFEST,
    fingerprints_match,
    source_hash_matches,
)


class SourceFingerprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries = json.loads(RELOCATION_MANIFEST.read_text(encoding="utf-8"))["sources"]

    def test_manifest_matches_the_actual_audited_files(self):
        for relative, entry in self.entries.items():
            with self.subTest(source=relative):
                actual = hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()
                self.assertEqual(actual, entry["relocated_sha256"])

    def test_saved_pre_move_and_current_hashes_are_accepted(self):
        for relative, entry in self.entries.items():
            current = entry["relocated_sha256"]
            self.assertTrue(source_hash_matches(entry["original_sha256"], current, relative))
            self.assertTrue(source_hash_matches(current, current, relative))

    def test_later_source_edits_reject_old_and_legacy_hashes(self):
        future = hashlib.sha256(b"a later algorithm change").hexdigest()
        legacy = hashlib.sha256(b"historical implementation").hexdigest()
        for relative, entry in self.entries.items():
            self.assertFalse(source_hash_matches(entry["original_sha256"], future, relative))
            self.assertFalse(source_hash_matches(legacy, future, relative, legacy_hashes=(legacy,)))
            self.assertTrue(source_hash_matches(future, future, relative))

    def test_no_alias_for_unknown_source_or_unknown_hash(self):
        relative, entry = next(iter(self.entries.items()))
        self.assertFalse(source_hash_matches("unknown", entry["relocated_sha256"], relative))
        self.assertFalse(source_hash_matches(entry["original_sha256"],
                                             entry["relocated_sha256"], "unlisted.py"))

    def test_relocation_cannot_hide_input_or_fingerprint_schema_changes(self):
        relative = "tools/visualization/create_model_orbits.py"
        entry = self.entries[relative]
        current = {"source": "geometry", "script": entry["relocated_sha256"], "manifest": "inputs"}
        saved = dict(current, script=entry["original_sha256"])
        self.assertTrue(fingerprints_match(saved, current, relative, hash_key="script"))
        for bad in [None, {}, dict(saved, source="changed"), dict(saved, manifest="changed"),
                    dict(saved, extra="unexpected")]:
            with self.subTest(saved=bad):
                self.assertFalse(fingerprints_match(bad, current, relative, hash_key="script"))

    def test_comparison_and_independent_verifier_preserve_sini_restriction(self):
        from pipeline import compare_laser_scans as comparison
        from pipeline import verify_laser_comparisons as verification

        entry = self.entries["pipeline/compare_laser_scans.py"]
        current = {"method_sha256": entry["relocated_sha256"], "leveling_sha256": "unchanged-leveling"}
        old = dict(current, method_sha256=entry["original_sha256"])
        legacy = dict(current, method_sha256=comparison.LEGACY_METHOD_SHA256)
        future = dict(current, method_sha256=hashlib.sha256(b"later comparison edit").hexdigest())
        for fingerprint in [current, future]:
            with patch.object(comparison, "method_fingerprint", return_value=fingerprint):
                for model in ["Sini_single", "Sini_threepiece", "Harmu_threepiece"]:
                    for saved in [old, legacy, dict(old, leveling_sha256="changed-leveling")]:
                        expected = (fingerprint == current and saved["leveling_sha256"] == current["leveling_sha256"]
                                    and (saved == old or not model.startswith("Sini_")))
                        registration = {"method_fingerprint": saved}
                        with self.subTest(model=model, saved=saved, current=fingerprint):
                            self.assertEqual(comparison.cached_method_matches(registration, model), expected)
                            self.assertEqual(verification.fingerprint_matches(registration, model, fingerprint), expected)


if __name__ == "__main__":
    unittest.main()
