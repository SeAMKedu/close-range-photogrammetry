"""Recognize audited source relocations without accepting later code changes.

New reports always record the actual source hash. Historical hashes are accepted
only while the current file matches the exact relocated hash in the manifest.
This module has no numerical dependencies, so Blender can use it as well.
"""
from pathlib import Path
import json


RELOCATION_MANIFEST = Path(__file__).with_name("source_relocations.json")


def source_hash_matches(saved, current, relative_source, *, legacy_hashes=()):
    """Compare SHA256 values calculated from source bytes by the caller."""
    if saved == current:
        return True
    manifest = json.loads(RELOCATION_MANIFEST.read_text(encoding="utf-8"))
    entry = manifest["sources"].get(relative_source)
    if entry is None or current != entry["relocated_sha256"]:
        return False
    return saved == entry["original_sha256"] or saved in legacy_hashes


def fingerprints_match(saved, current, relative_source, *, hash_key,
                       legacy_hashes=()):
    """Allow only the audited source hash to differ; inputs must match exactly."""
    if not isinstance(saved, dict) or saved.keys() != current.keys():
        return False
    if hash_key not in current:
        return False
    if any(saved[key] != value for key, value in current.items() if key != hash_key):
        return False
    return source_hash_matches(saved[hash_key], current[hash_key], relative_source,
                               legacy_hashes=legacy_hashes)
