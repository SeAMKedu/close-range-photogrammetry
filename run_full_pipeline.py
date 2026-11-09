"""Run the accepted Basler pipeline for every frame of a configured object."""
from __future__ import annotations

import json
import time
from pathlib import Path

from project_config import load_project_config
from run_registered_filter import run_registered_filter
from run_registration_baseline import run_registration
from run_sequence_baseline import run_sequence


def run_full_pipeline(sequence_config_key: str = "full_sequence") -> dict:
    started = time.perf_counter()
    sequence_report = run_sequence(sequence_config_key)
    if sequence_report["missing_or_failed_frame_keys"]:
        raise RuntimeError(
            "Full reconstruction has missing frames: "
            f"{sequence_report['missing_or_failed_frame_keys']}"
        )

    registration_report = run_registration(
        sequence_config_key,
        "registration_baseline",
    )
    filter_report = run_registered_filter(sequence_config_key)
    overall = {
        "object": sequence_report["object"],
        "requested_frames": len(sequence_report["requested_frame_keys"]),
        "reconstructed_frames": sequence_report["produced_frames"],
        "registered_frames": registration_report["registered_frames"],
        "registration_failures": registration_report["failed_frame_keys"],
        "registration_review_frames": registration_report[
            "review_recommended_frame_keys"
        ],
        "triangulated_points": sequence_report["total_triangulated_points"],
        "frame_filtered_points": sequence_report["total_filtered_points"],
        "registered_points": registration_report["merged_points"],
        "plane_filtered_points": filter_report["filtered_points"],
        "plane_filter_retained_percent": filter_report["retained_percent"],
        "plane_residual_after_mm": filter_report["residual_mm_after"],
        "processing_time_s": time.perf_counter() - started,
    }
    config = load_project_config(Path("project_config.yaml"))
    sequence = config[sequence_config_key]
    output_root = (
        Path(config["img_root"])
        / sequence["output_folder"]
        / sequence["object_name"]
    )
    (output_root / "full_pipeline_report.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    print(json.dumps(overall, indent=2))
    return overall


def main() -> None:
    run_full_pipeline()


if __name__ == "__main__":
    main()
