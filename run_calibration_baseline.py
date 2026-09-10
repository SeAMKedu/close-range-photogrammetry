"""Run existing calibration functions without GUI and record acceptance checks."""
from pathlib import Path
import json
import platform
import subprocess
import cv2
from project_config import load_project_config
from calibrate_cam_with_chessboards import calibrate_camera_from_chessboard_images
from calibrate_stereo_with_chessboards import calibrate_multicamera_extrinsics, save_extrinsics_json


def main() -> None:
    config = load_project_config(Path("project_config.yaml"))
    root = Path(config["img_root"])
    output = root / config["folders"]["calibration_files"]
    output.mkdir(parents=True, exist_ok=True)
    report = {"config": config, "python": platform.python_version(),
              "opencv": cv2.__version__, "git_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], text=True).strip(), "intrinsics": {}}
    paths = []
    for camera in config["camera_names"]:
        path = output / f"{camera}{config['filenames']['calibration_result_ending']}"
        paths.append(path)
        result = calibrate_camera_from_chessboard_images(
            root / config["folders"]["intrinsic_images"] / camera,
            tuple(config["checkerboard"]["board_size"]), config["checkerboard"]["square_size"],
            save_corner_images=True, corner_output_folder=output / "debug" / "intrinsics" / camera,
            save_json_path=path)
        report["intrinsics"][camera] = {"rms_px": result.rms_error,
            "used_images": len(result.used_images), "failed_images": [str(p) for p in result.failed_images],
            "image_size": result.image_size, "passes_0_5_px": result.rms_error < 0.5}
        for folder in [root / config["folders"]["extrinsic_images"] / camera,
                       *[p / camera for p in (root / config["folders"]["objects"]).iterdir() if p.is_dir()]]:
            for image_path in folder.glob("*.jpg"):
                image = cv2.imread(str(image_path))
                if image is None or (image.shape[1], image.shape[0]) != result.image_size:
                    raise ValueError(f"Unreadable image or calibration resolution mismatch: {image_path}")
    result = calibrate_multicamera_extrinsics(
        paths, root / config["folders"]["extrinsic_images"], config["camera_names"],
        tuple(config["checkerboard"]["board_size"]), config["checkerboard"]["square_size"],
        reference_camera_index=config["reference_camera_index"], square_size_unit=config["checkerboard"]["unit"],
        save_corner_images=True, corner_output_folder=output / "debug" / "extrinsics",
        output_json_path=output / "extrinsics_baseline_candidate.json",
        frame_key_offsets=config.get("stereo_sync", {}).get("frame_key_offsets"),
        frame_cycle_length=config.get("stereo_sync", {}).get("frame_cycle_length"))
    report["stereo"] = {config["camera_names"][r.target_camera_index]: {
        "rms_px": r.rms_error, "used_frames": r.used_frames, "passes_1_px": r.rms_error < 1.0}
        for r in result.stereo_results}
    report["all_image_resolutions_match"] = True
    report["accepted"] = all(r["passes_0_5_px"] for r in report["intrinsics"].values()) and all(
        r["passes_1_px"] for r in report["stereo"].values())
    (output / "baseline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["accepted"]:
        raise SystemExit("Calibration rejected: see baseline_report.json. Reconstruction must not proceed.")
    save_extrinsics_json(result, output / config["filenames"]["extrinsics_result"])


if __name__ == "__main__":
    main()
