
# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path
import yaml
from pipeline.paths import PROJECT_ROOT


def load_project_config(path: Path) -> dict:
    """Load the project YAML configuration file."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config

def make_run_suffix(config: dict) -> str:
    """Create a folder suffix from debug settings."""
    debug = config.get("debug", {})

    if not debug.get("enabled", False):
        return ""

    suffix_parts = []

    reference_camera_index = config["reference_camera_index"]

    target_camera_indices = debug.get("target_camera_indices")
    if target_camera_indices is not None:
        cameras = "_".join(
            f"cam{idx}" for idx in target_camera_indices
        )
        suffix_parts.append(f"cam{reference_camera_index}_{cameras}")

    frame_keys = debug.get("frame_keys_to_use")
    if frame_keys is not None:
        suffix_parts.append(f"frames_{'_'.join(frame_keys)}")

    if not suffix_parts:
        return "_debug"

    return "_" + "_".join(suffix_parts)
