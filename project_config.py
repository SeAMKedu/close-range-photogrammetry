from pathlib import Path
import yaml


def load_project_config(path: Path) -> dict:
    """Load the project YAML configuration file."""
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config