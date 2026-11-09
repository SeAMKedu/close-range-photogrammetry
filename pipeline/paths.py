"""Stable project paths, independent of the script or current directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
TOOLS_DIR = PROJECT_ROOT / "tools"
