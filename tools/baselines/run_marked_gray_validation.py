"""Run the Basler and scanner validation for marked_terminal_gray."""
from __future__ import annotations

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from pipeline.compare_profile_reference import run_profile_comparison
from tools.baselines.run_full_pipeline import run_full_pipeline


def main() -> None:
    run_full_pipeline("marked_gray_full_sequence")
    run_profile_comparison(
        "marked_gray_full_sequence",
        "marked_gray_profile_comparison",
    )


if __name__ == "__main__":
    main()
