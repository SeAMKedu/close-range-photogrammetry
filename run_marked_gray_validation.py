"""Run the Basler and scanner validation for marked_terminal_gray."""
from __future__ import annotations

from compare_profile_reference import run_profile_comparison
from run_full_pipeline import run_full_pipeline


def main() -> None:
    run_full_pipeline("marked_gray_full_sequence")
    run_profile_comparison(
        "marked_gray_full_sequence",
        "marked_gray_profile_comparison",
    )


if __name__ == "__main__":
    main()
