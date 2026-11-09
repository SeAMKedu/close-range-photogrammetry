"""Run the full Basler reconstruction and scanner comparison for terminal_gray."""
from __future__ import annotations

from compare_profile_reference import run_profile_comparison
from run_full_pipeline import run_full_pipeline


def main() -> None:
    run_full_pipeline("gray_full_sequence")
    run_profile_comparison("gray_full_sequence", "gray_profile_comparison")


if __name__ == "__main__":
    main()
