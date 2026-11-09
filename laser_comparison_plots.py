"""Scientific figures for rigid photogrammetry/reference-cloud comparisons.

Plots do not register, filter, rescale or otherwise change either input cloud.
All distance statistics and CDFs use every supplied distance. Only point-cloud
overlays are deterministically subsampled to make overlapping samples legible.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
import textwrap

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "3dlaatu-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np


PHOTO_COLOR = "#D55E00"
REFERENCE_COLOR = "#0072B2"
COLOR_LIMIT_MM = 2.0


def _validate(points: np.ndarray, distances: np.ndarray, label: str):
    points = np.asarray(points, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"{label} must be a nonempty N x 3 array")
    if distances.ndim != 1 or len(distances) != len(points):
        raise ValueError(f"{label} distances must contain one value per point")
    if not np.isfinite(points).all() or not np.isfinite(distances).all():
        raise ValueError(f"{label} contains non-finite coordinates or distances")
    if (distances < 0).any():
        raise ValueError(f"{label} distances must be nonnegative")
    return points, distances


def _indices(size: int, maximum: int) -> np.ndarray:
    """A stable sample that includes the first and last stored point."""
    return np.linspace(0, size - 1, min(size, maximum), dtype=np.int64)


def _stats(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(len(values)),
        "median_mm": float(np.median(values)),
        "rmse_mm": float(np.sqrt(np.mean(np.square(values)))),
        "p95_mm": float(np.percentile(values, 95)),
        "maximum_mm": float(values.max()),
        "at_or_below_0_5_mm_percent": float(100 * np.mean(values <= 0.5)),
        "at_or_below_1_mm_percent": float(100 * np.mean(values <= 1.0)),
    }


def _equal_limits(axes, points: np.ndarray, dimensions: tuple[int, int]):
    minima = points[:, dimensions].min(axis=0)
    maxima = points[:, dimensions].max(axis=0)
    span = np.maximum(maxima - minima, 0.5)
    for ax in axes:
        ax.set_xlim(minima[0] - 0.05 * span[0], maxima[0] + 0.05 * span[0])
        ax.set_ylim(minima[1] - 0.05 * span[1], maxima[1] + 0.05 * span[1])
        ax.set_aspect("equal", adjustable="box")


def plot_comparison(
    name: str,
    photo_aligned: np.ndarray,
    reference: np.ndarray,
    forward_dist: np.ndarray,
    reverse_dist: np.ndarray,
    output_dir: Path,
    metadata: dict | None = None,
) -> dict[str, str]:
    """Write top/side overlays, a distance map and full-sample empirical CDFs.

    ``forward_dist`` is photo-to-reference nearest-3D-point distance;
    ``reverse_dist`` is reference-to-photo nearest-3D-point distance. Coordinates
    and distances must already be expressed in millimetres in one frame.
    Return absolute paths under keys ``png``, ``pdf`` and ``text``.
    """
    photo, forward = _validate(photo_aligned, forward_dist, "photogrammetry")
    scan, reverse = _validate(reference, reverse_dist, "reference")
    metadata = metadata or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "comparison"
    paths = {kind: output_dir / f"{stem}_comparison.{suffix}"
             for kind, suffix in (("png", "png"), ("pdf", "pdf"), ("text", "txt"))}
    fs, rs = _stats(forward), _stats(reverse)
    photo_indices, scan_indices = _indices(len(photo), 30000), _indices(len(scan), 30000)
    shown_photo, shown_scan = photo[photo_indices], scan[scan_indices]
    all_coordinates = np.vstack((photo, scan))
    side_axis = int(np.argmax(np.ptp(all_coordinates[:, :2], axis=0)))
    side_label = "XY"[side_axis]
    above_cap = int(np.count_nonzero(forward > COLOR_LIMIT_MM))

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelsize": 10, "axes.titlesize": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.2))
        fig.subplots_adjust(left=0.075, right=0.95, top=0.88, bottom=0.18,
                            wspace=0.30, hspace=0.36)
        ax_xy, ax_side, ax_map, ax_cdf = axes.flat
        fig.suptitle(f"{name}: laser reference comparison", fontsize=15, y=0.973)
        status = str(metadata.get("status", "Provisional rigid registration; inspect overlays"))
        fig.text(0.5, 0.935, textwrap.fill(status.replace("_", " "), 125),
                 ha="center", va="center", fontsize=9)

        for ax, dims in ((ax_xy, (0, 1)), (ax_side, (side_axis, 2))):
            ax.scatter(shown_scan[:, dims[0]], shown_scan[:, dims[1]], s=1.4,
                       c=REFERENCE_COLOR, alpha=0.48, linewidths=0, rasterized=True)
            ax.scatter(shown_photo[:, dims[0]], shown_photo[:, dims[1]], s=1.3,
                       c=PHOTO_COLOR, alpha=0.55, linewidths=0, rasterized=True)
        ax_xy.set_title("(a) Registered clouds: top view", loc="left")
        ax_xy.set_xlabel("Scanner X (mm)")
        ax_xy.set_ylabel("Scanner Y (mm)")
        ax_xy.legend(handles=[
            Line2D([], [], marker="o", linestyle="none", color=REFERENCE_COLOR,
                   markersize=4, label=f"Laser: {len(shown_scan):,} / {len(scan):,} shown"),
            Line2D([], [], marker="o", linestyle="none", color=PHOTO_COLOR,
                   markersize=4, label=f"Photo: {len(shown_photo):,} / {len(photo):,} shown")],
            fontsize=7.5, loc="best", framealpha=0.92)
        ax_side.set_title(f"(b) Registered clouds: {side_label}Z side view", loc="left")
        ax_side.set_xlabel(f"Scanner {side_label} (mm)")
        ax_side.set_ylabel("Scanner Z (mm)")
        _equal_limits([ax_xy, ax_map], all_coordinates, (0, 1))
        _equal_limits([ax_side], all_coordinates, (side_axis, 2))

        # Render every photogrammetry sample; larger residuals are drawn last.
        order = np.argsort(forward, kind="stable")
        mapped = ax_map.scatter(photo[order, 0], photo[order, 1], c=forward[order],
                                s=1.7, cmap="viridis", vmin=0, vmax=COLOR_LIMIT_MM,
                                linewidths=0, rasterized=True)
        ax_map.set_title("(c) Photo-to-laser 3D distance", loc="left")
        ax_map.set_xlabel("Scanner X (mm)")
        ax_map.set_ylabel("Scanner Y (mm)")
        colorbar = fig.colorbar(mapped, ax=ax_map, fraction=0.048, pad=0.035,
                               extend="max", ticks=[0, 0.5, 1, 1.5, 2])
        colorbar.set_label("Nearest-point distance (mm)")
        ax_map.text(0.03, 0.03, f"{above_cap:,} samples\nabove 2 mm",
                    transform=ax_map.transAxes, fontsize=7.5, va="bottom", ha="left",
                    bbox={"facecolor": "white", "alpha": 1.0, "edgecolor": "none",
                          "boxstyle": "square,pad=0.25"})

        for values, label, color, linestyle in (
            (forward, "Photo to laser", PHOTO_COLOR, "-"),
            (reverse, "Laser to photo", REFERENCE_COLOR, "--"),
        ):
            ordered = np.sort(values)
            percent = 100 * np.arange(1, len(values) + 1) / len(values)
            ax_cdf.step(np.r_[0, ordered], np.r_[0, percent], where="post",
                        color=color, linestyle=linestyle, linewidth=1.5,
                        label=label, rasterized=True)
        maximum = max(float(forward.max()), float(reverse.max()), 0.1)
        ax_cdf.set_xscale("symlog", linthresh=0.1, linscale=0.5)
        ax_cdf.set_xlim(0, maximum * 1.05)
        ticks = [x for x in [0, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]
                 if x <= maximum * 1.05]
        ax_cdf.set_xticks(ticks)
        ax_cdf.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax_cdf.set_ylim(0, 100)
        ax_cdf.set_yticks([0, 25, 50, 75, 95, 100])
        ax_cdf.set_title("(d) Cumulative distribution: all points", loc="left")
        ax_cdf.set_xlabel("Nearest-point distance (mm)")
        ax_cdf.set_ylabel("Samples at or below distance (%)")
        ax_cdf.grid(axis="y", alpha=0.25)
        ax_cdf.legend(loc="lower right", fontsize=8)
        ax_cdf.text(0.02, 0.97, "X axis: linear below 0.1 mm, logarithmic above",
                    transform=ax_cdf.transAxes, fontsize=7.5, va="top")
        for ax in axes.flat:
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(direction="out")

        fig.text(0.075, 0.124,
                 f"Photo to laser: median {fs['median_mm']:.3f} mm | RMSE {fs['rmse_mm']:.3f} mm | "
                 f"95th percentile {fs['p95_mm']:.3f} mm | max {fs['maximum_mm']:.3f} mm", fontsize=9)
        fig.text(0.075, 0.096,
                 f"Laser to photo: median {rs['median_mm']:.3f} mm | RMSE {rs['rmse_mm']:.3f} mm | "
                 f"95th percentile {rs['p95_mm']:.3f} mm | max {rs['maximum_mm']:.3f} mm", fontsize=9)
        excluded_face = metadata.get("accuracy_comparison_eligible") is False
        fig.text(0.075, 0.064,
                 "OPPOSITE PHYSICAL FACE: diagnostic distances only; excluded from photogrammetry accuracy assessment."
                 if excluded_face else
                 "Distances describe agreement after rigid alignment, not calibrated measurement uncertainty. "
                 "No evaluation-distance trimming.", fontsize=8)
        fig.text(0.075, 0.040,
                 "The photographed Harmu face is scan 0003, identified by the rounded protrusions in its recessed pockets."
                 if excluded_face else
                 ("Sini: fixed horizontal handedness conversion based on asymmetric recesses, followed by rigid alignment."
                  if metadata.get('registration', {}).get('source_coordinate_conversion') else
                  "Reverse distances also reflect missing photogrammetric coverage. Top and side views use equal physical axis scales."),
                 fontsize=8)
        fig.savefig(paths["png"], dpi=220, facecolor="white")
        fig.savefig(paths["pdf"], dpi=220, facecolor="white")
        plt.close(fig)

    description = {
        "name": str(name),
        "coordinate_unit": "mm",
        "metric": "raw unsigned nearest-3D-point distance after rigid registration",
        "photo_to_laser": fs,
        "laser_to_photo": rs,
        "forward_color_cap_mm": COLOR_LIMIT_MM,
        "forward_points_above_color_cap": above_cap,
        "overlay_photo_points_displayed": int(len(shown_photo)),
        "overlay_reference_points_displayed": int(len(shown_scan)),
        "cdf_samples": "all supplied points in each direction, without distance trimming",
        "side_view": f"{side_label}Z, equal physical axis scales",
        "interpretation": ("Opposite physical face: diagnostic only, excluded from accuracy assessment."
                           if metadata.get("accuracy_comparison_eligible") is False else
                           "Post-alignment agreement, not calibrated measurement uncertainty. "
                           "Reverse distances also measure missing photogrammetric coverage."),
        "metadata": metadata,
    }
    paths["text"].write_text(json.dumps(description, indent=2, default=str) + "\n", encoding="utf-8")
    return {key: str(path.resolve()) for key, path in paths.items()}
