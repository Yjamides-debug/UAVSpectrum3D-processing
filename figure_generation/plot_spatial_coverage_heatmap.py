#!/usr/bin/env python3
"""Generate a publication-style measured spatial-coverage heatmap."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty source-data table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def save_publication_figure(fig, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.png", dpi=600, bbox_inches="tight")


def gaussian_density(
    pair_x: np.ndarray,
    pair_y: np.ndarray,
    weights: np.ndarray,
    bandwidth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    margin = 1.8 * bandwidth_m
    x_axis = np.linspace(pair_x.min() - margin, pair_x.max() + margin, 300)
    y_axis = np.linspace(pair_y.min() - margin, pair_y.max() + margin, 210)
    xx, yy = np.meshgrid(x_axis, y_axis)
    density = np.zeros_like(xx)
    for x_m, y_m, weight in zip(pair_x, pair_y, weights):
        squared_distance = (xx - x_m) ** 2 + (yy - y_m) ** 2
        density += weight * np.exp(-0.5 * squared_distance / bandwidth_m**2)
    density /= density.max()
    density[density < 0.012] = np.nan
    return xx, yy, density


def build_source_tables(valid: list[dict], output: Path) -> tuple[list[dict], list[dict]]:
    pair_stats: dict[tuple[float, float], dict] = defaultdict(
        lambda: {"record_count": 0, "height_counts": defaultdict(int), "z_values": []}
    )
    for row in valid:
        x_m = float(row["x_raw_m"])
        y_m = float(row["y_raw_m"])
        z_m = float(row["z_relative_label_m"])
        stats = pair_stats[(x_m, y_m)]
        stats["record_count"] += 1
        stats["height_counts"][int(round(z_m))] += 1
        stats["z_values"].append(z_m)

    horizontal_rows = []
    for pair_id, ((x_m, y_m), stats) in enumerate(sorted(pair_stats.items()), start=1):
        horizontal_rows.append(
            {
                "horizontal_pair_id": pair_id,
                "x_raw_m": f"{x_m:.6f}",
                "y_raw_m": f"{y_m:.6f}",
                "record_count": stats["record_count"],
                "observed_rounded_height_cells": len(stats["height_counts"]),
                "min_z_relative_label_m": f"{min(stats['z_values']):.6f}",
                "max_z_relative_label_m": f"{max(stats['z_values']):.6f}",
            }
        )

    grid_rows = []
    for grid_m in (10.0, 20.0):
        horizontal_cells: dict[tuple[int, int], dict] = defaultdict(
            lambda: {"record_count": 0, "height_cells": set()}
        )
        for row in valid:
            ix = int(round(float(row["x_raw_m"]) / grid_m))
            iy = int(round(float(row["y_raw_m"]) / grid_m))
            iz = int(round(float(row["z_relative_label_m"])))
            horizontal_cells[(ix, iy)]["record_count"] += 1
            horizontal_cells[(ix, iy)]["height_cells"].add(iz)
        for (ix, iy), stats in sorted(horizontal_cells.items()):
            grid_rows.append(
                {
                    "grid_m": f"{grid_m:.0f}",
                    "ix": ix,
                    "iy": iy,
                    "x_center_m": f"{ix * grid_m:.6f}",
                    "y_center_m": f"{iy * grid_m:.6f}",
                    "record_count": stats["record_count"],
                    "observed_space_height_cells": len(stats["height_cells"]),
                }
            )

    write_csv(output / "spatial_coverage_horizontal_summary.csv", horizontal_rows)
    write_csv(output / "spatial_coverage_grid_summary.csv", grid_rows)
    return horizontal_rows, grid_rows


def grid_count_array(rows: list[dict], grid_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in rows if float(row["grid_m"]) == grid_m]
    ix_values = np.arange(min(int(row["ix"]) for row in selected), max(int(row["ix"]) for row in selected) + 1)
    iy_values = np.arange(min(int(row["iy"]) for row in selected), max(int(row["iy"]) for row in selected) + 1)
    counts = np.full((len(iy_values), len(ix_values)), np.nan)
    ix_lookup = {value: index for index, value in enumerate(ix_values)}
    iy_lookup = {value: index for index, value in enumerate(iy_values)}
    for row in selected:
        counts[iy_lookup[int(row["iy"])], ix_lookup[int(row["ix"])]] = int(row["record_count"])
    return ix_values * grid_m, iy_values * grid_m, counts


def plot_spatial_coverage(work_dir: Path, bandwidth_m: float = 12.0) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    records = load_csv(work_dir / "02_standardized" / "standardized_records.csv")
    valid = [
        row
        for row in records
        if row.get("position_valid", "").lower() == "true"
        and row.get("x_raw_m", "")
        and row.get("y_raw_m", "")
        and row.get("z_relative_label_m", "")
    ]
    if not valid:
        raise ValueError("No valid exported-coordinate records are available for coverage plotting.")

    output = work_dir / "03_quality_audits"
    horizontal_rows, grid_rows = build_source_tables(valid, output)
    pair_x = np.array([float(row["x_raw_m"]) for row in horizontal_rows])
    pair_y = np.array([float(row["y_raw_m"]) for row in horizontal_rows])
    pair_n = np.array([int(row["record_count"]) for row in horizontal_rows])
    xx, yy, density = gaussian_density(pair_x, pair_y, pair_n, bandwidth_m)

    pair_height = np.zeros((len(horizontal_rows), 40), dtype=float)
    pair_lookup = {
        (float(row["x_raw_m"]), float(row["y_raw_m"])): index
        for index, row in enumerate(horizontal_rows)
    }
    for row in valid:
        pair_index = pair_lookup[(float(row["x_raw_m"]), float(row["y_raw_m"]))]
        height_index = int(round(float(row["z_relative_label_m"])))
        if 1 <= height_index <= 40:
            pair_height[pair_index, height_index - 1] += 1
    pair_height[pair_height == 0] = np.nan

    grid_arrays = {
        grid_m: grid_count_array(grid_rows, grid_m) for grid_m in (10.0, 20.0)
    }
    maximum_count = max(
        np.nanmax(pair_height),
        *(np.nanmax(values[2]) for values in grid_arrays.values()),
    )
    count_norm = LogNorm(vmin=1, vmax=maximum_count)

    fig = plt.figure(figsize=(7.2, 4.65), constrained_layout=True)
    layout = fig.add_gridspec(
        2,
        5,
        width_ratios=(1.08, 1.08, 1.0, 1.0, 0.075),
        height_ratios=(1.08, 1.0),
    )
    ax_a = fig.add_subplot(layout[:, :2])
    ax_b = fig.add_subplot(layout[0, 2:4])
    ax_c = fig.add_subplot(layout[1, 2])
    ax_d = fig.add_subplot(layout[1, 3])
    count_colorbar_axis = fig.add_subplot(layout[:, 4])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

    levels = np.linspace(0.02, 1.0, 16)
    density_plot = ax_a.contourf(xx, yy, density, levels=levels, cmap=cmap, extend="max")
    ax_a.contour(xx, yy, density, levels=(0.1, 0.3, 0.5, 0.7), colors="white", linewidths=0.45, alpha=0.8)
    ax_a.scatter(pair_x, pair_y, s=8, c="white", edgecolors="#263238", linewidths=0.3, zorder=3)
    density_colorbar_axis = ax_a.inset_axes((0.12, 0.91, 0.76, 0.025))
    colorbar_a = fig.colorbar(
        density_plot,
        cax=density_colorbar_axis,
        orientation="horizontal",
    )
    colorbar_a.set_label("Relative sampling density (KDE)", labelpad=2)
    density_colorbar_axis.xaxis.set_ticks_position("top")
    density_colorbar_axis.xaxis.set_label_position("top")
    colorbar_a.set_ticks((0.2, 0.4, 0.6, 0.8, 1.0))
    ax_a.set_title("a  KDE sampling density", loc="left", fontweight="bold")
    ax_a.set_xlabel("Local x (m)")
    ax_a.set_ylabel("Local y (m)")
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.text(
        0.02,
        0.02,
        f"{len(valid):,} records; {len(horizontal_rows)} exported coordinate pairs\nGaussian bandwidth = {bandwidth_m:g} m (visualization only)",
        transform=ax_a.transAxes,
        fontsize=6.2,
        color="#263238",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
    )

    height_plot = ax_b.imshow(
        pair_height,
        origin="lower",
        aspect="auto",
        extent=(0.5, 40.5, 0.5, len(horizontal_rows) + 0.5),
        cmap=cmap,
        norm=count_norm,
        interpolation="nearest",
    )
    ax_b.set_title("b  Position-by-height coverage", loc="left", fontweight="bold")
    ax_b.set_xlabel("Rounded relative-height index (m)")
    ax_b.set_ylabel("Exported horizontal pair ID")
    ax_b.set_yticks((1, 7, 13, 19, 26))

    for axis, grid_m, panel in ((ax_c, 10.0, "c"), (ax_d, 20.0, "d")):
        x_centres, y_centres, counts = grid_arrays[grid_m]
        image_plot = axis.imshow(
            counts,
            origin="lower",
            extent=(
                x_centres.min() - grid_m / 2,
                x_centres.max() + grid_m / 2,
                y_centres.min() - grid_m / 2,
                y_centres.max() + grid_m / 2,
            ),
            cmap=cmap,
            norm=count_norm,
            interpolation="nearest",
            aspect="equal",
        )
        occupied_xy = int(np.isfinite(counts).sum())
        axis.set_title(f"{panel}  {grid_m:g} m grid", loc="left", fontweight="bold")
        axis.set_xlabel("Local x-cell centre (m)")
        axis.set_ylabel("Local y-cell centre (m)" if grid_m == 10.0 else "")
        axis.text(
            0.03,
            0.04,
            f"{occupied_xy} occupied x-y cells",
            transform=axis.transAxes,
            fontsize=6.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
        )

    fig.colorbar(
        height_plot,
        cax=count_colorbar_axis,
        label="Measured records (log scale)",
    )

    fig.suptitle(
        "Measured spatial coverage and sampling density",
        x=0.01,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    save_publication_figure(fig, output, "spatial_coverage")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--bandwidth-m", type=float, default=12.0)
    args = parser.parse_args()
    configure_matplotlib()
    plot_spatial_coverage(args.work_dir, args.bandwidth_m)


if __name__ == "__main__":
    main()
