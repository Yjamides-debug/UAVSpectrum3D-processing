#!/usr/bin/env python3
"""Generate v0.4.0 coverage, QC, and reference-reuse manuscript figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .plot_spatial_coverage_heatmap import (
        configure_matplotlib as configure_coverage_matplotlib,
        plot_spatial_coverage as plot_spatial_coverage_heatmap,
    )
except ImportError:
    from plot_spatial_coverage_heatmap import (
        configure_matplotlib as configure_coverage_matplotlib,
        plot_spatial_coverage as plot_spatial_coverage_heatmap,
    )


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


def save_publication_figure(fig, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.png", dpi=600, bbox_inches="tight")


def configure_matplotlib():
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


def plot_qc(work_dir: Path) -> None:
    import matplotlib.pyplot as plt

    qc = json.loads(
        (work_dir / "03_quality_audits" / "five_dimension_qc_summary.json").read_text(
            encoding="utf-8"
        )
    )
    source = json.loads(
        (work_dir / "01_source_audit" / "source_audit_summary.json").read_text(
            encoding="utf-8"
        )
    )
    grid = json.loads(
        (work_dir / "07_grid_sensitivity" / "grid_sensitivity_summary.json").read_text(
            encoding="utf-8"
        )
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.35), constrained_layout=True)
    axes = axes.ravel()
    blue, orange, grey, red = "#4C78A8", "#E58C50", "#5B6770", "#B44C4C"

    axes[0].bar(["Raw rows", "Spatial rows"], [source["input_records"], source["records_with_valid_position"]], color=[grey, blue])
    axes[0].set_ylabel("Records")
    axes[0].set_title("a  Record retention", loc="left", fontweight="bold")
    axes[0].text(0.5, 0.88, "5,700 total\n5,468 with position", transform=axes[0].transAxes, ha="center")

    axes[1].bar(["Latitude", "Longitude", "Pairs"], [12, 3, 26], color=[blue, blue, orange])
    axes[1].set_ylabel("Unique exported values")
    axes[1].set_title("b  GPS quantization", loc="left", fontweight="bold")

    axes[2].bar(["Missing time", "Duplicate time", "Long interval"], [0, 0, 0], color=grey)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Count")
    axes[2].set_title("c  Temporal completeness", loc="left", fontweight="bold")
    axes[2].text(0.5, 0.65, "All audited counts = 0", transform=axes[2].transAxes, ha="center")

    axes[3].bar(["201-bin spectra", "Floor-like", "Positive candidate"], [1145700, qc["power_quality"]["floor_like_values"], qc["power_quality"]["positive_power_candidate_values"]], color=[blue, red, orange])
    axes[3].set_yscale("log")
    axes[3].set_ylabel("Record-frequency entries (log)")
    axes[3].set_title("d  Spectral quality flags", loc="left", fontweight="bold")
    axes[3].tick_params(axis="x", rotation=22)

    occupancy = [
        grid["configuration"][0]["space_height_occupancy_ratio"] * 100,
        grid["configuration"][2]["space_height_occupancy_ratio"] * 100,
    ]
    axes[4].bar(["10 m", "20 m"], occupancy, color=[blue, orange])
    axes[4].set_ylabel("Occupied x-y-z cells within bounds (%)")
    axes[4].set_title("e  Tensor occupancy", loc="left", fontweight="bold")
    for i, value in enumerate(occupancy):
        axes[4].text(i, value + 0.08, f"{value:.2f}%", ha="center")

    axes[5].axis("off")
    axes[5].text(0.02, 0.92, "Interpretation", fontweight="bold", fontsize=8)
    axes[5].text(0.02, 0.76, "Flags are retained diagnostics.\n10/20 m are aggregation grids.\nNo panel establishes GPS accuracy.", va="top", linespacing=1.5)
    for axis in axes[:5]:
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.7)
    save_publication_figure(fig, work_dir / "03_quality_audits", "quality_control_summary")
    plt.close(fig)


def plot_spatial_coverage(work_dir: Path) -> None:
    """Show measured-coordinate coverage without implying positioning accuracy."""
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

    pair_stats: dict[tuple[float, float], dict] = defaultdict(
        lambda: {"record_count": 0, "height_cells": set(), "z_values": []}
    )
    xyz_counts: dict[tuple[float, float, int], int] = defaultdict(int)
    for row in valid:
        x_m = float(row["x_raw_m"])
        y_m = float(row["y_raw_m"])
        z_m = float(row["z_relative_label_m"])
        iz = int(round(z_m))
        stats = pair_stats[(x_m, y_m)]
        stats["record_count"] += 1
        stats["height_cells"].add(iz)
        stats["z_values"].append(z_m)
        xyz_counts[(x_m, y_m, iz)] += 1

    horizontal_rows = []
    for pair_id, ((x_m, y_m), stats) in enumerate(sorted(pair_stats.items()), start=1):
        horizontal_rows.append(
            {
                "horizontal_pair_id": pair_id,
                "x_raw_m": f"{x_m:.6f}",
                "y_raw_m": f"{y_m:.6f}",
                "record_count": stats["record_count"],
                "observed_rounded_height_cells": len(stats["height_cells"]),
                "min_z_relative_label_m": f"{min(stats['z_values']):.6f}",
                "max_z_relative_label_m": f"{max(stats['z_values']):.6f}",
            }
        )

    grid_rows = []
    grid_plot_data = {}
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
        plotted = []
        for (ix, iy), stats in sorted(horizontal_cells.items()):
            item = {
                "grid_m": f"{grid_m:.0f}",
                "ix": ix,
                "iy": iy,
                "x_center_m": f"{ix * grid_m:.6f}",
                "y_center_m": f"{iy * grid_m:.6f}",
                "record_count": stats["record_count"],
                "observed_space_height_cells": len(stats["height_cells"]),
            }
            grid_rows.append(item)
            plotted.append(item)
        grid_plot_data[grid_m] = plotted

    output = work_dir / "03_quality_audits"
    write_csv(output / "spatial_coverage_horizontal_summary.csv", horizontal_rows)
    write_csv(output / "spatial_coverage_grid_summary.csv", grid_rows)

    blue, orange, dark = "#4C78A8", "#E58C50", "#313A40"
    fig = plt.figure(figsize=(7.2, 5.35), constrained_layout=True)
    grid_spec = fig.add_gridspec(2, 2, height_ratios=(1.08, 1.0))
    ax_a = fig.add_subplot(grid_spec[0, 0])
    ax_b = fig.add_subplot(grid_spec[0, 1], projection="3d")
    ax_c = fig.add_subplot(grid_spec[1, 0])
    ax_d = fig.add_subplot(grid_spec[1, 1])

    pair_x = np.array([float(row["x_raw_m"]) for row in horizontal_rows])
    pair_y = np.array([float(row["y_raw_m"]) for row in horizontal_rows])
    pair_n = np.array([int(row["record_count"]) for row in horizontal_rows])
    pair_sizes = 18 + 125 * np.sqrt(pair_n / pair_n.max())
    scatter_a = ax_a.scatter(
        pair_x,
        pair_y,
        s=pair_sizes,
        c=pair_n,
        cmap="Blues",
        norm=LogNorm(vmin=max(1, pair_n.min()), vmax=pair_n.max()),
        edgecolor=dark,
        linewidth=0.35,
        alpha=0.88,
    )
    fig.colorbar(scatter_a, ax=ax_a, label="Records per exported coordinate pair", pad=0.02)
    ax_a.set_title("a  Exported-coordinate record density", loc="left", fontweight="bold")
    ax_a.set_xlabel("Local x (m)")
    ax_a.set_ylabel("Local y (m)")
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.grid(color="#DDDDDD", linewidth=0.5, alpha=0.7)
    ax_a.text(0.02, 0.03, f"{len(valid):,} records; {len(horizontal_rows)} horizontal pairs", transform=ax_a.transAxes)

    xyz = sorted(xyz_counts)
    xyz_n = np.array([xyz_counts[item] for item in xyz])
    scatter_b = ax_b.scatter(
        [item[0] for item in xyz],
        [item[1] for item in xyz],
        [item[2] for item in xyz],
        c=[item[2] for item in xyz],
        s=8 + 26 * np.sqrt(xyz_n / xyz_n.max()),
        cmap="cividis",
        alpha=0.78,
        edgecolor="none",
    )
    fig.colorbar(
        scatter_b,
        ax=ax_b,
        label="Rounded relative-height index (m)",
        pad=0.08,
        shrink=0.72,
    )
    ax_b.set_title("b  Relative-height-labelled coverage", loc="left", fontweight="bold")
    ax_b.set_xlabel("Local x (m)", labelpad=2)
    ax_b.set_ylabel("Local y (m)", labelpad=2)
    ax_b.view_init(elev=24, azim=-58)

    for axis, grid_m, color, panel in (
        (ax_c, 10.0, blue, "c"),
        (ax_d, 20.0, orange, "d"),
    ):
        plotted = grid_plot_data[grid_m]
        heights = np.array([int(row["observed_space_height_cells"]) for row in plotted])
        points = axis.scatter(
            [float(row["x_center_m"]) for row in plotted],
            [float(row["y_center_m"]) for row in plotted],
            s=34 if grid_m == 10.0 else 62,
            marker="s",
            c=heights,
            cmap="Oranges" if color == orange else "Blues",
            vmin=1,
            vmax=max(heights.max(), 2),
            edgecolor=dark,
            linewidth=0.3,
        )
        fig.colorbar(points, ax=axis, label="Observed height cells per x-y cell", pad=0.02)
        occupied_xyz = sum(int(row["observed_space_height_cells"]) for row in plotted)
        axis.set_title(f"{panel}  {grid_m:.0f} m aggregation-grid coverage", loc="left", fontweight="bold")
        axis.set_xlabel("Local x-cell centre (m)")
        axis.set_ylabel("Local y-cell centre (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#DDDDDD", linewidth=0.5, alpha=0.7)
        axis.text(
            0.02,
            0.03,
            f"{len(plotted)} occupied x-y cells; {occupied_xyz} occupied x-y-z cells",
            transform=axis.transAxes,
        )

    fig.suptitle(
        "Measured spatial coverage from exported coordinates",
        x=0.01,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    save_publication_figure(fig, output, "spatial_coverage")
    plt.close(fig)


def plot_frequency_completion(work_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    rows = load_csv(work_dir / "06_frequency_completion" / "results_summary.csv")
    methods = ["frequency_mean", "linear_interpolation", "iterative_svd"]
    labels = {
        "frequency_mean": "Frequency mean",
        "linear_interpolation": "Linear interpolation",
        "iterative_svd": "Iterative SVD",
    }
    styles = {
        "frequency_mean": {
            "color": "#73777A",
            "marker": "o",
            "linestyle": (0, (3.0, 2.0)),
        },
        "linear_interpolation": {
            "color": "#4477AA",
            "marker": "s",
            "linestyle": "-",
        },
        "iterative_svd": {
            "color": "#D18C52",
            "marker": "^",
            "linestyle": "-",
        },
    }

    # Double-column composition (approximately 183 mm wide). Method identity is
    # encoded consistently by colour and marker in all three panels.
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.78), constrained_layout=False)
    fig.patch.set_facecolor("white")

    def panel_heading(axis, letter: str, title: str) -> None:
        axis.text(
            -0.17,
            1.075,
            f"({letter})",
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )
        axis.set_title(title, loc="left", fontsize=7.4, fontweight="normal", pad=7)

    def finish_axis(axis) -> None:
        axis.set_facecolor("white")
        axis.grid(axis="y", color="#D9DDE1", linewidth=0.45, alpha=0.75, zorder=0)
        axis.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=3.0,
            width=0.65,
            labelsize=6.5,
            color="#303438",
        )
        axis.spines["left"].set_color("#303438")
        axis.spines["bottom"].set_color("#303438")
        axis.spines["left"].set_linewidth(0.7)
        axis.spines["bottom"].set_linewidth(0.7)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))

    random_rows = [row for row in rows if row["protocol"] == "random_masking"]
    metric_specs = (
        ("RMSE", "rmse_db_mean", "rmse_db_std"),
        ("MAE", "mae_db_mean", "mae_db_std"),
    )
    metric_x = np.arange(len(metric_specs), dtype=float)
    method_offsets = dict(zip(methods, (-0.16, 0.0, 0.16)))
    for method in methods:
        row = next(row for row in random_rows if row["method"] == method)
        style = styles[method]
        values = [float(row[mean_key]) for _, mean_key, _ in metric_specs]
        spreads = [float(row[std_key]) for _, _, std_key in metric_specs]
        axes[0].errorbar(
            metric_x + method_offsets[method],
            values,
            yerr=spreads,
            fmt=style["marker"],
            linestyle="none",
            markersize=4.2,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.45,
            capsize=2.0,
            capthick=0.7,
            elinewidth=0.75,
            color=style["color"],
            zorder=3,
        )
    axes[0].set_xticks(metric_x, [item[0] for item in metric_specs])
    axes[0].set_ylabel("Error (dB)")
    axes[0].set_xlim(-0.42, 1.42)
    axes[0].set_ylim(0.75, 5.15)
    panel_heading(axes[0], "a", "Random masking\n50% observed")

    contiguous = [row for row in rows if row["protocol"] == "contiguous_block"]
    for axis, metric, error, title, panel in (
        (axes[1], "rmse_db_mean", "rmse_db_std", "RMSE (dB)", "b"),
        (axes[2], "mae_db_mean", "mae_db_std", "MAE (dB)", "c"),
    ):
        levels = sorted({float(row["level"]) for row in contiguous})
        level_x = np.arange(len(levels), dtype=float)
        for method in methods:
            selected = [
                next(row for row in contiguous if float(row["level"]) == level and row["method"] == method)
                for level in levels
            ]
            style = styles[method]
            axis.errorbar(
                level_x,
                [float(row[metric]) for row in selected],
                yerr=[float(row[error]) for row in selected],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=4.2,
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markeredgewidth=0.45,
                capsize=2.0,
                capthick=0.7,
                elinewidth=0.75,
                linewidth=1.05,
                color=style["color"],
                zorder=3 if method != "frequency_mean" else 2,
            )
        axis.set_xticks(level_x, [f"{level:.0%}" for level in levels])
        axis.set_xlabel("Hidden contiguous frequency fraction")
        axis.set_ylabel(title)
        axis.set_xlim(-0.12, len(levels) - 0.88)
        panel_heading(axis, panel, "Contiguous block masking")

    axes[1].set_ylim(1.6, 9.05)
    axes[2].set_ylim(0.9, 5.65)
    for axis in axes:
        finish_axis(axis)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=styles[method]["color"],
            marker=styles[method]["marker"],
            linestyle=styles[method]["linestyle"],
            linewidth=1.05,
            markersize=4.2,
            markerfacecolor=styles[method]["color"],
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=labels[method],
        )
        for method in methods
    ]
    fig.suptitle(
        "Frequency completion under different masking strategies",
        x=0.07,
        y=0.985,
        ha="left",
        fontsize=8.6,
        fontweight="normal",
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.895),
        ncol=3,
        frameon=False,
        fontsize=6.5,
        handlelength=2.1,
        columnspacing=1.6,
        handletextpad=0.55,
    )
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.23, top=0.72, wspace=0.39)
    save_publication_figure(fig, work_dir / "06_frequency_completion", "frequency_completion_summary")
    plt.close(fig)


def generate(work_dir: Path) -> None:
    configure_matplotlib()
    plot_qc(work_dir)
    configure_coverage_matplotlib()
    plot_spatial_coverage_heatmap(work_dir)
    plot_frequency_completion(work_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(args.work_dir)


if __name__ == "__main__":
    main()
