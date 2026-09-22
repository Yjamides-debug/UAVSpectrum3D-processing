#!/usr/bin/env python3
"""Stage 07: quantify aggregation-grid and quality-policy sensitivity."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavspectrum3d_pipeline.config import DEFAULT_WORK_DIR
from uavspectrum3d_pipeline.utils import write_csv, write_json


def load_product(directory: Path) -> tuple[dict, list[dict]]:
    with np.load(directory / "matrix_product.npz") as arrays:
        product = {name: arrays[name].copy() for name in arrays.files}
    with (directory / "cell_coordinates.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        cells = list(csv.DictReader(handle))
    return product, cells


def distribution(values: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in np.percentile(values, [25, 50, 75, 100]))


def configuration_rows(
    products: dict[str, tuple[dict, list[dict]]]
) -> list[dict]:
    rows = []
    for grid, (product, cells) in products.items():
        grid_m = float(grid.replace("grid_", "").replace("m", ""))
        tensor_path = (
            Path(product["_directory"]) / "tensor_product.npz"
        )
        with np.load(tensor_path) as tensor:
            spatial_height_bound = int(np.prod(tensor["sample_count"].shape[:3]))
        horizontal_cells = len({(int(row["ix"]), int(row["iy"])) for row in cells})
        for policy, mask in (
            ("all", product["observed_mask"]),
            ("strict_quality", product["quality_mask"]),
        ):
            counts = product["sample_count"][mask]
            q1, median, q3, maximum = distribution(counts)
            rows.append(
                {
                    "configuration": f"{grid_m:g}m_{policy}",
                    "grid_m": grid_m,
                    "quality_policy": policy,
                    "observed_space_height_cells": len(cells),
                    "unique_horizontal_grid_cells": horizontal_cells,
                    "bounding_space_height_cells": spatial_height_bound,
                    "space_height_occupancy_ratio": len(cells) / spatial_height_bound,
                    "frequency_bins": product["observed_mask"].shape[1],
                    "observed_entries": int(product["observed_mask"].sum()),
                    "strict_quality_entries": int(product["quality_mask"].sum()),
                    "active_entries_under_policy": int(mask.sum()),
                    "strict_quality_retention_ratio": float(
                        product["quality_mask"].sum() / product["observed_mask"].sum()
                    ),
                    "sample_count_q1": q1,
                    "sample_count_median": median,
                    "sample_count_q3": q3,
                    "sample_count_max": maximum,
                }
            )
    return rows


def mean_spectrum_sensitivity(
    products: dict[str, tuple[dict, list[dict]]]
) -> list[dict]:
    rows = []
    for grid, (product, _) in products.items():
        grid_m = float(grid.replace("grid_", "").replace("m", ""))
        power = product["power_dbm_arithmetic_mean"]
        spectra = {}
        for policy, mask in (
            ("all", product["observed_mask"]),
            ("strict_quality", product["quality_mask"]),
        ):
            spectra[policy] = np.array(
                [np.mean(power[:, index][mask[:, index]]) for index in range(power.shape[1])]
            )
        difference = spectra["strict_quality"] - spectra["all"]
        rows.append(
            {
                "grid_m": grid_m,
                "mae_db": float(np.mean(np.abs(difference))),
                "rmse_db": float(np.sqrt(np.mean(difference**2))),
                "pearson": float(np.corrcoef(spectra["all"], spectra["strict_quality"])[0, 1]),
                "max_abs_difference_db": float(np.max(np.abs(difference))),
            }
        )
    return rows


def reaggregate_10m(
    product: dict, cells: list[dict]
) -> dict[tuple[int, int, int, int], tuple[float, int, int, int]]:
    totals: dict[tuple[int, int, int, int], list[float]] = defaultdict(
        lambda: [0.0, 0, 0, 0]
    )
    power = product["power_dbm_arithmetic_mean"]
    count = product["sample_count"]
    for row_index, cell in enumerate(cells):
        ix20 = map_10m_index_to_20m(int(cell["ix"]))
        iy20 = map_10m_index_to_20m(int(cell["iy"]))
        iz = int(cell["iz"])
        for frequency_index in np.flatnonzero(product["observed_mask"][row_index]):
            key = (ix20, iy20, iz, int(frequency_index))
            weight = int(count[row_index, frequency_index])
            totals[key][0] += float(power[row_index, frequency_index]) * weight
            totals[key][1] += weight
            totals[key][2] += int(product["floor_count"][row_index, frequency_index])
            totals[key][3] += int(product["positive_count"][row_index, frequency_index])
    return {
        key: (value[0] / value[1], int(value[1]), int(value[2]), int(value[3]))
        for key, value in totals.items()
    }


def map_10m_index_to_20m(index: int) -> int:
    """Apply the predefined floor-division mapping used for scale sensitivity.

    The native grids use rounded coordinates and therefore are not strictly
    nested; this mapping does not assert implementation equivalence.
    """
    return index // 2


def native_entries(
    product: dict, cells: list[dict], policy: str
) -> dict[tuple[int, int, int, int], float]:
    values = {}
    power = product["power_dbm_arithmetic_mean"]
    for row_index, cell in enumerate(cells):
        base = (int(cell["ix"]), int(cell["iy"]), int(cell["iz"]))
        mask = product["observed_mask"] if policy == "all" else product["quality_mask"]
        for frequency_index in np.flatnonzero(mask[row_index]):
            values[(*base, int(frequency_index))] = float(power[row_index, frequency_index])
    return values


def cross_grid_rows(
    products: dict[str, tuple[dict, list[dict]]]
) -> tuple[list[dict], list[dict]]:
    product10, cells10 = products["grid_10m"]
    product20, cells20 = products["grid_20m"]
    summaries = []
    common_rows = []
    reaggregated_all = reaggregate_10m(product10, cells10)
    for policy in ("all", "strict_quality"):
        reaggregated = {
            key: value
            for key, value in reaggregated_all.items()
            if policy == "all" or (value[2] == 0 and value[3] == 0)
        }
        native = native_entries(product20, cells20, policy)
        common = sorted(set(reaggregated) & set(native))
        estimate = np.array([reaggregated[key][0] for key in common])
        reference = np.array([native[key] for key in common])
        difference = estimate - reference
        summaries.append(
            {
                "policy": policy,
                "reaggregated_10m_entries": len(reaggregated),
                "native_20m_entries": len(native),
                "common_entries": len(common),
                "common_fraction_of_native": len(common) / max(len(native), 1),
                "mae_db": float(np.mean(np.abs(difference))),
                "rmse_db": float(np.sqrt(np.mean(difference**2))),
                "mean_signed_difference_db": float(np.mean(difference)),
                "max_abs_difference_db": float(np.max(np.abs(difference))),
                "pearson": float(np.corrcoef(reference, estimate)[0, 1]),
            }
        )
        for key, predicted, measured in zip(common, estimate, reference):
            common_rows.append(
                {
                    "policy": policy,
                    "ix_20m": key[0],
                    "iy_20m": key[1],
                    "iz": key[2],
                    "frequency_index": key[3],
                    "reaggregated_10m_power_dbm": predicted,
                    "native_20m_power_dbm": measured,
                    "difference_db": predicted - measured,
                }
            )
    return summaries, common_rows


def make_figure(
    output: Path,
    configuration: list[dict],
    cross_summary: list[dict],
    common_rows: list[dict],
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

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
    all_rows = [row for row in configuration if row["quality_policy"] == "all"]
    strict_rows = [row for row in configuration if row["quality_policy"] == "strict_quality"]
    labels = [f"{int(row['grid_m'])} m" for row in all_rows]
    colors = ["#4C78A8", "#E58C50"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    x = np.arange(len(labels))
    axes[0].bar(x, [100 * row["space_height_occupancy_ratio"] for row in all_rows], color=colors)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Occupied bounding cells (%)")
    axes[0].set_title("a  Sparse tensor occupancy", loc="left", fontweight="bold")
    axes[1].bar(x, [row["sample_count_median"] for row in all_rows], color=colors)
    for index, row in enumerate(all_rows):
        axes[1].vlines(index, row["sample_count_q1"], row["sample_count_q3"], color="#222222", linewidth=1.2)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Contributing samples per entry")
    axes[1].set_title("b  Aggregation density", loc="left", fontweight="bold")
    selected = [row for row in common_rows if row["policy"] == "strict_quality"]
    if len(selected) > 4000:
        selected = selected[:: max(1, len(selected) // 4000)]
    native = np.array([row["native_20m_power_dbm"] for row in selected])
    reagg = np.array([row["reaggregated_10m_power_dbm"] for row in selected])
    axes[2].scatter(native, reagg, s=4, alpha=0.18, color="#4C78A8", rasterized=True)
    lower = float(min(native.min(), reagg.min()))
    upper = float(max(native.max(), reagg.max()))
    axes[2].plot([lower, upper], [lower, upper], color="#333333", linewidth=0.8)
    metric = next(row for row in cross_summary if row["policy"] == "strict_quality")
    axes[2].text(0.04, 0.96, f"RMSE = {metric['rmse_db']:.2f} dB\nr = {metric['pearson']:.3f}", transform=axes[2].transAxes, va="top")
    axes[2].set_xlabel("Native 20 m power (dBm)")
    axes[2].set_ylabel("10 m reaggregated to 20 m (dBm)")
    axes[2].set_title("c  Cross-grid agreement", loc="left", fontweight="bold")
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.7)
    fig.savefig(output / "grid_sensitivity.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    args = parser.parse_args()
    products_root = args.work_dir / "05_analysis_products" / "raw_coordinates"
    output = args.work_dir / "07_grid_sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    products = {}
    for grid in ("grid_10m", "grid_20m"):
        product, cells = load_product(products_root / grid)
        product["_directory"] = str(products_root / grid)
        products[grid] = (product, cells)
    configuration = configuration_rows(products)
    spectrum = mean_spectrum_sensitivity(products)
    cross_summary, common_rows = cross_grid_rows(products)
    write_csv(output / "configuration_summary.csv", configuration)
    write_csv(output / "mean_spectrum_quality_policy_sensitivity.csv", spectrum)
    write_csv(output / "cross_grid_10m_to_20m_summary.csv", cross_summary)
    write_csv(output / "cross_grid_common_entries.csv", common_rows)
    write_json(
        output / "grid_sensitivity_summary.json",
        {
            "interpretation": "aggregation-grid, predefined floor-division index mapping, and quality-policy sensitivity; native rounded grids are not strictly nested; not implementation equivalence or positioning accuracy",
            "configuration": configuration,
            "mean_spectrum_quality_policy_sensitivity": spectrum,
            "cross_grid": cross_summary,
        },
    )
    make_figure(output, configuration, cross_summary, common_rows)
    report = [
        "# Aggregation-grid sensitivity",
        "",
        "The 10 m and 20 m values are aggregation settings, not positioning resolutions.",
        "The comparison quantifies storage occupancy, contributing-sample density,",
        "quality-policy sensitivity and agreement on entries shared after the predefined",
        "ix20=floor(ix10/2), iy20=floor(iy10/2) mapping. Because both native grids",
        "use rounded coordinates, they are not strictly nested. This is not a proof of",
        "implementation equivalence and does not identify either grid as spatial truth.",
        "",
        f"- 10 m observed space-height cells: {configuration[0]['observed_space_height_cells']}.",
        f"- 20 m observed space-height cells: {configuration[2]['observed_space_height_cells']}.",
        f"- 10 m bounding-tensor occupancy: {configuration[0]['space_height_occupancy_ratio']:.2%}.",
        f"- 20 m bounding-tensor occupancy: {configuration[2]['space_height_occupancy_ratio']:.2%}.",
        f"- Strict-quality cross-grid RMSE: {cross_summary[1]['rmse_db']:.3f} dB.",
        f"- Strict-quality cross-grid Pearson correlation: {cross_summary[1]['pearson']:.4f}.",
    ]
    (output / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Stage 07 complete: {output}")


if __name__ == "__main__":
    main()
