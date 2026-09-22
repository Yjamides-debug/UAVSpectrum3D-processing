#!/usr/bin/env python3
"""Stage 08: test downstream sensitivity to horizontal-coordinate representation."""

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


SEEDS = tuple(range(10))
TRAIN_FRACTION = 0.80
GRID_M = 20.0
HORIZONTAL_LENGTH_SCALES_M = (20.0, 40.0, 80.0)
PRIMARY_HORIZONTAL_LENGTH_SCALE_M = 40.0
VERTICAL_LENGTH_SCALE_M = 10.0
RIDGE = 0.1


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def grouped_holdout_split(
    spatial_indices: np.ndarray,
    raw_group_by_index: dict[int, tuple[float, float]],
    seed: int,
    train_fraction: float = TRAIN_FRACTION,
) -> tuple[np.ndarray, np.ndarray, set[tuple[float, float]]]:
    """Split whole raw-coordinate groups so train and test records stay disjoint."""
    groups = sorted(set(raw_group_by_index.values()))
    rng = np.random.default_rng(seed)
    n_train_groups = int(round(train_fraction * len(groups)))
    train_group_indices = set(
        rng.choice(np.arange(len(groups)), size=n_train_groups, replace=False).tolist()
    )
    train_groups = {groups[index] for index in train_group_indices}
    train_records = np.array(
        [index for index in spatial_indices if raw_group_by_index[int(index)] in train_groups],
        dtype=int,
    )
    test_records = np.array(
        [index for index in spatial_indices if raw_group_by_index[int(index)] not in train_groups],
        dtype=int,
    )
    if set(train_records) & set(test_records):
        raise ValueError("Train and test record IDs overlap.")
    return train_records, test_records, train_groups


def aggregate_training(
    record_indices: np.ndarray,
    coordinates: np.ndarray,
    heights: np.ndarray,
    powers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for record_index in record_indices:
        x, y = coordinates[record_index]
        key = (
            int(round(x / GRID_M)),
            int(round(y / GRID_M)),
            int(round(heights[record_index])),
        )
        groups[key].append(int(record_index))
    keys = sorted(groups)
    features = np.array(
        [[ix * GRID_M, iy * GRID_M, iz] for ix, iy, iz in keys], dtype=float
    )
    spectra = np.vstack(
        [np.mean(powers[np.asarray(groups[key], dtype=int)], axis=0) for key in keys]
    )
    return features, spectra


def scaled_squared_distance(
    left: np.ndarray, right: np.ndarray, horizontal_scale_m: float
) -> np.ndarray:
    scales = np.array(
        [horizontal_scale_m, horizontal_scale_m, VERTICAL_LENGTH_SCALE_M],
        dtype=float,
    )
    difference = left[:, None, :] / scales - right[None, :, :] / scales
    return np.sum(difference * difference, axis=2)


def kernel_ridge_predict(
    train_features: np.ndarray,
    train_spectra: np.ndarray,
    test_features: np.ndarray,
    horizontal_scale_m: float,
) -> np.ndarray:
    mean_spectrum = np.mean(train_spectra, axis=0)
    centered = train_spectra - mean_spectrum
    kernel = np.exp(
        -0.5
        * scaled_squared_distance(
            train_features, train_features, horizontal_scale_m
        )
    )
    system = kernel + RIDGE * np.eye(len(train_features))
    coefficients = np.linalg.solve(system, centered)
    test_kernel = np.exp(
        -0.5
        * scaled_squared_distance(test_features, train_features, horizontal_scale_m)
    )
    return test_kernel @ coefficients + mean_spectrum


def metric_row(
    truth: np.ndarray, prediction: np.ndarray, evaluation_mask: np.ndarray
) -> dict:
    error = prediction[evaluation_mask] - truth[evaluation_mask]
    absolute = np.abs(error)
    return {
        "test_entries": int(evaluation_mask.sum()),
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "mae_db": float(np.mean(absolute)),
        "median_abs_error_db": float(np.median(absolute)),
    }


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    groups = sorted(
        {
            (row["coordinate_source"], row["horizontal_length_scale_m"])
            for row in rows
        }
    )
    for coordinate_source, length_scale in groups:
        selected = [
            row
            for row in rows
            if row["coordinate_source"] == coordinate_source
            and row["horizontal_length_scale_m"] == length_scale
        ]
        summary.append(
            {
                "coordinate_source": coordinate_source,
                "horizontal_length_scale_m": length_scale,
                "seeds": len(selected),
                "train_cells_mean": float(np.mean([row["train_cells"] for row in selected])),
                "rmse_db_mean": float(np.mean([row["rmse_db"] for row in selected])),
                "rmse_db_std": float(np.std([row["rmse_db"] for row in selected], ddof=1)),
                "mae_db_mean": float(np.mean([row["mae_db"] for row in selected])),
                "mae_db_std": float(np.std([row["mae_db"] for row in selected], ddof=1)),
            }
        )
    return summary


def paired_differences(rows: list[dict]) -> list[dict]:
    output = []
    for length_scale in HORIZONTAL_LENGTH_SCALES_M:
        raw = {
            row["seed"]: row
            for row in rows
            if row["coordinate_source"] == "raw"
            and row["horizontal_length_scale_m"] == length_scale
        }
        for source in ("rts", "quantization_aware"):
            candidate = {
                row["seed"]: row
                for row in rows
                if row["coordinate_source"] == source
                and row["horizontal_length_scale_m"] == length_scale
            }
            differences = np.array(
                [candidate[seed]["rmse_db"] - raw[seed]["rmse_db"] for seed in SEEDS]
            )
            output.append(
                {
                    "coordinate_source": source,
                    "reference": "raw",
                    "horizontal_length_scale_m": length_scale,
                    "paired_seeds": len(differences),
                    "rmse_difference_db_mean": float(np.mean(differences)),
                    "rmse_difference_db_std": float(np.std(differences, ddof=1)),
                    "seeds_with_lower_rmse_than_raw": int(np.sum(differences < 0)),
                }
            )
    return output


def make_figure(output: Path, rows: list[dict], summary: list[dict]) -> None:
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
    labels = ["Raw GPS", "RTS", "Quant.-aware"]
    sources = ["raw", "rts", "quantization_aware"]
    colors = ["#5B6770", "#4C78A8", "#E58C50"]
    primary = [
        row for row in rows if row["horizontal_length_scale_m"] == PRIMARY_HORIZONTAL_LENGTH_SCALE_M
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    for index, (source, color) in enumerate(zip(sources, colors)):
        values = [row["rmse_db"] for row in primary if row["coordinate_source"] == source]
        axes[0].scatter(
            np.full(len(values), index) + np.linspace(-0.08, 0.08, len(values)),
            values,
            s=13,
            color=color,
            alpha=0.75,
        )
        axes[0].plot([index - 0.15, index + 0.15], [np.mean(values)] * 2, color="#111111", linewidth=1.2)
    axes[0].set_xticks(range(3), labels, rotation=18, ha="right")
    axes[0].set_ylabel("Held-out RMSE (dB)")
    axes[0].set_title("a  Same held-out records", loc="left", fontweight="bold")
    raw_by_seed = {
        row["seed"]: row["rmse_db"]
        for row in primary
        if row["coordinate_source"] == "raw"
    }
    for index, (source, color) in enumerate(zip(sources[1:], colors[1:])):
        differences = [
            row["rmse_db"] - raw_by_seed[row["seed"]]
            for row in primary
            if row["coordinate_source"] == source
        ]
        axes[1].scatter(
            np.full(len(differences), index) + np.linspace(-0.07, 0.07, len(differences)),
            differences,
            s=13,
            color=color,
            alpha=0.75,
        )
        axes[1].plot([index - 0.14, index + 0.14], [np.mean(differences)] * 2, color="#111111", linewidth=1.2)
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_xticks(range(2), labels[1:], rotation=18, ha="right")
    axes[1].set_ylabel("Paired RMSE difference from raw (dB)")
    axes[1].set_title("b  Representation sensitivity", loc="left", fontweight="bold")
    for source, label, color in zip(sources, labels, colors):
        selected = [row for row in summary if row["coordinate_source"] == source]
        selected.sort(key=lambda row: row["horizontal_length_scale_m"])
        axes[2].errorbar(
            [row["horizontal_length_scale_m"] for row in selected],
            [row["rmse_db_mean"] for row in selected],
            yerr=[row["rmse_db_std"] for row in selected],
            marker="o",
            capsize=2,
            color=color,
            label=label,
        )
    axes[2].set_xlabel("Horizontal kernel length scale (m)")
    axes[2].set_ylabel("Held-out RMSE (dB)")
    axes[2].set_title("c  Parameter robustness", loc="left", fontweight="bold")
    axes[2].legend()
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.7)
    fig.savefig(output / "coordinate_sensitivity.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    args = parser.parse_args()
    output = args.work_dir / "08_coordinate_sensitivity"
    split_dir = output / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    standardized = load_csv(args.work_dir / "02_standardized" / "standardized_records.csv")
    reconstructed = load_csv(
        args.work_dir
        / "04_trajectory_reconstruction"
        / "reconstructed_horizontal_coordinates.csv"
    )
    reconstructed_by_index = {int(row["record_index"]): row for row in reconstructed}
    with np.load(args.work_dir / "02_standardized" / "spectra_and_quality.npz") as arrays:
        powers = arrays["power_dbm"].copy()
        quality_flags = arrays["quality_flags"].copy()
    n_records = len(standardized)
    heights = np.full(n_records, np.nan, dtype=float)
    coordinate_arrays = {
        "raw": np.full((n_records, 2), np.nan, dtype=float),
        "rts": np.full((n_records, 2), np.nan, dtype=float),
        "quantization_aware": np.full((n_records, 2), np.nan, dtype=float),
    }
    spatial_indices = []
    raw_group_by_index = {}
    for row in standardized:
        record_index = int(row["record_index"])
        if row["position_valid"].lower() != "true" or row.get("core_spatial_product_included", "").lower() != "true":
            continue
        reconstructed_row = reconstructed_by_index[record_index]
        spatial_indices.append(record_index)
        heights[record_index] = float(row["z_relative_label_m"])
        coordinate_arrays["raw"][record_index] = [float(row["x_raw_m"]), float(row["y_raw_m"])]
        coordinate_arrays["rts"][record_index] = [float(reconstructed_row["x_rts_m"]), float(reconstructed_row["y_rts_m"])]
        coordinate_arrays["quantization_aware"][record_index] = [
            float(reconstructed_row["x_quantization_aware_m"]),
            float(reconstructed_row["y_quantization_aware_m"]),
        ]
        raw_group_by_index[record_index] = (
            round(float(row["x_raw_m"]), 6),
            round(float(row["y_raw_m"]), 6),
        )
    spatial_indices = np.asarray(spatial_indices, dtype=int)
    groups = sorted(set(raw_group_by_index.values()))
    if len(groups) != 26:
        raise ValueError(f"Expected 26 raw horizontal coordinate groups, found {len(groups)}.")
    rows = []
    for seed in SEEDS:
        train_records, test_records, train_groups = grouped_holdout_split(
            spatial_indices, raw_group_by_index, seed
        )
        np.savez_compressed(
            split_dir / f"raw_horizontal_group_holdout_seed_{seed}.npz",
            train_record_indices=train_records,
            test_record_indices=test_records,
            train_raw_horizontal_groups=np.asarray(sorted(train_groups), dtype=float),
            test_raw_horizontal_groups=np.asarray(
                sorted(set(groups) - train_groups), dtype=float
            ),
        )
        evaluation_mask = quality_flags[test_records] == 0
        for source, coordinates in coordinate_arrays.items():
            train_features, train_spectra = aggregate_training(
                train_records, coordinates, heights, powers
            )
            test_features = np.column_stack(
                [coordinates[test_records], heights[test_records]]
            )
            for horizontal_scale in HORIZONTAL_LENGTH_SCALES_M:
                prediction = kernel_ridge_predict(
                    train_features,
                    train_spectra,
                    test_features,
                    horizontal_scale,
                )
                rows.append(
                    {
                        "seed": seed,
                        "coordinate_source": source,
                        "horizontal_length_scale_m": horizontal_scale,
                        "vertical_length_scale_m": VERTICAL_LENGTH_SCALE_M,
                        "ridge": RIDGE,
                        "train_raw_horizontal_groups": len(train_groups),
                        "test_raw_horizontal_groups": len(groups) - len(train_groups),
                        "train_records": len(train_records),
                        "test_records": len(test_records),
                        "train_cells": len(train_features),
                        **metric_row(
                            powers[test_records], prediction, evaluation_mask
                        ),
                    }
                )
    summary = summarize(rows)
    paired = paired_differences(rows)
    write_csv(output / "results_by_seed.csv", rows)
    write_csv(output / "results_summary.csv", summary)
    write_csv(output / "paired_rmse_differences.csv", paired)
    metadata = {
        "protocol_version": "1.0",
        "task": "coordinate-representation sensitivity for spatial spectrum reconstruction",
        "split": "grouped holdout by the 26 raw horizontal coordinate pairs",
        "same_test_records_across_coordinate_sources": True,
        "train_fraction_of_raw_horizontal_groups": TRAIN_FRACTION,
        "seeds": list(SEEDS),
        "coordinate_sources": list(coordinate_arrays),
        "model": "fixed Gaussian-kernel ridge regression on 20 m aggregated training cells",
        "horizontal_length_scales_m": list(HORIZONTAL_LENGTH_SCALES_M),
        "primary_horizontal_length_scale_m": PRIMARY_HORIZONTAL_LENGTH_SCALE_M,
        "vertical_length_scale_m": VERTICAL_LENGTH_SCALE_M,
        "ridge": RIDGE,
        "training_power_policy": "all finite measured values",
        "evaluation_power_policy": "record-frequency entries with quality_flags == 0",
        "interpretation": "downstream representation sensitivity; not real-flight GPS accuracy or recovered coordinate truth",
    }
    write_json(output / "protocol_metadata.json", metadata)
    make_figure(output, rows, summary)
    primary_summary = [
        row
        for row in summary
        if row["horizontal_length_scale_m"] == PRIMARY_HORIZONTAL_LENGTH_SCALE_M
    ]
    report = [
        "# Coordinate-representation sensitivity",
        "",
        "All coordinate sources use the same raw-horizontal-group splits, held-out",
        "record IDs, measured power targets, quality evaluation mask and fixed kernel",
        "regression settings. Only the horizontal coordinate representation changes.",
        "Model-derived coordinates use the available trajectory sequence and therefore",
        "the experiment measures downstream sensitivity rather than navigation accuracy.",
        "",
    ]
    for row in primary_summary:
        report.append(
            f"- {row['coordinate_source']}: RMSE {row['rmse_db_mean']:.3f} +/- "
            f"{row['rmse_db_std']:.3f} dB at the primary 40 m length scale."
        )
    (output / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Stage 08 complete: {output}")


if __name__ == "__main__":
    main()
