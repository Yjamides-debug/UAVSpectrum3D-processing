#!/usr/bin/env python3
"""Audit batch-concentrated power flags and downstream batch influence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSING_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROCESSING_DIR))

from uavspectrum3d_pipeline.products import build_matrix_product


FLOOR_THRESHOLD_DBM = -149.9
POSITIVE_THRESHOLD_DBM = 0.0
COMPLETION_METHODS = ("frequency_mean", "linear_interpolation", "iterative_svd")


def load_stage_module(filename: str, module_name: str):
    path = PROCESSING_DIR / "stages" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import stage module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FREQUENCY_STAGE = load_stage_module(
    "06_frequency_completion_reference.py", "frequency_completion_stage"
)
COORDINATE_STAGE = load_stage_module(
    "08_coordinate_sensitivity.py", "coordinate_sensitivity_stage"
)


def finite_metric(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(left) & np.isfinite(right)
    if not np.any(valid):
        return {
            "common_frequency_bins": 0,
            "mae_db": float("nan"),
            "rmse_db": float("nan"),
            "median_abs_difference_db": float("nan"),
            "max_abs_difference_db": float("nan"),
            "mean_signed_difference_db": float("nan"),
        }
    difference = right[valid] - left[valid]
    absolute = np.abs(difference)
    return {
        "common_frequency_bins": int(valid.sum()),
        "mae_db": float(np.mean(absolute)),
        "rmse_db": float(np.sqrt(np.mean(difference**2))),
        "median_abs_difference_db": float(np.median(absolute)),
        "max_abs_difference_db": float(np.max(absolute)),
        "mean_signed_difference_db": float(np.mean(difference)),
    }


def load_release(release: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    records = pd.read_csv(release / "processed/records/standardized_records.csv")
    with np.load(release / "processed/records/spectra_and_quality.npz") as arrays:
        power = arrays["power_dbm"].astype(float)
        frequency_hz = arrays["frequencies_hz"].astype(np.int64)
        quality_flags = arrays["quality_flags"].astype(np.uint8)
        record_indices = arrays["record_indices"].astype(int)
    if power.shape != (len(records), len(frequency_hz)):
        raise ValueError("Record and spectrum shapes disagree.")
    if not np.array_equal(record_indices, records["record_index"].to_numpy(dtype=int)):
        raise ValueError("Record indices are not aligned with spectrum rows.")
    if quality_flags.shape != power.shape:
        raise ValueError("Quality flags are not aligned with power values.")
    return records, power, frequency_hz, quality_flags


def batch_label(source_file: str) -> str:
    label = source_file.removeprefix("recording_2025-11-19_").removesuffix(".csv")
    return label.replace("linefrom9mto1m", "line 9-1m")


def summarize_batches(
    records: pd.DataFrame,
    power: np.ndarray,
    frequencies_hz: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    spectrum_rows: list[dict] = []
    for source_file, group in records.groupby("source_file", sort=False):
        indices = group["record_index"].to_numpy(dtype=int)
        values = power[indices]
        finite = np.isfinite(values)
        positive = finite & (values > POSITIVE_THRESHOLD_DBM)
        floor_like = finite & (values <= FLOOR_THRESHOLD_DBM)
        affected_records = positive.any(axis=1)
        affected_frequencies = positive.any(axis=0)
        positive_values = values[positive]
        all_median = np.nanmedian(values, axis=0)
        positive_masked_median = np.nanmedian(
            np.where(positive, np.nan, values), axis=0
        )
        policy = str(group["height_label_policy"].iloc[0])
        z_values = group["z_relative_label_m"].dropna()
        rows.append(
            {
                "source_file": source_file,
                "batch_label": batch_label(source_file),
                "height_label_policy": policy,
                "relative_height_label_m": (
                    float(z_values.iloc[0]) if policy.startswith("constant_") and len(z_values) else np.nan
                ),
                "records": int(len(indices)),
                "finite_power_values": int(finite.sum()),
                "floor_like_values": int(floor_like.sum()),
                "positive_power_candidate_values": int(positive.sum()),
                "positive_candidate_fraction_percent": float(positive.mean() * 100),
                "records_with_positive_candidate": int(affected_records.sum()),
                "records_with_positive_candidate_percent": float(
                    affected_records.mean() * 100
                ),
                "frequency_bins_with_positive_candidate": int(
                    affected_frequencies.sum()
                ),
                "positive_candidate_min_dbm": (
                    float(np.min(positive_values)) if len(positive_values) else np.nan
                ),
                "positive_candidate_median_dbm": (
                    float(np.median(positive_values)) if len(positive_values) else np.nan
                ),
                "positive_candidate_max_dbm": (
                    float(np.max(positive_values)) if len(positive_values) else np.nan
                ),
            }
        )
        for frequency_index, frequency in enumerate(frequencies_hz):
            spectrum_rows.append(
                {
                    "source_file": source_file,
                    "batch_label": batch_label(source_file),
                    "frequency_index": frequency_index,
                    "frequency_hz": int(frequency),
                    "frequency_mhz": float(frequency / 1e6),
                    "power_median_all_finite_dbm": float(all_median[frequency_index]),
                    "power_median_positive_masked_dbm": float(
                        positive_masked_median[frequency_index]
                    ),
                    "positive_candidate_count": int(positive[:, frequency_index].sum()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(spectrum_rows)


def robust_batch_deviation(
    records: pd.DataFrame,
    batch_spectra: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed_files = records.loc[
        records["height_label_policy"].astype(str).str.startswith("constant_"),
        "source_file",
    ].drop_duplicates().tolist()
    selected = batch_spectra[batch_spectra["source_file"].isin(fixed_files)].copy()
    pivot_all = selected.pivot(
        index="source_file", columns="frequency_index", values="power_median_all_finite_dbm"
    ).loc[fixed_files]
    pivot_masked = selected.pivot(
        index="source_file",
        columns="frequency_index",
        values="power_median_positive_masked_dbm",
    ).loc[fixed_files]
    reference_all = pivot_all.median(axis=0).to_numpy(dtype=float)
    reference_masked = pivot_masked.median(axis=0).to_numpy(dtype=float)
    masked_values = pivot_masked.to_numpy(dtype=float)
    centre = np.nanmedian(masked_values, axis=0)
    mad = np.nanmedian(np.abs(masked_values - centre[None, :]), axis=0)
    robust_z = np.full_like(masked_values, np.nan, dtype=float)
    valid_mad = mad > 0
    robust_z[:, valid_mad] = (
        0.6745
        * (masked_values[:, valid_mad] - centre[None, valid_mad])
        / mad[None, valid_mad]
    )
    summary_rows: list[dict] = []
    frequency_rows: list[dict] = []
    frequency_hz = (
        selected.drop_duplicates("frequency_index")
        .sort_values("frequency_index")["frequency_hz"]
        .to_numpy(dtype=np.int64)
    )
    for batch_index, source_file in enumerate(fixed_files):
        metrics_all = finite_metric(reference_all, pivot_all.loc[source_file].to_numpy())
        metrics_masked = finite_metric(
            reference_masked, pivot_masked.loc[source_file].to_numpy()
        )
        summary_rows.append(
            {
                "source_file": source_file,
                "batch_label": batch_label(source_file),
                "all_finite_median_absolute_deviation_from_batch_reference_db": metrics_all[
                    "median_abs_difference_db"
                ],
                "all_finite_rmse_from_batch_reference_db": metrics_all["rmse_db"],
                "positive_masked_median_absolute_deviation_from_batch_reference_db": metrics_masked[
                    "median_abs_difference_db"
                ],
                "positive_masked_rmse_from_batch_reference_db": metrics_masked[
                    "rmse_db"
                ],
                "frequency_bins_with_abs_robust_z_ge_3_5": int(
                    np.sum(np.abs(robust_z[batch_index]) >= 3.5)
                ),
                "median_abs_robust_z": float(
                    np.nanmedian(np.abs(robust_z[batch_index]))
                ),
            }
        )
        for frequency_index, frequency in enumerate(frequency_hz):
            frequency_rows.append(
                {
                    "source_file": source_file,
                    "batch_label": batch_label(source_file),
                    "frequency_index": frequency_index,
                    "frequency_hz": int(frequency),
                    "reference_power_median_all_finite_dbm": float(
                        reference_all[frequency_index]
                    ),
                    "reference_power_median_positive_masked_dbm": float(
                        reference_masked[frequency_index]
                    ),
                    "batch_power_median_all_finite_dbm": float(
                        pivot_all.loc[source_file].iloc[frequency_index]
                    ),
                    "batch_power_median_positive_masked_dbm": float(
                        pivot_masked.loc[source_file].iloc[frequency_index]
                    ),
                    "positive_masked_robust_z": float(robust_z[batch_index, frequency_index]),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(frequency_rows)


def pooled_spectrum(
    power: np.ndarray, indices: np.ndarray, mask_positive: bool
) -> np.ndarray:
    values = power[indices]
    if mask_positive:
        values = np.where(values > POSITIVE_THRESHOLD_DBM, np.nan, values)
    return np.nanmedian(values, axis=0)


def leave_one_fixed_batch_out_spectra(
    records: pd.DataFrame, power: np.ndarray
) -> pd.DataFrame:
    fixed = records["height_label_policy"].astype(str).str.startswith("constant_")
    fixed_indices = records.loc[fixed, "record_index"].to_numpy(dtype=int)
    fixed_files = records.loc[fixed, "source_file"].drop_duplicates().tolist()
    baseline_all = pooled_spectrum(power, fixed_indices, mask_positive=False)
    baseline_masked = pooled_spectrum(power, fixed_indices, mask_positive=True)
    rows: list[dict] = []
    for source_file in fixed_files:
        keep = fixed & ~records["source_file"].eq(source_file)
        indices = records.loc[keep, "record_index"].to_numpy(dtype=int)
        all_metrics = finite_metric(
            baseline_all, pooled_spectrum(power, indices, mask_positive=False)
        )
        masked_metrics = finite_metric(
            baseline_masked, pooled_spectrum(power, indices, mask_positive=True)
        )
        rows.append(
            {
                "excluded_source_file": source_file,
                "excluded_batch_label": batch_label(source_file),
                "excluded_records": int((fixed & records["source_file"].eq(source_file)).sum()),
                **{f"all_finite_{key}": value for key, value in all_metrics.items()},
                **{
                    f"positive_masked_{key}": value
                    for key, value in masked_metrics.items()
                },
            }
        )
    return pd.DataFrame(rows)


def core_rows(records: pd.DataFrame, excluded_source: str | None = None) -> list[dict]:
    selected = records["core_spatial_product_included"].astype(str).str.lower().eq("true")
    if excluded_source is not None:
        selected &= ~records["source_file"].eq(excluded_source)
    return records.loc[selected].to_dict(orient="records")


def product_spectrum(product: dict, strict: bool) -> np.ndarray:
    values = product["power_dbm_median"].astype(float)
    mask = product["quality_mask"] if strict else product["observed_mask"]
    return np.nanmedian(np.where(mask, values, np.nan), axis=0)


def leave_one_core_batch_out_matrices(
    records: pd.DataFrame,
    power: np.ndarray,
    frequency_hz: np.ndarray,
) -> tuple[pd.DataFrame, dict[tuple[str, float], dict]]:
    core_files = records.loc[
        records["core_spatial_product_included"].astype(str).str.lower().eq("true"),
        "source_file",
    ].drop_duplicates().tolist()
    conditions: list[str | None] = [None, *core_files]
    products: dict[tuple[str, float], dict] = {}
    rows: list[dict] = []
    baseline_spectra: dict[tuple[float, bool], np.ndarray] = {}
    for excluded in conditions:
        condition = "none" if excluded is None else excluded
        selected_rows = core_rows(records, excluded)
        for grid_m in (10.0, 20.0):
            product = build_matrix_product(
                selected_rows,
                power,
                frequency_hz,
                grid_m,
                "x_raw_m",
                "y_raw_m",
            )
            products[(condition, grid_m)] = product
            all_spectrum = product_spectrum(product, strict=False)
            strict_spectrum = product_spectrum(product, strict=True)
            if excluded is None:
                baseline_spectra[(grid_m, False)] = all_spectrum
                baseline_spectra[(grid_m, True)] = strict_spectrum
            all_metrics = finite_metric(
                baseline_spectra.get((grid_m, False), all_spectrum), all_spectrum
            )
            strict_metrics = finite_metric(
                baseline_spectra.get((grid_m, True), strict_spectrum), strict_spectrum
            )
            rows.append(
                {
                    "excluded_source_file": condition,
                    "excluded_batch_label": "None" if excluded is None else batch_label(excluded),
                    "grid_m": grid_m,
                    "remaining_core_records": len(selected_rows),
                    "observed_cells": len(product["cells"]),
                    "observed_cell_frequency_entries": int(product["observed_mask"].sum()),
                    "strict_quality_entries": int(product["quality_mask"].sum()),
                    "positive_flagged_entries": int((product["positive_count"] > 0).sum()),
                    "total_contributing_samples": int(product["sample_count"].sum()),
                    **{f"all_finite_{key}": value for key, value in all_metrics.items()},
                    **{f"strict_quality_{key}": value for key, value in strict_metrics.items()},
                }
            )
    return pd.DataFrame(rows), products


def frequency_completion_sensitivity(
    products: dict[tuple[str, float], dict]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for (condition, grid_m), product in products.items():
        if grid_m != 20.0:
            continue
        data = {
            "power_dbm_arithmetic_mean": product["power_dbm_arithmetic_mean"],
            "quality_mask": product["quality_mask"],
        }
        eligible = FREQUENCY_STAGE.choose_rows(data)
        matrix = data["power_dbm_arithmetic_mean"][eligible]
        base = data["quality_mask"][eligible]

        def evaluate(protocol: str, level: float, seed: int, train: np.ndarray, test: np.ndarray) -> None:
            for method in COMPLETION_METHODS:
                prediction = FREQUENCY_STAGE.predict(matrix, train, method)
                metric = FREQUENCY_STAGE.metrics(matrix, prediction, test)
                rows.append(
                    {
                        "excluded_source_file": condition,
                        "excluded_batch_label": "None" if condition == "none" else batch_label(condition),
                        "eligible_rows": len(eligible),
                        "protocol": protocol,
                        "level": level,
                        "seed": seed,
                        "method": method,
                        **metric,
                    }
                )

        for fraction in FREQUENCY_STAGE.RANDOM_FRACTIONS:
            for seed in FREQUENCY_STAGE.SEEDS:
                train, test = FREQUENCY_STAGE.split_random(base, fraction, seed)
                evaluate("random_masking", fraction, seed, train, test)
        for fraction in FREQUENCY_STAGE.CONTIGUOUS_FRACTIONS:
            for seed in FREQUENCY_STAGE.SEEDS:
                train, test, _ = FREQUENCY_STAGE.split_contiguous(
                    base, fraction, seed
                )
                evaluate("contiguous_block", fraction, seed, train, test)
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(
            [
                "excluded_source_file",
                "excluded_batch_label",
                "eligible_rows",
                "protocol",
                "level",
                "method",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            seeds=("seed", "count"),
            rmse_db_mean=("rmse_db", "mean"),
            rmse_db_std=("rmse_db", "std"),
            mae_db_mean=("mae_db", "mean"),
            mae_db_std=("mae_db", "std"),
        )
    )
    baseline = summary[summary["excluded_source_file"].eq("none")][
        ["protocol", "level", "method", "rmse_db_mean", "mae_db_mean"]
    ].rename(
        columns={
            "rmse_db_mean": "baseline_rmse_db_mean",
            "mae_db_mean": "baseline_mae_db_mean",
        }
    )
    summary = summary.merge(baseline, on=["protocol", "level", "method"], how="left")
    summary["rmse_change_from_baseline_db"] = (
        summary["rmse_db_mean"] - summary["baseline_rmse_db_mean"]
    )
    summary["mae_change_from_baseline_db"] = (
        summary["mae_db_mean"] - summary["baseline_mae_db_mean"]
    )
    return detail, summary


def coordinate_sensitivity(
    release: Path,
    records: pd.DataFrame,
    power: np.ndarray,
    quality_flags: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reconstructed = pd.read_csv(
        release
        / "reference_reuse/trajectory_reconstruction/coordinates/reconstructed_horizontal_coordinates.csv"
    )
    reconstructed = reconstructed.set_index("record_index")
    n_records = len(records)
    heights = records["z_relative_label_m"].to_numpy(dtype=float)
    coordinate_arrays = {
        "raw": records[["x_raw_m", "y_raw_m"]].to_numpy(dtype=float),
        "rts": np.full((n_records, 2), np.nan, dtype=float),
        "quantization_aware": np.full((n_records, 2), np.nan, dtype=float),
    }
    for record_index, row in reconstructed.iterrows():
        coordinate_arrays["rts"][int(record_index)] = [row["x_rts_m"], row["y_rts_m"]]
        coordinate_arrays["quantization_aware"][int(record_index)] = [
            row["x_quantization_aware_m"],
            row["y_quantization_aware_m"],
        ]
    core_mask = records["core_spatial_product_included"].astype(str).str.lower().eq("true")
    core_files = records.loc[core_mask, "source_file"].drop_duplicates().tolist()
    conditions: list[str | None] = [None, *core_files]
    group_by_index = {
        int(row.record_index): (round(float(row.x_raw_m), 6), round(float(row.y_raw_m), 6))
        for row in records.loc[core_mask].itertuples()
    }
    split_dir = (
        release
        / "reference_reuse/trajectory_reconstruction/sensitivity_analysis/splits"
    )
    rows: list[dict] = []
    for excluded in conditions:
        selected_mask = core_mask.copy()
        if excluded is not None:
            selected_mask &= ~records["source_file"].eq(excluded)
        selected_indices = records.loc[selected_mask, "record_index"].to_numpy(dtype=int)
        for seed in COORDINATE_STAGE.SEEDS:
            with np.load(split_dir / f"raw_horizontal_group_holdout_seed_{seed}.npz") as split:
                train_groups = {tuple(np.round(value, 6)) for value in split["train_raw_horizontal_groups"]}
            train_records = np.asarray(
                [index for index in selected_indices if group_by_index[int(index)] in train_groups],
                dtype=int,
            )
            test_records = np.asarray(
                [index for index in selected_indices if group_by_index[int(index)] not in train_groups],
                dtype=int,
            )
            evaluation_mask = quality_flags[test_records] == 0
            for source, coordinates in coordinate_arrays.items():
                train_features, train_spectra = COORDINATE_STAGE.aggregate_training(
                    train_records, coordinates, heights, power
                )
                test_features = np.column_stack(
                    [coordinates[test_records], heights[test_records]]
                )
                prediction = COORDINATE_STAGE.kernel_ridge_predict(
                    train_features,
                    train_spectra,
                    test_features,
                    COORDINATE_STAGE.PRIMARY_HORIZONTAL_LENGTH_SCALE_M,
                )
                metric = COORDINATE_STAGE.metric_row(
                    power[test_records], prediction, evaluation_mask
                )
                rows.append(
                    {
                        "excluded_source_file": "none" if excluded is None else excluded,
                        "excluded_batch_label": "None" if excluded is None else batch_label(excluded),
                        "seed": seed,
                        "coordinate_source": source,
                        "horizontal_length_scale_m": COORDINATE_STAGE.PRIMARY_HORIZONTAL_LENGTH_SCALE_M,
                        "train_records": len(train_records),
                        "test_records": len(test_records),
                        "train_cells": len(train_features),
                        **metric,
                    }
                )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(
            ["excluded_source_file", "excluded_batch_label", "coordinate_source"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "count"),
            train_cells_mean=("train_cells", "mean"),
            rmse_db_mean=("rmse_db", "mean"),
            rmse_db_std=("rmse_db", "std"),
            mae_db_mean=("mae_db", "mean"),
            mae_db_std=("mae_db", "std"),
        )
    )
    baseline = summary[summary["excluded_source_file"].eq("none")][
        ["coordinate_source", "rmse_db_mean", "mae_db_mean"]
    ].rename(
        columns={
            "rmse_db_mean": "baseline_rmse_db_mean",
            "mae_db_mean": "baseline_mae_db_mean",
        }
    )
    summary = summary.merge(baseline, on="coordinate_source", how="left")
    summary["rmse_change_from_baseline_db"] = (
        summary["rmse_db_mean"] - summary["baseline_rmse_db_mean"]
    )
    summary["mae_change_from_baseline_db"] = (
        summary["mae_db_mean"] - summary["baseline_mae_db_mean"]
    )
    return detail, summary


def save_nine_m_mask(
    output: Path,
    records: pd.DataFrame,
    power: np.ndarray,
    frequency_hz: np.ndarray,
) -> tuple[str, np.ndarray]:
    matches = records[
        records["height_label_policy"].astype(str).eq("constant_9m_from_filename")
    ]
    if len(matches["source_file"].unique()) != 1:
        raise ValueError("Expected one fixed-height 9 m batch.")
    source_file = str(matches["source_file"].iloc[0])
    indices = matches["record_index"].to_numpy(dtype=int)
    positive_mask = power[indices] > POSITIVE_THRESHOLD_DBM
    np.savez_compressed(
        output / "nine_m_positive_candidate_mask.npz",
        source_file=np.asarray(source_file),
        record_indices=indices,
        source_rows=matches["source_row"].to_numpy(dtype=int),
        timestamps=matches["timestamp"].astype(str).to_numpy(),
        frequencies_hz=frequency_hz,
        positive_candidate_mask=positive_mask,
    )
    return source_file, positive_mask


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
        }
    )


def make_figure(
    output: Path,
    batch_summary: pd.DataFrame,
    robust_summary: pd.DataFrame,
    spectral_lobo: pd.DataFrame,
    positive_mask_9m: np.ndarray,
    frequencies_hz: np.ndarray,
) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    neutral = "#7B8790"
    accent = "#D97732"
    dark = "#30363B"
    fixed = robust_summary.copy()
    fixed_labels = fixed["batch_label"].tolist()
    highlight = [accent if label.endswith("9m") else neutral for label in fixed_labels]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)

    ordered = batch_summary.copy()
    x = np.arange(len(ordered))
    colors = [accent if label.endswith("9m") else neutral for label in ordered["batch_label"]]
    axes[0, 0].bar(x, ordered["positive_power_candidate_values"], color=colors, width=0.72)
    axes[0, 0].set_xticks(x, ordered["batch_label"], rotation=55, ha="right")
    axes[0, 0].set_ylabel("Positive-power candidates")
    axes[0, 0].set_title("a  Batch concentration", loc="left", fontweight="bold")
    axes[0, 0].grid(axis="y", color="#E1E4E6", linewidth=0.5, zorder=0)
    for position, count in zip(x, ordered["positive_power_candidate_values"]):
        if count > 0:
            axes[0, 0].text(position, count + max(ordered["positive_power_candidate_values"]) * 0.012, f"{int(count):,}", ha="center", va="bottom", fontsize=5.4, rotation=90 if count < 100 else 0)
    axes[0, 0].set_ylim(0, max(ordered["positive_power_candidate_values"]) * 1.14)

    axes[0, 1].imshow(
        positive_mask_9m,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[frequencies_hz[0] / 1e6, frequencies_hz[-1] / 1e6, 1, positive_mask_9m.shape[0]],
        cmap=ListedColormap(["#F7F7F7", accent]),
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    axes[0, 1].set_xlabel("Centre frequency (MHz)")
    axes[0, 1].set_ylabel("9 m record order")
    axes[0, 1].set_title("b  9 m time-frequency flag mask", loc="left", fontweight="bold")

    y = np.arange(len(fixed))
    axes[1, 0].barh(
        y,
        fixed["positive_masked_median_absolute_deviation_from_batch_reference_db"],
        color=highlight,
        height=0.7,
    )
    axes[1, 0].set_yticks(y, fixed_labels)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Median spectral deviation (dB)")
    axes[1, 0].set_title("c  Robust batch deviation", loc="left", fontweight="bold")
    axes[1, 0].grid(axis="x", color="#E1E4E6", linewidth=0.5)

    lobo = spectral_lobo.copy()
    lobo_labels = lobo["excluded_batch_label"].tolist()
    lobo_colors = [accent if label.endswith("9m") else neutral for label in lobo_labels]
    y = np.arange(len(lobo))
    axes[1, 1].barh(
        y,
        lobo["positive_masked_median_abs_difference_db"],
        color=lobo_colors,
        height=0.7,
    )
    axes[1, 1].set_yticks(y, lobo_labels)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Change after leaving one batch out (dB)")
    axes[1, 1].set_title("d  Leave-one-batch-out influence", loc="left", fontweight="bold")
    axes[1, 1].grid(axis="x", color="#E1E4E6", linewidth=0.5)

    for axis in axes.flat:
        axis.tick_params(direction="out", length=2.8, width=0.6)
    fig.savefig(output / "batch_power_sensitivity.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("data/UAVSpectrum3D"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/batch_power_sensitivity"),
    )
    parser.add_argument(
        "--skip-downstream",
        action="store_true",
        help="Skip frequency-completion and coordinate-sensitivity reruns.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    records, power, frequency_hz, quality_flags = load_release(args.release)
    batch_summary, batch_spectra = summarize_batches(records, power, frequency_hz)
    robust_summary, robust_frequency = robust_batch_deviation(records, batch_spectra)
    spectral_lobo = leave_one_fixed_batch_out_spectra(records, power)
    matrix_lobo, products = leave_one_core_batch_out_matrices(
        records, power, frequency_hz
    )
    nine_m_source, positive_mask_9m = save_nine_m_mask(
        args.output, records, power, frequency_hz
    )

    batch_summary.to_csv(args.output / "batch_power_quality_summary.csv", index=False)
    batch_spectra.to_csv(args.output / "batch_median_spectra.csv", index=False)
    robust_summary.to_csv(args.output / "batch_robust_deviation_summary.csv", index=False)
    robust_frequency.to_csv(
        args.output / "batch_robust_deviation_by_frequency.csv", index=False
    )
    spectral_lobo.to_csv(
        args.output / "leave_one_fixed_batch_out_spectral_influence.csv", index=False
    )
    matrix_lobo.to_csv(
        args.output / "leave_one_core_batch_out_matrix_influence.csv", index=False
    )

    completion_detail = completion_summary = None
    coordinate_detail = coordinate_summary = None
    if not args.skip_downstream:
        completion_detail, completion_summary = frequency_completion_sensitivity(products)
        completion_detail.to_csv(
            args.output / "frequency_completion_batch_sensitivity_by_seed.csv",
            index=False,
        )
        completion_summary.to_csv(
            args.output / "frequency_completion_batch_sensitivity_summary.csv",
            index=False,
        )
        coordinate_detail, coordinate_summary = coordinate_sensitivity(
            args.release, records, power, quality_flags
        )
        coordinate_detail.to_csv(
            args.output / "coordinate_batch_sensitivity_by_seed.csv", index=False
        )
        coordinate_summary.to_csv(
            args.output / "coordinate_batch_sensitivity_summary.csv", index=False
        )

    make_figure(
        args.output,
        batch_summary,
        robust_summary,
        spectral_lobo,
        positive_mask_9m,
        frequency_hz,
    )

    nine_m = batch_summary[batch_summary["source_file"].eq(nine_m_source)].iloc[0]
    other_positive = int(
        batch_summary.loc[
            ~batch_summary["source_file"].eq(nine_m_source),
            "positive_power_candidate_values",
        ].sum()
    )
    nine_deviation = robust_summary[
        robust_summary["source_file"].eq(nine_m_source)
    ].iloc[0]
    nine_lobo = spectral_lobo[
        spectral_lobo["excluded_source_file"].eq(nine_m_source)
    ].iloc[0]
    spectral_rmse = spectral_lobo["positive_masked_rmse_db"]
    matrix_without_baseline = matrix_lobo[
        ~matrix_lobo["excluded_source_file"].eq("none")
    ]
    matrix_10m_rmse = matrix_without_baseline.loc[
        matrix_without_baseline["grid_m"].eq(10.0), "strict_quality_rmse_db"
    ]
    matrix_20m_rmse = matrix_without_baseline.loc[
        matrix_without_baseline["grid_m"].eq(20.0), "strict_quality_rmse_db"
    ]
    leave_one_batch_out = {
        "record_level_fixed_height_batches": int(
            spectral_lobo["excluded_source_file"].nunique()
        ),
        "core_spatial_batches": int(
            matrix_without_baseline["excluded_source_file"].nunique()
        ),
        "positive_masked_record_spectrum_rmse_db": {
            "minimum": float(spectral_rmse.min()),
            "maximum": float(spectral_rmse.max()),
        },
        "strict_quality_10m_rmse_db": {
            "minimum": float(matrix_10m_rmse.min()),
            "maximum": float(matrix_10m_rmse.max()),
        },
        "strict_quality_20m_rmse_db": {
            "minimum": float(matrix_20m_rmse.min()),
            "maximum": float(matrix_20m_rmse.max()),
        },
    }
    if completion_summary is not None:
        leave_one_batch_out.update(
            {
                "frequency_completion_eligible_rows": {
                    "minimum": int(completion_summary["eligible_rows"].min()),
                    "maximum": int(completion_summary["eligible_rows"].max()),
                },
                "frequency_completion_rmse_change_db": {
                    "minimum": float(
                        completion_summary["rmse_change_from_baseline_db"].min()
                    ),
                    "maximum": float(
                        completion_summary["rmse_change_from_baseline_db"].max()
                    ),
                },
            }
        )
    if coordinate_summary is not None:
        leave_one_batch_out["coordinate_mean_rmse_change_db"] = {
            "minimum": float(
                coordinate_summary["rmse_change_from_baseline_db"].min()
            ),
            "maximum": float(
                coordinate_summary["rmse_change_from_baseline_db"].max()
            ),
        }

    result = {
        "status": "complete",
        "release": "UAVSpectrum3D_v0.4.0",
        "diagnostic_directory": "diagnostics/batch_sensitivity",
        "total_positive_power_candidates": int(
            batch_summary["positive_power_candidate_values"].sum()
        ),
        "nine_m": {
            "source_file": nine_m_source,
            "positive_power_candidates": int(nine_m["positive_power_candidate_values"]),
            "share_of_all_positive_candidates_percent": float(
                100
                * nine_m["positive_power_candidate_values"]
                / batch_summary["positive_power_candidate_values"].sum()
            ),
            "affected_records": int(nine_m["records_with_positive_candidate"]),
            "records": int(nine_m["records"]),
            "affected_frequency_bins": int(
                nine_m["frequency_bins_with_positive_candidate"]
            ),
            "positive_masked_batch_deviation_median_db": float(
                nine_deviation[
                    "positive_masked_median_absolute_deviation_from_batch_reference_db"
                ]
            ),
            "positive_masked_leave_one_out_spectral_change_median_db": float(
                nine_lobo["positive_masked_median_abs_difference_db"]
            ),
        },
        "positive_candidates_outside_nine_m": other_positive,
        "leave_one_batch_out": leave_one_batch_out,
        "protocol": {
            "batch_deviation_reference": "per-frequency median of fixed-height batch median spectra",
            "batch_deviation_primary_power_policy": "mask only finite readings > 0 dBm",
            "leave_one_batch_out_scope": "all fixed-height batches for record-level spectra; all core fixed-height batches for matrices and downstream reference reuse",
            "frequency_completion": "released masking strategies, seeds, methods, SVD rank, and iterations regenerated after each core-batch exclusion",
            "coordinate_sensitivity": "ten raw-coordinate-group splits and a 40 m horizontal kernel regenerated after each core-batch exclusion",
        },
    }
    if completion_summary is not None:
        result["frequency_completion_conditions"] = int(
            completion_summary["excluded_source_file"].nunique()
        )
    if coordinate_summary is not None:
        result["coordinate_sensitivity_conditions"] = int(
            coordinate_summary["excluded_source_file"].nunique()
        )
    write_json(args.output / "experiment_summary.json", result)
    write_json(
        args.output / "protocol.json",
        {
            "experiment": "batch_power_sensitivity",
            "release": "UAVSpectrum3D_v0.4.0",
            "diagnostic_directory": "diagnostics/batch_sensitivity",
            "raw_values_modified": False,
            "core_products_modified": False,
            "floor_threshold_dbm": FLOOR_THRESHOLD_DBM,
            "positive_threshold_dbm": POSITIVE_THRESHOLD_DBM,
            "record_level_fixed_height_batches": int(
                spectral_lobo["excluded_source_file"].nunique()
            ),
            "core_spatial_batches": int(
                matrix_without_baseline["excluded_source_file"].nunique()
            ),
            "frequency_completion_seeds": 5,
            "coordinate_sensitivity_seeds": 10,
            "coordinate_sensitivity_primary_kernel_m": 40,
            "figure_format": "PNG",
            "figure_dpi": 600,
            "interpretation_notes": [
                "Nominal relative height and acquisition batch are coupled for most fixed-height runs.",
                "Leave-one-batch-out comparisons include changes in contributing observations and spatial or evaluation membership.",
                "Reference-reuse reruns retain the released protocol settings and serve as batch-composition sensitivity checks.",
            ],
        },
    )
    readme = """# Batch sensitivity diagnostics

This directory contains batch-level power-quality and leave-one-batch-out
diagnostics generated from the released UAVSpectrum3D v0.4.0 records. The
diagnostics add evidence about the influence of acquisition-batch composition;
they do not alter the original CSV files or the core record, matrix, or tensor
products.

## Scope

- Batch-level power-quality counts cover all 12 acquisition batches.
- Record-level fixed-height comparisons cover nine fixed-height batches
  (4,950 records), including the 3 m batch without usable horizontal
  coordinates.
- Spatial-product and downstream reference-reuse checks cover the eight
  fixed-height batches contributing to the core spatial products
  (4,718 records). The 3 m batch is absent from these checks because it does
  not contribute to horizontal indexing or aggregation.
- The frequency-completion and coordinate-representation checks retain the
  released protocol settings, fixed seeds, and evaluation definitions.

## Files

| File | Contents |
|---|---|
| `batch_power_quality_summary.csv` | Per-batch finite-value and quality-state counts |
| `batch_median_spectra.csv` | Per-batch median spectra under the evaluated power policies |
| `batch_robust_deviation_summary.csv` | Batch-level robust spectral-deviation summaries |
| `batch_robust_deviation_by_frequency.csv` | Frequency-resolved robust batch deviations |
| `leave_one_fixed_batch_out_spectral_influence.csv` | Record-level fixed-height leave-one-batch-out spectral changes |
| `leave_one_core_batch_out_matrix_influence.csv` | Leave-one-core-batch-out changes for the 10 m and 20 m matrices |
| `frequency_completion_batch_sensitivity_summary.csv` | Frequency-completion batch-sensitivity summaries |
| `frequency_completion_batch_sensitivity_by_seed.csv` | Per-seed frequency-completion results |
| `coordinate_batch_sensitivity_summary.csv` | Coordinate-representation batch-sensitivity summaries |
| `coordinate_batch_sensitivity_by_seed.csv` | Per-seed coordinate-representation results |
| `nine_m_positive_candidate_mask.npz` | Record-frequency mask for positive-power candidates in the 9 m batch |
| `batch_power_sensitivity.png` | Overview of batch quality and spectral-sensitivity diagnostics |
| `protocol.json` | Thresholds, scope, and interpretation metadata |
| `experiment_summary.json` | Machine-readable summary of the completed checks |

## Summary ranges

After readings above 0 dBm were masked, leave-one-fixed-batch-out RMSE for the
record-level median spectrum ranged from 0.048 to 1.062 dB. For strict-quality
aggregation products, leave-one-core-batch-out RMSE ranged from 0.189 to
0.535 dB at the 10 m aggregation scale and from 0.171 to 0.596 dB at the 20 m
scale. Across the eight core-batch exclusions, the frequency-completion
eligible-row count ranged from 47 to 59; RMSE changes across protocol-method
combinations ranged from -0.645 to 0.407 dB. Mean-RMSE changes in the
coordinate-representation check ranged from -1.858 to 0.721 dB.

The leave-one-batch-out values compare each rebuilt summary or reference task
with its complete-data counterpart. Batch removal changes both the contributing
observations and, where applicable, spatial coverage or evaluation membership.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
