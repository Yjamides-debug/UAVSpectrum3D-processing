#!/usr/bin/env python3
"""Stage 03: audit GPS elevation, power aggregation, and inertial fields."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavspectrum3d_pipeline.config import DEFAULT_INPUT_DIR, DEFAULT_WORK_DIR
from uavspectrum3d_pipeline.coordinates import infer_coordinate_step
from uavspectrum3d_pipeline.height import height_policy
from uavspectrum3d_pipeline.power import aggregate_power_dbm
from uavspectrum3d_pipeline.raw_io import SENSOR_FIELDS, load_raw_dataset
from uavspectrum3d_pipeline.utils import write_csv, write_json


def finite_csv(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def describe(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "unique_count": 0,
            "min": "",
            "median": "",
            "max": "",
            "mean": "",
            "std": "",
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": len(array),
        "unique_count": len(np.unique(array)),
        "min": f"{np.min(array):.6f}",
        "median": f"{np.median(array):.6f}",
        "max": f"{np.max(array):.6f}",
        "mean": f"{np.mean(array):.6f}",
        "std": f"{np.std(array):.6f}",
    }


def is_gps_jump_candidate(d_gps_m: float, d_motion_m: float | None) -> bool:
    """Flag a large GPS step only when expected motion cannot explain it."""
    if d_gps_m < 20.0:
        return False
    if d_motion_m is None or d_motion_m <= 0:
        return True
    return d_gps_m / d_motion_m >= 5.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--audit-grid-m", type=float, default=20.0)
    args = parser.parse_args()
    output_dir = args.work_dir / "03_quality_audits"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_raw_dataset(args.input_dir)
    standardized_path = args.work_dir / "02_standardized" / "standardized_records.csv"
    spectra_path = args.work_dir / "02_standardized" / "spectra_and_quality.npz"
    if not standardized_path.exists() or not spectra_path.exists():
        raise FileNotFoundError("Run 02_standardize_records.py before Stage 03.")
    with standardized_path.open("r", encoding="utf-8-sig", newline="") as handle:
        standardized = list(csv.DictReader(handle))
    with np.load(spectra_path) as arrays:
        powers = arrays["power_dbm"]
        frequencies = arrays["frequencies_hz"]

    elevation_groups: dict[str, list[float]] = defaultdict(list)
    for record in dataset.records:
        if record.elevation_gps_msl_m is not None:
            elevation_groups[record.source_file].append(record.elevation_gps_msl_m)
    altitude_rows = []
    fixed_offsets = []
    for info in dataset.files:
        values = elevation_groups[info.name]
        stats = describe(values)
        policy, nominal = height_policy(info.name)
        median = float(stats["median"]) if values else None
        offset = median - nominal if median is not None and nominal is not None else None
        if offset is not None:
            fixed_offsets.append(offset)
        altitude_rows.append(
            {
                "source_file": info.name,
                "height_label_policy": policy,
                "nominal_relative_height_m": "" if nominal is None else f"{nominal:.3f}",
                **stats,
                "gps_elevation_range_m": "" if not values else f"{max(values) - min(values):.6f}",
                "median_msl_minus_nominal_m": "" if offset is None else f"{offset:.6f}",
            }
        )
    write_csv(output_dir / "altitude_audit_by_file.csv", altitude_rows)

    sensor_rows = []
    for source_file in [info.name for info in dataset.files]:
        group = [record for record in dataset.records if record.source_file == source_file]
        for field in SENSOR_FIELDS:
            values = [record.sensors[field] for record in group if record.sensors[field] is not None]
            repeats = sum(left == right for left, right in zip(values, values[1:]))
            sensor_rows.append(
                {
                    "source_file": source_file,
                    "field": field,
                    **describe(values),
                    "consecutive_repeat_fraction": ""
                    if len(values) < 2
                    else f"{repeats / (len(values) - 1):.6f}",
                    "semantic_status": "raw_or_mcs_transformed_value_requires_axis_scale_verification",
                }
            )
    write_csv(output_dir / "inertial_field_audit.csv", sensor_rows)

    spatial_by_file: dict[str, list[dict]] = defaultdict(list)
    for row in standardized:
        if row["position_valid"].lower() == "true":
            spatial_by_file[row["source_file"]].append(row)
    transition_rows = []
    gps_rows = []
    for source_file, group in sorted(spatial_by_file.items()):
        group = sorted(group, key=lambda row: int(row["source_row"]))
        same_steps = 0
        same_with_motion = 0
        jumps = 0
        run_length = 1
        run_start = None
        max_run = 1
        max_span_s = 0.0
        displacements = []
        for previous, current in zip(group, group[1:]):
            previous_time = datetime.fromisoformat(previous["timestamp"])
            current_time = datetime.fromisoformat(current["timestamp"])
            delta_t = (current_time - previous_time).total_seconds()
            dx = float(current["x_raw_m"]) - float(previous["x_raw_m"])
            dy = float(current["y_raw_m"]) - float(previous["y_raw_m"])
            d_gps = math.hypot(dx, dy)
            speed = finite_csv(current["speed_mps"])
            d_motion = None if speed is None else speed * max(delta_t, 0.0)
            ratio = None if not d_motion or d_motion <= 0 else d_gps / d_motion
            same = d_gps <= 1e-9
            jump = is_gps_jump_candidate(d_gps, d_motion)
            if same:
                same_steps += 1
                same_with_motion += int(speed is not None and speed > 0.2)
                if run_length == 1:
                    run_start = previous_time
                run_length += 1
                max_run = max(max_run, run_length)
                if run_start is not None:
                    max_span_s = max(max_span_s, (current_time - run_start).total_seconds())
            else:
                displacements.append(d_gps)
                run_length = 1
                run_start = None
            jumps += int(jump)
            transition_rows.append(
                {
                    "source_file": source_file,
                    "previous_record_index": previous["record_index"],
                    "current_record_index": current["record_index"],
                    "delta_t_s": f"{delta_t:.6f}",
                    "d_gps_m": f"{d_gps:.6f}",
                    "speed_mps": "" if speed is None else f"{speed:.6f}",
                    "d_motion_m": "" if d_motion is None else f"{d_motion:.6f}",
                    "gps_motion_ratio": "" if ratio is None else f"{ratio:.6f}",
                    "same_coordinate": same,
                    "position_quality_flag": "gps_jump_candidate" if jump else "",
                }
            )
        latitudes = [float(row["latitude_raw"]) for row in group]
        longitudes = [float(row["longitude_raw"]) for row in group]
        pairs = {(latitude, longitude) for latitude, longitude in zip(latitudes, longitudes)}
        gps_rows.append(
            {
                "source_file": source_file,
                "records": len(group),
                "unique_latitudes": len(set(latitudes)),
                "unique_longitudes": len(set(longitudes)),
                "unique_horizontal_pairs": len(pairs),
                "latitude_step_deg_min": infer_coordinate_step(latitudes),
                "longitude_step_deg_min": infer_coordinate_step(longitudes),
                "same_coordinate_steps": same_steps,
                "same_coordinate_fraction": same_steps / max(len(group) - 1, 1),
                "same_coordinate_with_speed_gt_0_2_mps": same_with_motion,
                "max_same_coordinate_run_records": max_run,
                "max_same_coordinate_span_s": max_span_s,
                "nonzero_displacement_count": len(displacements),
                "nonzero_displacement_median_m": "" if not displacements else float(np.median(displacements)),
                "nonzero_displacement_max_m": "" if not displacements else float(np.max(displacements)),
                "gps_jump_candidate_count": jumps,
            }
        )
    all_latitudes = [float(row["latitude_raw"]) for row in standardized if row["position_valid"].lower() == "true"]
    all_longitudes = [float(row["longitude_raw"]) for row in standardized if row["position_valid"].lower() == "true"]
    all_pairs = set(zip(all_latitudes, all_longitudes))
    gps_rows.append(
        {
            "source_file": "ALL_FILES",
            "records": len(all_latitudes),
            "unique_latitudes": len(set(all_latitudes)),
            "unique_longitudes": len(set(all_longitudes)),
            "unique_horizontal_pairs": len(all_pairs),
            "latitude_step_deg_min": infer_coordinate_step(all_latitudes),
            "longitude_step_deg_min": infer_coordinate_step(all_longitudes),
            "same_coordinate_steps": sum(int(row["same_coordinate_steps"]) for row in gps_rows),
            "same_coordinate_fraction": sum(int(row["same_coordinate_steps"]) for row in gps_rows) / max(len(transition_rows), 1),
            "same_coordinate_with_speed_gt_0_2_mps": sum(int(row["same_coordinate_with_speed_gt_0_2_mps"]) for row in gps_rows),
            "max_same_coordinate_run_records": max(int(row["max_same_coordinate_run_records"]) for row in gps_rows),
            "max_same_coordinate_span_s": max(float(row["max_same_coordinate_span_s"]) for row in gps_rows),
            "nonzero_displacement_count": sum(int(row["nonzero_displacement_count"]) for row in gps_rows),
            "nonzero_displacement_median_m": float(np.median([float(row["d_gps_m"]) for row in transition_rows if float(row["d_gps_m"]) > 0])),
            "nonzero_displacement_max_m": max(float(row["d_gps_m"]) for row in transition_rows),
            "gps_jump_candidate_count": sum(int(row["gps_jump_candidate_count"]) for row in gps_rows),
        }
    )
    write_csv(output_dir / "gps_transition_audit.csv", transition_rows)
    write_csv(output_dir / "gps_summary_by_file.csv", gps_rows)

    grid_groups: dict[tuple[int, int, int, int], list[float]] = defaultdict(list)
    for row in standardized:
        if row.get("core_spatial_product_included", "").lower() != "true":
            continue
        x = finite_csv(row["x_raw_m"])
        y = finite_csv(row["y_raw_m"])
        z = finite_csv(row["z_relative_label_m"])
        record_index = int(row["record_index"])
        if x is None or y is None or z is None:
            continue
        ix = int(round(x / args.audit_grid_m))
        iy = int(round(y / args.audit_grid_m))
        iz = int(round(z))
        for frequency_index, value in enumerate(powers[record_index]):
            if np.isfinite(value):
                grid_groups[(ix, iy, iz, frequency_index)].append(float(value))

    aggregation_rows = []
    linear_minus_dbm = []
    median_minus_dbm = []
    for (ix, iy, iz, frequency_index), values in sorted(grid_groups.items()):
        stats = aggregate_power_dbm(np.asarray(values))
        difference_linear = (
            stats["power_dbm_from_linear_mean"] - stats["power_dbm_arithmetic_mean"]
        )
        difference_median = stats["power_dbm_median"] - stats["power_dbm_arithmetic_mean"]
        linear_minus_dbm.append(difference_linear)
        median_minus_dbm.append(difference_median)
        aggregation_rows.append(
            {
                "ix": ix,
                "iy": iy,
                "iz": iz,
                "frequency_hz": int(frequencies[frequency_index]),
                **{key: f"{value:.6f}" if isinstance(value, float) else value for key, value in stats.items()},
                "linear_mean_minus_dbm_mean_db": f"{difference_linear:.6f}",
                "median_minus_dbm_mean_db": f"{difference_median:.6f}",
            }
        )
    write_csv(output_dir / "power_aggregation_comparison_20m.csv", aggregation_rows)

    elevation_offset_std = float(np.std(fixed_offsets)) if fixed_offsets else None
    source_summary = json.loads(
        (args.work_dir / "01_source_audit" / "source_audit_summary.json").read_text(encoding="utf-8")
    )
    aggregate_gps = gps_rows[-1]
    five_dimension_qc = {
        "power_quality": {
            "floor_like_values": source_summary["floor_like_spectrum_values"],
            "positive_power_candidate_values": source_summary["positive_spectrum_values"],
            "policy": "annotate retained values; do not delete or replace",
        },
        "position_quality": {
            "valid_horizontal_records": len(all_latitudes),
            "unique_latitudes": aggregate_gps["unique_latitudes"],
            "unique_longitudes": aggregate_gps["unique_longitudes"],
            "unique_horizontal_pairs": aggregate_gps["unique_horizontal_pairs"],
            "same_coordinate_fraction": aggregate_gps["same_coordinate_fraction"],
            "gps_jump_candidates": aggregate_gps["gps_jump_candidate_count"],
        },
        "temporal_quality": {
            "missing_timestamps": source_summary["timestamp_missing"],
            "duplicate_timestamp_occurrences": source_summary["duplicate_timestamp_occurrences"],
            "nonpositive_intervals": source_summary["interval_nonpositive"],
            "intervals_gt_10s": source_summary["interval_gt_10s"],
        },
        "spectral_quality": {
            "frequency_bins": len(frequencies),
            "missing_power_values": source_summary["missing_spectrum_values"],
            "shared_frequency_axis_verified": True,
        },
        "spatial_coverage": {
            "horizontal_coordinate_pairs": aggregate_gps["unique_horizontal_pairs"],
            "aggregation_grid_metrics_added_in_stage_07": True,
        },
    }
    write_json(output_dir / "five_dimension_qc_summary.json", five_dimension_qc)
    write_json(
        output_dir / "quality_audit_summary.json",
        {
            "altitude": {
                "files_with_finite_gps_elevation": sum(bool(values) for values in elevation_groups.values()),
                "fixed_height_file_offset_std_m": elevation_offset_std,
                "decision": "retain elevation_gps_msl_m as an auxiliary raw field; do not replace relative-height labels before independent validation",
            },
            "power_aggregation": {
                "aggregated_cell_frequency_entries": len(aggregation_rows),
                "median_linear_minus_dbm_mean_db": float(np.median(linear_minus_dbm)),
                "p95_linear_minus_dbm_mean_db": float(np.percentile(linear_minus_dbm, 95)),
                "median_absolute_median_minus_dbm_mean_db": float(
                    np.median(np.abs(median_minus_dbm))
                ),
                "decision": "publish explicitly named dBm mean, linear-power mean converted to dBm, and dBm median",
            },
            "inertial": {
                "fields_audited": list(SENSOR_FIELDS),
                "decision": "do not fuse until units, axes, mounting rotation, bias, calibration, and synchronization are verified",
            },
            "five_dimension_qc": five_dimension_qc,
        },
    )
    print(f"Stage 03 complete: {output_dir}")


if __name__ == "__main__":
    main()
