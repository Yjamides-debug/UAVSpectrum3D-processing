#!/usr/bin/env python3
"""Stage 04: reconstruct horizontal coordinates without replacing raw GPS."""

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

from uavspectrum3d_pipeline.config import (
    DEFAULT_WORK_DIR,
    LAT_M_PER_DEG,
    LON_M_PER_DEG,
)
from uavspectrum3d_pipeline.coordinates import infer_coordinate_step, local_m_to_latlon
from uavspectrum3d_pipeline.trajectory import (
    TrackData,
    bounded_quantization_reconstruct,
    rts_reconstruct,
)
from uavspectrum3d_pipeline.utils import write_csv, write_json


def finite(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def build_track(source_file: str, rows: list[dict]) -> TrackData:
    rows = sorted(rows, key=lambda row: int(row["source_row"]))
    timestamps = [np.datetime64(row["timestamp"]) for row in rows]
    origin = timestamps[0]
    time_s = np.array([(value - origin) / np.timedelta64(1, "s") for value in timestamps], dtype=float)
    return TrackData(
        record_indices=np.array([int(row["record_index"]) for row in rows], dtype=np.int32),
        source_file=source_file,
        x_raw_m=np.array([float(row["x_raw_m"]) for row in rows]),
        y_raw_m=np.array([float(row["y_raw_m"]) for row in rows]),
        time_s=time_s,
        speed_mps=np.array([finite(row["speed_mps"]) or 0.0 for row in rows]),
        heading_deg=np.array(
            [finite(row["heading_true_north_deg"]) if finite(row["heading_true_north_deg"]) is not None else np.nan for row in rows]
        ),
    )


def position_errors(estimate: np.ndarray, truth: np.ndarray) -> dict:
    errors = np.linalg.norm(estimate - truth, axis=1)
    return {
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "median_m": float(np.median(errors)),
        "p95_m": float(np.percentile(errors, 95)),
        "max_m": float(np.max(errors)),
    }


def simulated_track(
    scenario: str,
    seed: int,
    cell_x_m: float,
    cell_y_m: float,
    n: int = 240,
) -> tuple[TrackData, np.ndarray]:
    rng = np.random.default_rng(seed)
    dt = rng.choice(np.array([0.7, 2.5]), size=n - 1, p=[0.55, 0.45])
    times = np.concatenate([[0.0], np.cumsum(dt)])
    if scenario == "line":
        truth = np.column_stack([2.2 * times, 0.6 * times])
    elif scenario == "circle":
        angle = times * 0.035
        truth = np.column_stack([75.0 * np.cos(angle), 75.0 * np.sin(angle)])
    else:
        heading = np.where(times < times[-1] / 3, 25.0, np.where(times < 2 * times[-1] / 3, 115.0, 210.0))
        speed = 2.2 + 0.5 * np.sin(times / 20.0)
        velocity = np.column_stack(
            [speed * np.sin(np.radians(heading)), speed * np.cos(np.radians(heading))]
        )
        truth = np.zeros((n, 2), dtype=float)
        truth[1:] = np.cumsum(velocity[1:] * dt[:, None], axis=0)
    displacement = np.diff(truth, axis=0)
    speed = np.concatenate([[0.0], np.linalg.norm(displacement, axis=1) / dt])
    heading = np.concatenate(
        [[np.nan], (np.degrees(np.arctan2(displacement[:, 0], displacement[:, 1])) + 360.0) % 360.0]
    )
    observed_speed = np.maximum(speed + rng.normal(0.0, 0.15, n), 0.0)
    observed_heading = (heading + rng.normal(0.0, 5.0, n)) % 360.0
    observed_heading[rng.random(n) > 0.5181] = np.nan
    raw_x = np.round(truth[:, 0] / cell_x_m) * cell_x_m
    raw_y = np.round(truth[:, 1] / cell_y_m) * cell_y_m
    track = TrackData(
        record_indices=np.arange(n, dtype=np.int32),
        source_file=f"simulation_{scenario}_{seed}",
        x_raw_m=raw_x,
        y_raw_m=raw_y,
        time_s=times,
        speed_mps=observed_speed,
        heading_deg=observed_heading,
    )
    return track, truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--min-speed-mps", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    output_dir = args.work_dir / "04_trajectory_reconstruction"
    output_dir.mkdir(parents=True, exist_ok=True)

    standardized_dir = args.work_dir / "02_standardized"
    with (standardized_dir / "standardized_records.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        all_rows = list(csv.DictReader(handle))
    metadata = json.loads(
        (standardized_dir / "standardization_metadata.json").read_text(encoding="utf-8")
    )
    spatial_rows = [row for row in all_rows if row["position_valid"].lower() == "true"]
    latitude_values = [float(row["latitude_raw"]) for row in spatial_rows]
    longitude_values = [float(row["longitude_raw"]) for row in spatial_rows]
    lat_step = infer_coordinate_step(latitude_values)
    lon_step = infer_coordinate_step(longitude_values)
    cell_y_m = lat_step * LAT_M_PER_DEG
    cell_x_m = (
        lon_step
        * math.cos(math.radians(metadata["reference_latitude"]))
        * LON_M_PER_DEG
    )
    if cell_x_m <= 0 or cell_y_m <= 0:
        raise ValueError("Cannot infer positive GPS quantization-cell dimensions.")

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in spatial_rows:
        groups[row["source_file"]].append(row)
    coordinate_rows = []
    method_rows = []
    for source_file, group in groups.items():
        track = build_track(source_file, group)
        rts_xy, rts_covariance = rts_reconstruct(
            track, cell_x_m, cell_y_m, args.min_speed_mps
        )
        bounded_xy, solver = bounded_quantization_reconstruct(
            track, cell_x_m, cell_y_m, args.min_speed_mps
        )
        raw_xy = np.column_stack([track.x_raw_m, track.y_raw_m])
        for method, estimate in (("rts", rts_xy), ("quantization_aware", bounded_xy)):
            shift = np.linalg.norm(estimate - raw_xy, axis=1)
            method_rows.append(
                {
                    "source_file": source_file,
                    "method": method,
                    "records": len(group),
                    "median_shift_from_raw_m": f"{np.median(shift):.6f}",
                    "p95_shift_from_raw_m": f"{np.percentile(shift, 95):.6f}",
                    "max_shift_from_raw_m": f"{np.max(shift):.6f}",
                    "changed_10m_cells": int(
                        np.sum(np.any(np.round(estimate / 10.0) != np.round(raw_xy / 10.0), axis=1))
                    ),
                    "changed_20m_cells": int(
                        np.sum(np.any(np.round(estimate / 20.0) != np.round(raw_xy / 20.0), axis=1))
                    ),
                    "solver_success": True if method == "rts" else solver["success"],
                }
            )
        for index, row in enumerate(sorted(group, key=lambda item: int(item["source_row"]))):
            rts_lat, rts_lon = local_m_to_latlon(
                rts_xy[index, 0],
                rts_xy[index, 1],
                metadata["reference_latitude"],
                metadata["reference_longitude"],
            )
            bounded_lat, bounded_lon = local_m_to_latlon(
                bounded_xy[index, 0],
                bounded_xy[index, 1],
                metadata["reference_latitude"],
                metadata["reference_longitude"],
            )
            coordinate_rows.append(
                {
                    "record_index": row["record_index"],
                    "record_id": row["record_id"],
                    "source_file": source_file,
                    "source_row": row["source_row"],
                    "timestamp": row["timestamp"],
                    "latitude_raw": row["latitude_raw"],
                    "longitude_raw": row["longitude_raw"],
                    "x_raw_m": row["x_raw_m"],
                    "y_raw_m": row["y_raw_m"],
                    "z_relative_label_m": row["z_relative_label_m"],
                    "z_legacy_protocol_m": row.get("z_legacy_protocol_m", ""),
                    "x_rts_m": f"{rts_xy[index, 0]:.6f}",
                    "y_rts_m": f"{rts_xy[index, 1]:.6f}",
                    "rts_std_x_m": f"{math.sqrt(max(rts_covariance[index, 0, 0], 0.0)):.6f}",
                    "rts_std_y_m": f"{math.sqrt(max(rts_covariance[index, 1, 1], 0.0)):.6f}",
                    "latitude_rts": f"{rts_lat:.10f}",
                    "longitude_rts": f"{rts_lon:.10f}",
                    "x_quantization_aware_m": f"{bounded_xy[index, 0]:.6f}",
                    "y_quantization_aware_m": f"{bounded_xy[index, 1]:.6f}",
                    "latitude_quantization_aware": f"{bounded_lat:.10f}",
                    "longitude_quantization_aware": f"{bounded_lon:.10f}",
                    "available_methods": "rts;quantization_aware_bounded_smoother",
                    "coordinate_status": "model_derived_not_ground_truth",
                }
            )
    write_csv(output_dir / "reconstructed_horizontal_coordinates.csv", coordinate_rows)
    write_csv(output_dir / "method_comparison_by_file.csv", method_rows)

    validation_rows = []
    for scenario_index, scenario in enumerate(("line", "circle", "turns")):
        for replicate in range(5):
            track, truth = simulated_track(
                scenario,
                args.seed + scenario_index * 100 + replicate,
                cell_x_m,
                cell_y_m,
            )
            raw = np.column_stack([track.x_raw_m, track.y_raw_m])
            rts, _ = rts_reconstruct(track, cell_x_m, cell_y_m, args.min_speed_mps)
            bounded, _ = bounded_quantization_reconstruct(
                track, cell_x_m, cell_y_m, args.min_speed_mps
            )
            for method, estimate in (
                ("raw_quantized_cell_center", raw),
                ("rts", rts),
                ("quantization_aware", bounded),
            ):
                validation_rows.append(
                    {
                        "validation": "data_calibrated_simulation",
                        "scenario": scenario,
                        "replicate": replicate,
                        "method": method,
                        **position_errors(estimate, truth),
                    }
                )
    for source_file, group in groups.items():
        track = build_track(source_file, group)
        truth_proxy = np.column_stack([track.x_raw_m, track.y_raw_m])
        coarse_track = TrackData(
            record_indices=track.record_indices,
            source_file=track.source_file,
            x_raw_m=np.round(track.x_raw_m / (2.0 * cell_x_m)) * (2.0 * cell_x_m),
            y_raw_m=np.round(track.y_raw_m / (2.0 * cell_y_m)) * (2.0 * cell_y_m),
            time_s=track.time_s,
            speed_mps=track.speed_mps,
            heading_deg=track.heading_deg,
        )
        coarse_raw = np.column_stack([coarse_track.x_raw_m, coarse_track.y_raw_m])
        coarse_rts, _ = rts_reconstruct(coarse_track, 2.0 * cell_x_m, 2.0 * cell_y_m, args.min_speed_mps)
        coarse_bounded, _ = bounded_quantization_reconstruct(
            coarse_track, 2.0 * cell_x_m, 2.0 * cell_y_m, args.min_speed_mps
        )
        for method, estimate in (
            ("raw_double_quantized", coarse_raw),
            ("rts", coarse_rts),
            ("quantization_aware", coarse_bounded),
        ):
            validation_rows.append(
                {
                    "validation": "double_quantization_proxy",
                    "scenario": source_file,
                    "replicate": "aggregate",
                    "method": method,
                    **position_errors(estimate, truth_proxy),
                }
            )

        position_mask = np.ones(len(track.x_raw_m), dtype=bool)
        position_mask[2:-2:5] = False
        held = ~position_mask
        held_rts, _ = rts_reconstruct(
            track, cell_x_m, cell_y_m, args.min_speed_mps, position_mask=position_mask
        )
        held_bounded, _ = bounded_quantization_reconstruct(
            track, cell_x_m, cell_y_m, args.min_speed_mps, position_mask=position_mask
        )
        for method, estimate in (("rts", held_rts), ("quantization_aware", held_bounded)):
            metrics = position_errors(estimate[held], truth_proxy[held])
            cell_hit = (
                (np.abs(estimate[held, 0] - truth_proxy[held, 0]) <= cell_x_m / 2.0)
                & (np.abs(estimate[held, 1] - truth_proxy[held, 1]) <= cell_y_m / 2.0)
            )
            validation_rows.append(
                {
                    "validation": "coordinate_holdout_proxy",
                    "scenario": source_file,
                    "replicate": "every_fifth_interior_record",
                    "method": method,
                    **metrics,
                    "proxy_cell_hit_fraction": float(np.mean(cell_hit)),
                }
            )
    write_csv(output_dir / "simulation_validation.csv", validation_rows)
    write_json(
        output_dir / "trajectory_metadata.json",
        {
            "records_reconstructed": len(coordinate_rows),
            "reference_latitude": metadata["reference_latitude"],
            "reference_longitude": metadata["reference_longitude"],
            "latitude_step_deg": lat_step,
            "longitude_step_deg": lon_step,
            "quantization_cell_width_x_m": cell_x_m,
            "quantization_cell_width_y_m": cell_y_m,
            "primary_output": "raw coordinates remain primary; RTS and quantization-aware horizontal coordinates are separate experimental products",
            "vertical_output": "z_relative_label_m is retained only for fixed-height records; dynamic records expose z_legacy_protocol_m as an auxiliary legacy field; z is not reconstructed",
            "coordinate_status": "model_derived_not_ground_truth",
            "rts_parameters": {
                "state": ["x_m", "y_m", "vx_mps", "vy_mps"],
                "transition": "constant velocity with per-record dt; dt floored at 1e-3 s",
                "process_noise_axis": "[[dt^3/3, dt^2/2], [dt^2/2, dt]] * acceleration_q",
                "acceleration_q": 1.0,
                "position_sigma_x_m": "max(quantization_cell_width_x_m/sqrt(12), 0.25)",
                "position_sigma_y_m": "max(quantization_cell_width_y_m/sqrt(12), 0.25)",
                "initial_state": "[x_raw_m[0], y_raw_m[0], 0, 0]",
                "initial_velocity_variance_m2ps2": 25.0,
                "minimum_speed_for_velocity_observation_mps": 0.2,
                "velocity_observation_sigma_mps": 1.0,
                "heading_convention": "degrees clockwise from true north; vx=v*sin(heading), vy=v*cos(heading)",
            },
            "quantization_aware_parameters": {
                "solver": "scipy.optimize.lsq_linear",
                "solver_method": "trf",
                "lsmr_tol": "auto",
                "tol": 1e-7,
                "max_iter": 2000,
                "minimum_speed_for_motion_constraint_mps": 0.2,
                "motion_sigma_m": 5.0,
                "acceleration_sigma_mps": 2.0,
                "acceleration_regularizer_semantics": "difference between adjacent segment velocities; not divided by time to form acceleration",
                "normal_position_bound_half_width_cells": 0.5,
                "jump_position_bound_half_width_cells": 1.5,
                "jump_minimum_gps_displacement_m": 20.0,
                "jump_minimum_gps_to_motion_ratio": 5.0,
                "jump_position_weight_divisor": 5.0,
                "masked_position_bounds": "global raw coordinate range expanded by one quantization cell per axis",
            },
            "limitations": [
                "No synchronous high-precision trajectory truth is available.",
                "Simulation validation does not establish real-flight absolute positioning accuracy.",
                "Gyro, Compass, and Tilt are not fused before their semantics and calibration are verified.",
            ],
        },
    )
    print(f"Stage 04 complete: {output_dir}")


if __name__ == "__main__":
    main()
