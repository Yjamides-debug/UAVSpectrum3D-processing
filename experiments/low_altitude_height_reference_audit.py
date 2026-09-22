#!/usr/bin/env python3
"""Audit whether 1-9 m fixed-height runs can support trajectory height labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np


LOW_HEIGHT_PATTERN = re.compile(r"_(1|7|9)m\.csv$", re.IGNORECASE)
TRAJECTORY_MARKERS = ("circle1", "circle2", "linefrom9mto1m")


def finite(value: str | None) -> float | None:
    try:
        result = float(value) if value not in (None, "") else math.nan
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def nominal_height(source_file: str) -> float | None:
    match = LOW_HEIGHT_PATTERN.search(source_file)
    return float(match.group(1)) if match else None


def coordinate(row: dict[str, str]) -> tuple[float, float] | None:
    latitude = finite(row.get("latitude_raw"))
    longitude = finite(row.get("longitude_raw"))
    if latitude is None or longitude is None:
        return None
    return latitude, longitude


def upper_speed_half(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return a sensitivity subset, not a flight/non-flight classification."""
    speeds = [finite(row.get("speed_mps")) for row in rows]
    valid = [value for value in speeds if value is not None]
    if not valid:
        return []
    threshold = quantile(valid, 0.5)
    return [
        row
        for row, speed in zip(rows, speeds)
        if speed is not None and speed >= threshold
    ]


def elevation_values(rows: list[dict[str, str]]) -> list[float]:
    return [
        value
        for row in rows
        if (value := finite(row.get("elevation_gps_msl_m"))) is not None
    ]


def common_language_probability(lower: list[float], higher: list[float]) -> float:
    """Return P(higher > lower) + 0.5 P(higher == lower)."""
    lower_sorted = np.sort(np.asarray(lower, dtype=float))
    higher_array = np.asarray(higher, dtype=float)
    left = np.searchsorted(lower_sorted, higher_array, side="left")
    right = np.searchsorted(lower_sorted, higher_array, side="right")
    return float(np.sum(left + 0.5 * (right - left)) / (len(lower_sorted) * len(higher_array)))


def batch_summary(source_file: str, rows: list[dict[str, str]]) -> dict:
    rows = sorted(rows, key=lambda row: int(row["source_row"]))
    elevations = elevation_values(rows)
    speeds = [value for row in rows if (value := finite(row.get("speed_mps"))) is not None]
    times = [parse_time(row["timestamp"]) for row in rows]
    intervals = [
        (current - previous).total_seconds()
        for previous, current in zip(times, times[1:])
        if current > previous
    ]
    upper = upper_speed_half(rows)
    upper_elevations = elevation_values(upper)
    return {
        "source_file": source_file,
        "nominal_relative_height_m": nominal_height(source_file),
        "records": len(rows),
        "duration_s": round((times[-1] - times[0]).total_seconds(), 6),
        "sampling_interval_median_s": round(quantile(intervals, 0.5), 6),
        "speed_q10_mps": round(quantile(speeds, 0.1), 6),
        "speed_q25_mps": round(quantile(speeds, 0.25), 6),
        "speed_median_mps": round(quantile(speeds, 0.5), 6),
        "speed_q75_mps": round(quantile(speeds, 0.75), 6),
        "speed_q90_mps": round(quantile(speeds, 0.9), 6),
        "gps_elevation_median_m": round(quantile(elevations, 0.5), 6),
        "gps_elevation_iqr_m": round(quantile(elevations, 0.75) - quantile(elevations, 0.25), 6),
        "upper_speed_half_records": len(upper),
        "upper_speed_half_elevation_median_m": round(quantile(upper_elevations, 0.5), 6),
        "upper_speed_half_elevation_iqr_m": round(
            quantile(upper_elevations, 0.75) - quantile(upper_elevations, 0.25), 6
        ),
        "unique_horizontal_pairs": len({coordinate(row) for row in rows if coordinate(row) is not None}),
    }


def coordinate_medians(rows: list[dict[str, str]]) -> dict[tuple[float, float], float]:
    groups: dict[tuple[float, float], list[float]] = defaultdict(list)
    for row in rows:
        coord = coordinate(row)
        elevation = finite(row.get("elevation_gps_msl_m"))
        if coord is not None and elevation is not None:
            groups[coord].append(elevation)
    return {coord: float(np.median(values)) for coord, values in groups.items()}


def matched_coordinate_rows(groups: dict[str, list[dict[str, str]]]) -> tuple[list[dict], list[dict]]:
    active = {source_file: coordinate_medians(upper_speed_half(rows)) for source_file, rows in groups.items()}
    details: list[dict] = []
    summaries: list[dict] = []
    for left_file, right_file in combinations(sorted(groups), 2):
        left_height = nominal_height(left_file)
        right_height = nominal_height(right_file)
        common = sorted(set(active[left_file]) & set(active[right_file]))
        differences = []
        for latitude, longitude in common:
            difference = active[right_file][(latitude, longitude)] - active[left_file][(latitude, longitude)]
            differences.append(difference)
            details.append(
                {
                    "left_source_file": left_file,
                    "right_source_file": right_file,
                    "left_nominal_height_m": left_height,
                    "right_nominal_height_m": right_height,
                    "nominal_height_difference_m": right_height - left_height,
                    "latitude": latitude,
                    "longitude": longitude,
                    "left_gps_elevation_median_m": round(active[left_file][(latitude, longitude)], 6),
                    "right_gps_elevation_median_m": round(active[right_file][(latitude, longitude)], 6),
                    "gps_elevation_difference_m": round(difference, 6),
                }
            )
        if differences:
            summaries.append(
                {
                    "left_source_file": left_file,
                    "right_source_file": right_file,
                    "nominal_height_difference_m": right_height - left_height,
                    "matched_horizontal_pairs": len(common),
                    "gps_elevation_difference_median_m": round(float(np.median(differences)), 6),
                    "gps_elevation_difference_min_m": round(float(np.min(differences)), 6),
                    "gps_elevation_difference_max_m": round(float(np.max(differences)), 6),
                }
            )
    return details, summaries


def leave_one_batch_out(groups: dict[str, list[dict[str, str]]]) -> list[dict]:
    output: list[dict] = []
    for target_file, target_rows_all in sorted(groups.items()):
        target_rows = upper_speed_half(target_rows_all)
        reference_values: dict[tuple[float, float], list[float]] = defaultdict(list)
        for source_file, rows in groups.items():
            if source_file == target_file:
                continue
            height = nominal_height(source_file)
            for row in upper_speed_half(rows):
                coord = coordinate(row)
                elevation = finite(row.get("elevation_gps_msl_m"))
                if coord is not None and elevation is not None and height is not None:
                    reference_values[coord].append(elevation - height)
        baselines = {coord: float(np.median(values)) for coord, values in reference_values.items()}
        predictions = []
        for row in target_rows:
            coord = coordinate(row)
            elevation = finite(row.get("elevation_gps_msl_m"))
            if coord in baselines and elevation is not None:
                predictions.append(elevation - baselines[coord])
        target_height = nominal_height(target_file)
        absolute_errors = [abs(value - target_height) for value in predictions]
        output.append(
            {
                "target_source_file": target_file,
                "nominal_relative_height_m": target_height,
                "upper_speed_half_records": len(target_rows),
                "coordinate_matched_records": len(predictions),
                "coordinate_match_fraction": round(len(predictions) / max(len(target_rows), 1), 6),
                "predicted_height_median_m": "" if not predictions else round(float(np.median(predictions)), 6),
                "absolute_error_median_m": "" if not absolute_errors else round(float(np.median(absolute_errors)), 6),
                "absolute_error_q90_m": "" if not absolute_errors else round(quantile(absolute_errors, 0.9), 6),
            }
        )
    return output


def separation_rows(groups: dict[str, list[dict[str, str]]]) -> list[dict]:
    by_height: dict[float, list[float]] = defaultdict(list)
    for source_file, rows in groups.items():
        by_height[nominal_height(source_file)].extend(elevation_values(upper_speed_half(rows)))
    output = []
    for lower, higher in combinations(sorted(by_height), 2):
        output.append(
            {
                "lower_nominal_height_m": lower,
                "higher_nominal_height_m": higher,
                "nominal_height_difference_m": higher - lower,
                "lower_records": len(by_height[lower]),
                "higher_records": len(by_height[higher]),
                "lower_gps_elevation_median_m": round(float(np.median(by_height[lower])), 6),
                "higher_gps_elevation_median_m": round(float(np.median(by_height[higher])), 6),
                "gps_median_difference_m": round(
                    float(np.median(by_height[higher]) - np.median(by_height[lower])), 6
                ),
                "common_language_probability": round(
                    common_language_probability(by_height[lower], by_height[higher]), 6
                ),
            }
        )
    return output


def trajectory_comparison(
    trajectory_groups: dict[str, list[dict[str, str]]],
    fixed_groups: dict[str, list[dict[str, str]]],
) -> list[dict]:
    reference_by_height: dict[float, list[float]] = defaultdict(list)
    for source_file, rows in fixed_groups.items():
        reference_by_height[nominal_height(source_file)].extend(elevation_values(upper_speed_half(rows)))
    reference_medians = {
        height: float(np.median(values)) for height, values in reference_by_height.items()
    }
    reference_min = min(quantile(values, 0.1) for values in reference_by_height.values())
    reference_max = max(quantile(values, 0.9) for values in reference_by_height.values())
    output = []
    for source_file, rows in sorted(trajectory_groups.items()):
        elevations = elevation_values(rows)
        nearest_counts = {height: 0 for height in reference_medians}
        for elevation in elevations:
            nearest = min(reference_medians, key=lambda height: abs(elevation - reference_medians[height]))
            nearest_counts[nearest] += 1
        output.append(
            {
                "source_file": source_file,
                "records": len(rows),
                "gps_elevation_median_m": round(float(np.median(elevations)), 6),
                "gps_elevation_q10_m": round(quantile(elevations, 0.1), 6),
                "gps_elevation_q90_m": round(quantile(elevations, 0.9), 6),
                "within_fixed_low_altitude_q10_q90_envelope_fraction": round(
                    sum(reference_min <= value <= reference_max for value in elevations) / len(elevations), 6
                ),
                **{
                    f"nearest_reference_{height:g}m_records": nearest_counts[height]
                    for height in sorted(nearest_counts)
                },
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    records_path = args.work_dir / "02_standardized" / "standardized_records.csv"
    output_dir = args.output_dir or args.work_dir / "10_motion_audit"
    with records_path.open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))

    fixed_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    trajectory_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        source_file = row["source_file"]
        if nominal_height(source_file) is not None and finite(row.get("elevation_gps_msl_m")) is not None:
            fixed_groups[source_file].append(row)
        elif any(marker in source_file.lower() for marker in TRAJECTORY_MARKERS):
            trajectory_groups[source_file].append(row)

    summaries = [batch_summary(source_file, rows) for source_file, rows in sorted(fixed_groups.items())]
    matched_details, matched_summaries = matched_coordinate_rows(fixed_groups)
    cross_validation = leave_one_batch_out(fixed_groups)
    separation = separation_rows(fixed_groups)
    trajectory = trajectory_comparison(trajectory_groups, fixed_groups)

    write_csv(output_dir / "low_altitude_reference_batch_summary.csv", summaries)
    write_csv(output_dir / "low_altitude_matched_coordinate_differences.csv", matched_details)
    write_csv(output_dir / "low_altitude_matched_coordinate_summary.csv", matched_summaries)
    write_csv(output_dir / "low_altitude_leave_one_batch_out.csv", cross_validation)
    write_csv(output_dir / "low_altitude_elevation_separation.csv", separation)
    write_csv(output_dir / "trajectory_vs_low_altitude_reference.csv", trajectory)

    seven_nine = next(
        row
        for row in separation
        if row["lower_nominal_height_m"] == 7.0 and row["higher_nominal_height_m"] == 9.0
    )
    summary = {
        "reference_heights_m": [1.0, 7.0, 9.0],
        "reference_batches": len(fixed_groups),
        "three_meter_reference_available": False,
        "speed_policy": (
            "No universal speed cutoff is inferred. The upper within-batch speed half is used only "
            "as a sensitivity subset because speed distributions differ strongly by batch."
        ),
        "sampling_policy": "All future smoothing or segmentation must use elapsed seconds, not row counts.",
        "seven_vs_nine_m_common_language_probability": seven_nine["common_language_probability"],
        "height_derivation_decision": (
            "GPS elevation, quantized horizontal coordinates and speed do not support an independently "
            "validated record-level height estimator. Use them to detect and audit trajectory stages; "
            "map detected stages to nominal labels only under an explicit flight-protocol constraint."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "low_altitude_height_reference_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Low-altitude height-reference audit complete: {output_dir}")


if __name__ == "__main__":
    main()
