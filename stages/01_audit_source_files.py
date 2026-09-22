#!/usr/bin/env python3
"""Stage 01: audit raw MCS files and record-level completeness."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavspectrum3d_pipeline.config import (
    DEFAULT_INPUT_DIR,
    DEFAULT_WORK_DIR,
    FLOOR_POWER_DBM,
    POSITIVE_POWER_DBM,
)
from uavspectrum3d_pipeline.height import height_policy
from uavspectrum3d_pipeline.raw_io import load_raw_dataset
from uavspectrum3d_pipeline.utils import write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    args = parser.parse_args()
    output_dir = args.work_dir / "01_source_audit"

    dataset = load_raw_dataset(args.input_dir)
    file_rows = []
    categories = Counter()
    for info in dataset.files:
        if info.parse_failures or info.short_rows or info.frequency_bins != 201:
            category = "invalid"
            reason = "parse, row-length, or frequency-axis failure"
        elif info.valid_position_rows == 0:
            category = "excluded"
            reason = "no valid horizontal coordinates"
        elif height_policy(info.name)[1] is None:
            category = "supplementary"
            reason = "supplementary trajectory records retained outside core coordinate-indexed aggregation products"
        else:
            category = "valid"
            reason = "fixed-height records included in core coordinate-indexed aggregation products"
        categories[category] += 1
        file_rows.append(
            {
                "raw_file": info.name,
                "category": category,
                "input_rows": info.input_rows,
                "frequency_bins": info.frequency_bins,
                "gps_columns_present": info.gps_columns_present,
                "valid_position_rows": info.valid_position_rows,
                "parse_failures": info.parse_failures,
                "short_rows": info.short_rows,
                "reason": reason,
            }
        )

    record_rows = []
    for record in dataset.records:
        finite = np.isfinite(record.powers_dbm)
        record_rows.append(
            {
                "record_index": record.record_index,
                "record_id": f"{record.source_file}:{record.source_row}",
                "source_file": record.source_file,
                "source_row": record.source_row,
                "timestamp_valid": record.timestamp is not None,
                "position_valid": record.position_valid,
                "elevation_valid": record.elevation_gps_msl_m is not None,
                "speed_valid": record.speed_knots_raw is not None,
                "heading_valid": record.heading_true_north_deg is not None,
                "measure_unit": record.measure_unit,
                "finite_frequency_bins": int(finite.sum()),
                "missing_frequency_bins": int((~finite).sum()),
            }
        )

    records_by_file = defaultdict(list)
    for record in dataset.records:
        records_by_file[record.source_file].append(record)
    category_by_file = {row["raw_file"]: row["category"] for row in file_rows}
    measurement_rows = []
    for info in dataset.files:
        records = sorted(records_by_file[info.name], key=lambda item: item.source_row)
        timestamps = [record.timestamp for record in records if record.timestamp is not None]
        duplicate_timestamps = len(timestamps) - len(set(timestamps))
        intervals = [
            (right - left).total_seconds()
            for left, right in zip(timestamps, timestamps[1:])
        ]
        powers = (
            np.vstack([record.powers_dbm for record in records])
            if records
            else np.empty((0, len(dataset.frequencies_hz)))
        )
        finite = np.isfinite(powers)
        finite_per_record = finite.sum(axis=1) if len(records) else np.array([], dtype=int)
        measurement_rows.append(
            {
                "source_file": info.name,
                "category": category_by_file[info.name],
                "records": len(records),
                "timestamp_missing": sum(record.timestamp is None for record in records),
                "duplicate_timestamp_occurrences": duplicate_timestamps,
                "interval_nonpositive": sum(value <= 0 for value in intervals),
                "interval_gt_10s": sum(value > 10 for value in intervals),
                "interval_median_s": "" if not intervals else f"{np.median(intervals):.6f}",
                "missing_latitude": sum(record.latitude is None for record in records),
                "missing_longitude": sum(record.longitude is None for record in records),
                "missing_elevation": sum(record.elevation_gps_msl_m is None for record in records),
                "missing_speed": sum(record.speed_knots_raw is None for record in records),
                "missing_heading": sum(record.heading_true_north_deg is None for record in records),
                "missing_spectrum_values": int((~finite).sum()),
                "floor_like_spectrum_values": int((finite & (powers <= FLOOR_POWER_DBM)).sum()),
                "positive_spectrum_values": int((finite & (powers > POSITIVE_POWER_DBM)).sum()),
                "valid_frequency_bins_min": "" if not len(finite_per_record) else int(finite_per_record.min()),
                "valid_frequency_bins_median": "" if not len(finite_per_record) else float(np.median(finite_per_record)),
                "valid_frequency_bins_max": "" if not len(finite_per_record) else int(finite_per_record.max()),
            }
        )

    all_powers = dataset.power_matrix_dbm
    all_finite = np.isfinite(all_powers)
    all_timestamps = [record.timestamp for record in dataset.records if record.timestamp is not None]
    measurement_rows.append(
        {
            "source_file": "ALL_FILES",
            "category": "aggregate",
            "records": len(dataset.records),
            "timestamp_missing": sum(record.timestamp is None for record in dataset.records),
            "duplicate_timestamp_occurrences": sum(
                len([record for record in records_by_file[name] if record.timestamp is not None])
                - len({record.timestamp for record in records_by_file[name] if record.timestamp is not None})
                for name in records_by_file
            ),
            "interval_nonpositive": sum(int(row["interval_nonpositive"]) for row in measurement_rows),
            "interval_gt_10s": sum(int(row["interval_gt_10s"]) for row in measurement_rows),
            "interval_median_s": "",
            "missing_latitude": sum(record.latitude is None for record in dataset.records),
            "missing_longitude": sum(record.longitude is None for record in dataset.records),
            "missing_elevation": sum(record.elevation_gps_msl_m is None for record in dataset.records),
            "missing_speed": sum(record.speed_knots_raw is None for record in dataset.records),
            "missing_heading": sum(record.heading_true_north_deg is None for record in dataset.records),
            "missing_spectrum_values": int((~all_finite).sum()),
            "floor_like_spectrum_values": int((all_finite & (all_powers <= FLOOR_POWER_DBM)).sum()),
            "positive_spectrum_values": int((all_finite & (all_powers > POSITIVE_POWER_DBM)).sum()),
            "valid_frequency_bins_min": int(all_finite.sum(axis=1).min()),
            "valid_frequency_bins_median": float(np.median(all_finite.sum(axis=1))),
            "valid_frequency_bins_max": int(all_finite.sum(axis=1).max()),
        }
    )

    write_csv(output_dir / "file_audit.csv", file_rows)
    write_csv(output_dir / "record_completeness.csv", record_rows)
    write_csv(output_dir / "measurement_quality_summary.csv", measurement_rows)
    write_json(
        output_dir / "source_audit_summary.json",
        {
            "raw_files": len(dataset.files),
            "valid_files": categories["valid"],
            "invalid_files": categories["invalid"],
            "excluded_files": categories["excluded"],
            "input_records": len(dataset.records),
            "records_with_valid_position": sum(record.position_valid for record in dataset.records),
            "frequency_bins": len(dataset.frequencies_hz),
            "frequency_min_hz": int(dataset.frequencies_hz.min()),
            "frequency_max_hz": int(dataset.frequencies_hz.max()),
            "timestamp_missing": measurement_rows[-1]["timestamp_missing"],
            "duplicate_timestamp_occurrences": measurement_rows[-1]["duplicate_timestamp_occurrences"],
            "interval_nonpositive": measurement_rows[-1]["interval_nonpositive"],
            "interval_gt_10s": measurement_rows[-1]["interval_gt_10s"],
            "missing_spectrum_values": measurement_rows[-1]["missing_spectrum_values"],
            "floor_like_spectrum_values": measurement_rows[-1]["floor_like_spectrum_values"],
            "positive_spectrum_values": measurement_rows[-1]["positive_spectrum_values"],
            "raw_files_unchanged": True,
        },
    )
    print(f"Stage 01 complete: {output_dir}")


if __name__ == "__main__":
    main()
