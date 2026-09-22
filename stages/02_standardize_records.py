#!/usr/bin/env python3
"""Stage 02: standardize fields while preserving all raw power observations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavspectrum3d_pipeline.config import DEFAULT_INPUT_DIR, DEFAULT_WORK_DIR
from uavspectrum3d_pipeline.coordinates import coordinate_reference, latlon_to_local_m
from uavspectrum3d_pipeline.height import (
    assign_formal_relative_height_labels,
    assign_legacy_protocol_height_labels,
    height_policy,
    is_core_spatial_height_record,
)
from uavspectrum3d_pipeline.power import power_quality_flags
from uavspectrum3d_pipeline.raw_io import SENSOR_FIELDS, load_raw_dataset
from uavspectrum3d_pipeline.utils import csv_number, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    args = parser.parse_args()
    output_dir = args.work_dir / "02_standardized"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_raw_dataset(args.input_dir)
    ref_lat, ref_lon = coordinate_reference(dataset.records)
    formal_height_labels = assign_formal_relative_height_labels(dataset.records)
    legacy_height_labels = assign_legacy_protocol_height_labels(dataset.records)
    power_matrix = dataset.power_matrix_dbm
    observed_mask = np.isfinite(power_matrix)
    quality_flags = power_quality_flags(power_matrix)

    rows = []
    for record in dataset.records:
        x_m = y_m = None
        if record.position_valid:
            x_m, y_m = latlon_to_local_m(record.latitude, record.longitude, ref_lat, ref_lon)
        policy, _ = height_policy(record.source_file)
        row = {
            "record_index": record.record_index,
            "record_id": f"{record.source_file}:{record.source_row}",
            "source_file": record.source_file,
            "source_row": record.source_row,
            "timestamp": record.timestamp_raw,
            "latitude_raw": csv_number(record.latitude, 8),
            "longitude_raw": csv_number(record.longitude, 8),
            "elevation_gps_msl_m": csv_number(record.elevation_gps_msl_m, 3),
            "x_raw_m": csv_number(x_m, 6),
            "y_raw_m": csv_number(y_m, 6),
            "z_relative_label_m": csv_number(formal_height_labels[record.record_index], 6),
            "z_legacy_protocol_m": csv_number(legacy_height_labels[record.record_index], 6),
            "height_label_policy": policy,
            "height_label_source": (
                "filename_nominal_height"
                if formal_height_labels[record.record_index] is not None
                else "legacy_time_protocol"
                if legacy_height_labels[record.record_index] is not None
                else "none"
            ),
            "height_label_status": (
                "formal_core_label"
                if formal_height_labels[record.record_index] is not None
                else "legacy_unvalidated_auxiliary"
                if legacy_height_labels[record.record_index] is not None
                else "unavailable"
            ),
            "core_spatial_product_included": is_core_spatial_height_record(
                record.source_file, record.position_valid
            ),
            "speed_knots_raw": csv_number(record.speed_knots_raw, 6),
            "speed_mps": csv_number(record.speed_mps, 6),
            "heading_true_north_deg": csv_number(record.heading_true_north_deg, 6),
            "measure_unit": record.measure_unit,
            "position_valid": record.position_valid,
            "finite_frequency_bins": int(observed_mask[record.record_index].sum()),
        }
        for sensor in SENSOR_FIELDS:
            row[sensor.lower().replace("-", "_")] = csv_number(record.sensors[sensor], 6)
        rows.append(row)

    write_csv(output_dir / "standardized_records.csv", rows)
    np.savez_compressed(
        output_dir / "spectra_and_quality.npz",
        power_dbm=power_matrix,
        frequencies_hz=dataset.frequencies_hz,
        observed_mask=observed_mask,
        quality_flags=quality_flags,
        record_indices=np.arange(len(dataset.records), dtype=np.int32),
    )
    write_json(
        output_dir / "standardization_metadata.json",
        {
            "records": len(dataset.records),
            "frequency_bins": len(dataset.frequencies_hz),
            "reference_latitude": ref_lat,
            "reference_longitude": ref_lon,
            "speed_conversion": "speed_mps = speed_knots_raw * 0.5144444444444445",
            "elevation_semantics": "raw GPS antenna height above mean sea level as exported",
            "relative_height_semantics": "z_relative_label_m is populated only from fixed-height filenames; z_legacy_protocol_m stores unvalidated historical time labels for circular and descending flights",
            "core_spatial_product_policy": "coordinate-indexed aggregation products include only records with valid horizontal coordinates and filename-derived fixed-height labels",
            "power_semantics": "raw dBm values retained without replacement or calibration",
        },
    )
    print(f"Stage 02 complete: {output_dir}")


if __name__ == "__main__":
    main()
