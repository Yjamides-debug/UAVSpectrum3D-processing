#!/usr/bin/env python3
"""Stage 05: build raw-coordinate and reconstructed-coordinate matrix products."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavspectrum3d_pipeline.config import DEFAULT_WORK_DIR
from uavspectrum3d_pipeline.products import build_matrix_product
from uavspectrum3d_pipeline.tensor import matrix_to_tensor, validate_tensor_against_matrix
from uavspectrum3d_pipeline.utils import write_csv, write_json


def write_product(
    output_dir: Path,
    product: dict,
    grid_m: float,
    coordinate_source: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in product.items() if key != "cells"}
    np.savez_compressed(output_dir / "matrix_product.npz", **arrays)
    tensor = matrix_to_tensor(product)
    validate_tensor_against_matrix(product, tensor)
    np.savez_compressed(output_dir / "tensor_product.npz", **tensor)
    cell_rows = [
        {
            "cell_id": index,
            "ix": ix,
            "iy": iy,
            "iz": iz,
            "x_center_m": f"{ix * grid_m:.6f}",
            "y_center_m": f"{iy * grid_m:.6f}",
            "z_relative_label_center_m": f"{iz:.6f}",
        }
        for index, (ix, iy, iz) in enumerate(product["cells"])
    ]
    write_csv(output_dir / "cell_coordinates.csv", cell_rows)
    write_json(
        output_dir / "metadata.json",
        {
            "coordinate_source": coordinate_source,
            "grid_m": grid_m,
            "z_index_step_m": 1.0,
            "cells": len(product["cells"]),
            "frequencies": len(product["frequencies_hz"]),
            "aggregation_statistics": [
                "power_dbm_arithmetic_mean",
                "power_dbm_from_linear_mean",
                "power_dbm_median",
            ],
            "grid_semantics": "post-processing aggregation grid; not positioning resolution",
            "height_semantics": "protocol-derived relative-height label",
            "tensor_representation": "regular x-y-z-frequency axes with NaN/unobserved cells; no interpolation",
            "tensor_shape": list(tensor["power_dbm_arithmetic_mean"].shape),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--grid-sizes-m", default="10,20")
    args = parser.parse_args()
    output_dir = args.work_dir / "05_analysis_products"
    obsolete = output_dir / "reconstructed_coordinates"
    if obsolete.exists():
        shutil.rmtree(obsolete)
    standardized_dir = args.work_dir / "02_standardized"
    trajectory_dir = args.work_dir / "04_trajectory_reconstruction"

    with (standardized_dir / "standardized_records.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        standardized = list(csv.DictReader(handle))
    with (trajectory_dir / "reconstructed_horizontal_coordinates.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reconstructed = list(csv.DictReader(handle))
    with np.load(standardized_dir / "spectra_and_quality.npz") as arrays:
        powers = arrays["power_dbm"].copy()
        frequencies = arrays["frequencies_hz"].copy()
    core_rows = [
        row
        for row in standardized
        if row.get("core_spatial_product_included", "").lower() == "true"
    ]

    grid_sizes = sorted({float(value.strip()) for value in args.grid_sizes_m.split(",") if value.strip()})
    summary_rows = []
    for grid_m in grid_sizes:
        label = f"grid_{grid_m:g}m"
        raw_product = build_matrix_product(
            core_rows,
            powers,
            frequencies,
            grid_m,
            "x_raw_m",
            "y_raw_m",
        )
        rts_product = build_matrix_product(
            reconstructed,
            powers,
            frequencies,
            grid_m,
            "x_rts_m",
            "y_rts_m",
        )
        bounded_product = build_matrix_product(
            reconstructed,
            powers,
            frequencies,
            grid_m,
            "x_quantization_aware_m",
            "y_quantization_aware_m",
        )
        write_product(
            output_dir / "raw_coordinates" / label,
            raw_product,
            grid_m,
            "raw_quantized_GPS_coordinates",
        )
        write_product(
            output_dir / "rts_coordinates_experimental" / label,
            rts_product,
            grid_m,
            "model_derived_RTS_horizontal_coordinates_experimental",
        )
        write_product(
            output_dir / "quantization_aware_coordinates_experimental" / label,
            bounded_product,
            grid_m,
            "model_derived_quantization_aware_horizontal_coordinates_experimental",
        )
        for source, product in (
            ("raw_coordinates", raw_product),
            ("rts_coordinates_experimental", rts_product),
            ("quantization_aware_coordinates_experimental", bounded_product),
        ):
            summary_rows.append(
                {
                    "coordinate_source": source,
                    "grid_m": grid_m,
                    "observed_cells": len(product["cells"]),
                    "observed_cell_frequency_entries": int(product["observed_mask"].sum()),
                    "strict_quality_entries": int(product["quality_mask"].sum()),
                    "total_contributing_samples": int(product["sample_count"].sum()),
                }
            )
    write_csv(output_dir / "product_summary.csv", summary_rows)
    print(f"Stage 05 complete: {output_dir}")


if __name__ == "__main__":
    main()
