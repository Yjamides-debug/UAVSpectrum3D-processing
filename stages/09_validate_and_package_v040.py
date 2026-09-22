#!/usr/bin/env python3
"""Stage 09: validate stage contracts and assemble the v0.4.0 release package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavspectrum3d_pipeline.config import (
    DEFAULT_INPUT_DIR,
    DEFAULT_RELEASE_DIR,
    DEFAULT_WORK_DIR,
)
from figure_generation.plot_v040_validation_figures import generate as generate_validation_figures
from uavspectrum3d_pipeline.height import height_policy
from uavspectrum3d_pipeline.release_docs import write_release_documentation
from uavspectrum3d_pipeline.tensor import validate_tensor_against_matrix
from uavspectrum3d_pipeline.utils import write_csv, write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_matrix(path: Path) -> dict:
    with np.load(path) as arrays:
        dbm_mean = arrays["power_dbm_arithmetic_mean"]
        linear_mean = arrays["power_dbm_from_linear_mean"]
        median = arrays["power_dbm_median"]
        count = arrays["sample_count"]
        observed = arrays["observed_mask"]
        quality = arrays["quality_mask"]
        flags = arrays["quality_flags"]
        shapes = {array.shape for array in (dbm_mean, linear_mean, median, count, observed, quality, flags)}
        if len(shapes) != 1:
            raise ValueError(f"Array shapes disagree in {path}.")
        if not np.array_equal(observed, count > 0):
            raise ValueError(f"Observed mask disagrees with sample_count in {path}.")
        if np.any(quality & (~observed | (flags != 0))):
            raise ValueError(f"Quality-mask contract failed in {path}.")
        if np.any(np.isnan(dbm_mean[observed])) or np.any(~np.isnan(dbm_mean[~observed])):
            raise ValueError(f"dBm-mean missingness contract failed in {path}.")
        if np.any(linear_mean[observed] + 1e-4 < dbm_mean[observed]):
            raise ValueError(f"Linear mean violates the log-domain Jensen check in {path}.")
        return {
            "shape": list(dbm_mean.shape),
            "observed_entries": int(observed.sum()),
            "quality_entries": int(quality.sum()),
            "contributing_samples": int(count.sum()),
        }


def public_raw_name(name: str) -> str:
    if name.lower().startswith("recording_"):
        return name.replace(" ", "_")
    prefix = "扫描 Recording_ "
    suffix = name[len(prefix) :] if name.startswith(prefix) else name
    return "recording_" + suffix.replace(" ", "_")


def public_record_rows(rows: list[dict], name_mapping: dict[str, str]) -> list[dict]:
    """Replace source filenames in release tables while preserving manifest traceability."""
    public_rows = []
    for row in rows:
        converted = dict(row)
        original_name = row.get("source_file", "")
        public_name = name_mapping.get(original_name, original_name)
        converted["source_file"] = public_name
        if "record_id" in converted:
            converted["record_id"] = f"{public_name}:{row.get('source_row', '')}"
        public_rows.append(converted)
    return public_rows


def reset_release_directory(release: Path) -> None:
    """Remove only a specifically named generated dataset directory."""
    allowed_name = release.name == "UAVSpectrum3D" or release.name.startswith(
        "UAVSpectrum3D_v"
    )
    if not allowed_name:
        raise ValueError(
            "Refusing to replace a release directory unless its name is "
            "'UAVSpectrum3D' or starts with 'UAVSpectrum3D_v'."
        )
    if release.exists():
        shutil.rmtree(release)
    release.mkdir(parents=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    args = parser.parse_args()
    generate_validation_figures(args.work_dir)
    validation_dir = args.work_dir / "09_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    source_summary = json.loads(
        (args.work_dir / "01_source_audit" / "source_audit_summary.json").read_text(encoding="utf-8")
    )
    standard_dir = args.work_dir / "02_standardized"
    standard_rows = load_csv(standard_dir / "standardized_records.csv")
    core_spatial_rows = [
        row
        for row in standard_rows
        if row.get("core_spatial_product_included", "").lower() == "true"
    ]
    dynamic_rows = [
        row for row in standard_rows
        if row.get("position_valid", "").lower() == "true"
        and row.get("core_spatial_product_included", "").lower() != "true"
        and row.get("z_legacy_protocol_m", "")
    ]
    standardization_metadata = json.loads(
        (standard_dir / "standardization_metadata.json").read_text(encoding="utf-8")
    )
    trajectory_dir = args.work_dir / "04_trajectory_reconstruction"
    reconstructed_rows = load_csv(trajectory_dir / "reconstructed_horizontal_coordinates.csv")
    trajectory_metadata = json.loads(
        (trajectory_dir / "trajectory_metadata.json").read_text(encoding="utf-8")
    )
    quality_summary = json.loads(
        (args.work_dir / "03_quality_audits" / "quality_audit_summary.json").read_text(
            encoding="utf-8"
        )
    )
    five_dimension_qc = json.loads(
        (args.work_dir / "03_quality_audits" / "five_dimension_qc_summary.json").read_text(
            encoding="utf-8"
        )
    )
    gps_summary_rows = load_csv(
        args.work_dir / "03_quality_audits" / "gps_summary_by_file.csv"
    )
    trajectory_validation_rows = load_csv(trajectory_dir / "simulation_validation.csv")
    frequency_completion_dir = args.work_dir / "06_frequency_completion"
    frequency_completion_summary = load_csv(frequency_completion_dir / "results_summary.csv")
    frequency_completion_splits = load_csv(frequency_completion_dir / "split_audit.csv")
    frequency_completion_protocol = json.loads(
        (frequency_completion_dir / "protocol_metadata.json").read_text(encoding="utf-8")
    )
    grid_sensitivity_dir = args.work_dir / "07_grid_sensitivity"
    grid_sensitivity = json.loads(
        (grid_sensitivity_dir / "grid_sensitivity_summary.json").read_text(
            encoding="utf-8"
        )
    )
    coordinate_sensitivity_dir = args.work_dir / "08_coordinate_sensitivity"
    coordinate_sensitivity_protocol = json.loads(
        (coordinate_sensitivity_dir / "protocol_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    coordinate_sensitivity_results = load_csv(
        coordinate_sensitivity_dir / "results_summary.csv"
    )
    coordinate_sensitivity_by_seed = load_csv(
        coordinate_sensitivity_dir / "results_by_seed.csv"
    )
    with np.load(standard_dir / "spectra_and_quality.npz") as arrays:
        power_shape = list(arrays["power_dbm"].shape)
        finite_power_values = int(np.isfinite(arrays["power_dbm"]).sum())
        frequencies = arrays["frequencies_hz"].copy()

    checks = {
        "record_count_matches": len(standard_rows) == source_summary["input_records"],
        "frequency_shape_matches": power_shape == [len(standard_rows), 201],
        "spatial_record_count_matches": len(reconstructed_rows)
        == source_summary["records_with_valid_position"],
        "spatial_expansion_matches": len(reconstructed_rows) * 201 == 1_099_068,
        "core_expansion_matches": len(core_spatial_rows) * 201 == 948_318,
        "dynamic_record_count_matches": len(dynamic_rows) == 750,
        "raw_expansion_matches": len(standard_rows) * 201 == 1_145_700,
        "core_spatial_products_have_fixed_height_labels": all(
            row["height_label_policy"].startswith("constant_")
            and row["position_valid"].lower() == "true"
            for row in core_spatial_rows
        ),
        "frequency_completion_summary_present": bool(frequency_completion_summary),
        "frequency_completion_split_count_matches": frequency_completion_protocol["split_count"]
        == len(frequency_completion_splits),
        "frequency_completion_partitions_valid": all(
            row["partition_ok"].lower() == "true" for row in frequency_completion_splits
        ),
        "five_dimension_qc_present": set(five_dimension_qc)
        == {
            "power_quality",
            "position_quality",
            "temporal_quality",
            "spectral_quality",
            "spatial_coverage",
        },
        "gps_quantization_counts_match": (
            five_dimension_qc["position_quality"]["unique_latitudes"],
            five_dimension_qc["position_quality"]["unique_longitudes"],
            five_dimension_qc["position_quality"]["unique_horizontal_pairs"],
        )
        == (12, 3, 26),
        "gps_aggregate_row_present": any(
            row["source_file"] == "ALL_FILES" for row in gps_summary_rows
        ),
        "grid_sensitivity_has_10m_and_20m": {
            float(row["grid_m"]) for row in grid_sensitivity["configuration"]
        }
        == {10.0, 20.0},
        "cross_grid_metrics_finite": all(
            np.isfinite(float(row[metric]))
            for row in grid_sensitivity["cross_grid"]
            for metric in ("mae_db", "rmse_db", "pearson")
        ),
        "coordinate_sensitivity_uses_10_seeds": coordinate_sensitivity_protocol["seeds"]
        == list(range(10)),
        "coordinate_sensitivity_same_test_records": coordinate_sensitivity_protocol[
            "same_test_records_across_coordinate_sources"
        ]
        is True,
        "coordinate_sources_complete": set(
            coordinate_sensitivity_protocol["coordinate_sources"]
        )
        == {"raw", "rts", "quantization_aware"},
        "coordinate_length_scales_complete": set(
            coordinate_sensitivity_protocol["horizontal_length_scales_m"]
        )
        == {20.0, 40.0, 80.0},
        "coordinate_result_design_complete": len(coordinate_sensitivity_by_seed)
        == 10 * 3 * 3,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Stage-level validation failed: {failed}")

    matrix_checks = {}
    coordinate_sources = (
        "raw_coordinates",
        "rts_coordinates_experimental",
        "quantization_aware_coordinates_experimental",
    )
    for coordinate_source in coordinate_sources:
        for grid in ("grid_10m", "grid_20m"):
            key = f"{coordinate_source}/{grid}"
            matrix_checks[key] = validate_matrix(
                args.work_dir
                / "05_analysis_products"
                / coordinate_source
                / grid
                / "matrix_product.npz"
            )
            if coordinate_source == "raw_coordinates":
                product_dir = args.work_dir / "05_analysis_products" / coordinate_source / grid
                with np.load(product_dir / "matrix_product.npz") as arrays:
                    matrix_product = {name: arrays[name].copy() for name in arrays.files}
                with np.load(product_dir / "tensor_product.npz") as arrays:
                    tensor_product = {name: arrays[name].copy() for name in arrays.files}
                cell_rows = load_csv(product_dir / "cell_coordinates.csv")
                matrix_product["cells"] = [
                    (int(row["ix"]), int(row["iy"]), int(row["iz"])) for row in cell_rows
                ]
                validate_tensor_against_matrix(matrix_product, tensor_product)

    release = args.release_dir
    reset_release_directory(release)
    for directory in (
        release / "raw" / "original_csv",
        release / "metadata",
        release / "processed" / "records",
        release / "processed" / "records" / "fixed_height",
        release / "processed" / "records" / "dynamic_trajectory",
        release / "processed" / "matrix" / "raw_coordinates",
        release / "processed" / "tensor" / "raw_coordinates",
        release / "diagnostics",
        release / "diagnostics" / "07_grid_sensitivity",
        release / "reference_reuse" / "frequency_completion",
        release / "reference_reuse" / "trajectory_reconstruction" / "coordinates",
        release / "reference_reuse" / "trajectory_reconstruction" / "matrix",
        release
        / "reference_reuse"
        / "trajectory_reconstruction"
        / "sensitivity_analysis",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    raw_manifest = []
    public_name_mapping = {}
    file_audit = {row["raw_file"]: row for row in load_csv(args.work_dir / "01_source_audit" / "file_audit.csv")}
    for source in sorted(args.input_dir.glob("*.csv")):
        public_name = public_raw_name(source.name)
        public_name_mapping[source.name] = public_name
        public_path = release / "raw" / "original_csv" / public_name
        shutil.copy2(source, public_path)
        source_hash = sha256(source)
        public_hash = sha256(public_path)
        if source_hash != public_hash:
            raise ValueError(f"Raw-file copy hash mismatch for {source.name}.")
        audit = file_audit[source.name]
        _, fixed_height = height_policy(source.name)
        core_coordinate_included = audit["category"] == "valid" and fixed_height is not None
        manifest_reason = audit["reason"]
        if audit["category"] == "valid" and fixed_height is None:
            manifest_reason = (
                "supplementary trajectory file retained in raw and standardized records; "
                "not included in core coordinate-indexed aggregation products"
            )
        elif core_coordinate_included:
            manifest_reason = "fixed-height file included in core coordinate-indexed aggregation products"
        raw_manifest.append(
            {
                "original_file": source.name,
                "public_file": public_name,
                "category": audit["category"],
                "input_rows": audit["input_rows"],
                "valid_position_rows": audit["valid_position_rows"],
                "coordinate_product_included": core_coordinate_included,
                "reason": manifest_reason,
                "sha256": source_hash,
            }
        )
    checks["raw_copy_hashes_match"] = True
    write_csv(release / "metadata" / "raw_file_manifest.csv", raw_manifest)
    write_csv(
        release / "metadata" / "frequency_bins.csv",
        [
            {"frequency_index": index, "frequency_hz": int(value)}
            for index, value in enumerate(frequencies)
        ],
    )
    write_json(
        release / "metadata" / "dataset_metadata.json",
        {
            "dataset_name": "UAVSpectrum3D",
            "package_version": "0.4.0",
            "package_status": "final public release published through Zenodo",
            "dataset_doi": "10.5281/zenodo.21983541",
            "schema_version": "3.0",
            "records": len(standard_rows),
            "spatial_records": len(reconstructed_rows),
            "core_spatial_product_records": len(core_spatial_rows),
            "frequency_bins": len(frequencies),
            "frequency_min_hz": int(frequencies.min()),
            "frequency_max_hz": int(frequencies.max()),
            "frequency_step_hz": int(frequencies[1] - frequencies[0]),
            "rbw_hz": 1_000_000,
            "raw_power_values_retained": True,
            "license": "CC-BY-4.0",
            "absolute_altitude_reconstruction_included": False,
            "analysis_height_axis": "z_relative_label_m for fixed-height records only",
            "auxiliary_gps_elevation_field": "elevation_gps_msl_m",
            "core_spatial_product_policy": "only fixed-height batches with valid horizontal coordinates enter core coordinate-indexed aggregation products",
            "supplementary_trajectory_policy": "circular and descending-line files are retained in raw and standardized records; their z_legacy_protocol_m labels are auxiliary only and they are excluded from core spatial-height aggregation products",
            "trajectory_status": "reference-reuse model outputs; not core data or ground truth",
            "grid_semantics": "10 m and 20 m are aggregation grids, not positioning resolution",
            "approved_airspace_boundary": "metadata/approved_airspace_boundary.csv; four WGS84 vertices transcribed from an author-retained operating-approval record; approval document not distributed",
        },
    )
    write_json(
        release / "metadata" / "coordinate_reference.json",
        {
            "reference_latitude_deg": standardization_metadata["reference_latitude"],
            "reference_longitude_deg": standardization_metadata["reference_longitude"],
            "horizontal_coordinate_source": "exported WGS84 latitude and longitude",
            "x_formula": "(longitude - reference_longitude) * cos(reference_latitude) * 111320",
            "y_formula": "(latitude - reference_latitude) * 110540",
            "coordinate_use": "local indexing and aggregation within the measurement extent",
            "positioning_precision_claim": "none; exported coordinates are quantized",
        },
    )
    write_csv(
        release / "metadata" / "approved_airspace_boundary.csv",
        [
            {"vertex_order": 1, "longitude_deg": 104.32138888888889, "latitude_deg": 30.3175, "x_m": -246.42790621104962, "y_m": -355.1087392097534, "crs": "WGS84", "source_provenance": "transcribed_from_author_retained_operating_approval_record", "public_scope": "boundary_vertices_only; approval document not included"},
            {"vertex_order": 2, "longitude_deg": 104.32388888888889, "latitude_deg": 30.334166666666665, "x_m": -6.19569856850821, "y_m": 1487.224594123475, "crs": "WGS84", "source_provenance": "transcribed_from_author_retained_operating_approval_record", "public_scope": "boundary_vertices_only; approval document not included"},
            {"vertex_order": 3, "longitude_deg": 104.34472222222222, "latitude_deg": 30.330833333333334, "x_m": 1995.7393651207024, "y_m": 1118.7579274571435, "crs": "WGS84", "source_provenance": "transcribed_from_author_retained_operating_approval_record", "public_scope": "boundary_vertices_only; approval document not included"},
            {"vertex_order": 4, "longitude_deg": 104.34055555555555, "latitude_deg": 30.31361111111111, "x_m": 1595.3523523831334, "y_m": -784.9865169874021, "crs": "WGS84", "source_provenance": "transcribed_from_author_retained_operating_approval_record", "public_scope": "boundary_vertices_only; approval document not included"},
        ],
    )
    write_json(
        release / "metadata" / "quality_flag_definitions.json",
        {
            "dtype": "uint8",
            "bit_0": {
                "value": 1,
                "name": "floor_like",
                "condition": "power_dbm <= -149.9",
            },
            "bit_1": {
                "value": 2,
                "name": "positive_power_candidate",
                "condition": "power_dbm > 0",
            },
            "record_semantics": "flags annotate retained observations without replacement or deletion",
            "aggregate_semantics": "a bit is set if any finite contributing sample sets that bit",
            "quality_mask": "observed_mask AND quality_flags == 0",
        },
    )

    write_csv(
        release / "processed" / "records" / "standardized_records.csv",
        public_record_rows(standard_rows, public_name_mapping),
    )
    write_csv(
        release / "processed" / "records" / "fixed_height" / "core_record_indices.csv",
        [{"record_index": row["record_index"], "record_id": f"{public_name_mapping.get(row['source_file'], row['source_file'])}:{row['source_row']}", "source_file": public_name_mapping.get(row["source_file"], row["source_file"]), "z_relative_label_m": row["z_relative_label_m"]} for row in core_spatial_rows],
    )
    write_csv(
        release / "processed" / "records" / "dynamic_trajectory" / "trajectory_record_indices.csv",
        [{"record_index": row["record_index"], "record_id": f"{public_name_mapping.get(row['source_file'], row['source_file'])}:{row['source_row']}", "source_file": public_name_mapping.get(row["source_file"], row["source_file"]), "trajectory_type": "circle" if "circle" in row["source_file"].lower() else "descending_line"} for row in dynamic_rows],
    )
    write_csv(
        release / "processed" / "records" / "dynamic_trajectory" / "legacy_protocol_height_labels.csv",
        [{"record_index": row["record_index"], "record_id": f"{public_name_mapping.get(row['source_file'], row['source_file'])}:{row['source_row']}", "source_file": public_name_mapping.get(row["source_file"], row["source_file"]), "z_legacy_protocol_m": row["z_legacy_protocol_m"], "height_label_policy": row["height_label_policy"]} for row in dynamic_rows],
    )
    shutil.copy2(standard_dir / "spectra_and_quality.npz", release / "processed" / "records")
    reference_trajectory = release / "reference_reuse" / "trajectory_reconstruction"
    for name in ("reconstructed_horizontal_coordinates.csv", "method_comparison_by_file.csv"):
        write_csv(
            reference_trajectory / "coordinates" / name,
            public_record_rows(load_csv(trajectory_dir / name), public_name_mapping),
        )
    shutil.copy2(
        trajectory_dir / "simulation_validation.csv",
        reference_trajectory / "simulation_validation.csv",
    )
    shutil.copy2(
        trajectory_dir / "trajectory_metadata.json",
        reference_trajectory / "trajectory_metadata.json",
    )
    products_source = args.work_dir / "05_analysis_products"
    for coordinate_source in coordinate_sources:
        for grid in ("grid_10m", "grid_20m"):
            if coordinate_source == "raw_coordinates":
                destination = release / "processed" / "matrix" / "raw_coordinates" / grid
            else:
                destination = reference_trajectory / "matrix" / coordinate_source / grid
            destination.mkdir(parents=True, exist_ok=True)
            for source in (products_source / coordinate_source / grid).glob("*"):
                if source.is_file():
                    if source.name != "tensor_product.npz":
                        shutil.copy2(source, destination / source.name)
            if coordinate_source == "raw_coordinates":
                tensor_destination = release / "processed" / "tensor" / "raw_coordinates" / grid
                tensor_destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(
                    products_source / coordinate_source / grid / "tensor_product.npz",
                    tensor_destination / "tensor_product.npz",
                )
    product_summary_rows = load_csv(products_source / "product_summary.csv")
    write_csv(
        release / "metadata" / "aggregation_product_summary.csv",
        [row for row in product_summary_rows if row["coordinate_source"] == "raw_coordinates"],
    )
    write_csv(
        reference_trajectory / "experimental_product_summary.csv",
        [row for row in product_summary_rows if row["coordinate_source"] != "raw_coordinates"],
    )
    for source in frequency_completion_dir.rglob("*"):
        if source.is_file():
            destination = release / "reference_reuse" / "frequency_completion" / source.relative_to(frequency_completion_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    for stage_name in ("01_source_audit", "03_quality_audits"):
        stage = args.work_dir / stage_name
        destination = release / "diagnostics" / stage_name
        destination.mkdir(parents=True, exist_ok=True)
        for source in stage.glob("*"):
            if source.is_file():
                shutil.copy2(source, destination / source.name)
    for source in grid_sensitivity_dir.glob("*"):
        if source.is_file():
            shutil.copy2(
                source,
                release / "diagnostics" / "07_grid_sensitivity" / source.name,
            )
    coordinate_sensitivity_release = (
        reference_trajectory / "sensitivity_analysis"
    )
    for source in coordinate_sensitivity_dir.rglob("*"):
        if source.is_file():
            destination = coordinate_sensitivity_release / source.relative_to(
                coordinate_sensitivity_dir
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    checks["core_uses_raw_coordinates_only"] = not any(
        "experimental" in path.as_posix()
        for path in (release / "processed").rglob("*")
    )
    checks["trajectory_outputs_separated_from_core"] = (
        not (release / "processed" / "trajectory_reconstruction").exists()
        and reference_trajectory.exists()
    )
    checks["absolute_altitude_reconstruction_omitted"] = True
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Final package validation failed: {failed}")

    validation_summary = {
        "status": "passed",
        "checks": checks,
        "matrix_checks": matrix_checks,
       "raw_records": len(standard_rows),
       "spatial_records": len(reconstructed_rows),
        "core_spatial_product_records": len(core_spatial_rows),
       "raw_record_frequency_observations": len(standard_rows) * 201,
       "spatial_record_frequency_observations": len(reconstructed_rows) * 201,
        "core_spatial_product_record_frequency_observations": len(core_spatial_rows) * 201,
        "finite_raw_power_values": finite_power_values,
        "remaining_limitations": trajectory_metadata["limitations"],
    }
    validation_summary["frequency_completion"] = {
        "summary_rows": len(frequency_completion_summary),
        "split_count": len(frequency_completion_splits),
        "protocol_version": frequency_completion_protocol["protocol_version"],
        "eligible_rows": frequency_completion_protocol["eligible_rows"],
        "evaluation_metrics": frequency_completion_protocol["evaluation_metrics"],
        "protocols": sorted({row["protocol"] for row in frequency_completion_summary}),
        "methods": sorted({row["method"] for row in frequency_completion_summary}),
    }
    validation_summary["five_dimension_qc"] = five_dimension_qc
    validation_summary["grid_sensitivity"] = {
        "interpretation": grid_sensitivity["interpretation"],
        "configuration": grid_sensitivity["configuration"],
        "mean_spectrum_quality_policy_sensitivity": grid_sensitivity[
            "mean_spectrum_quality_policy_sensitivity"
        ],
        "cross_grid": grid_sensitivity["cross_grid"],
    }
    validation_summary["coordinate_representation_sensitivity"] = {
        "protocol": coordinate_sensitivity_protocol,
        "results_summary": coordinate_sensitivity_results,
    }
    write_json(validation_dir / "validation_summary.json", validation_summary)

    trajectory_summary = {}
    for validation_name in (
        "data_calibrated_simulation",
        "double_quantization_proxy",
        "coordinate_holdout_proxy",
    ):
        for method in ("rts", "quantization_aware"):
            selected = [
                row
                for row in trajectory_validation_rows
                if row["validation"] == validation_name and row["method"] == method
            ]
            if not selected:
                continue
            key = f"{validation_name}_{method}"
            trajectory_summary[key] = {
                "median_rmse_m": float(np.median([float(row["rmse_m"]) for row in selected])),
                "mean_cell_hit_fraction": None
                if validation_name != "coordinate_holdout_proxy"
                else float(
                    np.mean([float(row["proxy_cell_hit_fraction"]) for row in selected])
                ),
            }
    write_json(validation_dir / "trajectory_validation_summary.json", trajectory_summary)

    grid_configuration = {
        row["configuration"]: row for row in grid_sensitivity["configuration"]
    }
    strict_cross_grid = next(
        row for row in grid_sensitivity["cross_grid"] if row["policy"] == "strict_quality"
    )
    coordinate_primary = {
        row["coordinate_source"]: row
        for row in coordinate_sensitivity_results
        if float(row["horizontal_length_scale_m"])
        == float(coordinate_sensitivity_protocol["primary_horizontal_length_scale_m"])
    }

    processing_report = f"""# UAVSpectrum3D v0.4.0 数据处理报告

## 1. 处理范围

本地正式包读取 {source_summary['raw_files']} 个原始 Aaronia MCS CSV 文件，共 {len(standard_rows):,} 条记录和 201 个频率点。原始文件只读，原始 dBm 数值未替换、未校正、未因质量标志删除。

## 2. 字段标准化

时间戳按 ISO 格式解析；Speed 从 knots 以 0.5144444444444445 换算为 m/s；原始 WGS84 坐标转换为局部米制坐标。`elevation_gps_msl_m` 保存仪器导出的海拔；定高记录的 `z_relative_label_m` 来自文件名名义高度，动态记录的 `z_legacy_protocol_m` 仅保存旧版时间协议假设，两者均不表示校准后的绝对 UAV 高度。

## 3. 质量审计

文件、记录完整性、高度、功率聚合和惯性字段分别审计。五维 QC 覆盖功率、位置、时间、频谱和空间覆盖：共标记 {five_dimension_qc['power_quality']['floor_like_values']:,} 个 floor-like 值和 {five_dimension_qc['power_quality']['positive_power_candidate_values']:,} 个 positive-power candidate；时间戳缺失、重复、非正间隔和超过 10 s 的间隔均为 0；所有记录均含 201 个有限频率值。标志仅用于诊断，不作为自动删除规则。

水平位置审计确认 5,468 条有效空间记录仅包含 12 个纬度值、3 个经度值和 26 个水平坐标对，相邻坐标重复比例为 {five_dimension_qc['position_quality']['same_coordinate_fraction']:.2%}，并按预定义的位移-运动不一致规则标记 {five_dimension_qc['position_quality']['gps_jump_candidates']} 个 GPS jump candidate。该结果用于描述坐标量化和使用限制，不等同于已知定位误差。

GPS 海拔在不同定高批次中的“批次中位海拔减名义相对高度”标准差为 {quality_summary['altitude']['fixed_height_file_offset_std_m']:.3f} m，表明原始 Elevation 不能未经校准就替换相对高度标签。20 m 单元-频率聚合中，线性功率均值转回 dBm 与 dBm 算术均值之差的中位数为 {quality_summary['power_aggregation']['median_linear_minus_dbm_mean_db']:.3f} dB，95% 分位数为 {quality_summary['power_aggregation']['p95_linear_minus_dbm_mean_db']:.3f} dB，因此三种统计量必须使用不同字段名。

Gyro、Compass 和 Tilt 已完成数值范围、唯一值和连续重复比例审计。由于物理单位、坐标轴与安装旋转、偏置、磁校准及记录同步尚未完整确认，这些字段当前不进入正式融合。

## 4. 水平轨迹参考复用实验

对 {len(reconstructed_rows):,} 条具有有效水平坐标的记录生成 RTS 和量化感知有界轨迹估计，仅作为 `reference_reuse/trajectory_reconstruction/` 下的复用示例。原始导出坐标是正式可追溯主产品。数据匹配仿真中 RTS 的运行 RMSE 中位数为 {trajectory_summary['data_calibrated_simulation_rts']['median_rmse_m']:.3f} m，量化感知方法为 {trajectory_summary['data_calibrated_simulation_quantization_aware']['median_rmse_m']:.3f} m；真实坐标留出代理的平均量化单元命中率分别为 {trajectory_summary['coordinate_holdout_proxy_rts']['mean_cell_hit_fraction']:.2%} 和 {trajectory_summary['coordinate_holdout_proxy_quantization_aware']['mean_cell_hit_fraction']:.2%}。不同代理指标没有给出一致胜者，因此不指定唯一推荐方法，也不把任一结果称为恢复的真实 GPS。动态轨迹不进行正式逐记录高度重建，本版本不进行绝对高度重建。

## 5. 分析产品、敏感性与频率补全协议

核心 `processed/` 目录发布原始量化坐标的 10 m、20 m 聚合矩阵及其规则轴四维张量；这些核心空间-高度产品仅使用 {len(core_spatial_rows):,} 条具有有效水平坐标且相对高度来自文件名定高标签的记录。圆形和下降直线批次保留在原始文件与标准化记录中，作为补充飞行记录和参考复用输入，不进入核心空间-高度聚合。张量是矩阵的无插值展开，未观测位置保持 `NaN` 并由掩膜标识。两个网格尺寸表示后处理聚合尺度，不表示定位分辨率。频率补全协议基于20 m矩阵中满足严格质量掩膜的原始观测，构造随机频点和连续频段遮蔽；被遮蔽值仍是实测值，仅作为留出评价参考。RTS 和量化感知坐标对应的聚合结果与代理验证共同放入 `reference_reuse/`，用于方法敏感性和读取示例，不与核心测量产品混合。

网格敏感性检查得到 10 m 和 20 m 产品分别含 {grid_configuration['10m_all']['observed_space_height_cells']} 和 {grid_configuration['20m_all']['observed_space_height_cells']} 个实测空间-高度单元，其规则边界张量占用率分别为 {grid_configuration['10m_all']['space_height_occupancy_ratio']:.2%} 和 {grid_configuration['20m_all']['space_height_occupancy_ratio']:.2%}。采用预定义的向下取整索引映射后，严格质量共同条目的 RMSE 为 {strict_cross_grid['rmse_db']:.3f} dB，Pearson 相关系数为 {strict_cross_grid['pearson']:.4f}。两个四舍五入网格并非严格嵌套；这些数值检验聚合尺度与索引定义的敏感性，不证明实现等价，也不判定某个网格为真实空间分辨率。

坐标表示敏感性使用 10 个按原始 26 个水平坐标组划分的留出种子，并对 raw、RTS 和量化感知坐标使用相同测试记录、功率目标和质量掩膜。在主要 40 m 核长度下，三者 RMSE 分别为 {float(coordinate_primary['raw']['rmse_db_mean']):.3f}、{float(coordinate_primary['rts']['rmse_db_mean']):.3f} 和 {float(coordinate_primary['quantization_aware']['rmse_db_mean']):.3f} dB。该实验仅表明下游重建对坐标表示敏感，不能证明模型坐标更接近真实飞行位置。
"""
    (validation_dir / "detailed_data_processing_report.md").write_text(processing_report, encoding="utf-8")
    technical_report = f"""# UAVSpectrum3D v0.4.0 技术验证报告

## 结构与完整性

- 标准记录：{len(standard_rows):,} 条。
- 记录-频率组合：{len(standard_rows) * 201:,} 个。
- 有效水平坐标记录：{len(reconstructed_rows):,} 条。
- 空间记录-频率组合：{len(reconstructed_rows) * 201:,} 个。
- 核心空间-高度产品记录：{len(core_spatial_rows):,} 条，仅包含定高批次。
- 核心空间-高度产品记录-频率组合：{len(core_spatial_rows) * 201:,} 个。
- 频率轴：{int(frequencies.min()) / 1e6:.2f}-{int(frequencies.max()) / 1e6:.2f} MHz，共 201 点，频率步长 350 kHz，RBW 1 MHz。

上述计数由数组形状、记录索引和频率轴联合验证，而不仅是人工乘法说明。

## 产品一致性

所有矩阵均通过形状一致性、`observed_mask == sample_count > 0`、质量掩膜和 NaN 语义检查。每个基于原始导出坐标构建的聚合矩阵还与对应的 `tensor_product.npz` 逐单元比较，确保张量只是规则轴展开而非插值或重算。线性功率平均转换后的 dBm 值同时通过 Jensen 关系检查，其值不低于同组 dBm 算术均值。

原始坐标产品在 10 m 和 20 m 聚合尺度下分别包含 {matrix_checks['raw_coordinates/grid_10m']['shape'][0]} 和 {matrix_checks['raw_coordinates/grid_20m']['shape'][0]} 个实际观测空间-高度单元。两种产品的总贡献样本数均为 {matrix_checks['raw_coordinates/grid_20m']['contributing_samples']:,}，与 {len(core_spatial_rows):,} 条核心空间-高度产品记录乘以 201 个频率点一致。

## 五维质量与聚合敏感性

功率、位置、时间、频谱和空间覆盖五个维度均生成机器可读摘要。审计确认 5,700 条记录的功率数组形状为 `5700 x 201`，无缺失功率值；5,468 条空间记录的坐标量化为 12 个纬度、3 个经度和 26 个水平坐标对。10 m 与 20 m 规则张量的实测空间-高度占用率仅为 {grid_configuration['10m_all']['space_height_occupancy_ratio']:.2%} 和 {grid_configuration['20m_all']['space_height_occupancy_ratio']:.2%}，因此未测体素不能被当作已有真值。采用 `ix20=floor(ix10/2), iy20=floor(iy10/2)` 的预定义映射后，10 m 条目与原生 20 m 产品在 {strict_cross_grid['common_entries']:,} 个严格质量共同条目上的 RMSE 为 {strict_cross_grid['rmse_db']:.3f} dB。两个以四舍五入生成的原生网格并非严格嵌套；这是聚合尺度与索引定义敏感性检查，不是实现等价性或定位精度验证。

## 受控留出复用检查

频率补全只对原本实测且满足质量条件的频点执行随机或连续频段遮蔽，因此每个测试目标都有保留的实测功率值。相比之下，完整 `x-y-z-frequency` 张量的空间占用率很低，未采样体素没有同期真值，故本版本不把严格四维空间补全的预测误差用作数据有效性证据。四维张量和掩膜仍正式发布，可供使用者设计具有额外真值或更强假设的复用研究。

坐标表示敏感性固定了分组留出、测试记录和核回归设置，仅改变水平坐标表示。在 40 m 核长度下，raw、RTS 与量化感知表示的留出 RMSE 为 {float(coordinate_primary['raw']['rmse_db_mean']):.3f} +/- {float(coordinate_primary['raw']['rmse_db_std']):.3f}、{float(coordinate_primary['rts']['rmse_db_mean']):.3f} +/- {float(coordinate_primary['rts']['rmse_db_std']):.3f} 和 {float(coordinate_primary['quantization_aware']['rmse_db_mean']):.3f} +/- {float(coordinate_primary['quantization_aware']['rmse_db_std']):.3f} dB。模型坐标利用了轨迹序列，且不存在独立导航真值，因此这些差异仅支持“结果对坐标表示敏感”，不支持 GPS 精度恢复声明。

## 轨迹参考复用的验证边界

参考复用分支提供数据匹配退化条件下的仿真、二次量化代理和真实坐标留出代理。RTS 在仿真检查中误差更低，而量化感知方法在坐标留出代理中具有更高单元命中率；结果并未支持将任一方法视为已恢复的真实轨迹。由于不存在同期 RTK 或飞控轨迹真值，这些检查不能建立实际飞行的绝对定位精度。模型坐标不得替代原始 GPS，也不得表述为高精度三维真值。GPS 绝对高度没有进入重建，因此无需对不存在的绝对高度产品作精度声明。
"""
    (validation_dir / "technical_validation_report.md").write_text(technical_report, encoding="utf-8")

    shutil.copy2(validation_dir / "detailed_data_processing_report.md", release / "diagnostics")
    shutil.copy2(validation_dir / "technical_validation_report.md", release / "diagnostics")
    shutil.copy2(validation_dir / "validation_summary.json", release / "diagnostics")
    shutil.copy2(validation_dir / "trajectory_validation_summary.json", release / "diagnostics")

    reference_readme = f"""# Experimental horizontal trajectory reference reuse

This directory contains model-derived horizontal coordinate examples for
{len(reconstructed_rows):,} records with usable exported latitude and longitude.
The RTS and quantization-aware estimates are retained to support sensitivity
analysis and reproducible method comparisons. They are not core measurement
coordinates, corrected GPS truth, or high-resolution three-dimensional flight
trajectories.

No synchronous RTK or flight-controller trajectory is available. Proxy metrics
do not identify a single winning method: the data-matched simulation median RMSE
is {trajectory_summary['data_calibrated_simulation_rts']['median_rmse_m']:.3f} m
for RTS and {trajectory_summary['data_calibrated_simulation_quantization_aware']['median_rmse_m']:.3f} m
for the quantization-aware method, while coordinate-holdout cell-hit fractions
are {trajectory_summary['coordinate_holdout_proxy_rts']['mean_cell_hit_fraction']:.2%}
and {trajectory_summary['coordinate_holdout_proxy_quantization_aware']['mean_cell_hit_fraction']:.2%},
respectively. These are method checks under proxy conditions, not measured
real-flight positioning errors.

The reference branch reconstructs neither GPS mean-sea-level elevation nor
absolute UAV altitude. `z_relative_label_m` is available only for fixed-height
records; dynamic records retain `z_legacy_protocol_m` only as an auxiliary
historical label.

The RTS reference uses state `[x, y, vx, vy]`, a per-record constant-velocity
transition, acceleration process-noise scale 1.0, position standard deviations
`max(cell_width/sqrt(12), 0.25 m)`, initial velocity variance 25 m2/s2, and
velocity observations above 0.2 m/s with 1.0 m/s standard deviation. Heading is
clockwise from true north, so `vx=v*sin(heading)` and `vy=v*cos(heading)`.

The quantization-aware reference uses position half-width bounds of 0.5 cell,
expanded to 1.5 cells for jump candidates. A jump candidate has GPS displacement
at least 20 m and either no positive motion displacement or a GPS-to-motion ratio
at least 5; its position weight is divided by 5. The motion-displacement and
adjacent-segment velocity-difference scales are 5 m and 2 m/s, respectively.
`scipy.optimize.lsq_linear` uses method `trf`,
`lsmr_tol=auto`, `tol=1e-7`, and `max_iter=2000`. Machine-readable definitions
are stored in `trajectory_metadata.json`.
"""
    (reference_trajectory / "README.md").write_text(reference_readme, encoding="utf-8")

    write_release_documentation(
        release,
        {
            "raw_records": len(standard_rows),
            "spatial_records": len(reconstructed_rows),
            "core_spatial_product_records": len(core_spatial_rows),
            "frequency_bins": len(frequencies),
            "raw_observations": len(standard_rows) * len(frequencies),
            "spatial_observations": len(reconstructed_rows) * len(frequencies),
            "core_spatial_product_observations": len(core_spatial_rows) * len(frequencies),
            "grid_10m_cells": matrix_checks["raw_coordinates/grid_10m"]["shape"][0],
            "grid_20m_cells": matrix_checks["raw_coordinates/grid_20m"]["shape"][0],
            "tensor_products": [
                "processed/tensor/raw_coordinates/grid_10m/tensor_product.npz",
                "processed/tensor/raw_coordinates/grid_20m/tensor_product.npz",
            ],
            "frequency_completion": "reference_reuse/frequency_completion/",
        },
    )
    shutil.copy2(
        Path(__file__).resolve().parents[1] / "requirements.txt",
        release / "requirements.txt",
    )

    batch_sensitivity_script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "batch_power_sensitivity.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(batch_sensitivity_script),
            "--release",
            str(release),
            "--output",
            str(release / "diagnostics" / "batch_sensitivity"),
        ],
        check=True,
    )

    checksum_rows = []
    checksum_path = release / "metadata" / "checksum_manifest.csv"
    for path in sorted(release.rglob("*")):
        if path.is_file() and path != checksum_path:
            checksum_rows.append(
                {
                    "relative_path": path.relative_to(release).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_csv(checksum_path, checksum_rows)
    print(f"Stage 09 complete: {release}")


if __name__ == "__main__":
    main()
