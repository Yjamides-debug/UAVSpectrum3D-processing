"""Focused contract tests for reusable v0.4.0 processing components."""

from __future__ import annotations

import math
import importlib.util
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from uavspectrum3d_pipeline.coordinates import latlon_to_local_m, local_m_to_latlon
from uavspectrum3d_pipeline.height import (
    assign_formal_relative_height_labels,
    assign_legacy_protocol_height_labels,
    assign_relative_height_labels,
    is_core_spatial_height_record,
)
from uavspectrum3d_pipeline.power import aggregate_power_dbm, power_quality_flags
from uavspectrum3d_pipeline.products import build_matrix_product


def load_stage_module(filename: str):
    path = Path(__file__).resolve().parents[1] / "stages" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load stage module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QUALITY_AUDIT_STAGE = load_stage_module("03_audit_height_power_inertial.py")
GRID_SENSITIVITY_STAGE = load_stage_module("07_grid_sensitivity.py")
COORDINATE_SENSITIVITY_STAGE = load_stage_module("08_coordinate_sensitivity.py")
PACKAGING_STAGE = load_stage_module("09_validate_and_package_v040.py")


class CoordinateTests(unittest.TestCase):
    def test_local_coordinate_round_trip(self) -> None:
        reference_latitude = 30.320712490855886
        reference_longitude = 104.32395336503292
        latitude = 30.3209
        longitude = 104.324
        x_m, y_m = latlon_to_local_m(
            latitude, longitude, reference_latitude, reference_longitude
        )
        recovered_latitude, recovered_longitude = local_m_to_latlon(
            x_m, y_m, reference_latitude, reference_longitude
        )
        self.assertAlmostEqual(latitude, recovered_latitude, places=12)
        self.assertAlmostEqual(longitude, recovered_longitude, places=12)


class PowerTests(unittest.TestCase):
    def test_log_and_linear_means_have_distinct_semantics(self) -> None:
        stats = aggregate_power_dbm(np.array([-50.0, -70.0]))
        self.assertAlmostEqual(stats["power_dbm_arithmetic_mean"], -60.0, places=10)
        self.assertAlmostEqual(stats["power_dbm_from_linear_mean"], -52.967086, places=5)
        self.assertAlmostEqual(stats["power_dbm_median"], -60.0, places=10)
        self.assertEqual(stats["sample_count"], 2)

    def test_linear_mean_is_not_below_log_domain_mean(self) -> None:
        values = np.array([-150.0, -80.0, -60.0, -45.0])
        stats = aggregate_power_dbm(values)
        self.assertGreaterEqual(
            stats["power_dbm_from_linear_mean"], stats["power_dbm_arithmetic_mean"]
        )

    def test_quality_flags_do_not_modify_values(self) -> None:
        values = np.array([-150.0, -70.0, 1.0, np.nan])
        original = values.copy()
        flags = power_quality_flags(values)
        np.testing.assert_equal(values, original)
        np.testing.assert_array_equal(flags, np.array([1, 0, 2, 0], dtype=np.uint8))


class HeightTests(unittest.TestCase):
    def test_formal_and_legacy_height_fields_are_separate(self) -> None:
        start = datetime(2025, 11, 19, 12, 0, 0)
        records = [
            SimpleNamespace(record_index=0, source_file="recording_9m.csv", timestamp=start),
            SimpleNamespace(record_index=1, source_file="recording_circle1.csv", timestamp=start),
            SimpleNamespace(record_index=2, source_file="recording_circle1.csv", timestamp=start + timedelta(seconds=10)),
        ]
        formal = assign_formal_relative_height_labels(records)
        legacy = assign_legacy_protocol_height_labels(records)
        self.assertEqual(formal, {0: 9.0, 1: None, 2: None})
        self.assertEqual(legacy[0], None)
        self.assertEqual(legacy[1], 9.0)
        self.assertEqual(legacy[2], 3.0)

    def test_circle_height_segments_use_elapsed_time_not_row_count(self) -> None:
        start = datetime(2025, 11, 19, 12, 0, 0)
        offsets = (0, 1, 9, 10)
        records = [
            SimpleNamespace(
                record_index=index,
                source_file="recording_circle1.csv",
                timestamp=start + timedelta(seconds=offset),
            )
            for index, offset in enumerate(offsets)
        ]
        labels = assign_relative_height_labels(records)
        self.assertEqual([labels[index] for index in range(4)], [9.0, 9.0, 3.0, 3.0])

    def test_core_spatial_height_records_require_fixed_height_filename(self) -> None:
        self.assertTrue(is_core_spatial_height_record("recording_2025-11-19_1057_9m.csv", True))
        self.assertFalse(is_core_spatial_height_record("recording_2025-11-19_1215_circle1.csv", True))
        self.assertFalse(is_core_spatial_height_record("recording_2025-11-19_1256_linefrom9mto1m.csv", True))
        self.assertFalse(is_core_spatial_height_record("recording_2025-11-19_1057_9m.csv", False))


class ProductTests(unittest.TestCase):
    def test_matrix_counts_masks_and_aggregate_flags_agree(self) -> None:
        rows = [
            {"record_index": "0", "x_raw_m": "0", "y_raw_m": "0", "z_relative_label_m": "1"},
            {"record_index": "1", "x_raw_m": "1", "y_raw_m": "1", "z_relative_label_m": "1"},
        ]
        powers = np.array([[-150.0, -70.0], [-80.0, 1.0]])
        product = build_matrix_product(
            rows,
            powers,
            np.array([860_000_000, 860_350_000]),
            10.0,
            "x_raw_m",
            "y_raw_m",
        )
        np.testing.assert_array_equal(product["sample_count"], [[2, 2]])
        np.testing.assert_array_equal(product["observed_mask"], [[True, True]])
        np.testing.assert_array_equal(product["quality_flags"], [[1, 2]])
        np.testing.assert_array_equal(product["quality_mask"], [[False, False]])

    def test_core_matrix_can_exclude_supplementary_trajectory_rows(self) -> None:
        rows = [
            {"record_index": "0", "x_raw_m": "0", "y_raw_m": "0", "z_relative_label_m": "9", "core_spatial_product_included": "True"},
            {"record_index": "1", "x_raw_m": "0", "y_raw_m": "0", "z_relative_label_m": "7", "core_spatial_product_included": "False"},
        ]
        core_rows = [
            row for row in rows if row["core_spatial_product_included"].lower() == "true"
        ]
        powers = np.array([[-70.0], [-40.0]])
        product = build_matrix_product(
            core_rows,
            powers,
            np.array([860_000_000]),
            10.0,
            "x_raw_m",
            "y_raw_m",
        )
        self.assertEqual(product["cells"], [(0, 0, 9)])
        np.testing.assert_array_equal(product["sample_count"], [[1]])
        np.testing.assert_allclose(product["power_dbm_arithmetic_mean"], [[-70.0]])


class QualityAuditTests(unittest.TestCase):
    def test_gps_jump_candidate_requires_large_unexplained_step(self) -> None:
        predicate = QUALITY_AUDIT_STAGE.is_gps_jump_candidate
        self.assertFalse(predicate(19.999, None))
        self.assertTrue(predicate(20.0, None))
        self.assertTrue(predicate(25.0, 5.0))
        self.assertFalse(predicate(25.0, 5.01))


class GridSensitivityTests(unittest.TestCase):
    def test_nested_grid_mapping_uses_floor_division_for_negative_indices(self) -> None:
        mapper = GRID_SENSITIVITY_STAGE.map_10m_index_to_20m
        self.assertEqual([mapper(value) for value in (-3, -2, -1, 0, 1, 2, 3)], [-2, -1, -1, 0, 0, 1, 1])


class CoordinateSensitivityTests(unittest.TestCase):
    def test_grouped_holdout_is_disjoint_and_representation_independent(self) -> None:
        indices = np.arange(8, dtype=int)
        group_by_index = {
            index: (float(index // 2), 0.0) for index in indices
        }
        first = COORDINATE_SENSITIVITY_STAGE.grouped_holdout_split(
            indices, group_by_index, seed=3, train_fraction=0.5
        )
        second = COORDINATE_SENSITIVITY_STAGE.grouped_holdout_split(
            indices, group_by_index, seed=3, train_fraction=0.5
        )
        train_records, test_records, train_groups = first
        self.assertFalse(set(train_records) & set(test_records))
        self.assertEqual(set(train_records) | set(test_records), set(indices))
        self.assertTrue(
            all(group_by_index[int(index)] in train_groups for index in train_records)
        )
        self.assertTrue(
            all(group_by_index[int(index)] not in train_groups for index in test_records)
        )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


class PackagingTests(unittest.TestCase):
    def test_public_raw_filename_conversion_is_idempotent(self) -> None:
        convert = PACKAGING_STAGE.public_raw_name
        public_name = "recording_2025-11-19_1057_9m.csv"
        self.assertEqual(convert(public_name), public_name)
        self.assertEqual(
            convert("扫描 Recording_ 2025-11-19_1057_9m.csv"), public_name
        )


if __name__ == "__main__":
    unittest.main()
