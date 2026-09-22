"""Build coordinate-frequency matrices with explicit aggregation semantics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .config import FLOOR_POWER_DBM, POSITIVE_POWER_DBM
from .power import aggregate_power_dbm


def build_matrix_product(
    record_rows: list[dict],
    powers_dbm: np.ndarray,
    frequencies_hz: np.ndarray,
    grid_m: float,
    x_field: str,
    y_field: str,
) -> dict:
    groups: dict[tuple[int, int, int, int], list[float]] = defaultdict(list)
    cells: set[tuple[int, int, int]] = set()
    for row in record_rows:
        try:
            record_index = int(row["record_index"])
            x_m = float(row[x_field])
            y_m = float(row[y_field])
            z_m = float(row["z_relative_label_m"])
        except (KeyError, TypeError, ValueError):
            continue
        ix = int(round(x_m / grid_m))
        iy = int(round(y_m / grid_m))
        iz = int(round(z_m))
        cells.add((ix, iy, iz))
        for frequency_index, value in enumerate(powers_dbm[record_index]):
            if np.isfinite(value):
                groups[(ix, iy, iz, frequency_index)].append(float(value))

    sorted_cells = sorted(cells, key=lambda item: (item[2], item[0], item[1]))
    cell_to_index = {cell: index for index, cell in enumerate(sorted_cells)}
    shape = (len(sorted_cells), len(frequencies_hz))
    dbm_mean = np.full(shape, np.nan, dtype=np.float32)
    linear_mean_dbm = np.full(shape, np.nan, dtype=np.float32)
    dbm_median = np.full(shape, np.nan, dtype=np.float32)
    sample_count = np.zeros(shape, dtype=np.int32)
    floor_count = np.zeros(shape, dtype=np.int32)
    positive_count = np.zeros(shape, dtype=np.int32)

    for (ix, iy, iz, frequency_index), values in groups.items():
        row_index = cell_to_index[(ix, iy, iz)]
        array = np.asarray(values, dtype=float)
        stats = aggregate_power_dbm(array)
        dbm_mean[row_index, frequency_index] = stats["power_dbm_arithmetic_mean"]
        linear_mean_dbm[row_index, frequency_index] = stats["power_dbm_from_linear_mean"]
        dbm_median[row_index, frequency_index] = stats["power_dbm_median"]
        sample_count[row_index, frequency_index] = stats["sample_count"]
        floor_count[row_index, frequency_index] = int(np.sum(array <= FLOOR_POWER_DBM))
        positive_count[row_index, frequency_index] = int(np.sum(array > POSITIVE_POWER_DBM))

    observed_mask = sample_count > 0
    quality_flags = np.zeros(shape, dtype=np.uint8)
    quality_flags[floor_count > 0] |= np.uint8(1)
    quality_flags[positive_count > 0] |= np.uint8(2)
    quality_mask = observed_mask & (quality_flags == 0)
    return {
        "cells": sorted_cells,
        "frequencies_hz": frequencies_hz,
        "power_dbm_arithmetic_mean": dbm_mean,
        "power_dbm_from_linear_mean": linear_mean_dbm,
        "power_dbm_median": dbm_median,
        "sample_count": sample_count,
        "floor_count": floor_count,
        "positive_count": positive_count,
        "observed_mask": observed_mask,
        "quality_flags": quality_flags,
        "quality_mask": quality_mask,
    }
