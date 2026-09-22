"""Power-quality flags and explicitly named aggregation statistics."""

from __future__ import annotations

import numpy as np

from .config import FLOOR_POWER_DBM, POSITIVE_POWER_DBM


POWER_FLAG_FLOOR_LIKE = np.uint8(1)
POWER_FLAG_POSITIVE_CANDIDATE = np.uint8(2)


def power_quality_flags(power_dbm: np.ndarray) -> np.ndarray:
    flags = np.zeros(power_dbm.shape, dtype=np.uint8)
    finite = np.isfinite(power_dbm)
    flags[finite & (power_dbm <= FLOOR_POWER_DBM)] |= POWER_FLAG_FLOOR_LIKE
    flags[finite & (power_dbm > POSITIVE_POWER_DBM)] |= POWER_FLAG_POSITIVE_CANDIDATE
    return flags


def aggregate_power_dbm(values_dbm: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values_dbm, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "power_dbm_arithmetic_mean": np.nan,
            "power_dbm_from_linear_mean": np.nan,
            "power_dbm_median": np.nan,
            "sample_count": 0,
        }
    linear_mw = np.power(10.0, values / 10.0)
    return {
        "power_dbm_arithmetic_mean": float(np.mean(values)),
        "power_dbm_from_linear_mean": float(10.0 * np.log10(np.mean(linear_mw))),
        "power_dbm_median": float(np.median(values)),
        "sample_count": int(len(values)),
    }
