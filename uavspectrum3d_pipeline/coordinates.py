"""Small-area WGS84/local-coordinate transforms and quantization utilities."""

from __future__ import annotations

import math
import statistics

from .config import LAT_M_PER_DEG, LON_M_PER_DEG


def coordinate_reference(records) -> tuple[float, float]:
    valid = [record for record in records if record.position_valid]
    if not valid:
        raise ValueError("No valid coordinates available.")
    return (
        statistics.mean(record.latitude for record in valid),
        statistics.mean(record.longitude for record in valid),
    )


def latlon_to_local_m(
    latitude: float, longitude: float, reference_latitude: float, reference_longitude: float
) -> tuple[float, float]:
    x_m = (
        (longitude - reference_longitude)
        * math.cos(math.radians(reference_latitude))
        * LON_M_PER_DEG
    )
    y_m = (latitude - reference_latitude) * LAT_M_PER_DEG
    return x_m, y_m


def local_m_to_latlon(
    x_m: float, y_m: float, reference_latitude: float, reference_longitude: float
) -> tuple[float, float]:
    latitude = reference_latitude + y_m / LAT_M_PER_DEG
    longitude = reference_longitude + x_m / (
        math.cos(math.radians(reference_latitude)) * LON_M_PER_DEG
    )
    return latitude, longitude


def infer_coordinate_step(values: list[float]) -> float:
    unique = sorted({round(value, 8) for value in values})
    differences = [right - left for left, right in zip(unique, unique[1:]) if right > left]
    return float(min(differences)) if differences else 0.0
