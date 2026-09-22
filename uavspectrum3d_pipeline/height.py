"""Relative-height labelling and GPS-elevation audit helpers."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def height_policy(source_file: str) -> tuple[str, float | None]:
    lower = source_file.lower()
    if "linefrom9mto1m" in lower:
        return "linear_time_9m_to_1m", None
    if "circle1" in lower or "circle2" in lower:
        return "four_equal_time_segments_9_7_5_3m", None
    match = re.search(r"_(\d+(?:\.\d+)?)m\.csv$", lower)
    if match:
        value = float(match.group(1))
        return f"constant_{value:g}m_from_filename", value
    return "unknown", None


def is_core_spatial_height_record(source_file: str, position_valid: bool) -> bool:
    """Return whether a record has a traceable height label for core products."""
    if not position_valid:
        return False
    _, fixed = height_policy(source_file)
    return fixed is not None


def assign_formal_relative_height_labels(records) -> dict[int, float | None]:
    """Return only filename-derived labels used by core products."""
    labels: dict[int, float | None] = {}
    for record in records:
        _, fixed = height_policy(record.source_file)
        labels[record.record_index] = fixed
    return labels


def assign_legacy_protocol_height_labels(records) -> dict[int, float | None]:
    """Return historical time-derived labels for dynamic-flight traceability."""
    groups: dict[str, list] = {}
    for record in records:
        groups.setdefault(record.source_file, []).append(record)
    labels: dict[int, float | None] = {}
    for source_file, group in groups.items():
        policy, fixed = height_policy(source_file)
        valid_times = [record.timestamp for record in group if record.timestamp is not None]
        start = min(valid_times) if valid_times else None
        end = max(valid_times) if valid_times else None
        for record in group:
            if fixed is not None:
                labels[record.record_index] = None
                continue
            if record.timestamp is None or start is None or end is None or end <= start:
                labels[record.record_index] = None
                continue
            ratio = (record.timestamp - start).total_seconds() / (end - start).total_seconds()
            ratio = min(1.0, max(0.0, ratio))
            if policy == "linear_time_9m_to_1m":
                labels[record.record_index] = 9.0 - 8.0 * ratio
            elif policy == "four_equal_time_segments_9_7_5_3m":
                labels[record.record_index] = (9.0, 7.0, 5.0, 3.0)[min(int(ratio * 4), 3)]
            else:
                labels[record.record_index] = None
    return labels


def assign_relative_height_labels(records) -> dict[int, float | None]:
    """Backward-compatible alias for the historical protocol labels."""
    return assign_legacy_protocol_height_labels(records)
