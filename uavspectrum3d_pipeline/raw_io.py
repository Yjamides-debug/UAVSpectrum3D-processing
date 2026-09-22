"""Read Aaronia MCS CSV files without modifying their reported values."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import EXPECTED_FREQUENCY_BINS, KNOT_TO_MPS


SENSOR_FIELDS = (
    "Tilt-X",
    "Tilt-Y",
    "Tilt-Z",
    "Compass-X",
    "Compass-Y",
    "Compass-Z",
    "Gyro-X",
    "Gyro-Y",
    "Gyro-Z",
)


def finite_number(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() in {"nan", "na", "n/a", "null"}:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None


@dataclass
class RawRecord:
    record_index: int
    source_file: str
    source_row: int
    timestamp_raw: str
    timestamp: datetime | None
    latitude: float | None
    longitude: float | None
    elevation_gps_msl_m: float | None
    speed_knots_raw: float | None
    heading_true_north_deg: float | None
    sensors: dict[str, float | None]
    measure_unit: str
    powers_dbm: np.ndarray

    @property
    def speed_mps(self) -> float | None:
        if self.speed_knots_raw is None:
            return None
        return self.speed_knots_raw * KNOT_TO_MPS

    @property
    def position_valid(self) -> bool:
        return self.latitude is not None and self.longitude is not None


@dataclass
class RawFileInfo:
    name: str
    input_rows: int
    frequency_bins: int
    gps_columns_present: bool
    valid_position_rows: int
    parse_failures: int
    short_rows: int


@dataclass
class RawDataset:
    records: list[RawRecord]
    frequencies_hz: np.ndarray
    files: list[RawFileInfo]

    @property
    def power_matrix_dbm(self) -> np.ndarray:
        if not self.records:
            return np.empty((0, len(self.frequencies_hz)), dtype=np.float32)
        return np.vstack([record.powers_dbm for record in self.records]).astype(np.float32)


def _frequency_columns(header: list[str]) -> list[tuple[int, int]]:
    columns = []
    for index, name in enumerate(header):
        try:
            frequency = int(float(name))
        except (TypeError, ValueError):
            continue
        if 100_000_000 <= frequency <= 10_000_000_000:
            columns.append((index, frequency))
    return columns


def load_raw_dataset(input_dir: Path) -> RawDataset:
    """Load all CSV rows and align every spectrum to one verified frequency axis."""
    records: list[RawRecord] = []
    file_infos: list[RawFileInfo] = []
    common_axis: np.ndarray | None = None

    for path in sorted(input_dir.glob("*.csv")):
        input_rows = 0
        parse_failures = 0
        short_rows = 0
        valid_positions = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            first = handle.readline()
            header_line = handle.readline()
            if not header_line:
                file_infos.append(RawFileInfo(path.name, 0, 0, False, 0, 1, 0))
                continue
            header = next(csv.reader([header_line.rstrip("\r\n")], delimiter=";"), [])
            indices = {name: index for index, name in enumerate(header)}
            frequency_columns = _frequency_columns(header)
            axis = np.array([frequency for _, frequency in frequency_columns], dtype=np.int64)
            if common_axis is None:
                common_axis = axis
            elif not np.array_equal(common_axis, axis):
                raise ValueError(f"Frequency axis mismatch in {path.name}.")

            def field(row: list[str], name: str) -> str | None:
                index = indices.get(name)
                return row[index] if index is not None and index < len(row) else None

            reader = csv.reader(handle, delimiter=";")
            for source_row, row in enumerate(reader, start=1):
                if not row:
                    continue
                input_rows += 1
                if len(row) < len(header):
                    short_rows += 1
                try:
                    powers = np.array(
                        [finite_number(row[index] if index < len(row) else None) for index, _ in frequency_columns],
                        dtype=float,
                    )
                    latitude = finite_number(field(row, "Latitude"))
                    longitude = finite_number(field(row, "Longitude"))
                    if latitude is not None and longitude is not None:
                        valid_positions += 1
                    sensors = {name: finite_number(field(row, name)) for name in SENSOR_FIELDS}
                    record = RawRecord(
                        record_index=len(records),
                        source_file=path.name,
                        source_row=source_row,
                        timestamp_raw=field(row, "Timestamp") or "",
                        timestamp=parse_timestamp(field(row, "Timestamp")),
                        latitude=latitude,
                        longitude=longitude,
                        elevation_gps_msl_m=finite_number(field(row, "Elevation")),
                        speed_knots_raw=finite_number(field(row, "Speed")),
                        heading_true_north_deg=finite_number(field(row, "Heading")),
                        sensors=sensors,
                        measure_unit=(field(row, "Measure Unit") or "").strip(),
                        powers_dbm=powers,
                    )
                except (IndexError, ValueError, TypeError):
                    parse_failures += 1
                    continue
                records.append(record)

        file_infos.append(
            RawFileInfo(
                name=path.name,
                input_rows=input_rows,
                frequency_bins=len(frequency_columns),
                gps_columns_present="Latitude" in indices and "Longitude" in indices,
                valid_position_rows=valid_positions,
                parse_failures=parse_failures,
                short_rows=short_rows,
            )
        )

    if common_axis is None:
        raise RuntimeError(f"No frequency axis found under {input_dir}.")
    if len(common_axis) != EXPECTED_FREQUENCY_BINS:
        raise ValueError(
            f"Expected {EXPECTED_FREQUENCY_BINS} frequency bins, found {len(common_axis)}."
        )
    return RawDataset(records=records, frequencies_hz=common_axis, files=file_infos)
