#!/usr/bin/env python3
"""Audit flight-stage candidates and motion consistency of derived coordinates.

This diagnostic is intentionally outside the nine-stage release pipeline.  It
does not remove records or rewrite the core dataset.  It compares exported
coordinates with the two model-derived horizontal coordinate sequences and
reports whether their step lengths are compatible with the recorded speed and
heading fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


METHODS = {
    "raw": ("x_raw_m", "y_raw_m"),
    "rts": ("x_rts_m", "y_rts_m"),
    "quantization_aware": ("x_quantization_aware_m", "y_quantization_aware_m"),
}
SPEED_THRESHOLDS = (0.3, 0.5, 0.7)


def short_batch_label(source_file: str) -> str:
    """Use compact ASCII labels in figures; retain full names in source tables."""
    stem = source_file.rsplit("/", 1)[-1].replace(".csv", "")
    marker = "recording_2025-11-19_"
    if marker in stem:
        stem = stem[stem.index(marker) :]
    elif "2025-11-19_" in stem:
        stem = stem[stem.index("2025-11-19_") + len("2025-11-19_") :]
    if stem.startswith(marker):
        stem = stem[len(marker) :]
    stem = stem.replace("linefrom9mto1m", "line_9to1m")
    return stem


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty audit table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value: str | None) -> float | None:
    try:
        result = float(value) if value not in (None, "") else math.nan
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def longest_boundary_run(speeds: np.ndarray, times: np.ndarray, threshold: float) -> dict:
    """Return start/end low-speed runs, measured in records and seconds.

    A low-speed record in the middle of a file is not classified as a ground
    candidate.  Only contiguous runs touching a file boundary are reported.
    """
    low = np.isfinite(speeds) & (speeds < threshold)

    def run_from_start() -> tuple[int, float]:
        n = 0
        while n < len(low) and low[n]:
            n += 1
        duration = float(times[n - 1] - times[0]) if n > 1 else 0.0
        return n, duration

    def run_from_end() -> tuple[int, float]:
        n = 0
        while n < len(low) and low[len(low) - 1 - n]:
            n += 1
        duration = float(times[-1] - times[len(times) - n]) if n > 1 else 0.0
        return n, duration

    start_n, start_s = run_from_start()
    end_n, end_s = run_from_end()
    return {
        "start_low_speed_records": start_n,
        "start_low_speed_duration_s": round(start_s, 3),
        "end_low_speed_records": end_n,
        "end_low_speed_duration_s": round(end_s, 3),
        "interior_low_speed_records": int(low.sum() - start_n - end_n),
    }


def circle_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Algebraic circle fit; returns centre x/y, radius and radial RMSE."""
    if len(x) < 3:
        return math.nan, math.nan, math.nan, math.nan
    matrix = np.column_stack((2.0 * x, 2.0 * y, np.ones(len(x))))
    target = x * x + y * y
    try:
        cx, cy, constant = np.linalg.lstsq(matrix, target, rcond=None)[0]
        radius_sq = constant + cx * cx + cy * cy
        if radius_sq <= 0:
            return math.nan, math.nan, math.nan, math.nan
        radius = math.sqrt(radius_sq)
        residual = np.hypot(x - cx, y - cy) - radius
        return float(cx), float(cy), radius, float(math.sqrt(np.mean(residual**2)))
    except np.linalg.LinAlgError:
        return math.nan, math.nan, math.nan, math.nan


def line_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit a principal axis and return angle, orthogonal RMSE and span."""
    if len(x) < 2:
        return math.nan, math.nan, math.nan
    points = np.column_stack((x, y))
    centred = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    axis = vh[0]
    orthogonal = centred @ vh[1]
    projection = centred @ axis
    angle = (math.degrees(math.atan2(axis[0], axis[1])) + 360.0) % 360.0
    return float(angle), float(math.sqrt(np.mean(orthogonal**2))), float(projection.max() - projection.min())


def build_audits(records: list[dict[str, str]], reconstructed: list[dict[str, str]]) -> tuple[list[dict], list[dict], list[dict]]:
    rec_by_id = {row["record_id"]: row for row in records}
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in reconstructed:
        if row["record_id"] in rec_by_id:
            groups[row["source_file"]].append(row)

    stage_rows: list[dict] = []
    step_rows: list[dict] = []
    geometry_rows: list[dict] = []

    for source_file, rows in sorted(groups.items()):
        rows.sort(key=lambda row: int(row["source_row"]))
        speeds = np.array([finite(rec_by_id[row["record_id"]].get("speed_mps")) or math.nan for row in rows])
        times = np.array([(parse_time(row["timestamp"]) - parse_time(rows[0]["timestamp"])).total_seconds() for row in rows])
        stage = {"source_file": source_file, "records": len(rows), "duration_s": round(float(times[-1]), 3)}
        for threshold in SPEED_THRESHOLDS:
            low = int(np.isfinite(speeds).sum() and np.sum(speeds < threshold))
            boundary = longest_boundary_run(speeds, times, threshold)
            prefix = f"speed_lt_{str(threshold).replace('.', 'p')}"
            stage[f"{prefix}_records"] = low
            stage[f"{prefix}_fraction"] = round(low / len(rows), 6)
            for key, value in boundary.items():
                stage[f"{prefix}_{key}"] = value
        stage_rows.append(stage)

        for method, (x_name, y_name) in METHODS.items():
            x = np.array([float(row[x_name]) for row in rows])
            y = np.array([float(row[y_name]) for row in rows])
            distances = np.hypot(np.diff(x), np.diff(y))
            dts = np.diff(times)
            valid = dts > 0
            implied_speed = np.full(len(distances), math.nan)
            implied_speed[valid] = distances[valid] / dts[valid]
            reported = speeds[1:]
            valid_motion = valid & np.isfinite(reported)
            speed_error = implied_speed[valid_motion] - reported[valid_motion]
            ratio_error = implied_speed[valid_motion] / np.maximum(reported[valid_motion], 0.2)
            heading = np.array([finite(rec_by_id[row["record_id"]].get("heading_true_north_deg")) or math.nan for row in rows])[1:]
            motion_heading = (np.degrees(np.arctan2(np.diff(x), np.diff(y))) + 360.0) % 360.0
            valid_heading = valid_motion & np.isfinite(heading)
            heading_error = np.abs((motion_heading[valid_heading] - heading[valid_heading] + 180.0) % 360.0 - 180.0)
            row = {
                "source_file": source_file,
                "method": method,
                "steps": int(valid.sum()),
                "median_step_m": round(float(np.median(distances[valid])), 6),
                "p95_step_m": round(float(np.percentile(distances[valid], 95)), 6),
                "max_step_m": round(float(np.max(distances[valid])), 6),
                "steps_gt_20m": int(np.sum(distances[valid] > 20.0)),
                "p95_implied_speed_mps": round(float(np.nanpercentile(implied_speed, 95)), 6),
                "max_implied_speed_mps": round(float(np.nanmax(implied_speed)), 6),
                "speed_comparisons": int(valid_motion.sum()),
                "speed_mae_mps": round(float(np.mean(np.abs(speed_error))), 6) if len(speed_error) else math.nan,
                "implied_speed_gt_reported_plus_2mps": int(np.sum(speed_error > 2.0)),
                "implied_to_reported_speed_ratio_gt_2": int(np.sum(ratio_error > 2.0)),
                "heading_comparisons": int(valid_heading.sum()),
                "heading_median_abs_error_deg": round(float(np.median(heading_error)), 6) if len(heading_error) else math.nan,
                "heading_p95_abs_error_deg": round(float(np.percentile(heading_error, 95)), 6) if len(heading_error) else math.nan,
            }
            step_rows.append(row)

            if "circle" in source_file or "linefrom" in source_file:
                if "circle" in source_file:
                    cx, cy, radius, rmse = circle_fit(x, y)
                    geometry_rows.append({"source_file": source_file, "method": method, "geometry": "circle", "fit_centre_x_m": round(cx, 6), "fit_centre_y_m": round(cy, 6), "fit_radius_m": round(radius, 6), "orthogonal_or_radial_rmse_m": round(rmse, 6)})
                else:
                    angle, rmse, span = line_fit(x, y)
                    geometry_rows.append({"source_file": source_file, "method": method, "geometry": "line", "fit_angle_true_north_deg": round(angle, 6), "orthogonal_or_radial_rmse_m": round(rmse, 6), "fit_span_m": round(span, 6)})

    return stage_rows, step_rows, geometry_rows


def make_figures(output: Path, stage_rows: list[dict], step_rows: list[dict], reconstructed: list[dict]) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.right": False, "axes.spines.top": False})
    blue, orange, grey, red = "#4C78A8", "#E58C50", "#5B6770", "#B44C4C"

    labels = [short_batch_label(row["source_file"]) for row in stage_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.0), constrained_layout=True)
    axes = axes.ravel()
    for threshold, color, offset in zip(SPEED_THRESHOLDS, (grey, blue, orange), (-0.22, 0.0, 0.22)):
        key = f"speed_lt_{str(threshold).replace('.', 'p')}_fraction"
        axes[0].bar(x + offset, [100 * float(row[key]) for row in stage_rows], width=0.2, color=color, label=f"v < {threshold:.1f} m/s")
    axes[0].set_xticks(x, labels, rotation=55, ha="right")
    axes[0].set_ylabel("Records below threshold (%)")
    axes[0].set_title("a  Low-speed records by batch", loc="left", fontweight="bold")
    axes[0].legend(fontsize=6)

    for threshold, color in zip(SPEED_THRESHOLDS, (grey, blue, orange)):
        key = f"speed_lt_{str(threshold).replace('.', 'p')}_start_low_speed_records"
        axes[1].plot(x, [row[key] for row in stage_rows], marker="o", color=color, label=f"v < {threshold:.1f} m/s")
    axes[1].set_xticks(x, labels, rotation=55, ha="right")
    axes[1].set_ylabel("Contiguous records at file start")
    axes[1].set_title("b  Boundary low-speed candidates", loc="left", fontweight="bold")
    axes[1].legend(fontsize=6)

    for method, color in (("raw", grey), ("rts", blue), ("quantization_aware", orange)):
        selected = [row for row in step_rows if row["method"] == method]
        selected.sort(key=lambda row: stage_rows.index(next(item for item in stage_rows if item["source_file"] == row["source_file"])))
        axes[2].plot(x, [row["max_step_m"] for row in selected], marker="o", color=color, label=method.replace("_", " "))
    axes[2].axhline(20.0, color=red, linestyle="--", linewidth=0.8, label="20 m check")
    axes[2].set_xticks(x, labels, rotation=55, ha="right")
    axes[2].set_ylabel("Maximum adjacent step (m)")
    axes[2].set_title("c  Within-batch trajectory continuity", loc="left", fontweight="bold")
    axes[2].legend(fontsize=6)

    for method, color in (("rts", blue), ("quantization_aware", orange)):
        selected = [row for row in step_rows if row["method"] == method]
        selected.sort(key=lambda row: stage_rows.index(next(item for item in stage_rows if item["source_file"] == row["source_file"])))
        axes[3].plot(x, [row["heading_median_abs_error_deg"] for row in selected], marker="o", color=color, label=method.replace("_", " "))
    axes[3].set_xticks(x, labels, rotation=55, ha="right")
    axes[3].set_ylabel("Median heading residual (degrees)")
    axes[3].set_title("d  Motion-direction consistency", loc="left", fontweight="bold")
    axes[3].legend(fontsize=6)
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.7)
    fig.suptitle("Flight-stage and trajectory-motion audit", x=0.01, ha="left", fontsize=9, fontweight="bold")
    fig.savefig(output / "trajectory_motion_audit.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # A separate trajectory plot is intentionally used for visual continuity;
    # the existing coordinate-sensitivity plot is a model-error figure.
    by_file: dict[str, list[dict]] = defaultdict(list)
    for row in reconstructed:
        by_file[row["source_file"]].append(row)
    selected_files = [f for f in sorted(by_file) if "circle" in f or "linefrom" in f]
    if selected_files:
        fig, axes = plt.subplots(1, len(selected_files), figsize=(3.6 * len(selected_files), 3.1), squeeze=False, constrained_layout=True)
        for axis, source_file in zip(axes[0], selected_files):
            rows = sorted(by_file[source_file], key=lambda row: int(row["source_row"]))
            for method, color, label in (("raw", grey, "Raw GPS"), ("rts", blue, "RTS"), ("quantization_aware", orange, "Quantization-aware")):
                x_name, y_name = METHODS[method]
                axis.plot([float(row[x_name]) for row in rows], [float(row[y_name]) for row in rows], color=color, linewidth=1.0, label=label)
            axis.scatter(float(rows[0]["x_raw_m"]), float(rows[0]["y_raw_m"]), color="#1B1B1B", s=18, zorder=4, label="Start")
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("Local x (m)")
            axis.set_ylabel("Local y (m)")
            axis.set_title(short_batch_label(source_file), loc="left", fontweight="bold")
            axis.grid(color="#DDDDDD", linewidth=0.5, alpha=0.7)
        axes[0][0].legend(fontsize=6)
        fig.suptitle("Within-batch horizontal coordinate sequences", x=0.01, ha="left", fontsize=9, fontweight="bold")
        fig.savefig(output / "trajectory_sequences.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    records_path = args.work_dir / "02_standardized" / "standardized_records.csv"
    reconstructed_path = args.work_dir / "04_trajectory_reconstruction" / "reconstructed_horizontal_coordinates.csv"
    output = args.output_dir or args.work_dir / "10_motion_audit"
    output.mkdir(parents=True, exist_ok=True)
    records = load_csv(records_path)
    reconstructed = load_csv(reconstructed_path)
    stage_rows, step_rows, geometry_rows = build_audits(records, reconstructed)
    write_csv(output / "flight_stage_audit_by_file.csv", stage_rows)
    write_csv(output / "trajectory_motion_consistency_by_file.csv", step_rows)
    write_csv(output / "trajectory_geometry_audit.csv", geometry_rows)
    summary = {
        "records_audited": len(reconstructed),
        "batches_audited": len(stage_rows),
        "speed_thresholds_mps": list(SPEED_THRESHOLDS),
        "interpretation": "speed thresholds identify boundary candidates only; interior low-speed records are not removed",
        "core_dataset_modified": False,
        "methods": list(METHODS),
    }
    (output / "trajectory_motion_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_figures(output, stage_rows, step_rows, reconstructed)
    print(f"Trajectory motion audit complete: {output}")


if __name__ == "__main__":
    main()
