#!/usr/bin/env python3
"""Run the nine independent UAVSpectrum3D v0.4.0 processing stages in order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from uavspectrum3d_pipeline.config import DEFAULT_INPUT_DIR, DEFAULT_RELEASE_DIR, DEFAULT_WORK_DIR


STAGES = (
    "01_audit_source_files.py",
    "02_standardize_records.py",
    "03_audit_height_power_inertial.py",
    "04_reconstruct_horizontal_trajectory.py",
    "05_build_analysis_products.py",
    "06_frequency_completion_reference.py",
    "07_grid_sensitivity.py",
    "08_coordinate_sensitivity.py",
    "09_validate_and_package_v040.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--start-stage", type=int, default=1, choices=range(1, 10))
    parser.add_argument("--stop-stage", type=int, default=9, choices=range(1, 10))
    args = parser.parse_args()
    if args.start_stage > args.stop_stage:
        parser.error("--start-stage must be less than or equal to --stop-stage")
    if args.start_stage <= 3 and not args.input_dir.is_dir():
        parser.error(
            f"raw CSV directory does not exist: {args.input_dir}. "
            "Unzip the dataset under data/UAVSpectrum3D or pass --input-dir."
        )

    script_dir = Path(__file__).resolve().parent / "stages"
    for stage_number, script in enumerate(STAGES, start=1):
        if not args.start_stage <= stage_number <= args.stop_stage:
            continue
        command = [
            sys.executable,
            str(script_dir / script),
            "--work-dir",
            str(args.work_dir),
        ]
        if stage_number in (1, 2, 3, 9):
            command.extend(["--input-dir", str(args.input_dir)])
        if stage_number == 9:
            command.extend(["--release-dir", str(args.release_dir)])
        print(f"Running Stage {stage_number}: {script}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
