#!/usr/bin/env python3
"""Stage 06: generate a controlled frequency-completion reference protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavspectrum3d_pipeline.config import DEFAULT_WORK_DIR
from uavspectrum3d_pipeline.utils import write_csv, write_json


SEEDS = (0, 1, 2, 3, 4)
RANDOM_FRACTIONS = (0.50,)
CONTIGUOUS_FRACTIONS = (0.10, 0.20, 0.30)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_matrix(directory: Path) -> dict:
    with np.load(directory / "matrix_product.npz") as arrays:
        required = (
            "power_dbm_arithmetic_mean",
            "observed_mask",
            "quality_mask",
            "sample_count",
            "frequencies_hz",
        )
        data = {key: arrays[key].copy() for key in required}
    if data["power_dbm_arithmetic_mean"].ndim != 2:
        raise ValueError("Frequency completion requires a cell-by-frequency matrix.")
    if not np.array_equal(data["observed_mask"], data["sample_count"] > 0):
        raise ValueError("Matrix observed_mask disagrees with sample_count.")
    if np.any(data["quality_mask"] & ~data["observed_mask"]):
        raise ValueError("Matrix quality_mask includes unobserved entries.")
    if np.any(~np.isfinite(data["power_dbm_arithmetic_mean"][data["observed_mask"]])):
        raise ValueError("Observed matrix values must be finite.")
    if np.any(np.isfinite(data["power_dbm_arithmetic_mean"][~data["observed_mask"]])):
        raise ValueError("Unobserved matrix values must be NaN.")
    return data


def choose_rows(data: dict, min_coverage: float = 0.80) -> np.ndarray:
    mask = data["quality_mask"]
    coverage = mask.sum(axis=1) / mask.shape[1]
    rows = np.flatnonzero(coverage >= min_coverage)
    if len(rows) == 0:
        raise ValueError("No rows satisfy frequency coverage threshold.")
    return rows


def split_random(base: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train = np.zeros_like(base, dtype=bool)
    test = np.zeros_like(base, dtype=bool)
    for row in range(base.shape[0]):
        available = np.flatnonzero(base[row])
        n_train = max(4, min(len(available) - 1, int(round(len(available) * fraction))))
        selected = rng.choice(available, size=n_train, replace=False)
        train[row, selected] = True
        test[row, available] = True
        test[row, selected] = False
    return train, test


def split_contiguous(base: np.ndarray, hidden_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_freq = base.shape[1]
    length = max(1, int(round(hidden_fraction * n_freq)))
    train = np.zeros_like(base, dtype=bool)
    test = np.zeros_like(base, dtype=bool)
    starts = np.full(base.shape[0], -1, dtype=np.int32)
    for row in range(base.shape[0]):
        valid = [start for start in range(1, n_freq - length)
                 if base[row, start:start + length].all()
                 and base[row, :start].any() and base[row, start + length:].any()]
        if not valid:
            raise ValueError(f"No internal contiguous block available for row {row}.")
        start = int(rng.choice(valid))
        starts[row] = start
        test[row, start:start + length] = True
        train[row] = base[row] & ~test[row]
    return train, test, starts


def predict(power: np.ndarray, train: np.ndarray, method: str, rank: int = 6) -> np.ndarray:
    values = power[train]
    global_mean = float(np.mean(values))
    column_means = np.array([
        np.mean(power[:, col][train[:, col]]) if np.any(train[:, col]) else global_mean
        for col in range(power.shape[1])
    ])
    output = np.broadcast_to(column_means, power.shape).copy()
    if method == "frequency_mean":
        return output
    if method == "linear_interpolation":
        for row in range(power.shape[0]):
            known = np.flatnonzero(train[row])
            if len(known) >= 2:
                missing = np.flatnonzero(~train[row])
                output[row, missing] = np.interp(missing, known, power[row, known])
        return output
    if method == "iterative_svd":
        filled = output.copy()
        filled[train] = power[train]
        missing = ~train
        effective_rank = min(rank, min(filled.shape))
        for _ in range(100):
            u, s, vt = np.linalg.svd(filled, full_matrices=False)
            approximation = (u[:, :effective_rank] * s[:effective_rank]) @ vt[:effective_rank]
            filled[missing] = approximation[missing]
            filled[train] = power[train]
        return filled
    raise ValueError(f"Unknown completion method: {method}")


def metrics(power: np.ndarray, prediction: np.ndarray, test: np.ndarray) -> dict:
    error = prediction[test] - power[test]
    return {
        "test_entries": int(test.sum()),
        "rmse_db": float(np.sqrt(np.mean(error ** 2))),
        "mae_db": float(np.mean(np.abs(error))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    args = parser.parse_args()
    matrix_dir = args.work_dir / "05_analysis_products" / "raw_coordinates" / "grid_20m"
    data = load_matrix(matrix_dir)
    eligible = choose_rows(data)
    power = data["power_dbm_arithmetic_mean"][eligible]
    base = data["quality_mask"][eligible]
    output = args.work_dir / "06_frequency_completion"
    split_dir = output / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    methods = ("frequency_mean", "linear_interpolation", "iterative_svd")
    rows = []
    split_rows = []

    def run(protocol: str, level: float, seed: int, train: np.ndarray, test: np.ndarray, starts=None) -> None:
        stem = f"{protocol}_{level:.2f}_seed_{seed}"
        payload = {"train_mask": train, "test_mask": test, "base_mask": base, "eligible_rows": eligible, "frequencies_hz": data["frequencies_hz"]}
        if starts is not None:
            payload["block_starts"] = starts
        np.savez_compressed(split_dir / f"{stem}.npz", **payload)
        split_rows.append({"protocol": protocol, "level": level, "seed": seed, "artifact": f"splits/{stem}.npz", "eligible_rows": len(eligible), "train_entries": int(train.sum()), "test_entries": int(test.sum()), "partition_ok": bool(np.array_equal(train | test, base) and not np.any(train & test))})
        for method in methods:
            result = metrics(power, predict(power, train, method), test)
            rows.append({"protocol": protocol, "level": level, "seed": seed, "method": method, **result})

    for fraction in RANDOM_FRACTIONS:
        for seed in SEEDS:
            run("random_masking", fraction, seed, *split_random(base, fraction, seed))
    for fraction in CONTIGUOUS_FRACTIONS:
        for seed in SEEDS:
            train, test, starts = split_contiguous(base, fraction, seed)
            run("contiguous_block", fraction, seed, train, test, starts)

    write_csv(output / "results_by_seed.csv", rows)
    write_csv(output / "split_audit.csv", split_rows)
    summary = []
    for protocol in ("random_masking", "contiguous_block"):
        for level in sorted({row["level"] for row in rows if row["protocol"] == protocol}):
            for method in methods:
                selected = [row for row in rows if row["protocol"] == protocol and row["level"] == level and row["method"] == method]
                summary.append({"protocol": protocol, "level": level, "method": method, "rmse_db_mean": float(np.mean([r["rmse_db"] for r in selected])), "rmse_db_std": float(np.std([r["rmse_db"] for r in selected], ddof=1)), "mae_db_mean": float(np.mean([r["mae_db"] for r in selected])), "mae_db_std": float(np.std([r["mae_db"] for r in selected], ddof=1))})
    write_csv(output / "results_summary.csv", summary)
    config = {
        "protocol_version": "1.0",
        "protocol": "controlled masking of originally observed, strict-quality matrix entries",
        "dataset_product": "processed/matrix/raw_coordinates/grid_20m",
        "matrix_shape": list(data["power_dbm_arithmetic_mean"].shape),
        "eligible_rows": int(len(eligible)),
        "quality_policy": "quality_mask",
        "random_observed_fractions": list(RANDOM_FRACTIONS),
        "contiguous_hidden_fractions": list(CONTIGUOUS_FRACTIONS),
        "seeds": list(SEEDS),
        "methods": list(methods),
        "rank": 6,
        "frequency_mean_definition": {
            "per_frequency_statistic": "mean over training entries in each frequency column",
            "empty_column_fallback": "global mean over all training entries",
        },
        "linear_interpolation_definition": {
            "scope": "row-wise over frequency-bin indices",
            "implementation": "numpy.interp",
            "outside_known_range": "nearest known endpoint value",
            "minimum_known_bins": 2,
        },
        "iterative_svd_definition": {
            "initialization": "frequency-column training means with global-mean fallback",
            "rank": 6,
            "iterations": 100,
            "stopping_rule": "fixed iteration count; no convergence threshold",
            "training_entries": "restored after every iteration",
        },
        "split_count": len(split_rows),
        "evaluation_metrics": ["rmse_db", "mae_db"],
        "target_semantics": "held-out measured matrix entries; not naturally missing frequency observations and not a calibrated field truth",
        "input_matrix_sha256": sha256_file(matrix_dir / "matrix_product.npz"),
    }
    write_json(output / "protocol_metadata.json", config)
    report = """# Frequency-completion reference protocol

Protocol version: 1.0. This is a reference-reuse experiment, not a core data
processing step and not a claim of naturally missing-frequency recovery.

## Input and eligibility

The input is the 20 m raw-coordinate `matrix_product.npz`. Only entries in its
`quality_mask` are eligible. Rows are retained when at least 80% of the 201
frequency bins pass that mask; the published protocol records the resulting
`eligible_rows` explicitly. The input matrix SHA-256 is stored in
`protocol_metadata.json`.

## Controlled splits

For each of five fixed seeds, `random_masking` samples 50% of each eligible
row's quality-valid entries for `train_mask`; the remaining quality-valid
entries form `test_mask`. For each of five fixed seeds and each hidden fraction
(10%, 20%, 30%), `contiguous_block` hides one internally located contiguous
frequency block per row and uses all other quality-valid entries for training.
Every split stores `base_mask`, `train_mask`, `test_mask`, `eligible_rows` and
`frequencies_hz`. The masks are disjoint and their union equals `base_mask`.

## Fixed reference methods and metrics

`frequency_mean` predicts each frequency from its training-set column mean and
falls back to the global mean of all training entries when a column has no
training value. `linear_interpolation` applies `numpy.interp` within each row
from at least two known frequency bins; values outside the known range use the
nearest known endpoint. `iterative_svd` starts from the column-mean prediction,
applies exactly 100 rank-6 SVD iterations with no convergence threshold, and
restores training entries after every iteration. Performance is calculated
only on held-out measured entries using
`RMSE = sqrt(mean((prediction - measured)^2))` and
`MAE = mean(abs(prediction - measured))`, both in dB.

The split-level results are in `results_by_seed.csv` and the mean and sample
standard deviation across seeds are in `results_summary.csv`. These results
characterize reuse under the stated masking protocol; they do not establish a
general algorithm ranking, physical field truth, or performance on naturally
missing sensor values.
"""
    (output / "README.md").write_text(report, encoding="utf-8")
    print(f"Stage 06 complete: {output}")


if __name__ == "__main__":
    main()
