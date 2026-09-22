"""Horizontal trajectory reconstruction for quantized GPS observations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear
from scipy.sparse import lil_matrix


@dataclass
class TrackData:
    record_indices: np.ndarray
    source_file: str
    x_raw_m: np.ndarray
    y_raw_m: np.ndarray
    time_s: np.ndarray
    speed_mps: np.ndarray
    heading_deg: np.ndarray


def _kalman_update(
    state: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    observation: np.ndarray,
    noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    innovation = measurement - observation @ state
    innovation_covariance = observation @ covariance @ observation.T + noise
    gain = covariance @ observation.T @ np.linalg.pinv(innovation_covariance)
    identity = np.eye(len(state))
    state = state + gain @ innovation
    covariance = (
        (identity - gain @ observation)
        @ covariance
        @ (identity - gain @ observation).T
        + gain @ noise @ gain.T
    )
    return state, covariance


def rts_reconstruct(
    track: TrackData,
    cell_width_x_m: float,
    cell_width_y_m: float,
    min_speed_mps: float = 0.2,
    acceleration_q: float = 1.0,
    velocity_sigma_mps: float = 1.0,
    position_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a constant-velocity Kalman filter followed by an RTS smoother."""
    n = len(track.x_raw_m)
    position_mask = np.ones(n, dtype=bool) if position_mask is None else position_mask
    filtered = np.zeros((n, 4), dtype=float)
    filtered_cov = np.zeros((n, 4, 4), dtype=float)
    predicted = np.zeros((n, 4), dtype=float)
    predicted_cov = np.zeros((n, 4, 4), dtype=float)
    transitions = [np.eye(4) for _ in range(n)]

    sigma_x = max(cell_width_x_m / math.sqrt(12.0), 0.25)
    sigma_y = max(cell_width_y_m / math.sqrt(12.0), 0.25)
    state = np.array([track.x_raw_m[0], track.y_raw_m[0], 0.0, 0.0], dtype=float)
    covariance = np.diag([sigma_x**2, sigma_y**2, 25.0, 25.0])
    h_position = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    h_velocity = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    r_position = np.diag([sigma_x**2, sigma_y**2])
    r_velocity = np.eye(2) * velocity_sigma_mps**2

    for index in range(n):
        if index:
            dt = max(track.time_s[index] - track.time_s[index - 1], 1e-3)
            transition = np.array(
                [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=float,
            )
            q_axis = np.array(
                [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]], dtype=float
            ) * acceleration_q
            process_noise = np.zeros((4, 4), dtype=float)
            process_noise[np.ix_([0, 2], [0, 2])] = q_axis
            process_noise[np.ix_([1, 3], [1, 3])] = q_axis
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process_noise
            transitions[index] = transition
        predicted[index] = state
        predicted_cov[index] = covariance
        if position_mask[index]:
            state, covariance = _kalman_update(
                state,
                covariance,
                np.array([track.x_raw_m[index], track.y_raw_m[index]]),
                h_position,
                r_position,
            )
        if (
            np.isfinite(track.speed_mps[index])
            and track.speed_mps[index] >= min_speed_mps
            and np.isfinite(track.heading_deg[index])
        ):
            angle = math.radians(track.heading_deg[index] % 360.0)
            velocity = np.array(
                [
                    track.speed_mps[index] * math.sin(angle),
                    track.speed_mps[index] * math.cos(angle),
                ]
            )
            state, covariance = _kalman_update(
                state, covariance, velocity, h_velocity, r_velocity
            )
        filtered[index] = state
        filtered_cov[index] = covariance

    smoothed = filtered.copy()
    smoothed_cov = filtered_cov.copy()
    for index in range(n - 2, -1, -1):
        transition = transitions[index + 1]
        gain = filtered_cov[index] @ transition.T @ np.linalg.pinv(predicted_cov[index + 1])
        smoothed[index] = filtered[index] + gain @ (
            smoothed[index + 1] - predicted[index + 1]
        )
        smoothed_cov[index] = filtered_cov[index] + gain @ (
            smoothed_cov[index + 1] - predicted_cov[index + 1]
        ) @ gain.T
    return smoothed[:, :2], smoothed_cov[:, :2, :2]


def bounded_quantization_reconstruct(
    track: TrackData,
    cell_width_x_m: float,
    cell_width_y_m: float,
    min_speed_mps: float = 0.2,
    motion_sigma_m: float = 5.0,
    acceleration_sigma_mps: float = 2.0,
    position_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Estimate a smooth path constrained to each reported GPS quantization cell."""
    n = len(track.x_raw_m)
    position_mask = np.ones(n, dtype=bool) if position_mask is None else position_mask
    raw = np.column_stack([track.x_raw_m, track.y_raw_m])
    jump = np.zeros(n, dtype=bool)
    for index in range(1, n):
        dt = track.time_s[index] - track.time_s[index - 1]
        if dt <= 0:
            continue
        gps_displacement = float(np.linalg.norm(raw[index] - raw[index - 1]))
        motion_displacement = (
            track.speed_mps[index] * dt if np.isfinite(track.speed_mps[index]) else 0.0
        )
        jump[index] = gps_displacement >= 20.0 and (
            motion_displacement <= 0 or gps_displacement / motion_displacement >= 5.0
        )
    lower = np.empty((n, 2), dtype=float)
    upper = np.empty((n, 2), dtype=float)
    global_lower = raw.min(axis=0) - np.array([cell_width_x_m, cell_width_y_m])
    global_upper = raw.max(axis=0) + np.array([cell_width_x_m, cell_width_y_m])
    for index in range(n):
        if not position_mask[index]:
            lower[index] = global_lower
            upper[index] = global_upper
        else:
            factor = 1.5 if jump[index] else 0.5
            lower[index] = raw[index] - np.array([cell_width_x_m, cell_width_y_m]) * factor
            upper[index] = raw[index] + np.array([cell_width_x_m, cell_width_y_m]) * factor
    sigma_x = max(cell_width_x_m / math.sqrt(12.0), 0.25)
    sigma_y = max(cell_width_y_m / math.sqrt(12.0), 0.25)

    rows: list[tuple[dict[int, float], float]] = []
    for index in range(n):
        if position_mask[index]:
            jump_scale = 5.0 if jump[index] else 1.0
            rows.append(
                ({2 * index: 1.0 / (sigma_x * jump_scale)}, raw[index, 0] / (sigma_x * jump_scale))
            )
            rows.append(
                (
                    {2 * index + 1: 1.0 / (sigma_y * jump_scale)},
                    raw[index, 1] / (sigma_y * jump_scale),
                )
            )

    motion_constraints = 0
    for index in range(1, n):
        dt = track.time_s[index] - track.time_s[index - 1]
        if (
            dt <= 0
            or not np.isfinite(track.speed_mps[index])
            or track.speed_mps[index] < min_speed_mps
            or not np.isfinite(track.heading_deg[index])
        ):
            continue
        angle = math.radians(track.heading_deg[index] % 360.0)
        expected = (
            track.speed_mps[index]
            * dt
            * np.array([math.sin(angle), math.cos(angle)])
        )
        for axis in (0, 1):
            rows.append(
                (
                    {
                        2 * index + axis: 1.0 / motion_sigma_m,
                        2 * (index - 1) + axis: -1.0 / motion_sigma_m,
                    },
                    expected[axis] / motion_sigma_m,
                )
            )
        motion_constraints += 1

    for index in range(2, n):
        dt_previous = track.time_s[index - 1] - track.time_s[index - 2]
        dt_current = track.time_s[index] - track.time_s[index - 1]
        if dt_previous <= 0 or dt_current <= 0:
            continue
        for axis in (0, 1):
            rows.append(
                (
                    {
                        2 * index + axis: 1.0 / (dt_current * acceleration_sigma_mps),
                        2 * (index - 1) + axis: -(
                            1.0 / dt_current + 1.0 / dt_previous
                        )
                        / acceleration_sigma_mps,
                        2 * (index - 2) + axis: 1.0
                        / (dt_previous * acceleration_sigma_mps),
                    },
                    0.0,
                )
            )

    matrix = lil_matrix((len(rows), 2 * n), dtype=float)
    target = np.zeros(len(rows), dtype=float)
    for row_index, (coefficients, value) in enumerate(rows):
        for column, coefficient in coefficients.items():
            matrix[row_index, column] = coefficient
        target[row_index] = value
    result = lsq_linear(
        matrix.tocsr(),
        target,
        bounds=(lower.ravel(), upper.ravel()),
        method="trf",
        lsmr_tol="auto",
        max_iter=2000,
        tol=1e-7,
    )
    return result.x.reshape(n, 2), {
        "success": bool(result.success),
        "status": int(result.status),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "iterations": int(result.nit),
        "jump_records": int(jump.sum()),
        "position_observations": int(position_mask.sum()),
        "motion_constraints": motion_constraints,
    }
