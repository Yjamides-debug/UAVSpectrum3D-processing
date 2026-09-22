"""Convert observed cell-by-frequency products to explicit spatial tensors."""

from __future__ import annotations

import numpy as np


def matrix_to_tensor(product: dict) -> dict:
    """Expand observed cells into regular x/y/z/f axes without interpolation."""
    cells = np.asarray(product["cells"], dtype=int)
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("cells must have shape (n_cells, 3)")
    if len(cells) == 0:
        raise ValueError("Cannot build a tensor without observed cells")
    axes = [np.arange(cells[:, index].min(), cells[:, index].max() + 1) for index in range(3)]
    shape = tuple(len(axis) for axis in axes) + (len(product["frequencies_hz"]),)

    tensor_fields = {
        "power_dbm_arithmetic_mean": np.full(shape, np.nan, dtype=np.float32),
        "power_dbm_from_linear_mean": np.full(shape, np.nan, dtype=np.float32),
        "power_dbm_median": np.full(shape, np.nan, dtype=np.float32),
        "sample_count": np.zeros(shape, dtype=np.int32),
        "floor_count": np.zeros(shape, dtype=np.int32),
        "positive_count": np.zeros(shape, dtype=np.int32),
        "observed_mask": np.zeros(shape, dtype=bool),
        "quality_flags": np.zeros(shape, dtype=np.uint8),
        "quality_mask": np.zeros(shape, dtype=bool),
    }
    for row_index, cell in enumerate(cells):
        index = tuple(int(cell[axis]) - int(axes[axis][0]) for axis in range(3))
        for key in tensor_fields:
            tensor_fields[key][index] = product[key][row_index]
    return {
        **tensor_fields,
        "frequencies_hz": np.asarray(product["frequencies_hz"]),
        "x_indices": axes[0],
        "y_indices": axes[1],
        "z_indices": axes[2],
        "source_cell_indices": cells,
    }


def validate_tensor_against_matrix(product: dict, tensor: dict) -> None:
    """Check that every observed matrix cell maps back to the same tensor slice."""
    cells = np.asarray(product["cells"], dtype=int)
    axes = [tensor[key] for key in ("x_indices", "y_indices", "z_indices")]
    for row_index, cell in enumerate(cells):
        index = tuple(int(cell[axis]) - int(axes[axis][0]) for axis in range(3))
        for key in (
            "power_dbm_arithmetic_mean",
            "power_dbm_from_linear_mean",
            "power_dbm_median",
            "sample_count",
            "floor_count",
            "positive_count",
            "observed_mask",
            "quality_flags",
            "quality_mask",
        ):
            if not np.array_equal(product[key][row_index], tensor[key][index]):
                raise ValueError(f"Tensor mismatch for {key} at matrix row {row_index}.")
    if not np.array_equal(tensor["observed_mask"], tensor["sample_count"] > 0):
        raise ValueError("Tensor observed_mask disagrees with sample_count > 0.")
    if np.any(tensor["quality_mask"] & ~tensor["observed_mask"]):
        raise ValueError("Tensor quality_mask includes unobserved entries.")
