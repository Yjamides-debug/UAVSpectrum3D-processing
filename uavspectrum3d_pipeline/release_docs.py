"""Write release documentation for the UAVSpectrum3D dataset package."""

from __future__ import annotations

from pathlib import Path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def write_release_documentation(release: Path, facts: dict) -> None:
    """Create human-readable metadata without inventing authors or identifiers."""
    raw_records = facts["raw_records"]
    spatial_records = facts["spatial_records"]
    core_spatial_product_records = facts["core_spatial_product_records"]
    frequency_bins = facts["frequency_bins"]
    raw_observations = facts["raw_observations"]
    spatial_observations = facts["spatial_observations"]
    core_spatial_product_observations = facts["core_spatial_product_observations"]
    grid_10m_cells = facts["grid_10m_cells"]
    grid_20m_cells = facts["grid_20m_cells"]

    _write(
        release / "README.md",
        f"""
# UAVSpectrum3D v0.4.0

UAVSpectrum3D is the final public package containing UAV-mounted broadband spectrum
measurements collected on 19 November 2025. It contains {raw_records:,} raw
measurement records, {frequency_bins} frequency bins from 860 to 930 MHz, and
{spatial_records:,} records with usable exported latitude and longitude. The
core coordinate-indexed aggregation products use {core_spatial_product_records:,}
fixed-height records; circular and descending-line records are retained as
supplementary trajectory records in the raw and standardized tables.

This package is the final v0.4.0 release published through Zenodo under CC BY
4.0. The dataset DOI is `10.5281/zenodo.21983541`.

## Package structure

```text
UAVSpectrum3D/
|-- raw/original_csv/
|-- metadata/
|-- processed/
|   |-- records/
|   |   |-- fixed_height/
|   |   `-- dynamic_trajectory/
|   |-- matrix/raw_coordinates/
|   `-- tensor/raw_coordinates/
|-- diagnostics/
`-- reference_reuse/
    |-- trajectory_reconstruction/
    `-- frequency_completion/
```

- `raw/original_csv/` preserves the 12 source CSV byte streams under ASCII
  public filenames. The original-to-public mapping and hashes are in
  `metadata/raw_file_manifest.csv`.
- `processed/records/` contains all record metadata and the shared unmodified dBm spectra.
- `metadata/approved_airspace_boundary.csv` contains the four ordered WGS84
  boundary vertices transcribed from the author-retained operating-approval
  record. The approval document itself is not distributed.
- `processed/records/fixed_height/` contains the index of the 4,718 records used by
  the core fixed-height products.
- `processed/records/dynamic_trajectory/` contains the index of the 750 circular
  and descending-line records and their legacy, unvalidated time-protocol labels.
- `processed/matrix/raw_coordinates/` contains 10 m and 20 m aggregation-grid
  products based on the exported, quantized GPS coordinates for fixed-height
  batches.
- `processed/tensor/raw_coordinates/` contains regular x-y-z-frequency views
  of the same products. Tensors contain no interpolation; unobserved cells are
  `NaN` and are identified by masks.
- `diagnostics/` contains source, quality, processing, and technical-validation
  evidence, including five-dimensional QC and 10/20 m aggregation sensitivity.
- `reference_reuse/trajectory_reconstruction/` contains model-derived
  horizontal-coordinate examples. They are not core measurements, recovered
  ground truth, or a replacement for the exported GPS coordinates. Its
  `sensitivity_analysis/` subdirectory compares downstream results under raw,
  RTS, and quantization-aware horizontal coordinate representations while
  keeping the held-out records fixed.

## Essential semantics

- Each record contributes {frequency_bins} power values. The complete raw
  spectrum array therefore contains {raw_observations:,} record-frequency
  observations; {spatial_observations:,} of these also have usable horizontal
  coordinates, and {core_spatial_product_observations:,} enter the core
  coordinate-indexed aggregation products.
- `elevation_gps_msl_m` is the GPS logger's exported mean-sea-level elevation
  field. It is auxiliary and is not used as the analysis height axis.
- `z_relative_label_m` is populated only for filename-recorded fixed-height
  batches. It is a relative-height label, not reconstructed absolute altitude.
- `z_legacy_protocol_m` is populated only for circular and descending-line
  records. It records the historical time-protocol assumption and is not used
  as a core vertical coordinate or measurement of flight height.
- No absolute GPS altitude reconstruction is included.
- `reference_reuse/frequency_completion/` contains a controlled masking
  protocol. Its targets are originally measured entries held out for evaluation,
  not naturally missing frequency observations.
- The 10 m and 20 m values denote post-processing aggregation grids, not GPS
  accuracy or spatial resolution.
- Raw dBm values are retained. Quality flags annotate values without replacing
  or deleting them.

Start with `metadata/data_dictionary.md`,
`metadata/processing_provenance.md`, and `DATA_CARD.md`. Verify file integrity
with `metadata/checksum_manifest.csv`.
""",
    )

    _write(
        release / "metadata" / "data_dictionary.md",
        """
# Data Dictionary

## Missing values and indexing

CSV fields that could not be parsed are empty. In aggregate NPZ products,
unobserved entries use `NaN` only in floating power-statistic arrays; count and
flag arrays are zero, and Boolean masks are false. `record_index`,
`frequency_index`, and `cell_id` are zero-based integer indices.

`metadata/approved_airspace_boundary.csv` stores four ordered WGS84 vertices
transcribed from the operating-approval record retained by the authors. The
public package supplies boundary vertices for reproducibility but does not
include the approval document.

## `processed/records/standardized_records.csv`

| Field | Definition | Unit / values |
|---|---|---|
| `record_index` | Row index shared with the first axis of `spectra_and_quality.npz` | integer |
| `record_id` | Stable public source filename and source-row pair | string |
| `source_file` | ASCII public raw filename | string |
| `source_row` | Data-row number within the source CSV after its two header lines | integer |
| `timestamp` | Timestamp exported by Aaronia MCS | ISO-like local timestamp; timezone unavailable |
| `latitude_raw` | Exported WGS84 latitude | decimal degrees north |
| `longitude_raw` | Exported WGS84 longitude | decimal degrees east |
| `elevation_gps_msl_m` | Exported GPS mean-sea-level elevation | m; auxiliary raw field |
| `x_raw_m`, `y_raw_m` | Local planar coordinates computed from raw latitude/longitude | m |
| `z_relative_label_m` | Filename-recorded nominal relative-height label for fixed-height batches | m; empty for dynamic trajectories; not absolute altitude |
| `z_legacy_protocol_m` | Historical time-derived label for circular and descending-line records | m; unvalidated auxiliary field |
| `height_label_policy` | Rule associated with the source file | controlled string |
| `height_label_source` | Origin of the height field | `filename_nominal_height`, `legacy_time_protocol`, or `none` |
| `height_label_status` | Validation status of the height field | `formal_core_label`, `legacy_unvalidated_auxiliary`, or `unavailable` |
| `core_spatial_product_included` | Whether the record contributes to core coordinate-indexed aggregation products | Boolean; true only for valid-position fixed-height records |
| `speed_knots_raw` | Exported speed | knot |
| `speed_mps` | `speed_knots_raw * 0.5144444444444445` | m s-1 |
| `heading_true_north_deg` | Exported heading | degrees clockwise from true north |
| `measure_unit` | Power unit string exported by the instrument | normally `dBm` |
| `position_valid` | Whether both latitude and longitude are finite | Boolean |
| `finite_frequency_bins` | Number of finite power values in the record | 0-201 |
| `tilt_x`, `tilt_y`, `tilt_z` | Exported tilt fields | raw logger values; physical interpretation not verified |
| `compass_x`, `compass_y`, `compass_z` | Exported compass fields | raw logger values; physical interpretation not verified |
| `gyro_x`, `gyro_y`, `gyro_z` | Exported gyro fields | raw logger values; physical interpretation not verified |

## `processed/records/spectra_and_quality.npz`

| Array | Shape | Definition |
|---|---:|---|
| `power_dbm` | records x frequency | Instrument-exported power values retained in dBm |
| `frequencies_hz` | frequency | Shared frequency axis in Hz |
| `observed_mask` | records x frequency | True where `power_dbm` is finite |
| `quality_flags` | records x frequency | Unsigned 8-bit quality bit field |
| `record_indices` | records | Index linking the NPZ arrays to the CSV table |

Quality flag bits are: bit 0 (`1`) for floor-like values satisfying
`power_dbm <= -149.9 dBm`, and bit 1 (`2`) for positive-power candidates
satisfying `power_dbm > 0 dBm`. A value of `3` means both bits are set, although
the two numerical conditions cannot occur simultaneously for one finite value.
Flags annotate observations; they do not remove or alter power values.

## `processed/matrix/raw_coordinates/grid_10m/` and `grid_20m/`

`cell_coordinates.csv` defines each observed `(ix, iy, iz)` cell and its grid
centre. Horizontal indices use `round(x_raw_m / grid_m)` and
`round(y_raw_m / grid_m)`. The vertical index uses
`round(z_relative_label_m / 1 m)`.

Only records with `core_spatial_product_included == true` contribute to these
core matrices. Circular and descending-line batches remain available in
`processed/records/standardized_records.csv`, the dynamic-trajectory index files,
and the raw archive, but are not used to form the released core space-height
cells because their relative-height progress is not independently validated.

`processed/records/fixed_height/core_record_indices.csv` and
`processed/records/dynamic_trajectory/trajectory_record_indices.csv` point into
the shared record table and `spectra_and_quality.npz`; spectra are not duplicated.
The dynamic-trajectory `legacy_protocol_height_labels.csv` is supplied only for
historical sensitivity analysis and reproduction of earlier processing.

`matrix_product.npz` contains cell x frequency arrays:

| Array | Definition |
|---|---|
| `frequencies_hz` | Shared frequency axis |
| `power_dbm_arithmetic_mean` | Arithmetic mean of contributing dBm numbers |
| `power_dbm_from_linear_mean` | Mean after conversion to mW, converted back to dBm |
| `power_dbm_median` | Median of contributing dBm numbers |
| `sample_count` | Number of finite contributing record-frequency values |
| `floor_count` | Number of contributions satisfying `power_dbm <= -149.9 dBm` |
| `positive_count` | Number of contributions satisfying `power_dbm > 0 dBm` |
| `observed_mask` | `sample_count > 0` |
| `quality_flags` | Bitwise OR of flags from all contributing samples |
| `quality_mask` | Observed and `quality_flags == 0` |

The 10 m and 20 m grid sizes are aggregation parameters. They must not be
reported as GPS precision, measurement footprint, or recovered spatial
resolution.

## `processed/tensor/raw_coordinates/`

Each `tensor_product.npz` is the regular-axis representation of the matching
raw-coordinate matrix. Arrays use the axis order `x_index x y_index x z_index x
frequency`. `x_indices`, `y_indices`, and `z_indices` give the integer index
values represented by each tensor axis. No interpolation or extrapolation is
performed when constructing the tensor. For cells absent from the matrix, the
floating power arrays remain `NaN`; count and flag arrays are zero; and
`observed_mask` and `quality_mask` are false. Axis array position zero is not
necessarily grid index zero: use `x_indices`, `y_indices`, and `z_indices` as
lookups (for example, `z_indices[0] == 1` and `z_indices[39] == 40`).

The 10 m tensor has shape `20 x 13 x 40 x 201`; the 20 m tensor has shape
`11 x 7 x 40 x 201`. These are regular storage bounds and do not imply dense
physical sampling or positioning resolution.

## Reference-reuse trajectory files

Files below `reference_reuse/trajectory_reconstruction/` contain RTS and
quantization-aware model outputs and their proxy validation. Coordinates with
`rts` or `quantization_aware` in their names are model-derived horizontal
estimates. They have no synchronous RTK or flight-controller ground truth and
are unsuitable as claims of recovered absolute GPS position.

`sensitivity_analysis/` contains ten grouped-holdout repetitions using the 26
raw horizontal coordinate pairs. Raw, RTS, and quantization-aware coordinates
use identical test record IDs and measured power targets. The results quantify
downstream coordinate-representation sensitivity, not navigation accuracy.

## `reference_reuse/frequency_completion/`

This directory stores deterministic split files (`*.npz`) and their audit table
for random-entry and contiguous-block frequency masking on the 20 m matrix.
`base_mask` is the strict quality mask, `train_mask` and `test_mask` are disjoint
and partition it, and `eligible_rows` links split rows to the source matrix.
The `results_by_seed.csv` and `results_summary.csv` files report fixed reference
methods. The test target is an originally observed matrix entry held out by the
protocol; it is not a naturally missing sensor value.

## Diagnostic and sensitivity evidence

`diagnostics/01_source_audit/measurement_quality_summary.csv` summarizes
file-, timestamp-, GPS-, and spectrum-field completeness. The files under
`diagnostics/03_quality_audits/` provide the five-dimensional QC summary and
its GPS-transition, height, power, and inertial-field evidence. Quality flags
identify review candidates and do not remove observations.

`diagnostics/07_grid_sensitivity/` contains the 10/20 m aggregation-grid and
quality-policy comparison. It applies the predefined
`ix20=floor(ix10/2), iy20=floor(iy10/2)` mapping; the native rounded grids are
not strictly nested. Agreement on shared entries tests aggregation-scale and
index-definition sensitivity, not implementation equivalence or positioning
accuracy.

`reference_reuse/trajectory_reconstruction/sensitivity_analysis/` contains the
grouped-holdout coordinate-representation protocol, deterministic split files,
per-seed results, summaries, and publication-format figures. It compares
downstream reconstruction under fixed evaluation records; it does not validate
recovered real-flight GPS coordinates.
""",
    )

    _write(
        release / "metadata" / "processing_provenance.md",
        """
# Processing Provenance

## Source and software location

The package is generated from the read-only CSV files in
`UAVSpectrum3D/raw/original_csv/` by the public processing repository
`UAVSpectrum3D-processing`. The entry point is `run_pipeline.py`; Python
dependencies are listed in the repository-level `requirements.txt` and copied
to the generated dataset package.

## Processing stages

1. `01_audit_source_files.py` inventories files, row counts, parse failures,
   GPS availability, and the common 201-bin frequency axis.
2. `02_standardize_records.py` parses timestamps and numeric fields, converts
   knots to metres per second, converts WGS84 latitude/longitude to a local
   planar index, assigns relative-height labels, marks core fixed-height
   product inclusion, and stores spectra and flags.
3. `03_audit_height_power_inertial.py` audits GPS elevation, alternative power
   aggregation semantics, and inertial-like fields without fusing them.
4. `04_reconstruct_horizontal_trajectory.py` creates experimental horizontal
   trajectory estimates and proxy validations for reference reuse only.
5. `05_build_analysis_products.py` builds fixed-height raw-coordinate core
   aggregation products and experimental coordinate reference products at 10 m
   and 20 m grid scales.
6. `06_frequency_completion_reference.py` constructs controlled random-entry
   and contiguous-block frequency masks on the final 20 m matrix and evaluates
   fixed reference methods against held-out measured entries.
7. `07_grid_sensitivity.py` compares 10 m and 20 m aggregation products and
   quality policies on entries shared under a predefined floor-division index
   mapping; the rounded native grids are not strictly nested.
8. `08_coordinate_sensitivity.py` evaluates raw and model-derived horizontal
   coordinate representations using identical grouped-holdout records.
9. `09_validate_and_package_v040.py` enforces cross-stage contracts, checks
   matrix–tensor equivalence, separates core measurements from reference-reuse
   outputs, writes documentation, and generates SHA-256 checksums.

## Coordinate and height provenance

The local reference is the mean of all finite exported latitude/longitude
records. The small-area conversion is
`x = (longitude - longitude_ref) * cos(latitude_ref) * 111320` and
`y = (latitude - latitude_ref) * 110540`, with angles handled consistently in
radians for the cosine.

Fixed-height filenames supply the relative-height labels used in the core
coordinate-indexed aggregation products. Circular and descending flights are
retained in the raw archive and standardized record table, but they are excluded
from the core space-height matrices and tensors because their vertical positions
are not independently validated. Their protocol-derived time labels remain only
as auxiliary standardized fields for traceability and reference reuse. The
exported GPS mean-sea-level elevation remains separate. No absolute altitude is
reconstructed.

## Power and quality provenance

Raw dBm values are not calibrated, clipped, replaced, or deleted. Record-level
quality bits mark floor-like (`<= -149.9 dBm`) and positive-power candidate
(`> 0 dBm`) values. Aggregated flags are set when any finite contribution
triggers a bit. Products expose arithmetic dBm means, linear-power means
converted back to dBm, and dBm medians as distinct quantities.

## Release status

This directory is the final `v0.4.0` package published through Zenodo under CC
BY 4.0. The dataset DOI is `10.5281/zenodo.21983541`.
Earlier local builds are superseded development artifacts.
""",
    )

    _write(
        release / "DATA_CARD.md",
        f"""
# UAVSpectrum3D Data Card

## Status and intended use

Version `0.4.0` is the final public package published through Zenodo under CC
BY 4.0 and described in the Scientific Data Data Descriptor. It supports spectrum-data inspection,
radio-environment mapping method development, aggregation-sensitivity studies,
and reproducible examples using sparse UAV measurements.

It is not a calibrated emissions inventory, a transmitter-identification
benchmark, or high-precision UAV navigation ground truth.

## Contents

- Acquisition date: 19 November 2025.
- Raw files: 12 Aaronia MCS CSV files.
- Raw records: {raw_records:,}.
- Records with usable latitude and longitude: {spatial_records:,}.
- Records contributing to core coordinate-indexed aggregation products:
  {core_spatial_product_records:,}.
- Frequency axis: 201 bins from 860 to 930 MHz, 350 kHz step, 1 MHz RBW.
- Raw record-frequency observations: {raw_observations:,}.
- Spatial record-frequency observations: {spatial_observations:,}.
- Core aggregation-product record-frequency observations:
  {core_spatial_product_observations:,}.
- Observed raw-coordinate space-height cells: {grid_10m_cells} at a 10 m grid
  and {grid_20m_cells} at a 20 m grid.

## Core and reference products

Core processed products retain record-level spectra and raw exported GPS
coordinates, plus raw-coordinate aggregation matrices for fixed-height batches.
Circular and descending-line files are retained as supplementary raw and
standardized records and do not contribute to the core space-height matrices.
Experimental RTS and quantization-aware horizontal trajectories are separated under
`reference_reuse/`; they are model outputs, not corrected observations.

## Quality controls

The pipeline verifies raw copy hashes, the 201-bin shared frequency axis, record
and array dimensions, mask/count consistency, NaN semantics, quality-bit
semantics, power-aggregation identities, five-dimensional QC outputs, 10/20 m
aggregation sensitivity, and grouped coordinate-representation sensitivity.
Extreme power values are flagged but retained.

## Known limitations

- One 3 m source file contributes 232 complete spectra but has no usable GPS
  position, so it is absent from coordinate-indexed products.
- Exported GPS coordinates are quantized and do not establish dense or
  high-accuracy spatial coverage.
- No synchronous RTK, flight-controller trajectory, or independent positioning
  truth is available.
- `z_relative_label_m` is not calibrated absolute altitude. Core aggregation
  products use only filename-derived fixed-height labels. Circular and
  descending-line records instead expose `z_legacy_protocol_m`, an auxiliary
  historical assumption that is not used as a core height axis.
- The exported GPS elevation varies inconsistently across fixed-height batches
  and is retained only as an auxiliary field.
- Tilt, compass, and gyro fields are not fused because their units, axes,
  mounting transform, calibration, and synchronization are not fully verified.
- Quality thresholds indicate candidates for cautious interpretation; they do
  not prove instrument failure or invalid measurement.

## Access, licence, and citation

The dataset licence is CC BY 4.0. The public dataset DOI is
`10.5281/zenodo.21983541`; see `CITATION.cff` for citation metadata.
""",
    )

    _write(
        release / "CITATION.cff",
        """
cff-version: 1.2.0
message: "Please cite the final public dataset record published through Zenodo."
type: dataset
title: "UAVSpectrum3D"
abstract: "UAV-mounted broadband spectrum measurements from 860 to 930 MHz with record-level metadata, quality flags, and spatial aggregation products."
authors:
  - family-names: "Fan"
    given-names: "Rong"
    affiliation: "College of Aviation Electronic and Electrical Engineering, Civil Aviation Flight University of China, China"
  - family-names: "Yu"
    given-names: "Qianqian"
    affiliation: "School of Information and Communication Engineering, University of Electronic Science and Technology of China, China"
  - family-names: "Tan"
    given-names: "Kang"
    affiliation: "School of Information and Communication Engineering, University of Electronic Science and Technology of China, China"
  - family-names: "Zhang"
    given-names: "Chao"
    affiliation: "Intelligent Policing Key Laboratory of Sichuan Province, Sichuan Police College, China"
    email: "galoiszhang@163.com"
  - family-names: "Liu"
    given-names: "Yipeng"
    affiliation: "School of Information and Communication Engineering, University of Electronic Science and Technology of China, China"
    email: "yipengliu@uestc.edu.cn"
version: "0.4.0"
license: "CC-BY-4.0"
doi: "10.5281/zenodo.21983541"
url: "https://doi.org/10.5281/zenodo.21983541"
keywords:
  - "UAV spectrum measurement"
  - "radio environment map"
  - "spectrum cartography"
  - "spatial spectrum data"
""",
    )

    _write(
        release / "LICENSE",
        """
UAVSpectrum3D dataset licence

Except where otherwise noted, this dataset is licensed under the Creative
Commons Attribution 4.0 International licence (CC BY 4.0).

You may share and adapt the material for any purpose, including commercially,
provided that you give appropriate credit, provide a link to the licence, and
indicate whether changes were made. You may not apply legal terms or
technological measures that legally restrict others from doing anything the
licence permits.

Licence summary: https://creativecommons.org/licenses/by/4.0/
Legal code: https://creativecommons.org/licenses/by/4.0/legalcode

This notice does not apply to third-party material that is explicitly identified
as being under different terms.
""",
    )

    _write(
        release / "metadata" / "build_notes.md",
        """
# Internal Build Notes

- `v0.4.0` is the final public package assembled by the current nine-stage
  pipeline and published through Zenodo.
- Earlier directories and local version labels were development artifacts and
  were not public repository releases.
- No public migration or compatibility claim from a previously published
  `v0.3.0` is made.
- The public dataset DOI and landing page are recorded in `CITATION.cff` and
  `metadata/dataset_metadata.json`. Author metadata should remain synchronized
  with the associated manuscript.
""",
    )
