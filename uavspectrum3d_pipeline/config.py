"""Path and numerical configuration shared by all v0.4.0 stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# A user can unzip the Zenodo archive under ``data/UAVSpectrum3D`` and run the
# pipeline without editing source code.  All three paths remain overridable
# from the command line for other layouts.
DEFAULT_INPUT_DIR = REPOSITORY_ROOT / "data" / "UAVSpectrum3D" / "raw" / "original_csv"
DEFAULT_WORK_DIR = REPOSITORY_ROOT / "work" / "v0.4.0"
DEFAULT_RELEASE_DIR = REPOSITORY_ROOT / "build" / "UAVSpectrum3D"

KNOT_TO_MPS = 0.5144444444444445
LAT_M_PER_DEG = 110_540.0
LON_M_PER_DEG = 111_320.0
FLOOR_POWER_DBM = -149.9
POSITIVE_POWER_DBM = 0.0
EXPECTED_FREQUENCY_BINS = 201


@dataclass(frozen=True)
class PipelinePaths:
    """Canonical stage paths; callers may override input/work/release roots."""

    input_dir: Path = DEFAULT_INPUT_DIR
    work_dir: Path = DEFAULT_WORK_DIR
    release_dir: Path = DEFAULT_RELEASE_DIR

    @property
    def source_audit_dir(self) -> Path:
        return self.work_dir / "01_source_audit"

    @property
    def standardized_dir(self) -> Path:
        return self.work_dir / "02_standardized"

    @property
    def quality_audit_dir(self) -> Path:
        return self.work_dir / "03_quality_audits"

    @property
    def trajectory_dir(self) -> Path:
        return self.work_dir / "04_trajectory_reconstruction"

    @property
    def products_dir(self) -> Path:
        return self.work_dir / "05_analysis_products"

    @property
    def validation_dir(self) -> Path:
        return self.work_dir / "06_validation"
