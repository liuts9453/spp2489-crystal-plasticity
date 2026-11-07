#!/usr/bin/env python3
"""Shared configuration and provenance helpers for EBSD--Neper calibration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = ROOT / "results" / "experimental_microstructure_calibration"
RESULTS_ROOT = (
    Path(os.environ.get("SPP2489_EBSD_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    .expanduser()
    .resolve()
)
DATA_ROOT_ENV = "SPP2489_EBSD_DATA_ROOT"
_NEPER_CONFIGURED = os.environ.get("SPP2489_NEPER")
_NEPER_DISCOVERED = shutil.which("neper")
NEPER = (
    Path(
        _NEPER_CONFIGURED
        or _NEPER_DISCOVERED
        or (Path.home() / "miniforge" / "envs" / "spp2489-neper" / "bin" / "neper")
    )
    .expanduser()
    .resolve()
)

EXPERIMENTAL_STATISTICS = RESULTS_ROOT / "experimental_statistics"
DISTRIBUTION_FITS = RESULTS_ROOT / "distribution_fits"
NEPER_RESULTS = RESULTS_ROOT / "neper"
CALIBRATED_PARAMETERS = RESULTS_ROOT / "calibrated_parameters"
VALIDATION = RESULTS_ROOT / "validation"
FIGURES = RESULTS_ROOT / "figures"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    filename: str
    material: str
    section_label: str
    expected_raw_count: int
    source_subdirectory: str
    alpha_semantics: str = "unknown"
    alpha_is_grain_axis_direction: bool = False


DATASETS: dict[str, DatasetSpec] = {
    "Al1_oBD": DatasetSpec(
        "Al1_oBD",
        "Al1_oBD.csv",
        "Al1",
        "oBD",
        2437,
        "src/Microstructure Reconstruction_HMnS/Feature Extraction/DRAGen/FJ/"
        "01 Data Treatment/02 CSV information",
        "section label assigned by filename (oBD -> 0 degrees)",
        False,
    ),
    "Al1_pBD": DatasetSpec(
        "Al1_pBD",
        "Al1_pBD.csv",
        "Al1",
        "pBD",
        848,
        "src/Microstructure Reconstruction_HMnS/Feature Extraction/DRAGen/FJ/"
        "01 Data Treatment/02 CSV information",
        "section label assigned by filename (pBD -> 90 degrees)",
        False,
    ),
    "Al0_00": DatasetSpec(
        "Al0_00",
        "GrainExtractionZ1_15_00_250x_B2_M1.csv",
        "Al0",
        "00",
        2632,
        "src/Microstructure Reconstruction_HMnS/Feature Extraction/DRAGen/IBF/"
        "03 Combination/02 CSV information",
        "constant section label in the supplied CSV (00 -> 0 degrees)",
        False,
    ),
    "Al0_90": DatasetSpec(
        "Al0_90",
        "GrainExtractionZ1_15_90_100x_B2_M1.csv",
        "Al0",
        "90",
        1519,
        "src/Microstructure Reconstruction_HMnS/Feature Extraction/DRAGen/IBF/"
        "03 Combination/02 CSV information",
        "constant section label in the supplied CSV (90 -> 90 degrees)",
        False,
    ),
}

CALIBRATION_SEEDS = (7301, 7302, 7303, 7304, 7305)
VALIDATION_SEEDS = (8301, 8302, 8303, 8304, 8305)
RVE_SIDE_LENGTHS_UM = (200.0, 400.0, 800.0)
MINIMUM_ECD_UM = 2.8


def ensure_result_directories() -> None:
    for path in (
        RESULTS_ROOT,
        EXPERIMENTAL_STATISTICS,
        DISTRIBUTION_FITS,
        NEPER_RESULTS,
        CALIBRATED_PARAMETERS,
        VALIDATION,
        FIGURES,
    ):
        path.mkdir(parents=True, exist_ok=True)


def configured_data_root(required: bool = True) -> Path | None:
    value = os.environ.get(DATA_ROOT_ENV)
    if not value:
        if required:
            raise FileNotFoundError(
                f"{DATA_ROOT_ENV} is not set. Point it to a read-only checkout or a "
                "directory containing the four private EBSD CSV files."
            )
        return None
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{DATA_ROOT_ENV} is not a directory: {root}")
    return root


def find_dataset_path(spec: DatasetSpec, data_root: Path | None = None) -> Path:
    root = data_root or configured_data_root(required=True)
    assert root is not None
    direct = root / spec.source_subdirectory / spec.filename
    if direct.is_file():
        return direct
    root_file = root / spec.filename
    if root_file.is_file():
        return root_file
    matches = sorted(root.rglob(spec.filename))
    if not matches:
        raise FileNotFoundError(f"Missing {spec.filename} beneath {root}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous private EBSD source {spec.filename}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def public_provenance(spec: DatasetSpec, source: Path) -> dict[str, Any]:
    """Return traceability without exposing the private checkout's absolute path."""
    return {
        "dataset": spec.key,
        "source_filename": source.name,
        "source_relative_directory": spec.source_subdirectory,
        "source_sha256": sha256(source),
        "source_size_bytes": source.stat().st_size,
        "expected_raw_count_reference": spec.expected_raw_count,
    }


def dataset_spec_payload(spec: DatasetSpec) -> dict[str, Any]:
    return asdict(spec)
