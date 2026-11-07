#!/usr/bin/env python3
"""Export aggregate-only v2 results without private EBSD rows or machine paths."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from common import ROOT, V2_RESULTS_ROOT


PUBLIC_ROOT = ROOT / "results" / "experimental_microstructure_calibration_v2_public"
SAFE_FILES = (
    "calibration_history.csv",
    "optimized_parameters.json",
    "independent_validation.json",
    "experimental_subsampling_uncertainty.json",
    "statistical_discrepancy_comparison.csv",
    "geometry_quality_summary.json",
    "mechanical_pilot_summary.json",
    "finalization_summary.json",
    "EXPERIMENTAL_NEPER_CALIBRATION_V2_REPORT.md",
)
PRIVATE_KEY_PARTS = (
    "path",
    "directory",
    "command",
    "source_sha256",
    "experimental_table_sha256",
    "orientation_mapping_sha256",
)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize(item)
            for key, item in value.items()
            if not any(part in key.lower() for part in PRIVATE_KEY_PARTS)
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, str) and value.startswith("/home/"):
        return "[private local path removed]"
    return value


def export(source: Path, destination: Path) -> Path:
    missing = [name for name in SAFE_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing v2 deliverables: " + ", ".join(missing))
    destination.mkdir(parents=True, exist_ok=True)
    for name in SAFE_FILES:
        source_path = source / name
        destination_path = destination / name
        if source_path.suffix == ".json":
            destination_path.write_text(
                json.dumps(
                    _sanitize(json.loads(source_path.read_text(encoding="utf-8"))),
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            shutil.copy2(source_path, destination_path)
    figures = sorted((source / "figures").glob("fig*.png"))
    figure_directory = destination / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    for path in figures:
        shutil.copy2(path, figure_directory / path.name)
    manifest = {
        "scope": "aggregate v2 EBSD–Neper calibration and synthetic FCC-CP pilot",
        "raw_or_cleaned_per_grain_experimental_tables_included": False,
        "synthetic_tessellation_files_included": False,
        "private_absolute_paths_included": False,
        "figure_count": len(figures),
        "safe_file_count": len(SAFE_FILES),
    }
    (destination / "PUBLIC_EXPORT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=V2_RESULTS_ROOT)
    parser.add_argument("--destination", type=Path, default=PUBLIC_ROOT)
    args = parser.parse_args()
    print(export(args.source.resolve(), args.destination.resolve()))
