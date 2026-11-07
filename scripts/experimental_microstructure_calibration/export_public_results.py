#!/usr/bin/env python3
"""Export a reviewed, aggregate-only subset of private calibration results."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from common import RESULTS_ROOT, ROOT


PUBLIC_RESULTS = ROOT / "results" / "experimental_microstructure_calibration_public"

SAFE_FILES = (
    "experimental_grain_statistics.csv",
    "experimental_distribution_parameters.json",
    "experimental_texture_statistics.json",
    "neper_calibration_history.csv",
    "neper_best_parameters.json",
    "neper_statistical_validation.json",
    "rve_representativeness.json",
    "EXPERIMENTAL_NEPER_CALIBRATION_REPORT.md",
)

PRIVATE_KEYS = {
    "source_sha256",
    "source_size_bytes",
    "cleaned_csv",
    "cleaning_audit_csv",
    "statistics_json",
    "validation_ensemble_csv",
    "representative_tessellation",
    "calibration_history_csv",
    "representativeness_csv",
}


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize(item)
            for key, item in value.items()
            if key not in PRIVATE_KEYS
            and "absolute" not in key.lower()
            and "private_source_root" not in key.lower()
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def export_public_results(
    source: Path = RESULTS_ROOT, destination: Path = PUBLIC_RESULTS
) -> Path:
    missing = [name for name in SAFE_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Calibration is incomplete; missing public deliverables: "
            + ", ".join(missing)
        )
    destination.mkdir(parents=True, exist_ok=True)
    for name in SAFE_FILES:
        source_path = source / name
        destination_path = destination / name
        if source_path.suffix == ".json":
            payload = _sanitize(json.loads(source_path.read_text(encoding="utf-8")))
            destination_path.write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        else:
            shutil.copy2(source_path, destination_path)
    figures = sorted((source / "figures").glob("fig*.png"))
    expected = {f"fig{index:02d}" for index in range(1, 11)}
    actual = {path.stem.split("_")[0] for path in figures}
    if actual != expected:
        raise RuntimeError(
            f"Expected ten reviewed figures {sorted(expected)}, found {sorted(actual)}"
        )
    figure_destination = destination / "figures"
    figure_destination.mkdir(parents=True, exist_ok=True)
    for path in figures:
        shutil.copy2(path, figure_destination / path.name)
    provenance = {
        "scope": "aggregate experimental statistics and synthetic Neper validation",
        "raw_private_csv_included": False,
        "cleaned_per_grain_tables_included": False,
        "private_absolute_paths_included": False,
        "synthetic_tessellations_with_sampled_experimental_orientations_included": False,
        "file_count": len(SAFE_FILES) + len(figures) + 1,
    }
    (destination / "PUBLIC_EXPORT_MANIFEST.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--destination", type=Path, default=PUBLIC_RESULTS)
    args = parser.parse_args()
    print(export_public_results(args.source.resolve(), args.destination.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
