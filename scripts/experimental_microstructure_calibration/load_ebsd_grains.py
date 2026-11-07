#!/usr/bin/env python3
"""Load private EBSD grain CSVs into one strict, unit-labelled table."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from common import DATASETS, DatasetSpec, find_dataset_path, public_provenance


CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "grain_id": ("grainid", "grainnumber", "id"),
    "phase": ("phase", "phaseid", "phasename"),
    "phi1": ("phi1", "euler1"),
    "Phi": ("phi", "euler2", "capitalphi"),
    "phi2": ("phi2", "euler3"),
    "semi_axis_a_um": ("a", "semiaxis1", "majoraxis", "ellipsea"),
    "semi_axis_b_um": ("b", "semiaxis2", "minoraxis", "ellipseb"),
    "grain_radius_um": (
        "grainradius",
        "grainradiusum",
        "radius",
        "radiusum",
        "equivalentradius",
        "equivalentradiusum",
    ),
    "grain_area_um2": (
        "grainarea",
        "grainareaum2",
        "area",
        "areaum2",
        "grainsize",
    ),
    "aspect_ratio_reported": ("aspectratio", "ar", "grainaspectratio"),
    "alpha": ("alpha", "angle", "orientationangle", "ellipseangle"),
    "boundary_grain": ("boundarygrain", "isboundary", "edgegrain"),
}

REQUIRED = (
    "grain_id",
    "phase",
    "phi1",
    "Phi",
    "phi2",
    "semi_axis_a_um",
    "semi_axis_b_um",
    "grain_area_um2",
    "aspect_ratio_reported",
    "alpha",
)


def _slug(value: str) -> str:
    # NFKD handles superscript digits (for example ``²`` -> ``2``), while
    # the two Unicode micro signs need an explicit transliteration.  Keeping
    # the unit in the slug avoids accidentally treating an unrelated area
    # column as square micrometres.
    normalized = unicodedata.normalize("NFKD", value.strip().lower())
    normalized = normalized.replace("μ", "u").replace("µ", "u")
    return re.sub(r"[^a-z0-9]", "", normalized)


def _detect_delimiter_and_decimal(path: Path) -> tuple[str, str]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[:8]
    sample = "\n".join(lines)
    delimiter = max((";", ",", "\t"), key=lambda value: sample.count(value))
    decimal = "," if delimiter == ";" and re.search(r"\d,\d", sample) else "."
    return delimiter, decimal


def _column_mapping(columns: Iterable[str]) -> dict[str, str]:
    slug_to_original = {_slug(str(column)): str(column) for column in columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        matches = [
            slug_to_original[name] for name in aliases if name in slug_to_original
        ]
        if len(matches) > 1:
            raise ValueError(f"Several columns map to {canonical}: {matches}")
        if matches:
            mapping[matches[0]] = canonical
    missing = [name for name in REQUIRED if name not in mapping.values()]
    if missing:
        raise ValueError(
            f"Missing required EBSD columns {missing}; available columns are {list(columns)}"
        )
    return mapping


def _infer_angle_unit(frame: pd.DataFrame, requested: str) -> str:
    if requested in ("degree", "radian"):
        return requested
    finite = frame[["phi1", "Phi", "phi2"]].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Euler angle columns contain no finite values")
    maximum = float(np.max(np.abs(finite)))
    if maximum > 2.0 * np.pi + 1.0e-6:
        return "degree"
    # A complete EBSD population confined below 2 pi can be radians, but could
    # also be an unusually narrow degree texture. Refuse silent guessing.
    raise ValueError(
        "Euler angle units are ambiguous because all values lie within ±2π. "
        "Pass --angle-unit degree or --angle-unit radian after checking metadata."
    )


def load_ebsd_csv(
    path: Path,
    spec: DatasetSpec,
    *,
    angle_unit: str = "auto",
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = path.expanduser().resolve()
    delimiter, decimal = _detect_delimiter_and_decimal(path)
    original = pd.read_csv(path, sep=delimiter, decimal=decimal, encoding="utf-8-sig")
    mapping = _column_mapping(original.columns)
    frame = original.rename(columns=mapping)[list(mapping.values())].copy()
    numeric = [
        "grain_id",
        "phi1",
        "Phi",
        "phi2",
        "semi_axis_a_um",
        "semi_axis_b_um",
        "grain_radius_um",
        "grain_area_um2",
        "aspect_ratio_reported",
        "alpha",
    ]
    for column in numeric:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    unit = _infer_angle_unit(frame, angle_unit)
    conversion = 180.0 / np.pi if unit == "radian" else 1.0
    for source, target in (
        ("phi1", "phi1_deg"),
        ("Phi", "Phi_deg"),
        ("phi2", "phi2_deg"),
        ("alpha", "alpha_deg"),
    ):
        frame[target] = frame[source] * conversion
    frame = frame.drop(columns=["phi1", "Phi", "phi2", "alpha"])
    frame.insert(0, "dataset", spec.key)
    frame.insert(1, "source_row", np.arange(2, len(frame) + 2, dtype=int))
    metadata: dict[str, object] = {
        **public_provenance(spec, path),
        "delimiter": "tab" if delimiter == "\t" else delimiter,
        "decimal": decimal,
        "input_angle_unit": unit,
        "canonical_angle_unit": "degree",
        "raw_row_count": int(len(frame)),
        "raw_count_matches_reference": len(frame) == spec.expected_raw_count,
        "column_mapping": mapping,
        "alpha_semantics": spec.alpha_semantics,
        "alpha_is_grain_axis_direction": spec.alpha_is_grain_axis_direction,
        "alpha_observed_unique_count": int(frame["alpha_deg"].nunique(dropna=True)),
        "alpha_observed_minimum_deg": float(frame["alpha_deg"].min()),
        "alpha_observed_maximum_deg": float(frame["alpha_deg"].max()),
    }
    return frame, metadata


def load_dataset(
    key: str, *, data_root: Path | None = None, angle_unit: str = "auto"
) -> tuple[pd.DataFrame, dict[str, object]]:
    spec = DATASETS[key]
    return load_ebsd_csv(
        find_dataset_path(spec, data_root), spec, angle_unit=angle_unit
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--angle-unit", choices=("auto", "degree", "radian"), default="auto"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    frame, metadata = load_dataset(
        args.dataset, data_root=args.data_root, angle_unit=args.angle_unit
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
