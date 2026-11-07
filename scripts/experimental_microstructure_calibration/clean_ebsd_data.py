#!/usr/bin/env python3
"""Apply explicit, audited EBSD grain filters without silently trimming tails."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATASETS, MINIMUM_ECD_UM
from load_ebsd_grains import load_dataset


@dataclass(frozen=True)
class CleaningPolicy:
    minimum_ecd_um: float | None = MINIMUM_ECD_UM
    maximum_aspect_ratio: float | None = None
    exclude_boundary_grains: bool = False


def add_derived_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["equivalent_diameter_um"] = 2.0 * np.sqrt(result["grain_area_um2"] / np.pi)
    larger = np.maximum(result["semi_axis_a_um"], result["semi_axis_b_um"])
    smaller = np.minimum(result["semi_axis_a_um"], result["semi_axis_b_um"])
    result["aspect_ratio_recomputed"] = larger / smaller
    result["aspect_ratio_relative_difference"] = (
        result["aspect_ratio_recomputed"] - result["aspect_ratio_reported"]
    ) / result["aspect_ratio_recomputed"]
    result["ellipse_fit_area_um2"] = (
        np.pi * result["semi_axis_a_um"] * result["semi_axis_b_um"]
    )
    result["ellipse_to_grain_area_ratio"] = (
        result["ellipse_fit_area_um2"] / result["grain_area_um2"]
    )
    return result


def clean_grains(
    frame: pd.DataFrame, policy: CleaningPolicy = CleaningPolicy()
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    raw = add_derived_geometry(frame)
    reason = pd.Series("kept", index=raw.index, dtype="object")

    def reject(mask: pd.Series, label: str) -> None:
        reason.loc[(reason == "kept") & mask.fillna(True)] = label

    reject(raw["grain_id"].isna(), "missing_grain_id")
    reject(raw["grain_id"].duplicated(keep=False), "duplicate_grain_id")
    reject(
        raw["phase"].isna() | (raw["phase"].astype(str).str.strip() == ""),
        "missing_phase",
    )
    finite_columns = [
        "phi1_deg",
        "Phi_deg",
        "phi2_deg",
        "semi_axis_a_um",
        "semi_axis_b_um",
        "grain_area_um2",
        "aspect_ratio_reported",
        "alpha_deg",
    ]
    reject(
        ~np.isfinite(raw[finite_columns]).all(axis=1), "missing_or_nonfinite_numeric"
    )
    reject(raw["grain_area_um2"] <= 0.0, "nonpositive_area")
    reject(
        (raw["semi_axis_a_um"] <= 0.0) | (raw["semi_axis_b_um"] <= 0.0),
        "nonpositive_ellipse_axis",
    )
    reject(
        (raw["Phi_deg"] < -1.0e-8)
        | (raw["Phi_deg"] > 180.0 + 1.0e-8)
        | (raw["phi1_deg"].abs() > 360.0 + 1.0e-8)
        | (raw["phi2_deg"].abs() > 360.0 + 1.0e-8),
        "invalid_bunge_euler_range",
    )
    reject(raw["aspect_ratio_recomputed"] < 1.0, "invalid_aspect_ratio")
    if policy.minimum_ecd_um is not None:
        reject(
            raw["equivalent_diameter_um"] < policy.minimum_ecd_um,
            "below_minimum_ecd",
        )
    if policy.maximum_aspect_ratio is not None:
        reject(
            raw["aspect_ratio_recomputed"] > policy.maximum_aspect_ratio,
            "above_maximum_aspect_ratio",
        )
    if policy.exclude_boundary_grains:
        if "boundary_grain" not in raw:
            raise ValueError(
                "Boundary exclusion was requested but the CSV has no boundary flag"
            )
        boundary = (
            raw["boundary_grain"]
            .astype(str)
            .str.lower()
            .isin(("1", "true", "yes", "y"))
        )
        reject(boundary, "boundary_grain")
    audited = raw.copy()
    audited["cleaning_status"] = reason
    cleaned = audited.loc[reason == "kept"].copy()
    counts = reason.value_counts().sort_index().to_dict()
    audit: dict[str, object] = {
        "policy": asdict(policy),
        "raw_count": int(len(raw)),
        "cleaned_count": int(len(cleaned)),
        "removed_count": int(len(raw) - len(cleaned)),
        "removal_counts_by_first_applicable_reason": {
            str(key): int(value) for key, value in counts.items() if key != "kept"
        },
        "tail_policy": (
            "No upper ECD or AR tail is removed unless maximum_aspect_ratio is "
            "explicitly supplied. The 2.8 um ECD threshold is reported separately."
        ),
    }
    return cleaned, audited, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--angle-unit", choices=("auto", "degree", "radian"), default="auto"
    )
    parser.add_argument("--minimum-ecd", type=float, default=MINIMUM_ECD_UM)
    parser.add_argument("--maximum-ar", type=float)
    parser.add_argument("--exclude-boundary", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame, source = load_dataset(
        args.dataset, data_root=args.data_root, angle_unit=args.angle_unit
    )
    policy = CleaningPolicy(
        minimum_ecd_um=args.minimum_ecd,
        maximum_aspect_ratio=args.maximum_ar,
        exclude_boundary_grains=args.exclude_boundary,
    )
    cleaned, audited, audit = clean_grains(frame, policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False)
    audited.to_csv(args.output.with_name(args.output.stem + "_audit.csv"), index=False)
    print(json.dumps({"source": source, "cleaning": audit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
