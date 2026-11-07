#!/usr/bin/env python3
"""Compute number-, area-, tail-, direction-, and joint EBSD statistics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from clean_ebsd_data import CleaningPolicy, add_derived_geometry, clean_grains
from common import DATASETS
from load_ebsd_grains import load_dataset


QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def weighted_quantile(
    values: np.ndarray, quantiles: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    order = np.argsort(values)
    ordered = np.asarray(values, dtype=float)[order]
    ordered_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(ordered_weights) - 0.5 * ordered_weights
    cumulative /= np.sum(ordered_weights)
    return np.interp(quantiles, cumulative, ordered)


def describe(values: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if weights is None:
        mean = float(np.mean(array))
        standard_deviation = float(np.std(array, ddof=1))
        quantile_values = np.quantile(array, QUANTILES)
    else:
        weight = np.array(weights, dtype=float, copy=True)
        weight /= np.sum(weight)
        mean = float(np.sum(weight * array))
        standard_deviation = float(np.sqrt(np.sum(weight * (array - mean) ** 2)))
        quantile_values = weighted_quantile(array, np.asarray(QUANTILES), weight)
    result = {
        "count": int(array.size),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "coefficient_of_variation": standard_deviation / mean,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }
    for probability, value in zip(QUANTILES, quantile_values):
        result[f"q{int(100 * probability):02d}"] = float(value)
    return result


def _axial_direction_statistics(
    alpha_deg: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    doubled = np.radians(2.0 * np.mod(alpha_deg, 180.0))
    normalized = weights / np.sum(weights)
    cosine = float(normalized @ np.cos(doubled))
    sine = float(normalized @ np.sin(doubled))
    return {
        "mean_axis_deg_mod_180": (0.5 * math.degrees(math.atan2(sine, cosine))) % 180.0,
        "resultant_length": math.hypot(cosine, sine),
        "circular_variance": 1.0 - math.hypot(cosine, sine),
    }


def _distribution_table(
    values: np.ndarray, weights: np.ndarray
) -> list[dict[str, float]]:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order] / np.sum(weights)
    number_cdf = np.arange(1, len(values) + 1, dtype=float) / len(values)
    area_cdf = np.cumsum(sorted_weights)
    return [
        {
            "value": float(value),
            "number_cdf": float(number),
            "area_weighted_cdf": float(area),
        }
        for value, number, area in zip(sorted_values, number_cdf, area_cdf)
    ]


def summarize_grains(
    frame: pd.DataFrame,
    *,
    alpha_is_grain_axis_direction: bool = False,
    alpha_semantics: str = "unknown",
) -> dict[str, object]:
    data = add_derived_geometry(frame)
    ecd = data["equivalent_diameter_um"].to_numpy(dtype=float)
    aspect = data["aspect_ratio_recomputed"].to_numpy(dtype=float)
    area = data["grain_area_um2"].to_numpy(dtype=float)
    alpha = data["alpha_deg"].to_numpy(dtype=float)
    pearson = stats.pearsonr(ecd, aspect)
    spearman = stats.spearmanr(ecd, aspect)
    size_bin = pd.qcut(ecd, q=min(5, len(np.unique(ecd))), duplicates="drop")
    conditional: list[dict[str, object]] = []
    for interval in size_bin.categories:
        mask = np.asarray(size_bin == interval)
        conditional.append(
            {
                "ecd_interval": str(interval),
                "count": int(np.sum(mask)),
                "ecd": describe(ecd[mask]),
                "aspect_ratio": describe(aspect[mask]),
            }
        )
    phases = {
        str(key): int(value)
        for key, value in data["phase"].astype(str).value_counts().items()
    }
    return {
        "grain_count": int(len(data)),
        "phase_counts": phases,
        "total_observed_grain_area_um2": float(np.sum(area)),
        "equivalent_diameter_um": {
            "number_weighted": describe(ecd),
            "area_weighted": describe(ecd, area),
            "empirical_cdf": _distribution_table(ecd, area),
        },
        "grain_area_um2": {
            "number_weighted": describe(area),
            "area_weighted": describe(area, area),
        },
        "aspect_ratio": {
            "definition": "max(a,b)/min(a,b), constrained to AR >= 1",
            "number_weighted": describe(aspect),
            "area_weighted": describe(aspect, area),
            "empirical_cdf": _distribution_table(aspect, area),
            "reported_vs_recomputed_max_abs_relative_difference": float(
                np.nanmax(np.abs(data["aspect_ratio_relative_difference"]))
            ),
        },
        "alpha_column_statistics": {
            "semantics": alpha_semantics,
            "usable_as_grain_axis_direction": alpha_is_grain_axis_direction,
            "number_weighted": _axial_direction_statistics(alpha, np.ones(len(data))),
            "area_weighted": _axial_direction_statistics(alpha, area),
        },
        "size_aspect_joint": {
            "pearson_r": float(pearson.statistic),
            "pearson_pvalue": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_pvalue": float(spearman.pvalue),
            "conditional_aspect_ratio_by_equal_count_ecd_bin": conditional,
        },
        "geometry_definition_checks": {
            "ellipse_fit_area_to_measured_grain_area": describe(
                data["ellipse_to_grain_area_ratio"].to_numpy(dtype=float)
            ),
            "grain_area_is_not_replaced_by_ellipse_fit_area": True,
        },
    }


def raw_and_cleaned_statistics(
    frame: pd.DataFrame,
    policy: CleaningPolicy,
    *,
    alpha_is_grain_axis_direction: bool = False,
    alpha_semantics: str = "unknown",
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    valid_for_raw = add_derived_geometry(frame)
    numerical = np.isfinite(
        valid_for_raw[
            [
                "grain_area_um2",
                "semi_axis_a_um",
                "semi_axis_b_um",
                "alpha_deg",
            ]
        ]
    ).all(axis=1)
    positive = (
        (valid_for_raw["grain_area_um2"] > 0)
        & (valid_for_raw["semi_axis_a_um"] > 0)
        & (valid_for_raw["semi_axis_b_um"] > 0)
    )
    raw_stats_frame = valid_for_raw.loc[numerical & positive].copy()
    cleaned, audited, cleaning = clean_grains(frame, policy)
    return (
        {
            "raw_valid_geometry": summarize_grains(
                raw_stats_frame,
                alpha_is_grain_axis_direction=alpha_is_grain_axis_direction,
                alpha_semantics=alpha_semantics,
            ),
            "cleaned": summarize_grains(
                cleaned,
                alpha_is_grain_axis_direction=alpha_is_grain_axis_direction,
                alpha_semantics=alpha_semantics,
            ),
            "cleaning_audit": cleaning,
        },
        cleaned,
        audited,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--angle-unit", choices=("auto", "degree", "radian"), default="auto"
    )
    parser.add_argument("--minimum-ecd", type=float, default=2.8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame, source = load_dataset(
        args.dataset, data_root=args.data_root, angle_unit=args.angle_unit
    )
    summary, cleaned, audited = raw_and_cleaned_statistics(
        frame,
        CleaningPolicy(minimum_ecd_um=args.minimum_ecd),
        alpha_is_grain_axis_direction=DATASETS[
            args.dataset
        ].alpha_is_grain_axis_direction,
        alpha_semantics=DATASETS[args.dataset].alpha_semantics,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"source": source, **summary}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    cleaned.to_csv(args.output.with_suffix(".cleaned.csv"), index=False)
    audited.to_csv(args.output.with_suffix(".audit.csv"), index=False)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
