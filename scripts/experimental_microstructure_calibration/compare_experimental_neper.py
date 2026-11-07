#!/usr/bin/env python3
"""Quantify experimental--Neper morphology and texture discrepancies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from extract_texture_statistics import ipf_coordinates, rotations_from_frame


def _normalized_wasserstein(target: np.ndarray, candidate: np.ndarray) -> float:
    scale = float(np.mean(target))
    return float(stats.wasserstein_distance(target, candidate) / scale)


def _cdf_supremum(target: np.ndarray, candidate: np.ndarray) -> float:
    return float(stats.ks_2samp(target, candidate).statistic)


def _ipf_histogram(frame: pd.DataFrame, bins: int = 18) -> np.ndarray:
    rotations = rotations_from_frame(frame)
    descriptors = []
    for axis in np.eye(3):
        coordinates = ipf_coordinates(rotations, axis)
        histogram = np.histogram2d(
            coordinates[:, 0],
            coordinates[:, 1],
            bins=(bins, bins),
            range=((0.0, 1.0), (0.0, 1.0)),
        )[0]
        histogram /= np.sum(histogram)
        descriptors.append(histogram.ravel())
    return np.concatenate(descriptors)


def _canonical_phase_labels(series: pd.Series) -> pd.Series:
    """Normalize numerically equivalent phase labels without guessing names."""
    stripped = series.astype(str).str.strip()
    numeric = pd.to_numeric(stripped, errors="coerce")
    labels = stripped.copy()
    finite = np.isfinite(numeric)
    labels.loc[finite] = numeric.loc[finite].map(lambda value: f"{value:.15g}")
    return labels


def _phase_texture_distance(
    experimental: pd.DataFrame, synthetic: pd.DataFrame
) -> tuple[float, dict[str, float]]:
    exp_phase = _canonical_phase_labels(experimental["phase"])
    syn_phase = _canonical_phase_labels(synthetic["phase"])
    phases = sorted(set(exp_phase) | set(syn_phase))
    exp_fractions = exp_phase.value_counts(normalize=True)
    syn_fractions = syn_phase.value_counts(normalize=True)
    phase_fraction_error = 0.5 * sum(
        abs(float(exp_fractions.get(phase, 0.0)) - float(syn_fractions.get(phase, 0.0)))
        for phase in phases
    )
    odf_distance = phase_fraction_error
    diagnostics: dict[str, float] = {
        "phase_fraction_total_variation": phase_fraction_error
    }
    for phase in phases:
        exp_group = experimental.loc[exp_phase == phase]
        syn_group = synthetic.loc[syn_phase == phase]
        if exp_group.empty or syn_group.empty:
            diagnostics[f"phase_{phase}_odf_total_variation"] = 1.0
            continue
        exp_odf = _ipf_histogram(exp_group)
        syn_odf = _ipf_histogram(syn_group)
        distance = float(0.5 * np.sum(np.abs(exp_odf - syn_odf)) / 3.0)
        diagnostics[f"phase_{phase}_odf_total_variation"] = distance
        odf_distance += float(exp_fractions.get(phase, 0.0)) * distance
    return min(float(odf_distance), 1.0), diagnostics


def _axial_histogram(alpha_deg: np.ndarray, bins: int = 18) -> np.ndarray:
    histogram = np.histogram(np.mod(alpha_deg, 180.0), bins=bins, range=(0.0, 180.0))[0]
    return histogram / np.sum(histogram)


def _size_orientation_correlations(frame: pd.DataFrame) -> np.ndarray:
    diameter = frame["equivalent_diameter_um"].to_numpy(dtype=float)
    rotations = rotations_from_frame(frame)
    values = []
    for axis in np.eye(3):
        coordinates = ipf_coordinates(rotations, axis)
        for column in range(2):
            coefficient = stats.spearmanr(diameter, coordinates[:, column]).statistic
            values.append(0.0 if not np.isfinite(coefficient) else coefficient)
    return np.asarray(values)


def comparison_metrics(
    experimental: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    weights: dict[str, float] | None = None,
    include_direction: bool = True,
) -> dict[str, object]:
    weights = weights or {
        "diameter": 1.0,
        "aspect": 1.0,
        "direction": 0.5,
        "correlation": 0.5,
        "odf": 1.0,
        "size_orientation": 0.5,
    }
    weights = dict(weights)
    if not include_direction:
        # The supplied SPP2489 CSVs use alpha as a section label (0/90), not
        # as a per-grain fitted-ellipse direction.  Keep the discrepancy as a
        # diagnostic, but do not optimize a synthetic morphology against it.
        weights["direction"] = 0.0
    exp_d = experimental["equivalent_diameter_um"].to_numpy(dtype=float)
    syn_d = synthetic["equivalent_diameter_um"].to_numpy(dtype=float)
    exp_a = experimental["aspect_ratio_recomputed"].to_numpy(dtype=float)
    syn_a = synthetic["aspect_ratio_recomputed"].to_numpy(dtype=float)
    exp_corr = float(stats.spearmanr(exp_d, exp_a).statistic)
    syn_corr = float(stats.spearmanr(syn_d, syn_a).statistic)
    odf_distance, texture_diagnostics = _phase_texture_distance(experimental, synthetic)
    exp_direction = _axial_histogram(experimental["alpha_deg"].to_numpy(dtype=float))
    syn_direction = _axial_histogram(synthetic["alpha_deg"].to_numpy(dtype=float))
    size_orientation_error = float(
        np.mean(
            np.abs(
                _size_orientation_correlations(experimental)
                - _size_orientation_correlations(synthetic)
            )
        )
        / 2.0
    )
    components = {
        "diameter": _normalized_wasserstein(exp_d, syn_d),
        "aspect": _normalized_wasserstein(exp_a, syn_a),
        "direction": float(0.5 * np.sum(np.abs(exp_direction - syn_direction))),
        "correlation": abs(exp_corr - syn_corr) / 2.0,
        "odf": odf_distance,
        "size_orientation": size_orientation_error,
    }
    objective = sum(weights[name] * components[name] for name in components) / sum(
        weights.values()
    )
    return {
        "objective": float(objective),
        "normalized_components": components,
        "weights": weights,
        "direction_target_used": include_direction,
        "diagnostics": {
            "diameter_ks": _cdf_supremum(exp_d, syn_d),
            "aspect_ratio_ks": _cdf_supremum(exp_a, syn_a),
            "experimental_size_ar_spearman": exp_corr,
            "synthetic_size_ar_spearman": syn_corr,
            "experimental_count": int(len(experimental)),
            "synthetic_count": int(len(synthetic)),
            **texture_diagnostics,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experimental_csv", type=Path)
    parser.add_argument("synthetic_csv", type=Path)
    parser.add_argument("--phase")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experimental = pd.read_csv(args.experimental_csv)
    synthetic = pd.read_csv(args.synthetic_csv)
    if args.phase is not None:
        experimental = experimental.loc[
            experimental["phase"].astype(str) == str(args.phase)
        ]
        synthetic = synthetic.loc[synthetic["phase"].astype(str) == str(args.phase)]
    result = comparison_metrics(experimental, synthetic)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
