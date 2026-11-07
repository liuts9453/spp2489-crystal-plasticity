#!/usr/bin/env python3
"""Estimate grain-level EBSD and matched-RVE finite-sample uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import V2_RESULTS_ROOT, write_json
from compare_experimental_neper import detailed_comparison_metrics


DEFAULT_COUNTS = {
    "Al1_oBD": 298,
    "Al1_pBD": 152,
    "Al0_00": 496,
    "Al0_90": 219,
}


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)),
        "q025": float(np.quantile(array, 0.025)),
        "q50": float(np.quantile(array, 0.5)),
        "q975": float(np.quantile(array, 0.975)),
    }


def _descriptors(frame: pd.DataFrame) -> dict[str, float]:
    diameter = frame["equivalent_diameter_um"].to_numpy(dtype=float)
    aspect = frame["aspect_ratio_recomputed"].to_numpy(dtype=float)
    return {
        "mean_ecd_um": float(np.mean(diameter)),
        "cv_ecd": float(np.std(diameter, ddof=1) / np.mean(diameter)),
        "mean_ar": float(np.mean(aspect)),
        "size_ar_spearman": float(stats.spearmanr(diameter, aspect).statistic),
    }


def bootstrap_dataset(
    frame: pd.DataFrame,
    *,
    dataset: str,
    matched_count: int,
    replicates: int = 500,
    seed: int = 2489,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    if matched_count > len(frame):
        raise ValueError("matched_count cannot exceed the cleaned EBSD grain count")
    rng = np.random.default_rng(seed)
    bootstrap_rows = []
    pair_rows = []
    for replicate in range(replicates):
        indices = rng.integers(0, len(frame), size=len(frame))
        sample = frame.iloc[indices].reset_index(drop=True)
        bootstrap_rows.append({"replicate": replicate, **_descriptors(sample)})

        first = frame.iloc[
            rng.choice(len(frame), size=matched_count, replace=False)
        ].reset_index(drop=True)
        second = frame.iloc[
            rng.choice(len(frame), size=matched_count, replace=False)
        ].reset_index(drop=True)
        metrics = detailed_comparison_metrics(first, second)
        pair_rows.append({"replicate": replicate, **metrics})

    bootstrap_table = pd.DataFrame(bootstrap_rows)
    pair_table = pd.DataFrame(pair_rows)
    summary = {
        "dataset": dataset,
        "cleaned_grain_count": int(len(frame)),
        "matched_rve_grain_count": int(matched_count),
        "bootstrap_replicates": int(replicates),
        "random_seed": int(seed),
        "resampling_level": "individual grains within one exported EBSD section",
        "limitation": (
            "Grain-level bootstrap quantifies finite-grain sampling only; it cannot "
            "estimate field-to-field or specimen-to-specimen variation because field "
            "identifiers are absent from the supplied CSV."
        ),
        "bootstrap_descriptors": {
            column: _quantiles(bootstrap_table[column].tolist())
            for column in bootstrap_table.columns
            if column != "replicate"
        },
        "matched_size_experiment_experiment_discrepancies": {
            column: _quantiles(pair_table[column].tolist())
            for column in pair_table.columns
            if column != "replicate"
        },
    }
    return summary, bootstrap_table, pair_table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cleaned_csv", type=Path)
    parser.add_argument("--dataset", required=True, choices=tuple(DEFAULT_COUNTS))
    parser.add_argument("--matched-count", type=int)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2489)
    parser.add_argument("--output", type=Path, default=V2_RESULTS_ROOT / "uncertainty")
    args = parser.parse_args()
    frame = pd.read_csv(args.cleaned_csv)
    count = args.matched_count or DEFAULT_COUNTS[args.dataset]
    summary, bootstrap_table, pair_table = bootstrap_dataset(
        frame,
        dataset=args.dataset,
        matched_count=count,
        replicates=args.replicates,
        seed=args.seed,
    )
    output = args.output / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    bootstrap_table.to_csv(output / "bootstrap_replicates.csv", index=False)
    pair_table.to_csv(output / "matched_size_exp_exp_replicates.csv", index=False)
    write_json(output / "experimental_subsampling_uncertainty.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
