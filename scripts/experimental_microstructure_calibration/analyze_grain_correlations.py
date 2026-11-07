#!/usr/bin/env python3
"""Export size--aspect-ratio conditional statistics and joint diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from extract_experimental_statistics import describe


def analyze_joint_statistics(frame: pd.DataFrame, bins: int = 5) -> dict[str, object]:
    diameter = frame["equivalent_diameter_um"].to_numpy(dtype=float)
    aspect = frame["aspect_ratio_recomputed"].to_numpy(dtype=float)
    area = frame["grain_area_um2"].to_numpy(dtype=float)
    edges = np.unique(np.quantile(diameter, np.linspace(0.0, 1.0, bins + 1)))
    identifiers = np.clip(np.digitize(diameter, edges[1:-1]), 0, len(edges) - 2)
    conditional = []
    for index in range(len(edges) - 1):
        mask = identifiers == index
        conditional.append(
            {
                "bin": index,
                "ecd_lower_um": float(edges[index]),
                "ecd_upper_um": float(edges[index + 1]),
                "count": int(np.sum(mask)),
                "aspect_ratio_number_weighted": describe(aspect[mask]),
                "aspect_ratio_area_weighted": describe(aspect[mask], area[mask]),
            }
        )
    pearson = stats.pearsonr(diameter, aspect)
    spearman = stats.spearmanr(diameter, aspect)
    return {
        "pearson": {"r": float(pearson.statistic), "pvalue": float(pearson.pvalue)},
        "spearman": {
            "rho": float(spearman.statistic),
            "pvalue": float(spearman.pvalue),
        },
        "equal_count_size_bins": conditional,
        "joint_target_note": (
            "Marginal ECD and AR matches do not imply this conditional distribution "
            "or rank correlation is matched."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cleaned_csv", type=Path)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_joint_statistics(pd.read_csv(args.cleaned_csv), args.bins)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
