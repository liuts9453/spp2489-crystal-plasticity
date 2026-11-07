#!/usr/bin/env python3
"""Compare baseline and improved holdouts against exp--exp uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import write_json


METRICS = (
    "ecd_wasserstein_normalized",
    "ar_wasserstein_normalized",
    "ecd_ks",
    "ar_ks",
    "mean_ecd_relative_error",
    "cv_ecd_relative_error",
    "mean_ar_relative_error",
    "ecd_q95_relative_error",
    "ecd_q99_relative_error",
    "ar_q95_relative_error",
    "size_ar_spearman_absolute_error",
    "size_conditioned_ar_error",
    "odf_number_total_variation",
    "odf_area_total_variation",
)


def compare(
    history: pd.DataFrame, exp_exp: pd.DataFrame, *, dataset: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    available_stages = set(history["stage"])
    holdout_stage = (
        "second_independent_holdout"
        if "second_independent_holdout" in available_stages
        else "independent_holdout"
    )
    holdout = history.loc[
        (history["stage"] == holdout_stage) & (history["status"] == "completed")
    ]
    rows = []
    for metric in METRICS:
        uncertainty = exp_exp[metric].to_numpy(dtype=float)
        row: dict[str, object] = {
            "dataset": dataset,
            "metric": metric,
            "exp_exp_mean": float(np.mean(uncertainty)),
            "exp_exp_q975": float(np.quantile(uncertainty, 0.975)),
        }
        for family in ("baseline", "improved"):
            values = holdout.loc[holdout["family"] == family, metric].to_numpy(
                dtype=float
            )
            row[f"{family}_mean"] = float(np.mean(values))
            row[f"{family}_standard_deviation"] = float(np.std(values, ddof=0))
            row[f"{family}_percentile_vs_exp_exp"] = float(
                np.mean(uncertainty <= np.mean(values))
            )
            row[f"{family}_exceeds_exp_exp_q975"] = bool(
                np.mean(values) > np.quantile(uncertainty, 0.975)
            )
        row["improvement_fraction"] = float(
            (row["baseline_mean"] - row["improved_mean"])
            / max(abs(row["baseline_mean"]), 1.0e-15)
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    summary = {
        "dataset": dataset,
        "holdout_stage": holdout_stage,
        "holdout_seeds_per_family": int(
            len(holdout.loc[holdout["family"] == "baseline"])
        ),
        "metrics": table.to_dict(orient="records"),
        "finite_sample_rule": (
            "A discrepancy is flagged beyond grain-count sampling when its five-seed "
            "holdout mean exceeds the 97.5th percentile of matched-size exp--exp pairs."
        ),
    }
    return table, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_csv", type=Path)
    parser.add_argument("exp_exp_csv", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    table, summary = compare(
        pd.read_csv(args.history_csv),
        pd.read_csv(args.exp_exp_csv),
        dataset=args.dataset,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    write_json(args.output_json, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
