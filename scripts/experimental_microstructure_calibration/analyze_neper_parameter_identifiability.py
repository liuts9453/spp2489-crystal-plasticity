#!/usr/bin/env python3
"""Diagnose response sensitivity and identifiability of four Neper parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import write_json
from optimize_neper_joint_statistics import PARAMETER_NAMES


RESPONSE_NAMES = (
    "morphology_objective",
    "mean_ecd_relative_error",
    "cv_ecd_relative_error",
    "mean_ar_relative_error",
    "size_ar_spearman_absolute_error",
    "size_conditioned_ar_error",
)


def analyze(history: pd.DataFrame) -> dict[str, object]:
    screening = history.loc[
        (history["status"] == "completed")
        & (history["family"] == "improved")
        & (history["stage"].isin(("screening", "local_refinement")))
    ].copy()
    if len(screening) < len(PARAMETER_NAMES) + 2:
        raise ValueError("Insufficient completed screening points for identifiability")
    correlations = {}
    regression = {}
    design = screening[list(PARAMETER_NAMES)].to_numpy(dtype=float)
    design_standard = (design - design.mean(axis=0)) / design.std(axis=0, ddof=1)
    singular_values = np.linalg.svd(design_standard, compute_uv=False)
    for response in RESPONSE_NAMES:
        response_values = screening[response].to_numpy(dtype=float)
        correlations[response] = {
            parameter: {
                "spearman": float(
                    stats.spearmanr(screening[parameter], response_values).statistic
                ),
                "p_value": float(
                    stats.spearmanr(screening[parameter], response_values).pvalue
                ),
            }
            for parameter in PARAMETER_NAMES
        }
        y_standard = (response_values - response_values.mean()) / max(
            response_values.std(ddof=1), 1.0e-15
        )
        coefficients, *_ = np.linalg.lstsq(
            np.column_stack((np.ones(len(screening)), design_standard)),
            y_standard,
            rcond=None,
        )
        predicted = (
            np.column_stack((np.ones(len(screening)), design_standard)) @ coefficients
        )
        denominator = np.sum(np.square(y_standard - y_standard.mean()))
        r_squared = 1.0 - np.sum(np.square(y_standard - predicted)) / max(
            denominator, 1.0e-15
        )
        regression[response] = {
            "standardized_linear_coefficients": dict(
                zip(PARAMETER_NAMES, map(float, coefficients[1:]))
            ),
            "r_squared": float(r_squared),
        }
    return {
        "screening_point_count": int(len(screening)),
        "design_singular_values": [float(value) for value in singular_values],
        "design_condition_number": float(singular_values[0] / singular_values[-1]),
        "parameter_response_spearman": correlations,
        "standardized_local_linear_models": regression,
        "interpretation_limit": (
            "Sensitivity correlations describe the sampled parameter box and do not "
            "prove global structural identifiability of the Neper generator."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(pd.read_csv(args.history_csv))
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
