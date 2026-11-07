#!/usr/bin/env python3
"""Re-screen candidate parameters on more seeds after a failed first holdout."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from common import write_json
from compare_experimental_neper import detailed_comparison_metrics, morphology_objective
from generate_ebsd_matched_neper import NeperMorphologyParameters, generate_rve
from optimize_neper_joint_statistics import BASELINE_PARAMETERS, PARAMETER_NAMES


ROBUST_CALIBRATION_SEEDS = (9401, 9402, 9403, 9404, 9405)
SECOND_HOLDOUT_SEEDS = (11301, 11302, 11303, 11304, 11305)
PBD_REFINEMENT_POINTS = {
    0: BASELINE_PARAMETERS["Al1_pBD"],
    14: (0.868373, 2.992787, 0.728813, 0.883449),
    18: (1.321932138635528, 2.8865138595900857, 0.5679521560683584, 0.9937825630779031),
    20: (1.195232, 2.141397, 0.815344, 0.269763),
    100: (1.20, 2.60, 0.80, 0.80),
    101: (1.20, 2.45, 0.90, 0.90),
}


def _workers(count: int) -> int:
    return min(count, max(1, int(os.environ.get("SPP2489_NEPER_JOBS", "4"))))


def _parameters(values: tuple[float, float, float, float]) -> NeperMorphologyParameters:
    return NeperMorphologyParameters(
        diameter_log_spread_scale=values[0],
        directional_ratio_x_over_y=values[1],
        aspect_ratio_proxy_scale=values[2],
        input_size_shape_rank_correlation=values[3],
    )


def refine(
    experimental: pd.DataFrame,
    output: Path,
    *,
    dataset: str,
    grain_count: int,
    points: dict[int, tuple[float, float, float, float]],
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)

    def evaluate(
        candidate_id: int,
        values: tuple[float, float, float, float],
        seed: int,
        stage: str,
        family: str,
    ) -> dict[str, object]:
        directory = (
            output / "runs" / family / f"candidate_{candidate_id:03d}" / f"seed_{seed}"
        )
        row: dict[str, object] = {
            "dataset": dataset,
            "stage": stage,
            "family": family,
            "candidate_id": candidate_id,
            "seed": seed,
            **dict(zip(PARAMETER_NAMES, values)),
        }
        try:
            manifest = generate_rve(
                experimental,
                directory,
                dataset_key=dataset,
                phase="all",
                side_length_um=400.0,
                seed=seed,
                parameters=_parameters(values),
                grain_count=grain_count,
                generation_timeout_s=300.0,
                morphology_time_limit_s=60.0,
            )
            metrics = detailed_comparison_metrics(
                experimental, pd.read_csv(directory / "synthetic_grains.csv")
            )
            row.update(
                {
                    "status": "completed",
                    "error": "",
                    "morphology_objective": morphology_objective(metrics),
                    "neper_wall_time_s": float(
                        manifest["generation"].get("wall_time_s", 0.0)
                    ),
                    **metrics,
                }
            )
        except (RuntimeError, OSError, ValueError) as error:
            row.update(
                {
                    "status": "failed",
                    "error": str(error),
                    "morphology_objective": np.nan,
                    "neper_wall_time_s": np.nan,
                }
            )
        return row

    tasks = [
        (candidate_id, values, seed, "robust_calibration", "candidate")
        for candidate_id, values in points.items()
        for seed in ROBUST_CALIBRATION_SEEDS
    ]
    with ThreadPoolExecutor(max_workers=_workers(len(tasks))) as executor:
        rows = list(executor.map(lambda item: evaluate(*item), tasks))
    table = pd.DataFrame(rows)
    completed = table.loc[table["status"] == "completed"]
    aggregate = completed.groupby("candidate_id")["morphology_objective"].agg(
        ["mean", "std", "max", "count"]
    )
    eligible = aggregate.loc[aggregate["count"] == len(ROBUST_CALIBRATION_SEEDS)]
    if eligible.empty:
        raise RuntimeError("No robust candidate completed every calibration seed")
    # A small worst-seed term avoids selecting an unstable low-mean candidate.
    score = eligible["mean"] + 0.25 * eligible["max"]
    selected_id = int(score.idxmin())
    selected = points[selected_id]

    holdout_tasks = []
    for family, candidate_id, values in (
        ("baseline", -1, BASELINE_PARAMETERS[dataset]),
        ("improved", selected_id, selected),
    ):
        holdout_tasks.extend(
            (candidate_id, values, seed, "second_independent_holdout", family)
            for seed in SECOND_HOLDOUT_SEEDS
        )
    with ThreadPoolExecutor(max_workers=_workers(len(holdout_tasks))) as executor:
        holdout_rows = list(executor.map(lambda item: evaluate(*item), holdout_tasks))
    rows.extend(holdout_rows)
    table = pd.DataFrame(rows)
    table.to_csv(output / "robust_calibration_history.csv", index=False)

    validation = {}
    for family, group in table.loc[
        (table["stage"] == "second_independent_holdout")
        & (table["status"] == "completed")
    ].groupby("family"):
        validation[str(family)] = {
            "successful_realizations": int(len(group)),
            "objective_mean": float(group["morphology_objective"].mean()),
            "objective_standard_deviation": float(
                group["morphology_objective"].std(ddof=0)
            ),
            "objective_maximum": float(group["morphology_objective"].max()),
            "metric_means": {
                key: float(group[key].mean())
                for key in detailed_comparison_metrics(experimental, experimental)
            },
            "seeds": [int(value) for value in group["seed"]],
        }
    failures = table.loc[table["status"] != "completed"]
    summary = {
        "dataset": dataset,
        "reason": (
            "The first selected candidate failed 2/5 initial holdouts; those seeds "
            "were retired from independent validation before this robust refinement."
        ),
        "candidate_points": {
            str(key): dict(zip(PARAMETER_NAMES, map(float, values)))
            for key, values in points.items()
        },
        "robust_calibration_seeds": list(ROBUST_CALIBRATION_SEEDS),
        "second_holdout_seeds_not_used_for_selection": list(SECOND_HOLDOUT_SEEDS),
        "selection_score": "mean objective + 0.25 * worst-seed objective",
        "selected_candidate_id": selected_id,
        "selected_parameters": asdict(_parameters(selected)),
        "selected_robust_calibration_objective_mean": float(
            eligible.loc[selected_id, "mean"]
        ),
        "selected_robust_calibration_objective_maximum": float(
            eligible.loc[selected_id, "max"]
        ),
        "independent_validation": validation,
        "failed_run_count": int(len(failures)),
        "failed_runs": failures[
            ["stage", "family", "candidate_id", "seed", "error"]
        ].to_dict(orient="records"),
    }
    write_json(output / "optimized_parameters_robust.json", summary)
    write_json(
        output / "independent_validation_robust.json",
        {
            "holdout_seeds": list(SECOND_HOLDOUT_SEEDS),
            "not_used_for_selection": True,
            "comparison": validation,
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cleaned_csv", type=Path)
    parser.add_argument("--dataset", default="Al1_pBD", choices=("Al1_pBD",))
    parser.add_argument("--grain-count", type=int, default=152)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = refine(
        pd.read_csv(args.cleaned_csv),
        args.output,
        dataset=args.dataset,
        grain_count=args.grain_count,
        points=PBD_REFINEMENT_POINTS,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
