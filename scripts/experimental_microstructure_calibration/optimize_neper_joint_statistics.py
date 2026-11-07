#!/usr/bin/env python3
"""Traceable four-parameter calibration using real periodic Neper geometries."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from common import V2_RESULTS_ROOT, write_json
from compare_experimental_neper import (
    MORPHOLOGY_OBJECTIVE_WEIGHTS,
    detailed_comparison_metrics,
    morphology_objective,
)
from generate_ebsd_matched_neper import (
    NeperMorphologyParameters,
    generate_rve,
)


PARAMETER_NAMES = (
    "diameter_log_spread_scale",
    "directional_ratio_x_over_y",
    "aspect_ratio_proxy_scale",
    "input_size_shape_rank_correlation",
)
DEFAULT_BOUNDS = np.asarray(
    ((0.6, 1.4), (1.0, 3.0), (0.5, 1.5), (-1.0, 1.0)), dtype=float
)
BASELINE_PARAMETERS = {
    "Al1_oBD": (0.95, 1.25660608, 1.0, 0.75),
    "Al1_pBD": (0.95, 2.45, 1.0, 1.0),
    "Al0_00": (0.95, 1.26, 1.0, 0.75),
    "Al0_90": (0.95, 2.25, 1.0, 1.0),
}
DEFAULT_COUNTS = {
    "Al1_oBD": 298,
    "Al1_pBD": 152,
    "Al0_00": 496,
    "Al0_90": 219,
}
SCREENING_SEED = 9301
CALIBRATION_SEEDS = (9301, 9302, 9303)
HOLDOUT_SEEDS = (10301, 10302, 10303, 10304, 10305)


def _workers(count: int) -> int:
    requested = int(os.environ.get("SPP2489_NEPER_JOBS", "4"))
    if requested < 1:
        raise ValueError("SPP2489_NEPER_JOBS must be positive")
    return min(requested, count)


def _candidate_matrix(
    dataset: str, count: int, bounds: np.ndarray = DEFAULT_BOUNDS
) -> list[np.ndarray]:
    if count < 4:
        raise ValueError("At least four Latin-hypercube candidates are required")
    sampler = qmc.LatinHypercube(d=4, seed=2489 + sum(map(ord, dataset)))
    matrix = qmc.scale(sampler.random(count), bounds[:, 0], bounds[:, 1])
    return [np.asarray(BASELINE_PARAMETERS[dataset], dtype=float), *matrix]


def _local_candidates(best: np.ndarray, bounds: np.ndarray) -> list[np.ndarray]:
    steps = np.asarray((0.08, 0.20, 0.12, 0.20), dtype=float)
    points = []
    for coordinate in range(4):
        for sign in (-1.0, 1.0):
            point = best.copy()
            point[coordinate] += sign * steps[coordinate]
            points.append(np.clip(point, bounds[:, 0], bounds[:, 1]))
    return points


def _parameters(values: np.ndarray) -> NeperMorphologyParameters:
    return NeperMorphologyParameters(
        diameter_log_spread_scale=float(values[0]),
        directional_ratio_x_over_y=float(values[1]),
        aspect_ratio_proxy_scale=float(values[2]),
        input_size_shape_rank_correlation=float(values[3]),
    )


def optimize(
    experimental: pd.DataFrame,
    output: Path,
    *,
    dataset: str,
    side_length_um: float = 400.0,
    grain_count: int | None = None,
    screening_candidates: int = 20,
    morphology_time_limit_s: float | None = None,
    morphology_iteration_limit: int | None = 60000,
    generation_timeout_s: float = 300.0,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    count = grain_count or DEFAULT_COUNTS[dataset]
    bounds = DEFAULT_BOUNDS.copy()
    rows: list[dict[str, object]] = []

    def evaluate(
        candidate_id: int,
        values: np.ndarray,
        seed: int,
        stage: str,
        family: str = "improved",
    ) -> dict[str, object]:
        parameters = _parameters(values)
        run_directory = (
            output / "runs" / family / f"candidate_{candidate_id:03d}" / f"seed_{seed}"
        )
        row: dict[str, object] = {
            "dataset": dataset,
            "family": family,
            "stage": stage,
            "candidate_id": candidate_id,
            "seed": seed,
            **dict(zip(PARAMETER_NAMES, map(float, values))),
        }
        try:
            manifest = generate_rve(
                experimental,
                run_directory,
                dataset_key=dataset,
                phase="all",
                side_length_um=side_length_um,
                seed=seed,
                parameters=parameters,
                grain_count=count,
                generation_timeout_s=generation_timeout_s,
                morphology_time_limit_s=morphology_time_limit_s,
                morphology_iteration_limit=morphology_iteration_limit,
            )
            synthetic = pd.read_csv(run_directory / "synthetic_grains.csv")
            metrics = detailed_comparison_metrics(experimental, synthetic)
            row.update(
                {
                    "status": "completed",
                    "error": "",
                    "morphology_objective": morphology_objective(metrics),
                    "neper_wall_time_s": float(
                        manifest["generation"].get("wall_time_s", 0.0)
                    ),
                    "cache_reused": bool(manifest["cache_reused"]),
                    "tessellation_sha256": manifest["tessellation_sha256"],
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
                    "cache_reused": False,
                    "tessellation_sha256": "",
                }
            )
        return row

    candidates = _candidate_matrix(dataset, screening_candidates, bounds)
    with ThreadPoolExecutor(max_workers=_workers(len(candidates))) as executor:
        screening_rows = list(
            executor.map(
                lambda item: evaluate(item[0], item[1], SCREENING_SEED, "screening"),
                enumerate(candidates),
            )
        )
    rows.extend(screening_rows)
    completed = [row for row in screening_rows if row["status"] == "completed"]
    if not completed:
        raise RuntimeError("All four-parameter screening candidates failed")
    best_screen = min(completed, key=lambda row: row["morphology_objective"])
    best_values = np.asarray([best_screen[name] for name in PARAMETER_NAMES])

    local = _local_candidates(best_values, bounds)
    first_local_id = len(candidates)
    with ThreadPoolExecutor(max_workers=_workers(len(local))) as executor:
        local_rows = list(
            executor.map(
                lambda item: evaluate(
                    first_local_id + item[0],
                    item[1],
                    SCREENING_SEED,
                    "local_refinement",
                ),
                enumerate(local),
            )
        )
    rows.extend(local_rows)

    first_seed_rows = [
        row
        for row in rows
        if row["status"] == "completed" and row["seed"] == SCREENING_SEED
    ]
    top_ids = [
        int(row["candidate_id"])
        for row in sorted(first_seed_rows, key=lambda row: row["morphology_objective"])[
            :5
        ]
    ]
    values_by_id = {
        int(row["candidate_id"]): np.asarray([row[name] for name in PARAMETER_NAMES])
        for row in first_seed_rows
    }
    additional_tasks = [
        (candidate_id, values_by_id[candidate_id], seed)
        for candidate_id in top_ids
        for seed in CALIBRATION_SEEDS[1:]
    ]
    with ThreadPoolExecutor(max_workers=_workers(len(additional_tasks))) as executor:
        rows.extend(
            executor.map(
                lambda item: evaluate(
                    item[0], item[1], item[2], "multi_seed_calibration"
                ),
                additional_tasks,
            )
        )

    table = pd.DataFrame(rows)
    calibration = table.loc[
        (table["candidate_id"].isin(top_ids))
        & (table["seed"].isin(CALIBRATION_SEEDS))
        & (table["status"] == "completed")
    ]
    grouped = calibration.groupby("candidate_id")["morphology_objective"].agg(
        ["mean", "std", "count"]
    )
    eligible = grouped.loc[grouped["count"] == len(CALIBRATION_SEEDS)]
    if eligible.empty:
        raise RuntimeError("No candidate completed all multi-seed calibration runs")
    selected_id = int(eligible["mean"].idxmin())
    selected_values = values_by_id[selected_id]
    selected_calibration_objective = float(eligible.loc[selected_id, "mean"])

    holdout_tasks = []
    for family, candidate_id, values in (
        ("baseline", -1, np.asarray(BASELINE_PARAMETERS[dataset], dtype=float)),
        ("improved", selected_id, selected_values),
    ):
        holdout_tasks.extend(
            (candidate_id, values, seed, family) for seed in HOLDOUT_SEEDS
        )
    with ThreadPoolExecutor(max_workers=_workers(len(holdout_tasks))) as executor:
        holdout_rows = list(
            executor.map(
                lambda item: evaluate(
                    item[0], item[1], item[2], "independent_holdout", item[3]
                ),
                holdout_tasks,
            )
        )
    rows.extend(holdout_rows)
    table = pd.DataFrame(rows)
    table.to_csv(output / "calibration_history.csv", index=False)

    holdout_summary: dict[str, object] = {}
    for family, group in table.loc[
        (table["stage"] == "independent_holdout") & (table["status"] == "completed")
    ].groupby("family"):
        holdout_summary[str(family)] = {
            "successful_realizations": int(len(group)),
            "objective_mean": float(group["morphology_objective"].mean()),
            "objective_standard_deviation": float(
                group["morphology_objective"].std(ddof=0)
            ),
            "metric_means": {
                column: float(group[column].mean())
                for column in detailed_comparison_metrics(experimental, experimental)
            },
            "seeds": [int(value) for value in group["seed"]],
        }
    failures = table.loc[table["status"] != "completed"]
    summary = {
        "dataset": dataset,
        "side_length_um": side_length_um,
        "grain_count": count,
        "search_method": "deterministic seeded Latin hypercube plus coordinate-local refinement",
        "parameter_names": list(PARAMETER_NAMES),
        "parameter_bounds": {
            name: [float(lower), float(upper)]
            for name, (lower, upper) in zip(PARAMETER_NAMES, bounds)
        },
        "screening_latin_hypercube_candidates": screening_candidates,
        "total_unique_candidate_count": len(candidates) + len(local),
        "screening_seed": SCREENING_SEED,
        "multi_seed_calibration_seeds": list(CALIBRATION_SEEDS),
        "independent_holdout_seeds_not_used_for_selection": list(HOLDOUT_SEEDS),
        "morphology_objective_weights": MORPHOLOGY_OBJECTIVE_WEIGHTS,
        "odf_in_geometry_objective": False,
        "selected_candidate_id": selected_id,
        "selected_parameters": asdict(_parameters(selected_values)),
        "neper_stopping_criterion": {
            "morphology_time_limit_s": morphology_time_limit_s,
            "morphology_iteration_limit": morphology_iteration_limit,
        },
        "selected_calibration_objective_mean": selected_calibration_objective,
        "selected_calibration_objective_standard_deviation": float(
            eligible.loc[selected_id, "std"]
        ),
        "selection_provenance": (
            "minimum mean morphology objective among candidates that completed all "
            "three calibration seeds; holdout results were not used for selection"
        ),
        "holdout_comparison": holdout_summary,
        "failed_run_count": int(len(failures)),
        "failed_runs": failures[
            ["family", "stage", "candidate_id", "seed", "error"]
        ].to_dict(orient="records"),
        "total_neper_wall_time_s": float(table["neper_wall_time_s"].sum()),
        "history_file": "calibration_history.csv",
    }
    write_json(output / "optimized_parameters.json", summary)
    write_json(
        output / "independent_validation.json",
        {
            "dataset": dataset,
            "holdout_seeds": list(HOLDOUT_SEEDS),
            "parameters_were_selected_without_holdout": True,
            "comparison": holdout_summary,
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cleaned_csv", type=Path)
    parser.add_argument("--dataset", required=True, choices=tuple(BASELINE_PARAMETERS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--side", type=float, default=400.0)
    parser.add_argument("--grain-count", type=int)
    parser.add_argument("--screening-candidates", type=int, default=20)
    stopping = parser.add_mutually_exclusive_group()
    stopping.add_argument("--morphology-time-limit", type=float)
    stopping.add_argument("--morphology-iteration-limit", type=int, default=60000)
    parser.add_argument("--generation-timeout", type=float, default=300.0)
    args = parser.parse_args()
    output = args.output or V2_RESULTS_ROOT / "calibration" / args.dataset
    result = optimize(
        pd.read_csv(args.cleaned_csv),
        output,
        dataset=args.dataset,
        side_length_um=args.side,
        grain_count=args.grain_count,
        screening_candidates=args.screening_candidates,
        morphology_time_limit_s=args.morphology_time_limit,
        morphology_iteration_limit=args.morphology_iteration_limit,
        generation_timeout_s=args.generation_timeout,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
