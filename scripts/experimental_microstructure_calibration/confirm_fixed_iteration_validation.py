#!/usr/bin/env python3
"""Confirm legacy Al1 selections with deterministic fixed-iteration Neper runs."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from common import RESULTS_ROOT, V2_RESULTS_ROOT, write_json
from compare_experimental_neper import (
    detailed_comparison_metrics,
    morphology_objective,
)
from generate_ebsd_matched_neper import (
    NeperMorphologyParameters,
    generate_rve,
)
from optimize_neper_joint_statistics import BASELINE_PARAMETERS, DEFAULT_COUNTS


SELECTION_FILES = {
    "Al1_oBD": (
        V2_RESULTS_ROOT / "calibration" / "Al1_oBD" / "optimized_parameters.json"
    ),
    "Al1_pBD": (
        V2_RESULTS_ROOT
        / "calibration"
        / "Al1_pBD_robust"
        / "optimized_parameters_robust.json"
    ),
}
DEFAULT_SEEDS = (12301, 12302, 12303, 12304, 12305)


def _parameters(mapping: dict[str, object]) -> NeperMorphologyParameters:
    return NeperMorphologyParameters(
        diameter_log_spread_scale=float(mapping["diameter_log_spread_scale"]),
        directional_ratio_x_over_y=float(mapping["directional_ratio_x_over_y"]),
        aspect_ratio_proxy_scale=float(mapping["aspect_ratio_proxy_scale"]),
        input_size_shape_rank_correlation=float(
            mapping["input_size_shape_rank_correlation"]
        ),
    )


def _same_parameters(
    left: NeperMorphologyParameters, right: NeperMorphologyParameters
) -> bool:
    return asdict(left) == asdict(right)


def confirm_dataset(
    dataset: str,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    iteration_limit: int = 60000,
    workers: int = 4,
) -> dict[str, object]:
    experimental = pd.read_csv(
        RESULTS_ROOT / "experimental_statistics" / f"{dataset}_cleaned.csv"
    )
    selection = json.loads(SELECTION_FILES[dataset].read_text(encoding="utf-8"))
    improved = _parameters(selection["selected_parameters"])
    baseline = _parameters(
        dict(
            zip(
                (
                    "diameter_log_spread_scale",
                    "directional_ratio_x_over_y",
                    "aspect_ratio_proxy_scale",
                    "input_size_shape_rank_correlation",
                ),
                BASELINE_PARAMETERS[dataset],
            )
        )
    )
    output = V2_RESULTS_ROOT / "fixed_iteration_confirmation" / dataset
    output.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, int, NeperMorphologyParameters]] = [
        ("baseline", seed, baseline) for seed in seeds
    ]
    identical_families = _same_parameters(baseline, improved)
    if not identical_families:
        tasks.extend(("improved", seed, improved) for seed in seeds)

    def evaluate(item: tuple[str, int, NeperMorphologyParameters]) -> dict[str, object]:
        family, seed, parameters = item
        run = output / "runs" / family / f"seed_{seed}"
        row: dict[str, object] = {
            "dataset": dataset,
            "family": family,
            "seed": seed,
            **asdict(parameters),
        }
        try:
            manifest = generate_rve(
                experimental,
                run,
                dataset_key=dataset,
                phase="all",
                side_length_um=400.0,
                seed=seed,
                parameters=parameters,
                grain_count=DEFAULT_COUNTS[dataset],
                generation_timeout_s=300.0,
                morphology_iteration_limit=iteration_limit,
            )
            metrics = detailed_comparison_metrics(
                experimental, pd.read_csv(run / "synthetic_grains.csv")
            )
            row.update(
                status="completed",
                error="",
                morphology_objective=morphology_objective(metrics),
                neper_wall_time_s=float(manifest["generation"]["wall_time_s"]),
                tessellation_sha256=manifest["tessellation_sha256"],
                grain_table_sha256=manifest["grain_table_sha256"],
                **metrics,
            )
        except (OSError, RuntimeError, ValueError) as error:
            row.update(
                status="failed",
                error=str(error),
                morphology_objective=float("nan"),
            )
        return row

    with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        rows = list(executor.map(evaluate, tasks))

    if identical_families:
        for row in list(rows):
            duplicate = row.copy()
            duplicate["family"] = "improved"
            duplicate["aliased_from_identical_baseline_parameters"] = True
            rows.append(duplicate)

    history = pd.DataFrame(rows).sort_values(["family", "seed"])
    history.to_csv(output / "fixed_iteration_history.csv", index=False)
    comparison: dict[str, object] = {}
    for family, group in history.loc[history["status"] == "completed"].groupby(
        "family"
    ):
        comparison[str(family)] = {
            "successful_realizations": int(len(group)),
            "objective_mean": float(group["morphology_objective"].mean()),
            "objective_standard_deviation": float(
                group["morphology_objective"].std(ddof=0)
            ),
            "metric_means": {
                metric: float(group[metric].mean())
                for metric in detailed_comparison_metrics(experimental, experimental)
            },
        }
    result = {
        "dataset": dataset,
        "purpose": (
            "confirmation of legacy time-stopped Al1 parameter selections with a "
            "deterministic fixed-iteration rule"
        ),
        "selection_was_not_changed_using_confirmation_seeds": True,
        "fixed_iteration_limit": iteration_limit,
        "confirmation_seeds": list(seeds),
        "families_have_identical_parameters": identical_families,
        "baseline_parameters": asdict(baseline),
        "improved_parameters": asdict(improved),
        "comparison": comparison,
        "failed_run_count": int((history["status"] != "completed").sum()),
    }
    write_json(output / "fixed_iteration_confirmation.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=("Al1_oBD", "Al1_pBD"), choices=SELECTION_FILES
    )
    parser.add_argument("--iteration-limit", type=int, default=60000)
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("SPP2489_NEPER_JOBS", "4"))
    )
    args = parser.parse_args()
    for dataset in args.datasets:
        result = confirm_dataset(
            dataset,
            iteration_limit=args.iteration_limit,
            workers=args.workers,
        )
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
