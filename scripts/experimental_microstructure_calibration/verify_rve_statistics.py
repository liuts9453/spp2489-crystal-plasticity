#!/usr/bin/env python3
"""Verify periodicity, realization variability, and RVE-size representativeness."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd

from common import RVE_SIDE_LENGTHS_UM, VALIDATION_SEEDS, write_json
from compare_experimental_neper import comparison_metrics
from generate_ebsd_matched_neper import (
    NeperMorphologyParameters,
    generate_rve,
)


def verify_representativeness(
    experimental: pd.DataFrame,
    output_directory: Path,
    *,
    dataset_key: str,
    phase: str,
    parameters: NeperMorphologyParameters,
    side_lengths_um: tuple[float, ...] = RVE_SIDE_LENGTHS_UM,
    seeds: tuple[int, ...] = VALIDATION_SEEDS,
    include_direction: bool = True,
) -> dict[str, object]:
    target = (
        experimental
        if phase.lower() == "all"
        else experimental.loc[experimental["phase"].astype(str) == str(phase)]
    )
    tasks = [(side, seed) for side in side_lengths_um for seed in seeds]
    generation_timeout_s = float(
        os.environ.get("SPP2489_NEPER_VALIDATION_TIMEOUT_S", "1800")
    )
    if generation_timeout_s <= 0.0:
        raise ValueError("SPP2489_NEPER_VALIDATION_TIMEOUT_S must be positive")
    morphology_time_limit_s = float(
        os.environ.get("SPP2489_NEPER_MORPHOLOGY_TIME_S", "90")
    )
    if morphology_time_limit_s <= 0.0:
        raise ValueError("SPP2489_NEPER_MORPHOLOGY_TIME_S must be positive")

    def evaluate(task: tuple[float, int]) -> dict[str, object]:
        side, seed = task
        run_directory = output_directory / f"side_{side:g}" / f"seed_{seed}"
        try:
            manifest = generate_rve(
                experimental,
                run_directory,
                dataset_key=dataset_key,
                phase=phase,
                side_length_um=side,
                seed=seed,
                parameters=parameters,
                generation_timeout_s=generation_timeout_s,
                morphology_time_limit_s=morphology_time_limit_s,
            )
        except (RuntimeError, OSError, ValueError) as error:
            return {
                "side_length_um": side,
                "seed": seed,
                "grain_count": "",
                "status": "failed",
                "error": str(error),
                "area_relative_error": np.nan,
                "objective": np.nan,
                **{
                    f"{key}_distance": np.nan
                    for key in (
                        "diameter",
                        "aspect",
                        "direction",
                        "correlation",
                        "odf",
                        "size_orientation",
                    )
                },
            }
        synthetic = pd.read_csv(run_directory / "synthetic_grains.csv")
        metrics = comparison_metrics(
            target, synthetic, include_direction=include_direction
        )
        return {
            "side_length_um": side,
            "seed": seed,
            "grain_count": manifest["grain_count"],
            "status": "completed",
            "error": "",
            "area_relative_error": abs(
                float(synthetic["grain_area_um2"].sum()) - side**2
            )
            / side**2,
            "objective": metrics["objective"],
            **{
                f"{key}_distance": value
                for key, value in metrics["normalized_components"].items()
            },
        }

    requested_workers = int(os.environ.get("SPP2489_NEPER_JOBS", "4"))
    if requested_workers < 1:
        raise ValueError("SPP2489_NEPER_JOBS must be a positive integer")
    with ThreadPoolExecutor(max_workers=min(len(tasks), requested_workers)) as executor:
        rows = list(executor.map(evaluate, tasks))
    table = pd.DataFrame(rows)
    output_directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_directory / "rve_representativeness.csv", index=False)
    size_summaries = []
    for side, group in table.groupby("side_length_um"):
        successful = group.loc[group["status"] == "completed"]
        failed = group.loc[group["status"] != "completed"]
        if successful.empty:
            continue
        size_summaries.append(
            {
                "side_length_um": float(side),
                "grain_count": int(successful["grain_count"].iloc[0]),
                "successful_realizations": int(len(successful)),
                "failed_realizations": int(len(failed)),
                "objective_mean": float(successful["objective"].mean()),
                "objective_standard_deviation": float(
                    successful["objective"].std(ddof=0)
                ),
                "objective_maximum": float(successful["objective"].max()),
                "maximum_area_relative_error": float(
                    successful["area_relative_error"].max()
                ),
            }
        )
    failed_tasks = [
        {
            "side_length_um": float(row.side_length_um),
            "seed": int(row.seed),
            "error": str(row.error),
        }
        for row in table.loc[table["status"] != "completed"].itertuples()
    ]
    result = {
        "dataset": dataset_key,
        "phase": phase,
        "validation_seeds": list(seeds),
        "direction_target_used": include_direction,
        "morphology_time_limit_s": morphology_time_limit_s,
        "parameters": asdict(parameters),
        "side_length_summaries": size_summaries,
        "failed_tasks": failed_tasks,
        "selection_rule": (
            "Select the smallest RVE whose ensemble mean discrepancy and realization "
            "standard deviation are acceptably close to the next larger RVE; do not "
            "select the single closest realization."
        ),
    }
    write_json(output_directory / "rve_representativeness.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cleaned_csv", type=Path)
    parser.add_argument("best_parameters_json", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase", default="all")
    parser.add_argument("--side", type=float, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-direction", action="store_true")
    parser.add_argument("--input-size-shape-rank-correlation", type=float)
    parser.add_argument("--diameter-spread", type=float)
    parser.add_argument("--directional-ratio", type=float)
    args = parser.parse_args()
    payload = json.loads(args.best_parameters_json.read_text(encoding="utf-8"))
    raw = payload["best_parameters"]
    allowed = {field.name for field in fields(NeperMorphologyParameters)}
    parameters = NeperMorphologyParameters(
        **{key: value for key, value in raw.items() if key in allowed}
    )
    if any(
        value is not None
        for value in (
            args.input_size_shape_rank_correlation,
            args.diameter_spread,
            args.directional_ratio,
        )
    ):
        overrides = {}
        if args.input_size_shape_rank_correlation is not None:
            overrides["input_size_shape_rank_correlation"] = (
                args.input_size_shape_rank_correlation
            )
        if args.diameter_spread is not None:
            overrides["diameter_log_spread_scale"] = args.diameter_spread
        if args.directional_ratio is not None:
            overrides["directional_ratio_x_over_y"] = args.directional_ratio
        parameters = NeperMorphologyParameters(**{**asdict(parameters), **overrides})
    result = verify_representativeness(
        pd.read_csv(args.cleaned_csv),
        args.output,
        dataset_key=args.dataset,
        phase=args.phase,
        parameters=parameters,
        side_lengths_um=tuple(args.side) if args.side else RVE_SIDE_LENGTHS_UM,
        include_direction=not args.exclude_direction,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
