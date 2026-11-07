#!/usr/bin/env python3
"""Deterministically calibrate Neper morphology parameters on fixed seeds."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from common import CALIBRATION_SEEDS, VALIDATION_SEEDS, write_json
from compare_experimental_neper import comparison_metrics
from generate_ebsd_matched_neper import (
    NeperMorphologyParameters,
    choose_grain_count,
    generate_rve,
)


def _parameter_tag(values: np.ndarray, aspect_ratio_proxy_scale: float | None) -> str:
    spread, ratio = values
    proxy = (
        "legacy"
        if aspect_ratio_proxy_scale is None
        else f"joint_{aspect_ratio_proxy_scale:.8f}"
    )
    return f"spread_{spread:.8f}_ratio_{ratio:.8f}_{proxy}".replace(".", "p")


def _parallel_workers(task_count: int) -> int:
    """Return a conservative worker count for independent Neper processes."""
    requested = int(os.environ.get("SPP2489_NEPER_JOBS", "4"))
    if requested < 1:
        raise ValueError("SPP2489_NEPER_JOBS must be a positive integer")
    return min(task_count, requested)


def calibrate(
    experimental: pd.DataFrame,
    output_directory: Path,
    *,
    dataset_key: str,
    phase: str,
    side_length_um: float = 200.0,
    calibration_seeds: tuple[int, ...] = CALIBRATION_SEEDS,
    validation_seeds: tuple[int, ...] = VALIDATION_SEEDS,
    maximum_evaluations: int = 24,
    include_direction: bool = True,
    aspect_ratio_proxy_scale: float | None = 1.0,
    input_size_shape_rank_correlation: float | None = None,
    candidate_points: tuple[tuple[float, float], ...] | None = None,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    target = (
        experimental
        if phase.lower() == "all"
        else experimental.loc[experimental["phase"].astype(str) == str(phase)]
    ).copy()
    if target.empty:
        raise ValueError(f"No target grains for phase {phase!r}")
    count = choose_grain_count(target, side_length_um)
    history: list[dict[str, object]] = []
    cache: dict[tuple[float, float], float] = {}
    candidate_timeout_s = float(
        os.environ.get("SPP2489_NEPER_CALIBRATION_TIMEOUT_S", "360")
    )
    if candidate_timeout_s <= 0.0:
        raise ValueError("SPP2489_NEPER_CALIBRATION_TIMEOUT_S must be positive")
    validation_timeout_s = float(
        os.environ.get("SPP2489_NEPER_VALIDATION_TIMEOUT_S", "1800")
    )
    if validation_timeout_s <= 0.0:
        raise ValueError("SPP2489_NEPER_VALIDATION_TIMEOUT_S must be positive")
    morphology_time_limit_s = float(
        os.environ.get("SPP2489_NEPER_MORPHOLOGY_TIME_S", "90")
    )
    if morphology_time_limit_s <= 0.0:
        raise ValueError("SPP2489_NEPER_MORPHOLOGY_TIME_S must be positive")
    # With a genuine per-grain direction target, ratios below and above one
    # carry distinct x/y information.  The supplied SPP2489 alpha column is a
    # section label, so only the unsigned elongation magnitude is identifiable.
    # In that case choose r_x/r_y >= 1 as a documented axis-label convention.
    ratio_lower_bound = 0.4 if include_direction else 1.0

    def objective(values: np.ndarray) -> float:
        clipped = np.clip(values, (0.4, ratio_lower_bound), (2.0, 3.0))
        key = tuple(float(f"{value:.8f}") for value in clipped)
        if key in cache:
            return cache[key]
        parameters = NeperMorphologyParameters(
            diameter_log_spread_scale=key[0],
            directional_ratio_x_over_y=key[1],
            aspect_ratio_proxy_scale=aspect_ratio_proxy_scale,
            input_size_shape_rank_correlation=input_size_shape_rank_correlation,
        )
        candidate_directory = (
            output_directory
            / "calibration_candidates"
            / _parameter_tag(clipped, aspect_ratio_proxy_scale)
        )

        def evaluate_seed(seed: int) -> dict[str, object]:
            run_directory = candidate_directory / f"seed_{seed}"
            try:
                generate_rve(
                    experimental,
                    run_directory,
                    dataset_key=dataset_key,
                    phase=phase,
                    side_length_um=side_length_um,
                    seed=seed,
                    parameters=parameters,
                    grain_count=count,
                    generation_timeout_s=candidate_timeout_s,
                    morphology_time_limit_s=morphology_time_limit_s,
                )
                synthetic = pd.read_csv(run_directory / "synthetic_grains.csv")
                return {
                    "seed": seed,
                    "metrics": comparison_metrics(
                        target, synthetic, include_direction=include_direction
                    ),
                }
            except (RuntimeError, OSError, ValueError) as error:
                return {"seed": seed, "error": str(error)}

        # Every seed writes to an independent directory.  Running those Neper
        # processes concurrently changes neither the prescribed seeds nor the
        # ensemble-average objective, while using the available CPU cores.
        with ThreadPoolExecutor(
            max_workers=_parallel_workers(len(calibration_seeds))
        ) as executor:
            seed_results = list(executor.map(evaluate_seed, calibration_seeds))
        failures = [result for result in seed_results if "error" in result]
        seed_metrics = [
            result["metrics"] for result in seed_results if "metrics" in result
        ]
        value = (
            1.0e6
            if failures
            else float(np.mean([metric["objective"] for metric in seed_metrics]))
        )
        row: dict[str, object] = {
            "evaluation": len(history) + 1,
            "diameter_log_spread_scale": key[0],
            "directional_ratio_x_over_y": key[1],
            "objective_mean": value,
            "objective_standard_deviation": (
                ""
                if failures
                else float(np.std([metric["objective"] for metric in seed_metrics]))
            ),
            "calibration_seeds": " ".join(str(seed) for seed in calibration_seeds),
            "status": "infeasible_generation" if failures else "completed",
            "failed_seeds": " ".join(str(item["seed"]) for item in failures),
            "failure_messages": " | ".join(str(item["error"]) for item in failures),
        }
        for component in (
            "diameter",
            "aspect",
            "direction",
            "correlation",
            "odf",
            "size_orientation",
        ):
            row[f"{component}_mean"] = (
                ""
                if failures
                else float(
                    np.mean(
                        [
                            metric["normalized_components"][component]
                            for metric in seed_metrics
                        ]
                    )
                )
            )
        history.append(row)
        cache[key] = value
        return value

    if candidate_points:
        for point in candidate_points:
            objective(np.asarray(point, dtype=float))
        optimizer_name = "deterministic reviewed candidate matrix"
        optimizer_success = all(row["status"] == "completed" for row in history)
        optimizer_message = (
            "All reviewed candidates completed"
            if optimizer_success
            else "Candidate matrix completed with explicitly recorded infeasible points"
        )
    else:
        result = optimize.minimize(
            objective,
            x0=np.array([1.0, max(1.25, ratio_lower_bound)]),
            method="Powell",
            bounds=((0.4, 2.0), (ratio_lower_bound, 3.0)),
            options={"maxfev": maximum_evaluations, "xtol": 0.025, "ftol": 0.005},
        )
        optimizer_name = "deterministic bounded Powell search"
        optimizer_success = bool(result.success)
        optimizer_message = str(result.message)
    history_path = output_directory / "neper_calibration_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    best_row = min(history, key=lambda row: float(row["objective_mean"]))
    best = NeperMorphologyParameters(
        diameter_log_spread_scale=float(best_row["diameter_log_spread_scale"]),
        directional_ratio_x_over_y=float(best_row["directional_ratio_x_over_y"]),
        aspect_ratio_proxy_scale=aspect_ratio_proxy_scale,
        input_size_shape_rank_correlation=input_size_shape_rank_correlation,
    )

    def validate_seed(seed: int) -> dict[str, object]:
        run_directory = output_directory / "validation_realizations" / f"seed_{seed}"
        generate_rve(
            experimental,
            run_directory,
            dataset_key=dataset_key,
            phase=phase,
            side_length_um=side_length_um,
            seed=seed,
            parameters=best,
            grain_count=count,
            generation_timeout_s=validation_timeout_s,
            morphology_time_limit_s=morphology_time_limit_s,
        )
        synthetic = pd.read_csv(run_directory / "synthetic_grains.csv")
        return {
            "seed": seed,
            **comparison_metrics(
                target, synthetic, include_direction=include_direction
            ),
        }

    with ThreadPoolExecutor(
        max_workers=_parallel_workers(len(validation_seeds))
    ) as executor:
        validation_metrics = list(executor.map(validate_seed, validation_seeds))
    summary = {
        "dataset": dataset_key,
        "phase": phase,
        "side_length_um": side_length_um,
        "grain_count": count,
        "optimizer": optimizer_name,
        "optimizer_success": optimizer_success,
        "optimizer_message": optimizer_message,
        "evaluation_count": len(history),
        "calibration_seeds": list(calibration_seeds),
        "validation_seeds_not_used_in_fit": list(validation_seeds),
        "direction_target_used": include_direction,
        "joint_size_shape_target_used": aspect_ratio_proxy_scale is not None,
        "aspect_ratio_proxy_scale": aspect_ratio_proxy_scale,
        "input_size_shape_rank_correlation": input_size_shape_rank_correlation,
        "morphology_time_limit_s": morphology_time_limit_s,
        "directional_ratio_bounds": [ratio_lower_bound, 3.0],
        "directional_ratio_interpretation": (
            "physical x/y direction calibrated from per-grain direction data"
            if include_direction
            else "unsigned major/minor elongation magnitude; x/y assignment is a convention"
        ),
        "best_parameters": asdict(best),
        "best_calibration_objective": float(best_row["objective_mean"]),
        "validation": validation_metrics,
        "validation_objective_mean": float(
            np.mean([item["objective"] for item in validation_metrics])
        ),
        "validation_objective_standard_deviation": float(
            np.std([item["objective"] for item in validation_metrics])
        ),
        "limitations": [
            "The experimental fitted-ellipse AR is mapped to a cell-wise ellipse-circularity proxy because the CSVs contain no measured grain perimeter.",
            "Neper's global aspratio controls directional elongation; achieved polygon AR is measured independently and is not assumed equal to the proxy.",
            "The empirical diameq target can retain a long tail or mixture, while the achieved polygon distribution remains an optimization result.",
            "Size--AR pairing is transmitted through paired cell-wise targets and then independently validated on achieved polygons.",
        ],
    }
    write_json(output_directory / "neper_best_parameters.json", summary)
    write_json(output_directory / "neper_statistical_validation.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cleaned_csv", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase", default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", type=float, default=200.0)
    parser.add_argument("--maximum-evaluations", type=int, default=24)
    parser.add_argument("--calibration-seed", type=int, action="append")
    parser.add_argument("--validation-seed", type=int, action="append")
    parser.add_argument("--aspect-ratio-proxy-scale", type=float, default=1.0)
    parser.add_argument("--input-size-shape-rank-correlation", type=float)
    parser.add_argument("--disable-joint-shape-target", action="store_true")
    parser.add_argument(
        "--candidate",
        action="append",
        metavar="SPREAD,RATIO",
        help="Reviewed deterministic candidate; may be repeated instead of Powell",
    )
    parser.add_argument("--exclude-direction", action="store_true")
    args = parser.parse_args()
    candidates = None
    if args.candidate:
        parsed = []
        for value in args.candidate:
            fields = value.split(",")
            if len(fields) != 2:
                parser.error(f"Invalid --candidate {value!r}; expected SPREAD,RATIO")
            parsed.append((float(fields[0]), float(fields[1])))
        candidates = tuple(parsed)
    summary = calibrate(
        pd.read_csv(args.cleaned_csv),
        args.output,
        dataset_key=args.dataset,
        phase=args.phase,
        side_length_um=args.side,
        calibration_seeds=(
            tuple(args.calibration_seed) if args.calibration_seed else CALIBRATION_SEEDS
        ),
        validation_seeds=(
            tuple(args.validation_seed) if args.validation_seed else VALIDATION_SEEDS
        ),
        maximum_evaluations=args.maximum_evaluations,
        include_direction=not args.exclude_direction,
        aspect_ratio_proxy_scale=(
            None if args.disable_joint_shape_target else args.aspect_ratio_proxy_scale
        ),
        input_size_shape_rank_correlation=args.input_size_shape_rank_correlation,
        candidate_points=candidates,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
