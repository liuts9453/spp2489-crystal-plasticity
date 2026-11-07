#!/usr/bin/env python3
"""Quantify mesh, time-integration, and realization uncertainty for morphology."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ID2_SCRIPT_DIR = SCRIPT_DIR.parent / "cp_inverse_design_2d"
for directory in (SCRIPT_DIR, ID2_SCRIPT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common import PROTOCOLS, write_json  # noqa: E402
from generate_morphology import generate_design  # noqa: E402
from morphology_common import (  # noqa: E402
    BASE_GEOMETRY_SEED,
    ROBUSTNESS_GEOMETRY_SEEDS,
    SENSITIVITY,
    VERIFICATION,
    baseline_texture,
    geometry_directory,
)
from texture_morphology_forward_solver import (  # noqa: E402
    run_texture_morphology_forward,
)


MESH_LEVELS = (0.70, 1.00, 1.50)
TIME_LEVELS = (0.70, 1.50)
REALIZATION_LEVELS = (1.00, 1.50)
PROPERTIES = ("proof_stress_MPa", "flow_stress_2pct_MPa")


def _read_coarse_sensitivity() -> dict[float, dict[str, object]]:
    path = SENSITIVITY.parent / "morphology_sensitivity.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 5:
        raise RuntimeError("Run the complete five-level sensitivity study first")
    return {float(row["input_a"]): row for row in rows}


def _compact(result: dict[str, object]) -> dict[str, object]:
    statistics = result["area_weighted_orientation_statistics"]
    dyad = np.asarray(statistics["area_weighted_sample_x_crystal_dyad"])
    return {
        "input_a": result["morphology_input_a"],
        "achieved_a": result["achieved_directional_extent_ratio_x_over_y"],
        "geometry_seed": result["geometry_seed"],
        "mesh_name": result["mesh_name"],
        "protocol": result["protocol"],
        "proof_stress_MPa": result["proof_stress_MPa"],
        "flow_stress_2pct_MPa": result["flow_stress_2pct_MPa"],
        "mean_accumulated_slip": result["mean_accumulated_slip"],
        "maximum_accumulated_slip": result["maximum_accumulated_slip"],
        "minimum_triangle_quality": result["mesh_metrics"][
            "minimum_triangle_quality"
        ],
        "maximum_aspect_ratio": result["mesh_metrics"]["maximum_aspect_ratio"],
        "area_coefficient_of_variation": result["area_coefficient_of_variation"],
        "orientation_dyad": dyad.tolist(),
        "wall_time_s": result["wall_time_s"],
        "failed_attempts": result["failed_attempts"],
        "accepted_increments": result["accepted_increments"],
        "all_checks_passed": result["all_checks_passed"],
        "run_directory": result["run_directory"],
    }


def _relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), 1.0e-30)


def run_verification(*, mpi_ranks: int = 8, reuse_cache: bool = True) -> dict[str, object]:
    eta, theta = baseline_texture()
    coarse = _read_coarse_sensitivity()
    VERIFICATION.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    fine_results: dict[float, dict[str, object]] = {}
    for a_value in MESH_LEVELS:
        generate_design(a_value, BASE_GEOMETRY_SEED, ("fine",))
        fine_results[a_value] = _compact(
            run_texture_morphology_forward(
                eta,
                theta,
                a_value,
                geometry_seed=BASE_GEOMETRY_SEED,
                mesh_name="fine",
                protocol=PROTOCOLS["adaptive_strict"],
                mpi_ranks=mpi_ranks,
                label_prefix="mesh_verification",
                reuse_cache=reuse_cache,
            )
        )
        write_json(VERIFICATION / "verification_progress.json", {"fine": fine_results})

    time_results: dict[float, dict[str, object]] = {}
    for a_value in TIME_LEVELS:
        time_results[a_value] = _compact(
            run_texture_morphology_forward(
                eta,
                theta,
                a_value,
                geometry_seed=BASE_GEOMETRY_SEED,
                mesh_name="coarse",
                protocol=PROTOCOLS["adaptive_refined"],
                mpi_ranks=mpi_ranks,
                label_prefix="time_verification",
                reuse_cache=reuse_cache,
            )
        )
        write_json(
            VERIFICATION / "verification_progress.json",
            {"fine": fine_results, "time": time_results},
        )

    realization_results: dict[int, dict[float, dict[str, object]]] = {}
    for seed in ROBUSTNESS_GEOMETRY_SEEDS:
        realization_results[seed] = {}
        for a_value in REALIZATION_LEVELS:
            generate_design(a_value, seed, ("coarse",))
            realization_results[seed][a_value] = _compact(
                run_texture_morphology_forward(
                    eta,
                    theta,
                    a_value,
                    geometry_seed=seed,
                    mesh_name="coarse",
                    protocol=PROTOCOLS["adaptive_strict"],
                    mpi_ranks=mpi_ranks,
                    label_prefix="realization_verification",
                    reuse_cache=reuse_cache,
                )
            )
            write_json(
                VERIFICATION / "verification_progress.json",
                {
                    "fine": fine_results,
                    "time": time_results,
                    "realizations": realization_results,
                },
            )

    mesh_comparisons = []
    for a_value, fine in fine_results.items():
        old = coarse[a_value]
        row: dict[str, object] = {"input_a": a_value}
        for property_name in PROPERTIES:
            coarse_value = float(old[property_name])
            fine_value = float(fine[property_name])
            row[f"coarse_{property_name}"] = coarse_value
            row[f"fine_{property_name}"] = fine_value
            row[f"absolute_change_{property_name}"] = abs(fine_value - coarse_value)
            row[f"relative_change_{property_name}"] = _relative(
                coarse_value, fine_value
            )
        mesh_comparisons.append(row)

    time_comparisons = []
    for a_value, refined in time_results.items():
        old = coarse[a_value]
        row = {"input_a": a_value}
        for property_name in PROPERTIES:
            default_value = float(old[property_name])
            refined_value = float(refined[property_name])
            row[f"default_{property_name}"] = default_value
            row[f"refined_{property_name}"] = refined_value
            row[f"absolute_change_{property_name}"] = abs(
                refined_value - default_value
            )
            row[f"relative_change_{property_name}"] = _relative(
                default_value, refined_value
            )
        time_comparisons.append(row)

    seed_2026 = {
        1.0: {
            **{key: float(coarse[1.0][key]) for key in PROPERTIES},
            "orientation_dyad": json.loads(
                (
                    geometry_directory(BASE_GEOMETRY_SEED, 1.0)
                    / "morphology_statistics.json"
                ).read_text(encoding="utf-8")
            )["orientation_statistics"]["area_weighted_sample_x_crystal_dyad"],
        },
        1.5: {
            **{key: float(coarse[1.5][key]) for key in PROPERTIES},
            "orientation_dyad": json.loads(
                (
                    geometry_directory(BASE_GEOMETRY_SEED, 1.5)
                    / "morphology_statistics.json"
                ).read_text(encoding="utf-8")
            )["orientation_statistics"]["area_weighted_sample_x_crystal_dyad"],
        },
    }
    paired = {BASE_GEOMETRY_SEED: seed_2026, **realization_results}
    realization_comparisons = []
    reference_dyad = np.asarray(seed_2026[1.0]["orientation_dyad"])
    for seed, values in paired.items():
        row = {
            "geometry_seed": seed,
            "area_weighted_texture_dyad_distance_from_seed2026_a1": {
                str(a_value): float(
                    np.linalg.norm(np.asarray(values[a_value]["orientation_dyad"]) - reference_dyad)
                )
                for a_value in REALIZATION_LEVELS
            },
        }
        for property_name in PROPERTIES:
            row[f"a1_{property_name}"] = values[1.0][property_name]
            row[f"a1p5_{property_name}"] = values[1.5][property_name]
            row[f"paired_contrast_{property_name}"] = (
                float(values[1.5][property_name])
                - float(values[1.0][property_name])
            )
        realization_comparisons.append(row)

    uncertainty: dict[str, object] = {}
    gate_by_property: dict[str, object] = {}
    for property_name in PROPERTIES:
        sensitivity_values = [float(row[property_name]) for row in coarse.values()]
        morphology_range = max(sensitivity_values) - min(sensitivity_values)
        mesh_envelope = max(
            float(row[f"absolute_change_{property_name}"])
            for row in mesh_comparisons
        )
        time_envelope = max(
            float(row[f"absolute_change_{property_name}"])
            for row in time_comparisons
        )
        numerical_envelope = max(mesh_envelope, time_envelope)
        a1_values = [float(values[1.0][property_name]) for values in paired.values()]
        realization_span = max(a1_values) - min(a1_values)
        contrasts = [
            float(row[f"paired_contrast_{property_name}"])
            for row in realization_comparisons
        ]
        same_sign = all(value > 0.0 for value in contrasts) or all(
            value < 0.0 for value in contrasts
        )
        minimum_paired_effect = min(abs(value) for value in contrasts)
        robust = bool(
            morphology_range > 2.0 * numerical_envelope
            and same_sign
            and minimum_paired_effect > 2.0 * numerical_envelope
        )
        uncertainty[property_name] = {
            "five_level_morphology_range_MPa": morphology_range,
            "mesh_envelope_MPa": mesh_envelope,
            "time_integration_envelope_MPa": time_envelope,
            "numerical_envelope_MPa": numerical_envelope,
            "a1_realization_span_MPa": realization_span,
            "paired_a1p5_minus_a1_contrasts_MPa": contrasts,
            "paired_contrasts_same_sign": same_sign,
            "minimum_absolute_paired_effect_MPa": minimum_paired_effect,
        }
        gate_by_property[property_name] = {
            "criterion": (
                "five-level range > 2*numerical envelope AND all paired a=1.5-a=1 "
                "contrasts have one sign AND minimum paired effect > 2*numerical envelope"
            ),
            "passes": robust,
        }

    enter_bo = any(item["passes"] for item in gate_by_property.values())
    summary = {
        "study": "morphology numerical and realization uncertainty",
        "texture": {"eta": eta, "theta_deg": theta},
        "mpi_ranks": mpi_ranks,
        "fine_results": list(fine_results.values()),
        "time_refined_results": list(time_results.values()),
        "realization_results": {
            str(seed): list(values.values())
            for seed, values in realization_results.items()
        },
        "mesh_comparisons": mesh_comparisons,
        "time_comparisons": time_comparisons,
        "realization_comparisons": realization_comparisons,
        "uncertainty": uncertainty,
        "scientific_gate": {
            "predeclared_rule": (
                "Enter 3-variable BO only if at least one macroscopic target has a reproducible "
                "paired morphology effect clearly above mesh/time error."
            ),
            "by_property": gate_by_property,
            "enter_three_variable_bo": enter_bo,
            "decision": (
                "execute three-variable BO"
                if enter_bo
                else "do not execute three-variable BO; morphology is not a resolved additional macroscopic design variable"
            ),
        },
        "all_checks_passed": all(
            item["all_checks_passed"]
            for item in [
                *fine_results.values(),
                *time_results.values(),
                *(
                    result
                    for values in realization_results.values()
                    for result in values.values()
                ),
            ]
        ),
        "wall_time_s": time.perf_counter() - started,
    }
    write_json(VERIFICATION / "verification_summary.json", summary)
    sensitivity_summary_path = SENSITIVITY.parent / "morphology_sensitivity_summary.json"
    sensitivity_summary = json.loads(sensitivity_summary_path.read_text(encoding="utf-8"))
    sensitivity_summary["decision"] = summary["scientific_gate"]
    sensitivity_summary["uncertainty"] = uncertainty
    write_json(sensitivity_summary_path, sensitivity_summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpi-ranks", type=int, default=8)
    parser.add_argument("--no-reuse", action="store_true")
    args = parser.parse_args()
    summary = run_verification(
        mpi_ranks=args.mpi_ranks, reuse_cache=not args.no_reuse
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
