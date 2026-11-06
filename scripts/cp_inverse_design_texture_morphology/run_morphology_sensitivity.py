#!/usr/bin/env python3
"""Run and summarize the five-level FCC-CP morphology sensitivity study."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ID2_SCRIPT_DIR = SCRIPT_DIR.parent / "cp_inverse_design_2d"
for directory in (SCRIPT_DIR, ID2_SCRIPT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common import PROTOCOLS, write_json  # noqa: E402
from generate_morphology import generate_design  # noqa: E402
from morphology_common import (  # noqa: E402
    BASE_GEOMETRY_SEED,
    DEFAULT_PROTOCOL,
    MORPHOLOGY_LEVELS,
    SENSITIVITY,
    baseline_texture,
    geometry_directory,
    mesh_metrics_path,
)
from texture_morphology_forward_solver import (  # noqa: E402
    run_texture_morphology_forward,
)


SUMMARY_COLUMNS = (
    "input_a",
    "achieved_a",
    "geometry_seed",
    "mesh_name",
    "eta",
    "theta_deg",
    "proof_stress_MPa",
    "flow_stress_2pct_MPa",
    "initial_effective_stiffness_MPa",
    "P11_1pct_MPa",
    "secant_slope_1_to_2pct_MPa",
    "mean_accumulated_slip",
    "maximum_accumulated_slip",
    "local_P11_range_MPa",
    "integrated_plastic_dissipation",
    "wall_time_s",
    "accepted_increments",
    "failed_attempts",
    "total_newton_iterations",
    "minimum_triangle_quality",
    "maximum_aspect_ratio",
    "area_coefficient_of_variation",
    "maximum_abs_P22_MPa",
    "maximum_abs_macro_shear",
    "hill_mandel_relative_residual_max",
    "maximum_periodicity_error",
    "converged",
    "all_checks_passed",
    "run_directory",
    "csv_path",
    "exodus_path",
)


def _summary_row(result: dict[str, object]) -> dict[str, object]:
    exodus = result.get("exodus", {})
    element_statistics = exodus.get("element_statistics", {})
    p11 = element_statistics.get("P11", {})
    metrics = result["mesh_metrics"]
    return {
        "input_a": result["morphology_input_a"],
        "achieved_a": result["achieved_directional_extent_ratio_x_over_y"],
        "geometry_seed": result["geometry_seed"],
        "mesh_name": result["mesh_name"],
        "eta": result["eta"],
        "theta_deg": result["theta_deg"],
        "proof_stress_MPa": result.get("proof_stress_MPa"),
        "flow_stress_2pct_MPa": result.get("flow_stress_2pct_MPa"),
        "initial_effective_stiffness_MPa": result.get(
            "initial_effective_stiffness_MPa"
        ),
        "P11_1pct_MPa": result.get("P11_1pct_MPa"),
        "secant_slope_1_to_2pct_MPa": result.get(
            "secant_slope_1_to_2pct_MPa"
        ),
        "mean_accumulated_slip": result.get("mean_accumulated_slip"),
        "maximum_accumulated_slip": result.get("maximum_accumulated_slip"),
        "local_P11_range_MPa": p11.get("range"),
        "integrated_plastic_dissipation": result.get(
            "integrated_plastic_dissipation"
        ),
        "wall_time_s": result.get("wall_time_s"),
        "accepted_increments": result.get("accepted_increments"),
        "failed_attempts": result.get("failed_attempts"),
        "total_newton_iterations": result.get("total_newton_iterations"),
        "minimum_triangle_quality": metrics["minimum_triangle_quality"],
        "maximum_aspect_ratio": metrics["maximum_aspect_ratio"],
        "area_coefficient_of_variation": result[
            "area_coefficient_of_variation"
        ],
        "maximum_abs_P22_MPa": result.get("maximum_abs_P22_MPa"),
        "maximum_abs_macro_shear": result.get("maximum_abs_macro_shear"),
        "hill_mandel_relative_residual_max": result.get(
            "hill_mandel_relative_residual_max"
        ),
        "maximum_periodicity_error": exodus.get("maximum_periodicity_error"),
        "converged": result.get("converged", False),
        "all_checks_passed": result.get("all_checks_passed", False),
        "run_directory": result.get("run_directory"),
        "csv_path": result.get("csv_path"),
        "exodus_path": result.get("exodus_path"),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=SUMMARY_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_morphology_statistics(seed: int, levels: tuple[float, ...]) -> None:
    output = SENSITIVITY.parent / "morphology_statistics.csv"
    fields = (
        "input_a",
        "geometry_seed",
        "grain_count",
        "domain_area_um2",
        "area_coefficient_of_variation",
        "minimum_grain_area_um2",
        "maximum_grain_area_um2",
        "area_weighted_directional_extent_ratio_x_over_y",
        "area_weighted_principal_aspect_ratio",
        "area_weighted_principal_axis_angle_deg",
        "area_weighted_mean_circularity",
    )
    rows = []
    for value in levels:
        path = geometry_directory(seed, value) / "morphology_statistics.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({key: data[key] for key in fields})
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_study(
    *,
    geometry_seed: int = BASE_GEOMETRY_SEED,
    protocol_name: str = DEFAULT_PROTOCOL.name,
    mpi_ranks: int = 8,
    reuse_cache: bool = True,
) -> dict[str, object]:
    SENSITIVITY.mkdir(parents=True, exist_ok=True)
    eta, theta = baseline_texture()
    protocol = PROTOCOLS[protocol_name]
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for a_value in MORPHOLOGY_LEVELS:
        generate_design(a_value, geometry_seed, ("coarse",))
        result = run_texture_morphology_forward(
            eta,
            theta,
            a_value,
            geometry_seed=geometry_seed,
            mesh_name="coarse",
            protocol=protocol,
            mpi_ranks=mpi_ranks,
            label_prefix="sensitivity",
            reuse_cache=reuse_cache,
        )
        rows.append(_summary_row(result))
        _write_csv(SENSITIVITY.parent / "morphology_sensitivity.csv", rows)
        if not result.get("converged"):
            raise RuntimeError(f"FEM failed for a={a_value}: {result.get('failure_reason')}")
    baseline = next(row for row in rows if abs(float(row["input_a"]) - 1.0) < 1e-14)
    proof_values = [float(row["proof_stress_MPa"]) for row in rows]
    flow_values = [float(row["flow_stress_2pct_MPa"]) for row in rows]
    slip_values = [float(row["mean_accumulated_slip"]) for row in rows]
    summary = {
        "study": "five-level finite-strain FCC-CP morphology sensitivity",
        "geometry_seed": geometry_seed,
        "texture": {"eta": eta, "theta_deg": theta},
        "protocol": protocol_name,
        "mpi_ranks": mpi_ranks,
        "levels": list(MORPHOLOGY_LEVELS),
        "all_converged": all(bool(row["converged"]) for row in rows),
        "all_physical_checks_passed": all(
            bool(row["all_checks_passed"]) for row in rows
        ),
        "delta_proof_stress_morphology_MPa": max(proof_values) - min(proof_values),
        "delta_flow_stress_morphology_MPa": max(flow_values) - min(flow_values),
        "normalized_proof_sensitivity": (max(proof_values) - min(proof_values))
        / float(baseline["proof_stress_MPa"]),
        "normalized_flow_sensitivity": (max(flow_values) - min(flow_values))
        / float(baseline["flow_stress_2pct_MPa"]),
        "delta_mean_accumulated_slip": max(slip_values) - min(slip_values),
        "wall_time_s": time.perf_counter() - started,
        "rows": rows,
        "decision": "pending numerical and realization uncertainty study",
    }
    _write_morphology_statistics(geometry_seed, MORPHOLOGY_LEVELS)
    write_json(
        SENSITIVITY.parent / "morphology_sensitivity_summary.json", summary
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=BASE_GEOMETRY_SEED)
    parser.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL.name)
    parser.add_argument("--mpi-ranks", type=int, default=8)
    parser.add_argument("--no-reuse", action="store_true")
    args = parser.parse_args()
    summary = run_study(
        geometry_seed=args.seed,
        protocol_name=args.protocol,
        mpi_ranks=args.mpi_ranks,
        reuse_cache=not args.no_reuse,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
