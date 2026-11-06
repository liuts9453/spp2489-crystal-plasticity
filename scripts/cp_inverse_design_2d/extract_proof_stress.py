#!/usr/bin/env python3
"""Extract proof/flow properties and physical checks from one Neper CP run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file


def read_history(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    if not rows:
        raise RuntimeError(f"Empty CSV history: {path}")
    required = {
        "time",
        "F11_bar",
        "F12_bar",
        "F21_bar",
        "F22_bar",
        "P11_bar",
        "P12_bar",
        "P21_bar",
        "P22_bar",
        "accumulated_slip_bar",
        "accumulated_slip_max",
        "slip_resistance_mean_bar",
        "det_fp_min",
        "det_fp_max",
        "plastic_dissipation_bar",
        "local_power_bar",
        "nonlinear_iterations",
        "step",
    }
    missing = sorted(required - rows[0].keys())
    if missing:
        raise RuntimeError(f"Missing CSV columns in {path}: {missing}")
    return rows


def physical_history(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    """Return initialized loading states and synthesize the known undeformed state.

    The generated MOOSE AuxVariables are zero in the INITIAL output row because
    their kernels execute at TIMESTEP_END. That row must not be interpreted as
    F=0 in a stiffness fit or discrete Hill--Mandel calculation.
    """

    initialized = [row for row in rows if row["time"] > 0.0 and row["F11_bar"] > 0.5]
    if not initialized:
        raise RuntimeError("No initialized loading states in CSV history")
    initial = {key: 0.0 for key in initialized[0]}
    initial.update(
        {
            "time": 0.0,
            "F11_bar": 1.0,
            "F22_bar": 1.0,
            "det_fp_min": 1.0,
            "det_fp_max": 1.0,
            "slip_resistance_mean_bar": 80.0,
        }
    )
    return [initial, *initialized]


def _exact_row(
    rows: list[dict[str, float]], time_value: float, tolerance: float = 2.0e-11
) -> dict[str, float]:
    matches = [row for row in rows if abs(row["time"] - time_value) <= tolerance]
    if len(matches) != 1:
        nearest = min(rows, key=lambda row: abs(row["time"] - time_value))
        raise RuntimeError(
            f"Expected one synchronized row at t={time_value}, found {len(matches)}; "
            f"nearest={nearest['time']}"
        )
    return matches[0]


def _linear_fit(
    points: list[dict[str, float]], maximum_strain: float
) -> dict[str, float]:
    selected = [row for row in points if 0.0 <= row["F11_bar"] - 1.0 <= maximum_strain]
    if len(selected) < 3:
        raise RuntimeError(
            f"Need at least three initialized points through strain={maximum_strain:g}; "
            f"found {len(selected)}"
        )
    strain = np.asarray([row["F11_bar"] - 1.0 for row in selected])
    stress = np.asarray([row["P11_bar"] for row in selected])
    matrix = np.column_stack((strain, np.ones_like(strain)))
    slope, intercept = np.linalg.lstsq(matrix, stress, rcond=None)[0]
    prediction = slope * strain + intercept
    residual = stress - prediction
    rmse = float(np.sqrt(np.mean(residual * residual)))
    denominator = max(float(np.ptp(stress)), abs(float(stress[-1])), 1.0)
    ss_total = float(np.sum((stress - np.mean(stress)) ** 2))
    r_squared = 1.0 - float(np.sum(residual * residual)) / max(ss_total, 1.0e-30)
    return {
        "maximum_fit_strain": maximum_strain,
        "point_count": len(selected),
        "slope_MPa": float(slope),
        "intercept_MPa": float(intercept),
        "rmse_MPa": rmse,
        "normalized_rmse": rmse / denominator,
        "r_squared": r_squared,
    }


def extract_offset_proof(rows: list[dict[str, float]]) -> dict[str, object]:
    initial = _linear_fit(rows, 0.001)
    half = _linear_fit(rows, 0.0005)
    slope = float(initial["slope_MPa"])
    slope_stability = abs(slope - float(half["slope_MPa"])) / abs(slope)
    previous: tuple[dict[str, float], float] | None = None
    crossing: dict[str, float] | None = None
    for row in rows:
        strain = row["F11_bar"] - 1.0
        if strain < 0.0:
            continue
        difference = row["P11_bar"] - slope * (strain - 0.002)
        if (
            previous is not None
            and previous[1] >= 0.0
            and difference <= 0.0
            and strain > 0.002
        ):
            old, old_difference = previous
            denominator = old_difference - difference
            fraction = old_difference / denominator if denominator != 0.0 else 0.0
            old_strain = old["F11_bar"] - 1.0
            proof_strain = old_strain + fraction * (strain - old_strain)
            proof_stress = old["P11_bar"] + fraction * (row["P11_bar"] - old["P11_bar"])
            crossing = {
                "proof_strain": proof_strain,
                "proof_F11": 1.0 + proof_strain,
                "proof_stress_MPa": proof_stress,
                "bracket_lower_strain": old_strain,
                "bracket_upper_strain": strain,
                "bracket_width": strain - old_strain,
                "interpolation_fraction": fraction,
            }
            break
        previous = (row, difference)
    return {
        "definition": "0.2% offset proof stress under the prescribed macroscopic plane-strain loading condition",
        "initial_fit": initial,
        "half_window_fit": half,
        "relative_slope_window_difference": slope_stability,
        "initial_stress_MPa": rows[0]["P11_bar"],
        "intersection_found": crossing is not None,
        "intersection": crossing,
    }


def _decode_names(array: np.ndarray) -> list[str]:
    return [b"".join(row).decode("utf-8").rstrip("\x00 ") for row in np.asarray(array)]


def _periodic_error(
    x_values: np.ndarray, y_values: np.ndarray, values: np.ndarray, axis: str
) -> tuple[int, int, float]:
    if axis == "x":
        first = np.flatnonzero(np.isclose(x_values, 0.0, atol=1.0e-8))
        second = np.flatnonzero(np.isclose(x_values, 200.0, atol=1.0e-8))
        first = first[np.argsort(y_values[first])]
        second = second[np.argsort(y_values[second])]
    else:
        first = np.flatnonzero(np.isclose(y_values, 0.0, atol=1.0e-8))
        second = np.flatnonzero(np.isclose(y_values, 200.0, atol=1.0e-8))
        first = first[np.argsort(x_values[first])]
        second = second[np.argsort(x_values[second])]
    if len(first) != len(second):
        return len(first), len(second), math.inf
    return (
        len(first),
        len(second),
        float(np.max(np.abs(values[second] - values[first]))),
    )


def exodus_metrics(path: Path) -> dict[str, object]:
    required_element = {
        "grain_id",
        "euler_angle_1",
        "euler_angle_2",
        "euler_angle_3",
        "F11",
        "F22",
        "P11",
        "P22",
        "von_mises",
        "accumulated_slip",
        "slip_resistance_mean",
        "active_slip_systems",
        "fp_11",
        "fp_12",
        "fp_21",
        "fp_22",
        "fp_33",
    }
    required_nodal = {
        "disp_x",
        "disp_y",
        "total_displacement_x",
        "total_displacement_y",
        "total_displacement_magnitude",
    }
    with netcdf_file(path, "r", mmap=False) as database:
        x_values = np.asarray(database.variables["coordx"].data).copy()
        y_values = np.asarray(database.variables["coordy"].data).copy()
        block_ids = [
            int(value) for value in np.asarray(database.variables["eb_prop1"].data)
        ]
        nodal_names = _decode_names(database.variables["name_nod_var"].data)
        element_names = _decode_names(database.variables["name_elem_var"].data)
        periodic: list[dict[str, object]] = []
        for variable_name in ("disp_x", "disp_y"):
            index = nodal_names.index(variable_name) + 1
            values = np.asarray(
                database.variables[f"vals_nod_var{index}"].data[-1]
            ).copy()
            for axis in ("x", "y"):
                first, second, error = _periodic_error(x_values, y_values, values, axis)
                periodic.append(
                    {
                        "variable": variable_name,
                        "axis": axis,
                        "minus_nodes": first,
                        "plus_nodes": second,
                        "maximum_error": error,
                    }
                )
        statistics: dict[str, dict[str, float]] = {}
        grain_means: dict[str, dict[str, float]] = {}
        for variable_name in (
            "P11",
            "F11",
            "accumulated_slip",
            "slip_resistance_mean",
            "active_slip_systems",
        ):
            index = element_names.index(variable_name) + 1
            all_values: list[float] = []
            per_grain: dict[str, float] = {}
            for block_number, block_id in enumerate(block_ids, start=1):
                variable = f"vals_elem_var{index}eb{block_number}"
                if variable not in database.variables:
                    continue
                values = np.asarray(database.variables[variable].data[-1]).copy()
                all_values.extend(values.tolist())
                per_grain[str(block_id)] = float(np.mean(values))
            array = np.asarray(all_values, dtype=float)
            statistics[variable_name] = {
                "minimum": float(array.min()),
                "maximum": float(array.max()),
                "mean": float(array.mean()),
                "range": float(np.ptp(array)),
            }
            grain_means[variable_name] = per_grain
    return {
        "readable": True,
        "block_ids": block_ids,
        "node_count": len(x_values),
        "periodic_pairs": periodic,
        "maximum_periodicity_error": max(item["maximum_error"] for item in periodic),
        "missing_required_fields": sorted(
            (required_element - set(element_names))
            | (required_nodal - set(nodal_names))
        ),
        "element_statistics": statistics,
        "grain_means": grain_means,
    }


def extract_properties(csv_path: Path, exodus_path: Path) -> dict[str, object]:
    rows = physical_history(read_history(csv_path))
    one_percent = _exact_row(rows, 0.5)
    final = _exact_row(rows, 1.0)
    proof = extract_offset_proof(rows)
    if not proof["intersection_found"]:
        raise RuntimeError(f"No valid 0.2% offset intersection in {csv_path}")
    proof_stress = float(proof["intersection"]["proof_stress_MPa"])
    hill_mandel: list[float] = []
    for old, new in zip(rows[:-1], rows[1:]):
        time_increment = new["time"] - old["time"]
        if time_increment <= 0.0:
            continue
        macro_power = sum(
            new[f"P{i}{j}_bar"]
            * (new[f"F{i}{j}_bar"] - old[f"F{i}{j}_bar"])
            / time_increment
            for i, j in ((1, 1), (1, 2), (2, 1), (2, 2))
        )
        local_power = new["local_power_bar"]
        hill_mandel.append(
            abs(local_power - macro_power)
            / max(abs(local_power), abs(macro_power), 1.0)
        )
    dissipation_integral = float(
        np.trapezoid(
            [row["plastic_dissipation_bar"] for row in rows],
            [row["time"] for row in rows],
        )
    )
    exodus = exodus_metrics(exodus_path)
    values = np.asarray(
        [[value for value in row.values()] for row in rows], dtype=float
    )
    result: dict[str, object] = {
        "proof_stress_MPa": proof_stress,
        "proof_extraction": proof,
        "flow_stress_2pct_MPa": final["P11_bar"],
        "P11_1pct_MPa": one_percent["P11_bar"],
        "initial_effective_stiffness_MPa": proof["initial_fit"]["slope_MPa"],
        "secant_slope_1_to_2pct_MPa": (final["P11_bar"] - one_percent["P11_bar"])
        / 0.01,
        "F22_final": final["F22_bar"],
        "maximum_abs_P22_MPa": max(abs(row["P22_bar"]) for row in rows[1:]),
        "maximum_abs_macro_shear": max(
            max(abs(row["F12_bar"]), abs(row["F21_bar"])) for row in rows[1:]
        ),
        "mean_accumulated_slip": final["accumulated_slip_bar"],
        "maximum_accumulated_slip": final["accumulated_slip_max"],
        "mean_slip_resistance_MPa": final["slip_resistance_mean_bar"],
        "minimum_det_Fp": final["det_fp_min"],
        "maximum_det_Fp": final["det_fp_max"],
        "final_plastic_dissipation_rate": final["plastic_dissipation_bar"],
        "integrated_plastic_dissipation": dissipation_integral,
        "hill_mandel_relative_residual_max": max(hill_mandel, default=math.nan),
        "accepted_increments": int(round(final["step"])),
        "accepted_newton_iterations": int(
            round(sum(row["nonlinear_iterations"] for row in rows[1:]))
        ),
        "constitutive_timing": {
            "update_calls": int(round(final["cp_update_calls"])),
            "update_seconds_inclusive_mpi_sum": final["cp_update_seconds"],
            "residual_seconds_inclusive_mpi_sum": final["cp_residual_seconds"],
            "jacobian_seconds_inclusive_mpi_sum": final["cp_jacobian_seconds"],
            "linear_solve_seconds_inclusive_mpi_sum": final["cp_linear_solve_seconds"],
            "line_search_seconds_inclusive_mpi_sum": final["cp_line_search_seconds"],
            "tangent_seconds_inclusive_mpi_sum": final["cp_tangent_seconds"],
            "local_iterations_total": int(round(final["cp_local_iterations_total"])),
            "substeps_total": int(round(final["cp_substeps_total"])),
            "maximum_local_residual": final["cp_local_residual_max"],
        },
        "history_is_finite": bool(np.isfinite(values).all()),
        "full_curve": [
            {
                "time": row["time"],
                "strain": row["F11_bar"] - 1.0,
                "F11": row["F11_bar"],
                "F22": row["F22_bar"],
                "P11_MPa": row["P11_bar"],
                "P22_MPa": row["P22_bar"],
            }
            for row in rows
        ],
        "exodus": exodus,
    }
    local = exodus["element_statistics"]
    result.update(
        {
            "minimum_local_slip_resistance_MPa": local["slip_resistance_mean"][
                "minimum"
            ],
            "maximum_local_P11_MPa": local["P11"]["maximum"],
            "minimum_local_P11_MPa": local["P11"]["minimum"],
            "maximum_active_slip_systems": local["active_slip_systems"]["maximum"],
        }
    )
    checks = {
        "reached_physical_end_time": abs(final["time"] - 1.0) <= 2.0e-11,
        "exact_synchronized_targets": abs(one_percent["F11_bar"] - 1.01) <= 2.0e-7
        and abs(final["F11_bar"] - 1.02) <= 2.0e-7,
        "proof_intersection": proof["intersection_found"],
        "initial_fit": proof["initial_fit"]["normalized_rmse"] < 5.0e-3,
        "P22_constraint": result["maximum_abs_P22_MPa"] < 1.0e-3,
        "macro_shear_constraints": result["maximum_abs_macro_shear"] < 1.0e-10,
        "plasticity_active": result["maximum_accumulated_slip"] > 1.0e-4,
        "finite_state": result["history_is_finite"],
        "positive_slip_resistance": result["minimum_local_slip_resistance_MPa"] > 0.0,
        "plastic_volume_error": max(
            abs(result["minimum_det_Fp"] - 1.0), abs(result["maximum_det_Fp"] - 1.0)
        )
        < 5.0e-4,
        "nonnegative_plastic_dissipation": result["integrated_plastic_dissipation"]
        >= -1.0e-10,
        "hill_mandel": result["hill_mandel_relative_residual_max"] < 5.0e-4,
        "periodic_BCs": exodus["maximum_periodicity_error"] < 1.0e-8,
        "twenty_grain_blocks": exodus["block_ids"] == list(range(1, 21)),
        "required_exodus_fields": not exodus["missing_required_fields"],
    }
    result["checks"] = checks
    result["all_checks_passed"] = all(checks.values())
    if not result["all_checks_passed"]:
        raise RuntimeError(f"Physical validation failed for {csv_path}: {checks}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("exodus", type=Path)
    args = parser.parse_args()
    print(json.dumps(extract_properties(args.csv, args.exodus), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
