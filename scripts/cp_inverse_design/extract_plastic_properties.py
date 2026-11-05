#!/usr/bin/env python3
"""Extract plastic-response objectives and physical checks from one CP RVE run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file


def _rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    if not rows:
        raise RuntimeError(f"Empty CSV history: {path}")
    required = {
        "time", "F11_bar", "F22_bar", "P11_bar", "P22_bar", "F12_bar", "F21_bar",
        "accumulated_slip_bar", "accumulated_slip_max", "slip_resistance_mean_bar",
        "det_fp_min", "det_fp_max", "plastic_dissipation_bar", "local_power_bar",
        "nonlinear_iterations", "step",
    }
    missing = sorted(required - rows[0].keys())
    if missing:
        raise RuntimeError(f"Missing CSV columns in {path}: {missing}")
    return rows


def _exact_row(rows: list[dict[str, float]], time_value: float, tolerance: float = 2.0e-11) -> dict[str, float]:
    candidates = [row for row in rows if abs(row["time"] - time_value) <= tolerance]
    if len(candidates) != 1:
        nearest = min(rows, key=lambda row: abs(row["time"] - time_value))
        raise RuntimeError(
            f"Expected one exact synchronized row at t={time_value}, found {len(candidates)}; "
            f"nearest t={nearest['time']}"
        )
    return candidates[0]


def _decode_names(array: np.ndarray) -> list[str]:
    return [b"".join(row).decode("utf-8").rstrip("\x00 ") for row in np.asarray(array)]


def _paired_error(x: np.ndarray, y: np.ndarray, values: np.ndarray, axis: str) -> float:
    if axis == "x":
        minus, plus = np.flatnonzero(np.isclose(x, x.min())), np.flatnonzero(np.isclose(x, x.max()))
        minus, plus = minus[np.argsort(y[minus])], plus[np.argsort(y[plus])]
    else:
        minus, plus = np.flatnonzero(np.isclose(y, y.min())), np.flatnonzero(np.isclose(y, y.max()))
        minus, plus = minus[np.argsort(x[minus])], plus[np.argsort(x[plus])]
    if len(minus) != len(plus):
        return math.inf
    return float(np.max(np.abs(values[plus] - values[minus])))


def exodus_metrics(path: Path) -> dict[str, object]:
    required_element = {
        "grain_id", "euler_angle_1", "euler_angle_2", "euler_angle_3", "F11", "P11",
        "von_mises", "accumulated_slip", "slip_resistance_mean", "active_slip_systems",
        "fp_11", "fp_12", "fp_21", "fp_22", "fp_33",
    }
    required_nodal = {
        "disp_x", "disp_y", "total_displacement_x", "total_displacement_y",
        "total_displacement_magnitude",
    }
    with netcdf_file(path, "r", mmap=False) as db:
        x = np.asarray(db.variables["coordx"].data).copy()
        y = np.asarray(db.variables["coordy"].data).copy()
        nodal_names = _decode_names(db.variables["name_nod_var"].data)
        element_names = _decode_names(db.variables["name_elem_var"].data)
        blocks = len(np.asarray(db.variables["eb_prop1"].data))
        periodic = []
        for name in ("disp_x", "disp_y"):
            index = nodal_names.index(name) + 1
            values = np.asarray(db.variables[f"vals_nod_var{index}"].data[-1]).copy()
            periodic.extend((_paired_error(x, y, values, "x"), _paired_error(x, y, values, "y")))
        ranges: dict[str, dict[str, float]] = {}
        for name in ("P11", "F11", "accumulated_slip", "slip_resistance_mean", "active_slip_systems"):
            index = element_names.index(name) + 1
            values: list[float] = []
            for block in range(1, blocks + 1):
                variable = f"vals_elem_var{index}eb{block}"
                if variable in db.variables:
                    values.extend(np.asarray(db.variables[variable].data[-1]).copy().tolist())
            array = np.asarray(values, dtype=float)
            ranges[name] = {
                "minimum": float(array.min()),
                "maximum": float(array.max()),
                "mean": float(array.mean()),
            }
    return {
        "readable": True,
        "missing_required_fields": sorted((required_element - set(element_names)) | (required_nodal - set(nodal_names))),
        "maximum_periodicity_error": max(periodic),
        "final_element_statistics": ranges,
    }


def extract_properties(csv_path: Path, exodus_path: Path) -> dict[str, object]:
    rows = _rows(csv_path)
    one = _exact_row(rows, 0.5)
    two = _exact_row(rows, 1.0)
    p1, p2 = one["P11_bar"], two["P11_bar"]
    hill_mandel = []
    for old, new in zip(rows[1:-1], rows[2:]):
        dt = new["time"] - old["time"]
        if dt <= 0.0:
            continue
        macro_power = sum(
            new[f"P{i}{j}_bar"] * (new[f"F{i}{j}_bar"] - old[f"F{i}{j}_bar"]) / dt
            for i, j in ((1, 1), (1, 2), (2, 1), (2, 2))
        )
        local_power = new["local_power_bar"]
        hill_mandel.append(abs(local_power - macro_power) / max(abs(local_power), abs(macro_power), 1.0))
    exodus = exodus_metrics(exodus_path)
    all_values = np.asarray([[value for value in row.values()] for row in rows], dtype=float)
    result: dict[str, object] = {
        "P11_at_F11_1p01_MPa": p1,
        "P11_at_F11_1p02_MPa": p2,
        "H_sec_MPa": (p2 - p1) / 0.01,
        "time_at_1pct_s": one["time"],
        "time_at_2pct_s": two["time"],
        "F11_at_1pct": one["F11_bar"],
        "F11_at_2pct": two["F11_bar"],
        "F22_final": two["F22_bar"],
        "maximum_abs_P22_MPa": max(abs(row["P22_bar"]) for row in rows[1:]),
        "maximum_abs_macro_shear": max(
            max(abs(row["F12_bar"]), abs(row["F21_bar"])) for row in rows[1:]
        ),
        "mean_accumulated_slip": two["accumulated_slip_bar"],
        "maximum_accumulated_slip": two["accumulated_slip_max"],
        "mean_slip_resistance_MPa": two["slip_resistance_mean_bar"],
        "minimum_det_Fp": two["det_fp_min"],
        "maximum_det_Fp": two["det_fp_max"],
        "plastic_dissipation": two["plastic_dissipation_bar"],
        "hill_mandel_relative_residual_max": max(hill_mandel, default=math.nan),
        "accepted_increments": int(round(two["step"])),
        "accepted_newton_iterations": int(round(sum(row["nonlinear_iterations"] for row in rows[1:]))),
        "history_is_finite": bool(np.isfinite(all_values).all()),
        "full_curve": [
            {"time": row["time"], "F11": row["F11_bar"], "P11_MPa": row["P11_bar"]}
            for row in rows[1:]
        ],
        "exodus": exodus,
    }
    stats = exodus["final_element_statistics"]
    result.update(
        {
            "maximum_local_P11_MPa": stats["P11"]["maximum"],
            "minimum_local_slip_resistance_MPa": stats["slip_resistance_mean"]["minimum"],
            "maximum_local_slip_resistance_MPa": stats["slip_resistance_mean"]["maximum"],
            "maximum_active_slip_systems": stats["active_slip_systems"]["maximum"],
        }
    )
    checks = {
        "reached_physical_end_time": abs(two["time"] - 1.0) <= 2.0e-11,
        "exact_synchronized_targets": abs(one["F11_bar"] - 1.01) <= 2.0e-7 and abs(two["F11_bar"] - 1.02) <= 2.0e-7,
        "P22_constraint": result["maximum_abs_P22_MPa"] < 1.0e-3,
        "macro_shear_constraints": result["maximum_abs_macro_shear"] < 1.0e-10,
        "plasticity_active": result["maximum_accumulated_slip"] > 1.0e-4,
        "finite_state": result["history_is_finite"],
        "positive_slip_resistance": result["minimum_local_slip_resistance_MPa"] > 0.0,
        "plastic_volume_error": max(abs(result["minimum_det_Fp"] - 1.0), abs(result["maximum_det_Fp"] - 1.0)) < 5.0e-4,
        "nonnegative_plastic_dissipation": result["plastic_dissipation"] >= -1.0e-10,
        "hill_mandel": result["hill_mandel_relative_residual_max"] < 5.0e-4,
        "periodic_BCs": exodus["maximum_periodicity_error"] < 1.0e-10,
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
