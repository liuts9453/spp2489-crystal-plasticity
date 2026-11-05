#!/usr/bin/env python3
"""Perform physical and numerical checks on completed CP RVE CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/crystal_plasticity"


def decode_names(array) -> list[str]:
    return [b"".join(row).decode("utf-8").rstrip("\x00 ") for row in np.asarray(array)]


def paired_error(x: np.ndarray, y: np.ndarray, values: np.ndarray, axis: str) -> float:
    if axis == "x":
        minus, plus = np.flatnonzero(np.isclose(x, x.min())), np.flatnonzero(np.isclose(x, x.max()))
        minus, plus = minus[np.argsort(y[minus])], plus[np.argsort(y[plus])]
    else:
        minus, plus = np.flatnonzero(np.isclose(y, y.min())), np.flatnonzero(np.isclose(y, y.max()))
        minus, plus = minus[np.argsort(x[minus])], plus[np.argsort(x[plus])]
    if len(minus) != len(plus):
        return math.inf
    return float(np.max(np.abs(values[plus] - values[minus])))


def exodus_checks(path: Path) -> dict[str, object]:
    required_element = {
        "grain_id", "euler_angle_1", "euler_angle_2", "euler_angle_3", "F11", "F22",
        "P11", "P22", "von_mises", "accumulated_slip", "fp_11", "fp_22", "fp_33",
        "slip_resistance_mean",
    }
    required_nodal = {"total_displacement_x", "total_displacement_y", "total_displacement_magnitude"}
    with netcdf_file(path, "r", mmap=False) as db:
        x = np.asarray(db.variables["coordx"].data).copy()
        y = np.asarray(db.variables["coordy"].data).copy()
        nodal_names = decode_names(db.variables["name_nod_var"].data)
        element_name_list = decode_names(db.variables["name_elem_var"].data)
        element_names = set(element_name_list)
        periodic_errors = []
        for name in ("disp_x", "disp_y"):
            index = nodal_names.index(name) + 1
            values = np.asarray(db.variables[f"vals_nod_var{index}"].data[-1]).copy()
            periodic_errors.extend((paired_error(x, y, values, "x"), paired_error(x, y, values, "y")))
        field_ranges = {}
        for name in ("F11", "P11", "accumulated_slip"):
            index = element_name_list.index(name) + 1
            values = []
            for block in range(1, len(np.asarray(db.variables["eb_prop1"].data)) + 1):
                values.extend(np.asarray(db.variables[f"vals_elem_var{index}eb{block}"].data[-1]).copy())
            field_ranges[name] = float(max(values) - min(values))
    return {
        "readable": True,
        "missing_required_fields": sorted((required_element - element_names) | (required_nodal - set(nodal_names))),
        "maximum_periodic_fluctuation_error": max(periodic_errors),
        "final_element_field_ranges": field_ranges,
    }


def rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def verify(path: Path) -> dict[str, object]:
    data = rows(path)
    final = data[-1]
    completed = abs(final["time"] - 1.0) < 1e-10 and abs(final["F11_bar"] - 1.02) < 2e-7
    grain_slips = [final[f"grain_{grain:02d}_slip"] for grain in range(1, 21)]
    same_orientation = "_same_" in path.stem
    hm_residuals: list[float] = []
    for old, new in zip(data[1:-1], data[2:]):
        dt = new["time"] - old["time"]
        if dt <= 0:
            continue
        macro_power = sum(
            new[f"P{i}{j}_bar"] * (new[f"F{i}{j}_bar"] - old[f"F{i}{j}_bar"]) / dt
            for i, j in ((1, 1), (1, 2), (2, 1), (2, 2))
        )
        local = new["local_power_bar"]
        hm_residuals.append(abs(local - macro_power) / max(abs(local), abs(macro_power), 1.0))

    dt_values = [new["time"] - old["time"] for old, new in zip(data[:-1], data[1:])]
    exodus_path = path.with_suffix(".e")
    exodus = exodus_checks(exodus_path) if exodus_path.exists() else {
        "readable": False, "missing_required_fields": ["missing Exodus"],
        "maximum_periodic_fluctuation_error": math.inf,
        "final_element_field_ranges": {},
    }
    checks = {
        "completed_to_2_percent": completed,
        "plasticity_active": final["accumulated_slip_max"] > 1e-4,
        "orientation_dependent_slip": statistics.pstdev(grain_slips) > 1e-4,
        "same_orientation_uniform_slip":
            max(grain_slips) - min(grain_slips) < 1e-10 if same_orientation else None,
        "same_orientation_zero_fluctuation":
            max(abs(final["fluctuation_l2_x"]), abs(final["fluctuation_l2_y"])) < 1e-9
            if same_orientation else None,
        "same_orientation_uniform_fields":
            bool(exodus["final_element_field_ranges"])
            and max(exodus["final_element_field_ranges"].values()) < 1e-8
            if same_orientation else None,
        "P22_constraint": abs(final["P22_bar"]) < 1e-3,
        "macro_shear_constraints": max(abs(final["F12_bar"]), abs(final["F21_bar"])) < 1e-10,
        "plastic_incompressibility": max(
            abs(final["det_fp_min"] - 1.0), abs(final["det_fp_max"] - 1.0)
        ) < 5e-4,
        "positive_plastic_dissipation": final["plastic_dissipation_bar"] >= -1e-10,
        "hill_mandel_relative_residual_max": max(hm_residuals, default=math.nan),
        "hill_mandel_pass": max(hm_residuals, default=1.0) < 5e-4,
        "equal_time_increments": max(dt_values, default=0.0) - min(dt_values, default=0.0) < 1e-10,
        "exodus_required_fields": not exodus["missing_required_fields"],
        "periodic_fluctuation_BCs": exodus["maximum_periodic_fluctuation_error"] < 1e-10,
    }
    return {
        "file": str(path.relative_to(ROOT)),
        "checks": checks,
        "final": {
            "F11": final["F11_bar"],
            "F22": final["F22_bar"],
            "P11_MPa": final["P11_bar"],
            "P22_MPa": final["P22_bar"],
            "accumulated_slip_mean": final["accumulated_slip_bar"],
            "accumulated_slip_max": final["accumulated_slip_max"],
            "grain_slip_min": min(grain_slips),
            "grain_slip_max": max(grain_slips),
            "det_Fp_min": final["det_fp_min"],
            "det_Fp_max": final["det_fp_max"],
            "plastic_dissipation_rate": final["plastic_dissipation_bar"],
        },
        "exodus": exodus,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.files or (
        sorted(RESULTS.glob("analytic_cp_[0-9]*_s*.csv"))
        + sorted(RESULTS.glob("analytic_cp_same_*_s*.csv"))
    )
    paths = [path if path.is_absolute() else ROOT / path for path in paths]
    reports = [verify(path) for path in paths if path.exists()]
    output = {"cases": reports}
    (RESULTS / "cp_verification.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for report in reports:
        print(report["file"])
        for name, value in report["checks"].items():
            print(f"  {name}: {value}")
    mandatory = (
        "completed_to_2_percent",
        "plasticity_active",
        "P22_constraint",
        "macro_shear_constraints",
        "plastic_incompressibility",
        "positive_plastic_dissipation",
        "hill_mandel_pass",
        "exodus_required_fields",
        "periodic_fluctuation_BCs",
    )
    valid = []
    for case in reports:
        checks = case["checks"]
        base = all(checks[key] for key in mandatory)
        if "_same_" in case["file"]:
            base = base and all(
                checks[key]
                for key in (
                    "same_orientation_uniform_slip",
                    "same_orientation_zero_fluctuation",
                    "same_orientation_uniform_fields",
                )
            )
        else:
            base = base and checks["orientation_dependent_slip"]
        valid.append(base)
    return 0 if reports and all(valid) else 2


if __name__ == "__main__":
    raise SystemExit(main())
