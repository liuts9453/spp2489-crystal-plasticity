#!/usr/bin/env python3
"""Extract comparable, machine-readable CP and hyperelastic benchmark metrics."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/crystal_plasticity"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def number(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return float(matches[-1]) if matches else None


def elapsed_seconds(value: str) -> float:
    fields = [float(item) for item in value.split(":")]
    if len(fields) == 3:
        return fields[0] * 3600 + fields[1] * 60 + fields[2]
    if len(fields) == 2:
        return fields[0] * 60 + fields[1]
    return fields[0]


def parse_log(path: Path) -> dict[str, float | int | None]:
    if not path.exists():
        return {}
    text = ANSI.sub("", path.read_text(encoding="utf-8", errors="replace"))
    wall_match = re.findall(r"Elapsed \(wall clock\) time .*?:\s*([0-9:.]+)", text)
    processors = number(r"Num Processors:\s+(\d+)", text)
    exit_status = number(r"Exit status:\s+(\d+)", text)
    return {
        "wall_seconds": elapsed_seconds(wall_match[-1]) if wall_match else None,
        "user_seconds": number(r"User time \(seconds\):\s*([0-9.eE+-]+)", text),
        "system_seconds": number(r"System time \(seconds\):\s*([0-9.eE+-]+)", text),
        "max_rss_kb": number(r"Maximum resident set size \(kbytes\):\s*([0-9.eE+-]+)", text),
        "mpi_ranks": int(processors) if processors else 1,
        "elements": int(number(r"Elems:\s+\n\s+Total:\s+(\d+)", text) or 0),
        "exit_status": int(exit_status) if exit_status is not None else -1,
        "timed_out": int("SPP2489_BENCHMARK_TIMEOUT_SECONDS=" in text),
        "failed_newton_iterations": sum(
            int(value)
            for value in re.findall(
                r"Nonlinear solve did not converge due to .*? iterations (\d+)", text
            )
        ),
    }


def parse_perfgraph(path: Path) -> list[dict[str, object]]:
    """Return the rank-zero PerfGraph table without pretending nested rows are additive."""
    if not path.exists():
        return []
    text = ANSI.sub("", path.read_text(encoding="utf-8", errors="replace"))
    if "Performance Graph:" not in text:
        return []
    table = text.split("Performance Graph:", 1)[1].split("Finished Executing", 1)[0]
    records: list[dict[str, object]] = []
    for line in table.splitlines():
        columns = line.split("|")
        if len(columns) < 11:
            continue
        section = columns[1].rstrip()
        try:
            calls = int(columns[2].strip())
            self_seconds = float(columns[3].strip())
            total_seconds = float(columns[7].strip())
        except ValueError:
            continue
        records.append(
            {
                "section": section.strip(),
                "nesting_spaces": len(section) - len(section.lstrip()),
                "calls": calls,
                "self_seconds_rank0": self_seconds,
                "inclusive_seconds_rank0": total_seconds,
            }
        )
    return records


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def as_float(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def infer_case(path: Path) -> tuple[str, int, int] | None:
    match = re.fullmatch(r"(analytic_cp|hyper)_(\d+)_s(\d+)\.csv", path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def main() -> None:
    summaries: list[dict[str, object]] = []
    steps_out: list[dict[str, object]] = []
    breakdown: list[dict[str, object]] = []
    for csv_path in sorted(RESULTS.glob("*.csv")):
        case = infer_case(csv_path)
        if not case:
            continue
        kind, mesh, requested_steps = case
        model = "cp" if kind == "analytic_cp" else kind
        rows = load_csv(csv_path)
        if not rows:
            continue
        final = rows[-1]
        log = parse_log(csv_path.with_suffix(".log"))
        final_time = as_float(final, "time") or 0.0
        final_f11 = as_float(final, "F11_bar") or 0.0
        completed = final_time >= 1.0 - 1e-10 and final_f11 >= 1.0199998 and log.get("exit_status") == 0
        ranks = int(log.get("mpi_ranks") or 1)
        cp_rank_seconds = as_float(final, "cp_update_seconds") if model == "cp" else None
        cp_wall_equiv = cp_rank_seconds / ranks if cp_rank_seconds is not None else None
        wall = log.get("wall_seconds")
        cp_fraction = cp_wall_equiv / wall if cp_wall_equiv is not None and wall else None
        actual_steps = int(round(as_float(final, "step") or max(0, len(rows) - 1)))
        elements = int(log.get("elements") or mesh * mesh)
        accepted_newton = sum(
            int(round(as_float(row, "nonlinear_iterations") or 0)) for row in rows[1:]
        )
        failed_newton = int(log.get("failed_newton_iterations") or 0)
        summary = {
            "case": csv_path.stem,
            "model": model,
            "mesh": mesh,
            "requested_steps": requested_steps,
            "actual_steps": actual_steps,
            "completed": int(completed),
            "wall_seconds": wall,
            "user_seconds": log.get("user_seconds"),
            "system_seconds": log.get("system_seconds"),
            "max_rss_kb": log.get("max_rss_kb"),
            "mpi_ranks": ranks,
            "elements": elements,
            "integration_points": 4 * elements,
            "final_F11": final_f11,
            "final_F22": as_float(final, "F22_bar"),
            "final_P11_MPa": as_float(final, "P11_bar"),
            "final_P22_MPa": as_float(final, "P22_bar"),
            "accumulated_slip_mean": as_float(final, "accumulated_slip_bar"),
            "accumulated_slip_max": as_float(final, "accumulated_slip_max"),
            "det_Fp_mean": as_float(final, "det_fp_bar"),
            "det_Fp_min": as_float(final, "det_fp_min"),
            "det_Fp_max": as_float(final, "det_fp_max"),
            "cp_update_rank_seconds": cp_rank_seconds,
            "cp_update_wall_equivalent_seconds": cp_wall_equiv,
            "cp_update_wall_fraction": cp_fraction,
            "newton_iterations_accepted_steps": accepted_newton,
            "newton_iterations_failed_attempts": failed_newton,
            "newton_iterations_total": accepted_newton + failed_newton,
            "linear_iterations_accepted_steps": sum(
                int(round(as_float(row, "linear_iterations") or 0)) for row in rows[1:]
            ),
            "failed_global_attempts": csv_path.with_suffix(".log").read_text(
                encoding="utf-8", errors="replace"
            ).count("Solve Did NOT Converge!") if csv_path.with_suffix(".log").exists() else 0,
            "final_cp_local_iterations_bar": as_float(final, "cp_local_iterations_bar"),
            "final_cp_local_substeps_max": as_float(final, "cp_local_substeps_max"),
            "final_cp_local_residual_max": as_float(final, "cp_local_residual_max"),
            "timed_out": log.get("timed_out", 0),
        }
        summaries.append(summary)
        breakdown.append(
            {
                "case": csv_path.stem,
                "wall_seconds_time_v": wall,
                "cp_update_rank_seconds": cp_rank_seconds,
                "cp_update_wall_equivalent_seconds": cp_wall_equiv,
                "non_cp_wall_equivalent_seconds":
                    max(float(wall) - cp_wall_equiv, 0.0)
                    if wall is not None and cp_wall_equiv is not None
                    else None,
                "note": (
                    "PerfGraph sections are inclusive and nested; do not add them. "
                    "The explicit CP timer is aggregated across MPI ranks and divided by the rank count "
                    "only for the wall-equivalent estimate."
                ),
                "perfgraph_rank0": parse_perfgraph(csv_path.with_suffix(".log")),
            }
        )

        previous_cp = as_float(rows[0], "cp_update_seconds") or 0.0
        previous_wall = as_float(rows[0], "benchmark_elapsed_seconds") or 0.0
        previous_f11 = as_float(rows[0], "F11_bar") or 1.0
        for index, row in enumerate(rows[1:], start=1):
            cumulative_cp = as_float(row, "cp_update_seconds")
            cumulative_wall = as_float(row, "benchmark_elapsed_seconds")
            current_f11 = as_float(row, "F11_bar")
            steps_out.append(
                {
                    "case": csv_path.stem,
                    "step_index": index,
                    "physical_time": as_float(row, "time"),
                    "delta_F11": current_f11 - previous_f11 if current_f11 is not None else None,
                    "F11": current_f11,
                    "P11_MPa": as_float(row, "P11_bar"),
                    "P22_MPa": as_float(row, "P22_bar"),
                    "newton_iterations": as_float(row, "nonlinear_iterations"),
                    "linear_iterations": as_float(row, "linear_iterations"),
                    "step_wall_seconds": cumulative_wall - previous_wall if cumulative_wall is not None else None,
                    "cp_update_step_rank_seconds": cumulative_cp - previous_cp if cumulative_cp is not None else None,
                    "accumulated_slip_mean": as_float(row, "accumulated_slip_bar"),
                }
            )
            if cumulative_cp is not None:
                previous_cp = cumulative_cp
            if cumulative_wall is not None:
                previous_wall = cumulative_wall
            if current_f11 is not None:
                previous_f11 = current_f11

    RESULTS.mkdir(parents=True, exist_ok=True)
    if summaries:
        with (RESULTS / "cp_runtime_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    if steps_out:
        with (RESULTS / "cp_step_statistics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(steps_out[0]))
            writer.writeheader()
            writer.writerows(steps_out)
    (RESULTS / "cp_performance_breakdown.json").write_text(
        json.dumps({"cases": breakdown}, indent=2) + "\n", encoding="utf-8"
    )

    parameters = {
        "elastic_constants_MPa": {"C11": 200000.0, "C12": 120000.0, "C44": 100000.0},
        "slip_systems": "12 FCC {111}<110>",
        "reference_slip_rate_per_s": 0.001,
        "rate_sensitivity_m_single_crystal": 0.05,
        "rate_sensitivity_m_rve": 0.10,
        "initial_slip_resistance_MPa": 80.0,
        "hardening_modulus_MPa": 400.0,
        "saturation_resistance_MPa": 160.0,
        "hardening_exponent": 1.0,
        "latent_hardening_ratio": 1.4,
        "total_loading_time_s": 1.0,
        "maximum_axial_stretch": 1.02,
        "macroscopic_constraints": {"P22_MPa": 0.0, "F12": 0.0, "F21": 0.0},
        "plane_strain_F33": 1.0,
        "global_solver": {
            "solve_type": "NEWTON",
            "line_search": "none",
            "nl_max_its": 25,
            "nl_rel_tol": 1e-6,
            "nl_abs_tol": 1e-10,
            "linear_relative_tolerance": 1e-8,
            "mpi_ranks": 4,
            "preconditioner": "ASM overlap=2, subdomain LU",
        },
        "local_solver": {
            "absolute_tolerance": 1e-10,
            "relative_tolerance": 1e-9,
            "maximum_iterations": 60,
            "maximum_internal_substeps": 64,
            "maximum_slip_increment_per_substep": 0.02,
        },
        "moose_commit": "11853d1623555a9e444ffb0725a4dc16f64fb255",
        "RVE_note": "Direct PK2 analytic-tangent implementation. m=0.05 is retained for single-crystal verification; the RVE uses the native-default-level m=0.10 because m=0.05 required repeated global continuation at the yield transition.",
    }
    (RESULTS / "cp_material_parameters.json").write_text(
        json.dumps(parameters, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(summaries)} cases and {len(steps_out)} load-step records")


if __name__ == "__main__":
    main()
