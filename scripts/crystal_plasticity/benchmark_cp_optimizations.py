#!/usr/bin/env python3
"""Benchmark the incremental A0--A5 FCC local-integrator configurations."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "spp2489-opt"
INPUT = ROOT / "examples/crystal_plasticity/polycrystal_20_cp.i"
RESULTS = ROOT / "results/crystal_plasticity/optimization"

CONFIGS = {
    "A0": ("legacy", False, False, False, "previous"),
    "A1": ("grouped", False, False, False, "previous"),
    "A2": ("grouped", True, False, False, "previous"),
    "A3": ("grouped", True, True, False, "previous"),
    "A4": ("grouped", True, True, True, "previous"),
    "A5": ("grouped", True, True, True, "elastic_blend"),
}


def boolean(value: bool) -> str:
    return "true" if value else "false"


def elapsed_seconds(value: str) -> float:
    parts = [float(item) for item in value.split(":")]
    if len(parts) == 3:
        return 3600 * parts[0] + 60 * parts[1] + parts[2]
    if len(parts) == 2:
        return 60 * parts[0] + parts[1]
    return parts[0]


def parse_log(path: Path) -> dict[str, float | int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    wall = re.findall(r"Elapsed \(wall clock\) time .*?:\s*([0-9:.]+)", text)
    number = lambda pattern, default=0.0: float(re.findall(pattern, text)[-1]) if re.findall(pattern, text) else default
    return {
        "wall_seconds": elapsed_seconds(wall[-1]) if wall else 0.0,
        "user_seconds": number(r"User time \(seconds\):\s*([0-9.eE+-]+)"),
        "max_rss_kb": int(number(r"Maximum resident set size \(kbytes\):\s*(\d+)")),
        "exit_status": int(number(r"Exit status:\s*(\d+)", -1)),
        "failed_global_attempts": text.count("Solve Did NOT Converge!"),
        "failed_global_newton_iterations": sum(
            int(value)
            for value in re.findall(
                r"Nonlinear solve did not converge due to .*? iterations (\d+)", text
            )
        ),
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def completed(path: Path) -> bool:
    if not path.exists():
        return False
    rows = read_rows(path)
    return bool(rows) and abs(float(rows[-1]["time"]) - 1.0) <= 1e-10


def run_case(
    configuration: str,
    mesh: int,
    steps: int,
    ranks: int,
    mode: str,
    orientation: str,
    force: bool,
) -> tuple[Path, list[str]]:
    solver, reuse, tensors, scalars, predictor = CONFIGS[configuration]
    directory = RESULTS / configuration
    directory.mkdir(parents=True, exist_ok=True)
    orientation_tag = "" if orientation == "random" else f"_{orientation}"
    stem_name = f"{mode}{orientation_tag}_{mesh}_s{steps}"
    stem = directory / stem_name
    csv_path = stem.with_suffix(".csv")
    run_name = f"optimization/{configuration}/{stem_name}"
    command = [
        "/usr/bin/time",
        "-v",
        "mpiexec",
        "-n",
        str(ranks),
        str(APP),
        "-i",
        str(INPUT),
        f"nx={mesh}",
        f"ny={mesh}",
        f"steps={steps}",
        f"run_name={run_name}",
        f"cp_local_solver={solver}",
        f"cp_reuse_lu={boolean(reuse)}",
        f"cp_optimize_tensors={boolean(tensors)}",
        f"cp_optimize_scalars={boolean(scalars)}",
        f"cp_predictor={predictor}",
        "--disable-perf-graph-live",
    ]
    if orientation == "same":
        for grain in range(1, 21):
            material = f"Materials/elasticity_g{grain:02d}"
            command.extend(
                [
                    f"{material}/euler_angle_1=37",
                    f"{material}/euler_angle_2=52",
                    f"{material}/euler_angle_3=11",
                ]
            )
    if mode == "fixed":
        command.extend(
            [
                "Executioner/TimeStepper/type=ConstantDT",
                f"Executioner/TimeStepper/dt={1.0 / steps:.17g}",
                "Executioner/nl_max_its=80",
                "--allow-unused",
            ]
        )
    if not force and completed(csv_path):
        print(f"[cached] {configuration} {mesh}x{mesh} {mode}")
        return csv_path, command

    print(f"[run] {configuration} {mesh}x{mesh} {mode}")
    with stem.with_suffix(".log").open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if process.returncode or not completed(csv_path):
        raise RuntimeError(f"Benchmark failed: {' '.join(command)}; see {stem.with_suffix('.log')}")
    return csv_path, command


def summarize(csv_path: Path, command: list[str], configuration: str, ranks: int) -> dict[str, object]:
    rows = read_rows(csv_path)
    final = rows[-1]
    log = parse_log(csv_path.with_suffix(".log"))
    accepted_newton = sum(int(round(float(row["nonlinear_iterations"]))) for row in rows[1:])
    cp_rank_seconds = float(final["cp_update_seconds"])
    record: dict[str, object] = {
        "configuration": configuration,
        "local_dimension": 18 if configuration == "A0" else 10,
        "case": csv_path.stem,
        "mesh": int(re.search(r"_(\d+)_s", csv_path.stem).group(1)),
        "requested_steps": int(re.search(r"_s(\d+)$", csv_path.stem).group(1)),
        "accepted_steps": int(round(float(final["step"]))),
        "wall_seconds": log["wall_seconds"],
        "user_seconds": log["user_seconds"],
        "peak_rss_kb": log["max_rss_kb"],
        "mpi_ranks": ranks,
        "cp_rank_seconds": cp_rank_seconds,
        "cp_wall_equivalent_seconds": cp_rank_seconds / ranks,
        "residual_wall_equivalent_seconds": float(final["cp_residual_seconds"]) / ranks,
        "jacobian_wall_equivalent_seconds": float(final["cp_jacobian_seconds"]) / ranks,
        "linear_solve_wall_equivalent_seconds": float(final["cp_linear_solve_seconds"]) / ranks,
        "line_search_wall_equivalent_seconds": float(final["cp_line_search_seconds"]) / ranks,
        "tangent_wall_equivalent_seconds": float(final["cp_tangent_seconds"]) / ranks,
        "material_update_calls": int(round(float(final["cp_update_calls"]))),
        "local_newton_iterations": int(round(float(final["cp_local_iterations_total"]))),
        "internal_substeps": int(round(float(final["cp_substeps_total"]))),
        "line_search_evaluations": int(round(float(final["cp_line_search_evaluations"]))),
        "failed_global_attempts": log["failed_global_attempts"],
        "accepted_global_newton_iterations": accepted_newton,
        "failed_global_newton_iterations": log["failed_global_newton_iterations"],
        "total_global_newton_iterations": accepted_newton
        + int(log["failed_global_newton_iterations"]),
        "final_P11_MPa": float(final["P11_bar"]),
        "final_P22_MPa": float(final["P22_bar"]),
        "final_det_Fp_mean": float(final["det_fp_bar"]),
        "final_plastic_dissipation": float(final["plastic_dissipation_bar"]),
        "command": " ".join(command),
    }
    return record


def write_summary(records: list[dict[str, object]]) -> None:
    baseline = {
        (record["case"].replace(record["configuration"], ""), record["mesh"]): record
        for record in records
        if record["configuration"] == "A0"
    }
    # Configuration is encoded by the directory rather than case name; use mesh/mode/steps identity.
    baseline = {
        (record["case"].split("_")[0], record["mesh"], record["requested_steps"]): record
        for record in records
        if record["configuration"] == "A0"
    }
    for record in records:
        key = (record["case"].split("_")[0], record["mesh"], record["requested_steps"])
        reference = baseline.get(key)
        if reference:
            record["P11_relative_error_vs_A0"] = abs(
                float(record["final_P11_MPa"]) - float(reference["final_P11_MPa"])
            ) / abs(float(reference["final_P11_MPa"]))
            record["local_speedup_vs_A0"] = float(reference["cp_wall_equivalent_seconds"]) / float(
                record["cp_wall_equivalent_seconds"]
            )
            record["wall_speedup_vs_A0"] = float(reference["wall_seconds"]) / float(
                record["wall_seconds"]
            )

    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_output = RESULTS / "optimization_summary.csv"
    fields = [key for key in records[0] if key != "command"]
    with csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in records)
    (RESULTS / "optimization_summary.json").write_text(
        json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8"
    )
    print(csv_output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--meshes", nargs="+", type=int, default=[16])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--mode", choices=("adaptive", "fixed"), default="adaptive")
    parser.add_argument("--orientation", choices=("random", "same"), default="random")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not APP.exists():
        raise SystemExit(f"Missing {APP}; build the application first")

    records = []
    for mesh in args.meshes:
        for configuration in args.configs:
            csv_path, command = run_case(
                configuration,
                mesh,
                args.steps,
                args.ranks,
                args.mode,
                args.orientation,
                args.force,
            )
            records.append(summarize(csv_path, command, configuration, args.ranks))
    write_summary(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
