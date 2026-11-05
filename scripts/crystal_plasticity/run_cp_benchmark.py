#!/usr/bin/env python3
"""Run or resume the SPP2489 crystal-plasticity performance benchmark.

The runner deliberately treats a CSV as cached only when it reaches the requested
physical end time and deformation.  Partial/failed runs are therefore never
reported as completed timings.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/crystal_plasticity"
APP = ROOT / "spp2489-opt"
CP_INPUT = ROOT / "examples/crystal_plasticity/polycrystal_20_cp.i"
HYPER_INPUT = ROOT / "examples/10_polycrystal_case_c_random_orientation.i"


def completed(csv_path: Path, target_f11: float) -> bool:
    if not csv_path.exists():
        return False
    try:
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        return bool(rows) and abs(float(rows[-1]["time"]) - 1.0) < 1e-10 and abs(
            float(rows[-1]["F11_bar"]) - target_f11
        ) < 2e-7
    except (KeyError, ValueError, OSError):
        return False


def command(kind: str, mesh: int, steps: int, strain: float, ranks: int, stem: str) -> list[str]:
    common = [
        "/usr/bin/time",
        "-v",
        "mpiexec",
        "-n",
        str(ranks),
        str(APP),
        "-i",
    ]
    petsc = [
        "Executioner/petsc_options_iname=-pc_type -pc_asm_overlap -sub_pc_type",
        "Executioner/petsc_options_value=asm 2 lu",
    ]
    if kind == "cp":
        return common + [
            str(CP_INPUT),
            f"nx={mesh}",
            f"ny={mesh}",
            f"steps={steps}",
            f"H11={strain}",
            f"run_name={stem}",
        ] + petsc

    base = RESULTS / stem
    return common + [
        str(HYPER_INPUT),
        f"nx={mesh}",
        f"ny={mesh}",
        f"H11={strain}",
        f"Executioner/dt={1.0 / steps:.17g}",
        "Executioner/nl_max_its=25",
        "Executioner/nl_rel_tol=1e-6",
        "Executioner/nl_abs_tol=1e-10",
        "Executioner/l_tol=1e-8",
        f"Outputs/csv/file_base={base}",
        f"Outputs/exodus/file_base={base}",
        "Outputs/perf_graph=true",
        f"Outputs/perf_graph_json_file={base}_perf_graph.json",
    ] + petsc


def run_one(kind: str, mesh: int, steps: int, strain: float, ranks: int, force: bool) -> bool:
    prefix = "analytic_cp" if kind == "cp" else kind
    stem = f"{prefix}_{mesh}_s{steps}"
    csv_path = RESULTS / f"{stem}.csv"
    log_path = RESULTS / f"{stem}.log"
    if not force and completed(csv_path, 1.0 + strain):
        print(f"[cached] {stem}")
        return True

    RESULTS.mkdir(parents=True, exist_ok=True)
    timeout = {16: 300, 32: 1200, 64: 1800}.get(mesh, 1800)
    cmd = command(kind, mesh, steps, strain, ranks, stem)
    print(f"[run] {stem} (timeout={timeout}s)")
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            ok = process.returncode == 0 and completed(csv_path, 1.0 + strain)
        except subprocess.TimeoutExpired:
            log.write(f"\nSPP2489_BENCHMARK_TIMEOUT_SECONDS={timeout}\n")
            ok = False
    print(f"[{'complete' if ok else 'failed/partial'}] {stem}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("cp", "hyper", "all"), default="all")
    parser.add_argument("--meshes", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--max-strain", type=float, default=0.02)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not APP.exists():
        raise SystemExit(f"Missing executable: {APP}; build with `conda run -n moose make -j4`")
    kinds = ("cp", "hyper") if args.kind == "all" else (args.kind,)
    status = [
        run_one(kind, mesh, args.steps, args.max_strain, args.ranks, args.force)
        for mesh in args.meshes
        for kind in kinds
    ]
    return 0 if all(status) else 2


if __name__ == "__main__":
    raise SystemExit(main())
