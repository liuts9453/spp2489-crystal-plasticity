#!/usr/bin/env python3
"""Run one isolated, cache-safe FCC CP texture homogenization evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path

from common import (
    A4_SOLVER,
    BASE_INPUT,
    CENTERS_INPUT,
    DEFAULT_PROTOCOL,
    EXECUTABLE,
    PROTOCOLS,
    ROOT,
    RUNS,
    TimeProtocol,
    canonical_hash,
    ensure_directories,
    simulation_configuration,
    write_json,
)
from extract_plastic_properties import extract_properties
from texture_generator import write_cp_material_include


def _last_accepted_state(csv_path: Path) -> dict[str, object]:
    if not csv_path.exists():
        return {"last_accepted_time_s": None, "last_accepted_F11": None, "last_accepted_step": None}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return {"last_accepted_time_s": None, "last_accepted_F11": None, "last_accepted_step": None}
    row = rows[-1]
    return {
        "last_accepted_time_s": float(row["time"]),
        "last_accepted_F11": float(row["F11_bar"]),
        "last_accepted_step": int(round(float(row["step"]))),
    }


def write_case_input(run_dir: Path, eta: float) -> Path:
    material_path = run_dir / "cp_materials.i"
    write_cp_material_include(material_path, eta)
    text = BASE_INPUT.read_text(encoding="utf-8")
    text = text.replace("!include ../polycrystal_20_centers.i", f"!include {CENTERS_INPUT}")
    text = text.replace("!include cp_20_random_materials.i", f"!include {material_path}")
    if str(material_path) not in text or str(CENTERS_INPUT) not in text:
        raise RuntimeError("Failed to replace CP input includes")
    case = run_dir / "case.i"
    case.write_text(f"# Isolated CP inverse-design case, eta={eta:.17g}\n" + text, encoding="utf-8")
    return case


def _completed_result(path: Path, requested: dict[str, object]) -> dict[str, object] | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("configuration") != requested or not result.get("converged"):
        return None
    if not Path(str(result.get("csv_path", ""))).exists() or not Path(str(result.get("exodus_path", ""))).exists():
        return None
    return result


def run_forward(
    eta: float,
    *,
    mesh: int = 32,
    protocol: TimeProtocol = DEFAULT_PROTOCOL,
    label: str | None = None,
    run_root: Path | None = None,
    reuse_cache: bool = True,
    mpi_ranks: int = 4,
) -> dict[str, object]:
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must lie in [0,1]")
    if mesh <= 0 or mpi_ranks <= 0:
        raise ValueError("mesh and mpi_ranks must be positive")
    if not EXECUTABLE.exists():
        raise FileNotFoundError(f"Build the application first: {EXECUTABLE}")
    ensure_directories()
    configuration = simulation_configuration(eta, mesh, protocol, mpi_ranks)
    key = canonical_hash(configuration)
    label = label or f"eta_{eta:.10f}_n{mesh}_{protocol.name}"
    if run_root is None:
        run_dir = RUNS / key
    else:
        run_dir = Path(run_root) / f"{label}_{key[:12]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if reuse_cache:
        cached = _completed_result(result_path, configuration)
        if cached is not None:
            cached = dict(cached)
            cached["reused_cache"] = True
            cached["requested_label"] = label
            print(
                f"REUSE {label}: eta={eta:.8f}, P11(1%)={cached['P11_at_F11_1p01_MPa']:.6f}, "
                f"P11(2%)={cached['P11_at_F11_1p02_MPa']:.6f}",
                flush=True,
            )
            return cached

    case = write_case_input(run_dir, eta)
    output_base = run_dir / "solution"
    csv_path = output_base.with_suffix(".csv")
    exodus_path = output_base.with_suffix(".e")
    log_path = run_dir / "run.log"
    command = [
        "/home/liu/miniforge/bin/conda",
        "run",
        "--no-capture-output",
        "-n",
        "moose",
        "/usr/bin/time",
        "-v",
        "mpiexec",
        "-n",
        str(mpi_ranks),
        str(EXECUTABLE),
        "-i",
        str(case),
        f"nx={mesh}",
        f"ny={mesh}",
        "H11=0.02",
        "cp_xm=0.10",
        f"cp_local_solver={A4_SOLVER['local_solver_mode']}",
        f"cp_reuse_lu={str(A4_SOLVER['reuse_tangent_factorization']).lower()}",
        f"cp_optimize_tensors={str(A4_SOLVER['optimized_tensor_evaluation']).lower()}",
        f"cp_optimize_scalars={str(A4_SOLVER['specialize_scalar_operations']).lower()}",
        f"cp_predictor={A4_SOLVER['local_predictor']}",
        f"run_name=cp_inverse_{key[:12]}",
        f"Outputs/csv/file_base={output_base}",
        f"Outputs/exodus/file_base={output_base}",
        *protocol.command_arguments(),
        "--disable-perf-graph-live",
    ]
    write_json(
        run_dir / "command.json",
        {"command": command, "cwd": str(ROOT), "configuration": configuration, "label": label},
    )
    print(f"RUN   {label}: eta={eta:.8f}, mesh={mesh}, protocol={protocol.name}", flush=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall = time.perf_counter() - start
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    failed_attempts = log_text.count("Solve Did NOT Converge!")
    failed_iterations = sum(
        int(value) for value in re.findall(r"did not converge due to .*? iterations (\d+)", log_text)
    )
    base: dict[str, object] = {
        "configuration": configuration,
        "cache_key": key,
        "eta": float(eta),
        "mesh": int(mesh),
        "mpi_ranks": int(mpi_ranks),
        "protocol": protocol.name,
        "label": label,
        "run_directory": str(run_dir),
        "input_path": str(case),
        "orientation_path": str(run_dir / "orientations.csv"),
        "csv_path": str(csv_path),
        "exodus_path": str(exodus_path),
        "log_path": str(log_path),
        "wall_time_s": wall,
        "returncode": completed.returncode,
        "failed_attempts": failed_attempts,
        "failed_newton_iterations": failed_iterations,
        "reused_cache": False,
    }
    if completed.returncode != 0 or not csv_path.exists() or not exodus_path.exists():
        failure = {
            **base,
            **_last_accepted_state(csv_path),
            "converged": False,
            "failure_reason": "nonzero return or missing output",
        }
        write_json(result_path, failure)
        print(f"FAIL  {label}: returncode={completed.returncode}; see {log_path}", flush=True)
        return failure
    try:
        properties = extract_properties(csv_path, exodus_path)
    except Exception as error:
        failure = {**base, "converged": False, "failure_reason": str(error)}
        write_json(result_path, failure)
        print(f"FAIL  {label}: postprocessing: {error}", flush=True)
        return failure
    accepted = int(properties["accepted_newton_iterations"])
    result = {
        **base,
        **properties,
        "converged": True,
        "total_newton_iterations": accepted + failed_iterations,
    }
    write_json(result_path, result)
    print(
        f"DONE  {label}: P11(1%)={result['P11_at_F11_1p01_MPa']:.6f}, "
        f"P11(2%)={result['P11_at_F11_1p02_MPa']:.6f}, "
        f"steps={result['accepted_increments']}, failures={failed_attempts}, wall={wall:.2f}s",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--mesh", type=int, default=32)
    parser.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL.name)
    parser.add_argument("--label")
    parser.add_argument("--mpi-ranks", type=int, default=4)
    parser.add_argument("--no-reuse", action="store_true")
    args = parser.parse_args()
    result = run_forward(
        args.eta,
        mesh=args.mesh,
        protocol=PROTOCOLS[args.protocol],
        label=args.label,
        reuse_cache=not args.no_reuse,
        mpi_ranks=args.mpi_ranks,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("converged") else 2


if __name__ == "__main__":
    raise SystemExit(main())
