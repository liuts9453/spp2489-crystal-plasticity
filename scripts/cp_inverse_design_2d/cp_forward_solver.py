#!/usr/bin/env python3
"""Run one cache-safe Neper-RVE finite-strain FCC CP homogenization."""

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
    CONDA_EXECUTABLE,
    CONVERGENCE,
    DEFAULT_PROTOCOL,
    EXECUTABLE,
    MESH_FILES,
    PROTOCOLS,
    ROOT,
    RUNS,
    TimeProtocol,
    canonical_hash,
    design_tag,
    ensure_directories,
    simulation_configuration,
    write_json,
)
from extract_proof_stress import extract_properties
from texture_parameterization import write_cp_material_include


GRAIN_ID_VALUES = " ".join(str(value) for value in range(1, 21))


def _last_accepted_state(csv_path: Path) -> dict[str, object]:
    if not csv_path.exists():
        return {
            "last_accepted_time_s": None,
            "last_accepted_F11": None,
            "last_accepted_step": None,
        }
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return {
            "last_accepted_time_s": None,
            "last_accepted_F11": None,
            "last_accepted_step": None,
        }
    row = rows[-1]
    return {
        "last_accepted_time_s": float(row["time"]),
        "last_accepted_F11": float(row["F11_bar"]),
        "last_accepted_step": int(round(float(row["step"]))),
    }


def write_case_input(run_directory: Path, eta: float, theta_deg: float) -> Path:
    material_path = run_directory / "cp_materials.i"
    write_cp_material_include(material_path, eta, theta_deg)
    text = BASE_INPUT.read_text(encoding="utf-8")
    centers = ROOT / "examples" / "polycrystal_20_centers.i"
    text = text.replace("!include ../polycrystal_20_centers.i", f"!include {centers}")
    text = text.replace(
        "!include cp_20_random_materials.i", f"!include {material_path}"
    )
    if str(material_path) not in text or str(centers) not in text:
        raise RuntimeError("Failed to isolate the CP input includes")
    case = run_directory / "case.i"
    case.write_text(
        f"# Neper CP inverse design: eta={eta:.17g}, theta={theta_deg:.17g} deg\n"
        + text,
        encoding="utf-8",
    )
    return case


def _completed_result(
    path: Path, requested_configuration: dict[str, object]
) -> dict[str, object] | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("configuration") != requested_configuration or not result.get(
        "converged"
    ):
        return None
    for name in ("csv_path", "exodus_path"):
        if not Path(str(result.get(name, ""))).exists():
            return None
    return result


def run_forward(
    eta: float,
    theta_deg: float,
    *,
    mesh_name: str = "coarse",
    protocol: TimeProtocol = DEFAULT_PROTOCOL,
    label: str | None = None,
    run_root: Path | None = None,
    reuse_cache: bool = True,
    mpi_ranks: int = 8,
    mesh_path: Path | None = None,
    mesh_manifest_path: Path | None = None,
    tessellation_path: Path | None = None,
    configuration_extra: dict[str, object] | None = None,
) -> dict[str, object]:
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must lie in [0,1]")
    if not -180.0 <= theta_deg <= 180.0:
        raise ValueError("theta must lie in [-180,180] degrees")
    if mesh_path is None and mesh_name not in MESH_FILES:
        raise ValueError(f"Unknown mesh: {mesh_name}")
    if mpi_ranks <= 0:
        raise ValueError("mpi_ranks must be positive")
    if not EXECUTABLE.exists():
        raise FileNotFoundError(f"Build the application first: {EXECUTABLE}")
    selected_mesh = Path(mesh_path) if mesh_path is not None else MESH_FILES[mesh_name]
    if not selected_mesh.exists():
        raise FileNotFoundError("Generate the Neper meshes before running FEM")
    ensure_directories()
    configuration = simulation_configuration(
        eta,
        theta_deg,
        mesh_name,
        protocol,
        mpi_ranks,
        mesh_path=selected_mesh,
        mesh_manifest_path=mesh_manifest_path,
        tessellation_path=tessellation_path,
        configuration_extra=configuration_extra,
    )
    cache_key = canonical_hash(configuration)
    label = label or f"{design_tag(eta, theta_deg)}_{mesh_name}_{protocol.name}"
    run_directory = (
        RUNS / cache_key
        if run_root is None
        else Path(run_root) / f"{label}_{cache_key[:12]}"
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    result_path = run_directory / "result.json"
    if reuse_cache:
        cached = _completed_result(result_path, configuration)
        if cached is not None:
            cached = dict(cached)
            cached["reused_cache"] = True
            cached["requested_label"] = label
            print(
                f"REUSE {label}: eta={eta:.7f}, theta={theta_deg:+.5f}, "
                f"P0.2={cached['proof_stress_MPa']:.6f}, P2%={cached['flow_stress_2pct_MPa']:.6f}",
                flush=True,
            )
            return cached

    case = write_case_input(run_directory, eta, theta_deg)
    output_base = run_directory / "solution"
    csv_path = output_base.with_suffix(".csv")
    exodus_path = output_base.with_suffix(".e")
    log_path = run_directory / "run.log"
    command = [
        str(CONDA_EXECUTABLE),
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
        "Mesh/inactive=grains",
        "Mesh/base/type=FileMeshGenerator",
        f"Mesh/base/file={selected_mesh}",
        "Mesh/pin/input=base",
        "Mesh/pin/coord='100 100 0'",
        "Mesh/pin/use_closest_node=true",
        "AuxKernels/grain_id/type=GrainPropertyAux",
        f"AuxKernels/grain_id/values='{GRAIN_ID_VALUES}'",
        "H11=0.02",
        "cp_xm=0.10",
        f"cp_local_solver={A4_SOLVER['local_solver_mode']}",
        f"cp_reuse_lu={str(A4_SOLVER['reuse_tangent_factorization']).lower()}",
        f"cp_optimize_tensors={str(A4_SOLVER['optimized_tensor_evaluation']).lower()}",
        f"cp_optimize_scalars={str(A4_SOLVER['specialize_scalar_operations']).lower()}",
        f"cp_predictor={A4_SOLVER['local_predictor']}",
        f"run_name=cp_id_2d_{cache_key[:12]}",
        f"Executioner/nl_rel_tol={CONVERGENCE['nl_rel_tol']:.17g}",
        f"Executioner/nl_abs_tol={CONVERGENCE['nl_abs_tol']:.17g}",
        f"Executioner/l_tol={CONVERGENCE['linear_tolerance']:.17g}",
        f"Outputs/csv/file_base={output_base}",
        f"Outputs/exodus/file_base={output_base}",
        "Outputs/perf_graph=false",
        *protocol.command_arguments(),
        "--allow-unused",
        "--disable-perf-graph-live",
    ]
    write_json(
        run_directory / "command.json",
        {
            "command": command,
            "cwd": str(ROOT),
            "configuration": configuration,
            "label": label,
        },
    )
    print(
        f"RUN   {label}: eta={eta:.7f}, theta={theta_deg:+.5f}, mesh={mesh_name}, "
        f"protocol={protocol.name}, MPI={mpi_ranks}",
        flush=True,
    )
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
    wall_time = time.perf_counter() - start
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    failed_attempts = log_text.count("Solve Did NOT Converge!")
    failed_iterations = sum(
        int(value)
        for value in re.findall(
            r"did not converge due to .*? iterations (\d+)", log_text
        )
    )
    rss_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", log_text)
    outlier_residuals = re.findall(
        r"Outlier Variable Residual Norms:\s*\n\s*\w+:\s*"
        r"(?:\x1b\[[0-9;]*m)?([0-9.eE+\-]+)",
        log_text,
    )
    final_outlier_residual_norm = (
        float(outlier_residuals[-1]) if outlier_residuals else None
    )
    base: dict[str, object] = {
        "configuration": configuration,
        "cache_key": cache_key,
        "eta": float(eta),
        "theta_deg": float(theta_deg),
        "mesh_name": mesh_name,
        "mesh_path": str(selected_mesh),
        "mpi_ranks": int(mpi_ranks),
        "protocol": protocol.name,
        "label": label,
        "run_directory": str(run_directory),
        "input_path": str(case),
        "orientation_path": str(run_directory / "orientations.csv"),
        "csv_path": str(csv_path),
        "exodus_path": str(exodus_path),
        "log_path": str(log_path),
        "wall_time_s": wall_time,
        "peak_rss_kib": int(rss_match.group(1)) if rss_match else None,
        "returncode": completed.returncode,
        "failed_attempts": failed_attempts,
        "failed_newton_iterations": failed_iterations,
        "final_outlier_residual_norm": final_outlier_residual_norm,
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
        print(
            f"FAIL  {label}: returncode={completed.returncode}; see {log_path}",
            flush=True,
        )
        return failure
    try:
        properties = extract_properties(csv_path, exodus_path)
    except Exception as error:
        failure = {**base, "converged": False, "failure_reason": str(error)}
        write_json(result_path, failure)
        print(f"FAIL  {label}: postprocessing: {error}", flush=True)
        return failure
    properties["checks"]["global_equilibrium"] = bool(
        final_outlier_residual_norm is not None and final_outlier_residual_norm < 1.0e-8
    )
    properties["all_checks_passed"] = all(properties["checks"].values())
    if not properties["all_checks_passed"]:
        failure = {
            **base,
            **properties,
            "converged": False,
            "failure_reason": "Global equilibrium residual validation failed",
        }
        write_json(result_path, failure)
        return failure
    result = {
        **base,
        **properties,
        "converged": True,
        "total_newton_iterations": int(properties["accepted_newton_iterations"])
        + failed_iterations,
    }
    write_json(result_path, result)
    print(
        f"DONE  {label}: P0.2={result['proof_stress_MPa']:.6f}, "
        f"P2%={result['flow_stress_2pct_MPa']:.6f}, steps={result['accepted_increments']}, "
        f"failures={failed_attempts}, wall={wall_time:.2f}s",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--mesh", choices=MESH_FILES, default="coarse")
    parser.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL.name)
    parser.add_argument("--label")
    parser.add_argument("--mpi-ranks", type=int, default=8)
    parser.add_argument("--no-reuse", action="store_true")
    args = parser.parse_args()
    result = run_forward(
        args.eta,
        args.theta,
        mesh_name=args.mesh,
        protocol=PROTOCOLS[args.protocol],
        label=args.label,
        reuse_cache=not args.no_reuse,
        mpi_ranks=args.mpi_ranks,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("converged") else 2


if __name__ == "__main__":
    raise SystemExit(main())
