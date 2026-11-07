#!/usr/bin/env python3
"""Build and run one EBSD-calibration FCC-CP mechanical pilot case."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, V2_RESULTS_ROOT, sha256, write_json


CP_SCRIPT_DIR = ROOT / "scripts" / "cp_inverse_design_2d"
BASE_INPUT = ROOT / "examples" / "crystal_plasticity" / "polycrystal_20_cp.i"
CENTERS = ROOT / "examples" / "polycrystal_20_centers.i"
EXECUTABLE = ROOT / "spp2489-opt"
CONDA = Path(
    os.environ.get("SPP2489_CONDA", Path.home() / "miniforge" / "bin" / "conda")
)


def _run_checked(command: list[str], log: Path | None = None) -> None:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        detail = f"; see {log}" if log is not None else ""
        raise RuntimeError(f"Command failed with status {completed.returncode}{detail}")


def _candidate_directory(calibration: Path, family: str, seed: int) -> Path:
    if family == "baseline":
        candidate_id = -1
    else:
        payload = json.loads(
            (calibration / "optimized_parameters.json").read_text(encoding="utf-8")
        )
        candidate_id = int(payload["selected_candidate_id"])
    return (
        calibration / "runs" / family / f"candidate_{candidate_id:03d}" / f"seed_{seed}"
    )


def _write_case(run_directory: Path, material: Path) -> Path:
    text = BASE_INPUT.read_text(encoding="utf-8")
    text = text.replace("!include ../polycrystal_20_centers.i", f"!include {CENTERS}")
    text = text.replace("!include cp_20_random_materials.i", f"!include {material}")
    case = run_directory / "case.i"
    case.write_text(
        "# Experimental-statistics FCC benchmark; phase-0 is assumed FCC only for this numerical pilot.\n"
        + text,
        encoding="utf-8",
    )
    return case


def build_and_run(
    *,
    dataset: str,
    family: str,
    geometry_seed: int,
    characteristic_length_um: float,
    mpi_ranks: int,
    calibration_root: Path,
    output_root: Path,
    run_fem: bool,
    geometric_smoothing_passes: int,
    maximum_vertex_move_fraction: float,
    smoothing_activation_threshold: float,
    reuse_existing_fem_output: bool,
) -> dict[str, object]:
    source = _candidate_directory(calibration_root, family, geometry_seed)
    tessellation = source / "rve_periodic_ebsd_matched.tess"
    grain_table = source / "synthetic_grains.csv"
    if not tessellation.is_file() or not grain_table.is_file():
        raise FileNotFoundError(f"Missing calibrated holdout geometry beneath {source}")
    grains = pd.read_csv(grain_table).sort_values("synthetic_grain_id")
    expected_ids = list(range(1, len(grains) + 1))
    if grains["synthetic_grain_id"].astype(int).tolist() != expected_ids:
        raise RuntimeError("Synthetic grain IDs are not consecutive")
    tag = f"h_{characteristic_length_um:g}".replace(".", "p")
    if geometric_smoothing_passes:
        activation = f"{smoothing_activation_threshold:g}".replace(".", "p")
        tag += f"_smooth_{geometric_smoothing_passes}_a_{activation}"
    run_directory = output_root / dataset / family / f"seed_{geometry_seed}" / tag
    run_directory.mkdir(parents=True, exist_ok=True)
    local_orientations = run_directory / "orientations.csv"
    grains[["synthetic_grain_id", "phi1_deg", "Phi_deg", "phi2_deg", "phase"]].rename(
        columns={"synthetic_grain_id": "grain_id"}
    ).to_csv(local_orientations, index=False)

    mesh = run_directory / "periodic_tri6.msh"
    _run_checked(
        [
            sys.executable,
            str(CP_SCRIPT_DIR / "neper_to_moose.py"),
            str(tessellation),
            str(mesh),
            "--characteristic-length",
            str(characteristic_length_um),
            "--domain-size",
            "400",
            "--expected-grains",
            str(len(grains)),
            "--geometric-smoothing-passes",
            str(geometric_smoothing_passes),
            "--maximum-vertex-move-fraction",
            str(maximum_vertex_move_fraction),
            "--smoothing-activation-threshold",
            str(smoothing_activation_threshold),
        ],
        run_directory / "mesh_conversion_driver.log",
    )
    mesh_metrics = run_directory / "mesh_metrics.json"
    _run_checked(
        [
            sys.executable,
            str(CP_SCRIPT_DIR / "validate_periodic_mesh.py"),
            str(mesh),
            "--output",
            str(mesh_metrics),
            "--expected-grains",
            str(len(grains)),
            "--domain-size",
            "400",
        ],
        run_directory / "mesh_validation_driver.log",
    )
    _run_checked(
        [
            sys.executable,
            str(CP_SCRIPT_DIR / "texture_parameterization.py"),
            "--euler-csv",
            str(local_orientations),
            "--description",
            f"{dataset} {family} holdout seed {geometry_seed} FCC numerical assumption",
            "--output-dir",
            str(run_directory),
        ],
        run_directory / "material_generation.log",
    )
    material = run_directory / "cp_materials.i"
    case = _write_case(run_directory, material)
    output_base = run_directory / "solution"
    result_path = run_directory / "result.json"
    mesh_payload = json.loads(mesh_metrics.read_text(encoding="utf-8"))
    original_areas = grains["grain_area_um2"].to_numpy(dtype=float)
    mesh_areas = np.asarray(
        [mesh_payload["grain_area_um2"][str(value)] for value in expected_ids],
        dtype=float,
    )
    area_change = mesh_areas - original_areas
    conversion_payload = json.loads(
        mesh.with_suffix(".conversion.json").read_text(encoding="utf-8")
    )
    preflight = {
        "dataset": dataset,
        "family": family,
        "geometry_seed": geometry_seed,
        "grain_count": int(len(grains)),
        "characteristic_length_um": characteristic_length_um,
        "tessellation_sha256": sha256(tessellation),
        "mesh_sha256": sha256(mesh),
        "orientation_mapping_sha256": sha256(local_orientations),
        "phase_assumption": (
            "single-phase FCC numerical benchmark; numeric phase label 0 is not "
            "independently identified as a material phase by this workflow"
        ),
        "mesh_metrics": mesh_payload,
        "geometry_level_correction": {
            **conversion_payload["geometric_smoothing"],
            "grain_area_l1_change_fraction": float(
                np.sum(np.abs(area_change)) / np.sum(original_areas)
            ),
            "grain_area_correlation_before_after": float(
                np.corrcoef(original_areas, mesh_areas)[0, 1]
            ),
            "maximum_absolute_grain_area_change_um2": float(
                np.max(np.abs(area_change))
            ),
            "topology_changed": False,
            "grain_ids_changed": False,
        },
        "fem_executed": False,
    }
    if not run_fem:
        write_json(result_path, preflight)
        return preflight
    if not EXECUTABLE.is_file():
        raise FileNotFoundError(f"Missing MOOSE executable: {EXECUTABLE}")
    grain_values = " ".join(str(value) for value in expected_ids)
    command = [
        str(CONDA),
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
        f"Mesh/base/file={mesh}",
        "Mesh/pin/input=base",
        "Mesh/pin/coord='200 200 0'",
        "Mesh/pin/use_closest_node=true",
        "AuxKernels/grain_id/type=GrainPropertyAux",
        f"AuxKernels/grain_id/values='{grain_values}'",
        "H11=0.02",
        "cp_xm=0.10",
        "cp_local_solver=grouped",
        "cp_reuse_lu=true",
        "cp_optimize_tensors=true",
        "cp_optimize_scalars=true",
        "cp_predictor=previous",
        "steps=40",
        "Executioner/nl_rel_tol=1e-10",
        "Executioner/nl_abs_tol=1e-10",
        "Executioner/l_tol=1e-8",
        f"Outputs/csv/file_base={output_base}",
        f"Outputs/exodus/file_base={output_base}",
        "Outputs/perf_graph=false",
        "Outputs/sync_times='0.5 1.0'",
        "--allow-unused",
        "--disable-perf-graph-live",
    ]
    write_json(run_directory / "command.json", {"command": command, **preflight})
    csv_path = output_base.with_suffix(".csv")
    exodus_path = output_base.with_suffix(".e")
    log = run_directory / "run.log"
    if reuse_existing_fem_output:
        missing = [path for path in (csv_path, exodus_path, log) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Cannot reuse FEM output; missing " + ", ".join(map(str, missing))
            )
        completed_returncode = 0
        wall_time = float(pd.read_csv(csv_path)["benchmark_elapsed_seconds"].iloc[-1])
    else:
        start = time.perf_counter()
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=os.environ.copy(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        wall_time = time.perf_counter() - start
        completed_returncode = completed.returncode
    result = {
        **preflight,
        "fem_executed": True,
        "returncode": completed_returncode,
        "wall_time_s": wall_time,
        "mpi_ranks": mpi_ranks,
        "csv_path": str(csv_path),
        "exodus_path": str(exodus_path),
        "log_path": str(log),
    }
    if completed_returncode == 0 and csv_path.is_file() and exodus_path.is_file():
        if str(CP_SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(CP_SCRIPT_DIR))
        from extract_proof_stress import extract_properties  # noqa: PLC0415

        result.update(extract_properties(csv_path, exodus_path, len(grains)))
        result["converged"] = bool(result["all_checks_passed"])
    else:
        log_text = log.read_text(encoding="utf-8", errors="replace")
        result.update(
            {
                "converged": False,
                "failure_reason": "nonzero return or missing output",
                "last_log_lines": log_text.splitlines()[-40:],
                "failed_global_attempts": log_text.count("Solve Did NOT Converge!"),
            }
        )
    log_text = log.read_text(encoding="utf-8", errors="replace")
    result["failed_global_attempts"] = log_text.count("Solve Did NOT Converge!")
    rss = re.search(
        r"Maximum resident set size \(kbytes\):\s*(\d+)",
        log_text,
    )
    result["peak_rss_kib"] = int(rss.group(1)) if rss else None
    write_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--family", required=True, choices=("baseline", "improved"))
    parser.add_argument("--geometry-seed", type=int, default=10303)
    parser.add_argument("--characteristic-length", type=float, default=6.0)
    parser.add_argument("--mpi-ranks", type=int, default=8)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=V2_RESULTS_ROOT / "mechanical"
    )
    parser.add_argument("--run-fem", action="store_true")
    parser.add_argument("--geometric-smoothing-passes", type=int, default=30)
    parser.add_argument("--maximum-vertex-move-fraction", type=float, default=0.5)
    parser.add_argument("--smoothing-activation-threshold", type=float, default=0.012)
    parser.add_argument("--reuse-existing-fem-output", action="store_true")
    args = parser.parse_args()
    calibration = (
        args.calibration_root or V2_RESULTS_ROOT / "calibration" / args.dataset
    )
    result = build_and_run(
        dataset=args.dataset,
        family=args.family,
        geometry_seed=args.geometry_seed,
        characteristic_length_um=args.characteristic_length,
        mpi_ranks=args.mpi_ranks,
        calibration_root=calibration,
        output_root=args.output_root,
        run_fem=args.run_fem,
        geometric_smoothing_passes=args.geometric_smoothing_passes,
        maximum_vertex_move_fraction=args.maximum_vertex_move_fraction,
        smoothing_activation_threshold=args.smoothing_activation_threshold,
        reuse_existing_fem_output=args.reuse_existing_fem_output,
    )
    print(json.dumps(result, indent=2))
    return 0 if not args.run_fem or result.get("converged") else 2


if __name__ == "__main__":
    raise SystemExit(main())
