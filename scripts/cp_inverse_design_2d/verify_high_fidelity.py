#!/usr/bin/env python3
"""Independent coarse, fine-mesh, and time-integration verification."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from common import (
    DEFAULT_PROTOCOL,
    EXODUS,
    PROTOCOLS,
    SUMMARY,
    TARGET,
    TARGET_PROPERTIES,
    VERIFICATION,
    write_json,
)
from cp_forward_solver import run_forward
from optimize_texture_2d import objective_and_errors


def _hidden_manifest() -> dict[str, object]:
    path = TARGET / "private_hidden_reference_manifest.json"
    if not path.exists():
        raise FileNotFoundError("Generate the hidden target before verification")
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_exodus(result: dict[str, object], name: str) -> str:
    EXODUS.mkdir(parents=True, exist_ok=True)
    destination = EXODUS / name
    shutil.copy2(Path(str(result["exodus_path"])), destination)
    return str(destination)


def _comparison(
    candidate: dict[str, object], reference: dict[str, object]
) -> dict[str, object]:
    e_proof, e_flow, objective = objective_and_errors(
        float(candidate["proof_stress_MPa"]),
        float(candidate["flow_stress_2pct_MPa"]),
        float(reference["proof_stress_MPa"]),
        float(reference["flow_stress_2pct_MPa"]),
    )
    return {
        "eta": candidate["eta"],
        "theta_deg": candidate["theta_deg"],
        "mesh_name": candidate["mesh_name"],
        "proof_stress_MPa": candidate["proof_stress_MPa"],
        "flow_stress_2pct_MPa": candidate["flow_stress_2pct_MPa"],
        "proof_extraction": candidate["proof_extraction"],
        "relative_error_proof": e_proof,
        "relative_error_flow": e_flow,
        "objective_J": objective,
        "wall_time_s": candidate["wall_time_s"],
        "accepted_increments": candidate["accepted_increments"],
        "failed_attempts": candidate["failed_attempts"],
        "total_newton_iterations": candidate["total_newton_iterations"],
        "mean_accumulated_slip": candidate["mean_accumulated_slip"],
        "maximum_accumulated_slip": candidate["maximum_accumulated_slip"],
        "maximum_abs_P22_MPa": candidate["maximum_abs_P22_MPa"],
        "hill_mandel_relative_residual_max": candidate[
            "hill_mandel_relative_residual_max"
        ],
        "maximum_periodicity_error": candidate["exodus"]["maximum_periodicity_error"],
        "all_checks_passed": candidate["all_checks_passed"],
        "csv_path": candidate["csv_path"],
        "exodus_path": candidate["exodus_path"],
        "run_directory": candidate["run_directory"],
        "full_curve": candidate["full_curve"],
    }


def _reference_summary(result: dict[str, object]) -> dict[str, object]:
    return {
        "eta": result["eta"],
        "theta_deg": result["theta_deg"],
        "mesh_name": result["mesh_name"],
        "protocol": result["protocol"],
        "proof_stress_MPa": result["proof_stress_MPa"],
        "flow_stress_2pct_MPa": result["flow_stress_2pct_MPa"],
        "proof_extraction": result["proof_extraction"],
        "initial_effective_stiffness_MPa": result["initial_effective_stiffness_MPa"],
        "wall_time_s": result["wall_time_s"],
        "accepted_increments": result["accepted_increments"],
        "failed_attempts": result["failed_attempts"],
        "total_newton_iterations": result["total_newton_iterations"],
        "mean_accumulated_slip": result["mean_accumulated_slip"],
        "maximum_accumulated_slip": result["maximum_accumulated_slip"],
        "all_checks_passed": result["all_checks_passed"],
        "csv_path": result["csv_path"],
        "exodus_path": result["exodus_path"],
        "run_directory": result["run_directory"],
        "full_curve": result["full_curve"],
    }


def verify_design(
    eta: float,
    theta_deg: float,
    *,
    tolerance: float = 0.01,
    mpi_ranks: int = 8,
    run_fine: bool = True,
    run_time_study: bool = True,
) -> dict[str, object]:
    manifest = _hidden_manifest()
    reference_eta = float(manifest["eta_reference"])
    reference_theta = float(manifest["theta_reference_deg"])
    target_coarse = manifest["result"]

    candidate_coarse = run_forward(
        eta,
        theta_deg,
        mesh_name="coarse",
        protocol=DEFAULT_PROTOCOL,
        label="optimized_independent_coarse",
        run_root=VERIFICATION / "independent",
        reuse_cache=False,
        mpi_ranks=mpi_ranks,
    )
    if not candidate_coarse.get("converged"):
        raise RuntimeError("Independent coarse candidate did not pass validation")
    coarse = _comparison(candidate_coarse, target_coarse)
    coarse["meets_dual_tolerance"] = bool(
        coarse["relative_error_proof"] < tolerance
        and coarse["relative_error_flow"] < tolerance
    )
    coarse["copied_exodus_path"] = _copy_exodus(candidate_coarse, "optimized_coarse.e")
    target_coarse_summary = _reference_summary(target_coarse)
    target_coarse_summary["copied_exodus_path"] = _copy_exodus(
        target_coarse, "hidden_target_coarse.e"
    )

    verification: dict[str, object] = {
        "hidden_design_disclosed_after_optimization": {
            "eta": reference_eta,
            "theta_deg": reference_theta,
        },
        "coarse_target": target_coarse_summary,
        "independent_coarse_candidate": coarse,
        "fine_requested": run_fine,
        "time_study_requested": run_time_study,
    }

    if run_fine:
        target_fine = run_forward(
            reference_eta,
            reference_theta,
            mesh_name="fine",
            protocol=DEFAULT_PROTOCOL,
            label="hidden_reference_fine",
            run_root=VERIFICATION / "fine",
            reuse_cache=False,
            mpi_ranks=mpi_ranks,
        )
        candidate_fine = run_forward(
            eta,
            theta_deg,
            mesh_name="fine",
            protocol=DEFAULT_PROTOCOL,
            label="optimized_fine",
            run_root=VERIFICATION / "fine",
            reuse_cache=False,
            mpi_ranks=mpi_ranks,
        )
        if not target_fine.get("converged") or not candidate_fine.get("converged"):
            raise RuntimeError("One or both fine-mesh verification runs failed")
        fine = _comparison(candidate_fine, target_fine)
        fine["meets_dual_tolerance"] = bool(
            fine["relative_error_proof"] < tolerance
            and fine["relative_error_flow"] < tolerance
        )
        fine["copied_exodus_path"] = _copy_exodus(candidate_fine, "optimized_fine.e")
        target_fine_summary = _reference_summary(target_fine)
        target_fine_summary["copied_exodus_path"] = _copy_exodus(
            target_fine, "hidden_target_fine.e"
        )
        verification["fine_target"] = target_fine_summary
        verification["fine_candidate"] = fine
        verification["target_coarse_to_fine"] = {
            "relative_proof_change": abs(
                float(target_fine["proof_stress_MPa"])
                - float(target_coarse["proof_stress_MPa"])
            )
            / abs(float(target_fine["proof_stress_MPa"])),
            "relative_flow_change": abs(
                float(target_fine["flow_stress_2pct_MPa"])
                - float(target_coarse["flow_stress_2pct_MPa"])
            )
            / abs(float(target_fine["flow_stress_2pct_MPa"])),
        }
        verification["candidate_coarse_to_fine"] = {
            "relative_proof_change": abs(
                float(candidate_fine["proof_stress_MPa"])
                - float(candidate_coarse["proof_stress_MPa"])
            )
            / abs(float(candidate_fine["proof_stress_MPa"])),
            "relative_flow_change": abs(
                float(candidate_fine["flow_stress_2pct_MPa"])
                - float(candidate_coarse["flow_stress_2pct_MPa"])
            )
            / abs(float(candidate_fine["flow_stress_2pct_MPa"])),
        }

    if run_time_study:
        time_target = run_forward(
            reference_eta,
            reference_theta,
            mesh_name="coarse",
            protocol=PROTOCOLS["adaptive_refined"],
            label="hidden_reference_time_refined",
            run_root=VERIFICATION / "time_integration",
            reuse_cache=False,
            mpi_ranks=mpi_ranks,
        )
        time_candidate = run_forward(
            eta,
            theta_deg,
            mesh_name="coarse",
            protocol=PROTOCOLS["adaptive_refined"],
            label="optimized_time_refined",
            run_root=VERIFICATION / "time_integration",
            reuse_cache=False,
            mpi_ranks=mpi_ranks,
        )
        if not time_target.get("converged") or not time_candidate.get("converged"):
            raise RuntimeError("One or both refined-time verification runs failed")
        verification["time_integration"] = {
            "target": _reference_summary(time_target),
            "candidate": _comparison(time_candidate, time_target),
            "target_default_to_refined": {
                "relative_proof_change": abs(
                    float(time_target["proof_stress_MPa"])
                    - float(target_coarse["proof_stress_MPa"])
                )
                / abs(float(time_target["proof_stress_MPa"])),
                "relative_flow_change": abs(
                    float(time_target["flow_stress_2pct_MPa"])
                    - float(target_coarse["flow_stress_2pct_MPa"])
                )
                / abs(float(time_target["flow_stress_2pct_MPa"])),
            },
            "candidate_default_to_refined": {
                "relative_proof_change": abs(
                    float(time_candidate["proof_stress_MPa"])
                    - float(candidate_coarse["proof_stress_MPa"])
                )
                / abs(float(time_candidate["proof_stress_MPa"])),
                "relative_flow_change": abs(
                    float(time_candidate["flow_stress_2pct_MPa"])
                    - float(candidate_coarse["flow_stress_2pct_MPa"])
                )
                / abs(float(time_candidate["flow_stress_2pct_MPa"])),
            },
        }

    write_json(VERIFICATION / "verification_summary.json", verification)
    if SUMMARY.exists():
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        summary["verification"] = verification
        write_json(SUMMARY, summary)
    return verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--target-tolerance", type=float, default=0.01)
    parser.add_argument("--mpi-ranks", type=int, default=8)
    parser.add_argument("--no-fine", action="store_true")
    parser.add_argument("--no-time-study", action="store_true")
    args = parser.parse_args()
    target = json.loads(TARGET_PROPERTIES.read_text(encoding="utf-8"))
    if not target.get("converged"):
        raise RuntimeError("Public target is missing or invalid")
    result = verify_design(
        args.eta,
        args.theta,
        tolerance=args.target_tolerance,
        mpi_ranks=args.mpi_ranks,
        run_fine=not args.no_fine,
        run_time_study=not args.no_time_study,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
