#!/usr/bin/env python3
"""Independent low/high-fidelity verification for the optimized CP texture."""

from __future__ import annotations

import argparse
import json
import shutil

from common import DEFAULT_PROTOCOL, EXODUS, TARGET, TARGET_PROPERTIES, VERIFICATION, write_json
from cp_forward_solver import run_forward
from random_search_baseline import errors_and_objective


def _hidden_manifest() -> dict[str, object]:
    path = TARGET / "private_hidden_reference_manifest.json"
    if not path.exists():
        raise FileNotFoundError("Generate the hidden target before verification")
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_exodus(result: dict[str, object], name: str) -> str:
    EXODUS.mkdir(parents=True, exist_ok=True)
    destination = EXODUS / name
    shutil.copy2(str(result["exodus_path"]), destination)
    return str(destination)


def _comparison(result: dict[str, object], target: dict[str, object]) -> dict[str, object]:
    e1, e2, objective = errors_and_objective(result, target)
    return {
        "eta": result["eta"],
        "mesh": result["mesh"],
        "P11_1pct_MPa": result["P11_at_F11_1p01_MPa"],
        "P11_2pct_MPa": result["P11_at_F11_1p02_MPa"],
        "H_sec_MPa": result["H_sec_MPa"],
        "relative_error_1pct": e1,
        "relative_error_2pct": e2,
        "objective_J": objective,
        "mean_accumulated_slip": result["mean_accumulated_slip"],
        "maximum_accumulated_slip": result["maximum_accumulated_slip"],
        "mean_slip_resistance_MPa": result["mean_slip_resistance_MPa"],
        "minimum_local_slip_resistance_MPa": result["minimum_local_slip_resistance_MPa"],
        "maximum_local_P11_MPa": result["maximum_local_P11_MPa"],
        "maximum_abs_P22_MPa": result["maximum_abs_P22_MPa"],
        "maximum_abs_macro_shear": result["maximum_abs_macro_shear"],
        "minimum_det_Fp": result["minimum_det_Fp"],
        "maximum_det_Fp": result["maximum_det_Fp"],
        "plastic_dissipation": result["plastic_dissipation"],
        "hill_mandel_relative_residual_max": result["hill_mandel_relative_residual_max"],
        "maximum_periodicity_error": result["exodus"]["maximum_periodicity_error"],
        "all_physical_checks_passed": result["all_checks_passed"],
        "wall_time_s": result["wall_time_s"],
        "accepted_increments": result["accepted_increments"],
        "failed_attempts": result["failed_attempts"],
        "total_newton_iterations": result["total_newton_iterations"],
        "csv_path": result["csv_path"],
        "exodus_path": result["exodus_path"],
        "run_directory": result["run_directory"],
        "full_curve": result["full_curve"],
    }


def verify_design(
    best_eta: float,
    target: dict[str, object],
    *,
    mesh: int = 32,
    tolerance: float = 0.01,
    high_fidelity: bool = False,
    mpi_ranks: int = 4,
) -> dict[str, object]:
    manifest = _hidden_manifest()
    reference_eta = float(manifest["eta_reference"])
    low_target_result = manifest["result"]
    if int(low_target_result["mesh"]) != mesh:
        raise RuntimeError("Hidden target and optimized candidate do not use the same low-fidelity mesh")
    best = run_forward(
        best_eta,
        mesh=mesh,
        protocol=DEFAULT_PROTOCOL,
        label=f"best_independent_n{mesh}",
        run_root=VERIFICATION / "independent",
        reuse_cache=False,
        mpi_ranks=mpi_ranks,
    )
    if not best.get("converged"):
        raise RuntimeError("Independent optimized-design FEM did not converge")
    low = _comparison(best, target)
    low["meets_tolerance"] = bool(
        low["relative_error_1pct"] <= tolerance and low["relative_error_2pct"] <= tolerance
    )
    low["copied_exodus_path"] = _copy_exodus(best, f"optimized_n{mesh}.e")
    target_low_exodus = _copy_exodus(low_target_result, f"hidden_target_n{mesh}.e")
    verification: dict[str, object] = {
        "reference_eta_disclosed_after_optimization": reference_eta,
        "independent_low_fidelity": low,
        "low_fidelity_target": {
            "mesh": mesh,
            "P11_1pct_MPa": target["target_P11_at_F11_1p01_MPa"],
            "P11_2pct_MPa": target["target_P11_at_F11_1p02_MPa"],
            "H_sec_MPa": target["target_H_sec_MPa"],
            "csv_path": low_target_result["csv_path"],
            "exodus_path": low_target_result["exodus_path"],
            "copied_exodus_path": target_low_exodus,
            "full_curve": low_target_result["full_curve"],
            "mean_accumulated_slip": low_target_result["mean_accumulated_slip"],
            "maximum_accumulated_slip": low_target_result["maximum_accumulated_slip"],
        },
        "high_fidelity_requested": high_fidelity,
    }
    if high_fidelity:
        reference64 = run_forward(
            reference_eta,
            mesh=64,
            protocol=DEFAULT_PROTOCOL,
            label="hidden_reference_n64",
            run_root=VERIFICATION / "high_fidelity",
            reuse_cache=False,
            mpi_ranks=mpi_ranks,
        )
        candidate64 = run_forward(
            best_eta,
            mesh=64,
            protocol=DEFAULT_PROTOCOL,
            label="optimized_n64",
            run_root=VERIFICATION / "high_fidelity",
            reuse_cache=False,
            mpi_ranks=mpi_ranks,
        )
        if not reference64.get("converged") or not candidate64.get("converged"):
            raise RuntimeError("One or both 64x64 verification simulations failed")
        high_target = {
            "target_P11_at_F11_1p01_MPa": reference64["P11_at_F11_1p01_MPa"],
            "target_P11_at_F11_1p02_MPa": reference64["P11_at_F11_1p02_MPa"],
        }
        high = _comparison(candidate64, high_target)
        high["meets_tolerance"] = bool(
            high["relative_error_1pct"] <= tolerance and high["relative_error_2pct"] <= tolerance
        )
        high["same_mesh_comparison"] = int(reference64["mesh"]) == int(candidate64["mesh"]) == 64
        high["copied_exodus_path"] = _copy_exodus(candidate64, "optimized_n64.e")
        verification["high_fidelity_candidate"] = high
        verification["high_fidelity_target"] = {
            "mesh": 64,
            "P11_1pct_MPa": reference64["P11_at_F11_1p01_MPa"],
            "P11_2pct_MPa": reference64["P11_at_F11_1p02_MPa"],
            "H_sec_MPa": reference64["H_sec_MPa"],
            "mean_accumulated_slip": reference64["mean_accumulated_slip"],
            "maximum_accumulated_slip": reference64["maximum_accumulated_slip"],
            "wall_time_s": reference64["wall_time_s"],
            "accepted_increments": reference64["accepted_increments"],
            "failed_attempts": reference64["failed_attempts"],
            "total_newton_iterations": reference64["total_newton_iterations"],
            "csv_path": reference64["csv_path"],
            "exodus_path": reference64["exodus_path"],
            "copied_exodus_path": _copy_exodus(reference64, "hidden_target_n64.e"),
            "full_curve": reference64["full_curve"],
        }
        verification["target_mesh_change_32_to_64"] = {
            "relative_P11_1pct": abs(
                reference64["P11_at_F11_1p01_MPa"] - target["target_P11_at_F11_1p01_MPa"]
            )
            / abs(reference64["P11_at_F11_1p01_MPa"]),
            "relative_P11_2pct": abs(
                reference64["P11_at_F11_1p02_MPa"] - target["target_P11_at_F11_1p02_MPa"]
            )
            / abs(reference64["P11_at_F11_1p02_MPa"]),
        }
        verification["candidate_mesh_change_32_to_64"] = {
            "relative_P11_1pct": abs(
                candidate64["P11_at_F11_1p01_MPa"] - best["P11_at_F11_1p01_MPa"]
            )
            / abs(candidate64["P11_at_F11_1p01_MPa"]),
            "relative_P11_2pct": abs(
                candidate64["P11_at_F11_1p02_MPa"] - best["P11_at_F11_1p02_MPa"]
            )
            / abs(candidate64["P11_at_F11_1p02_MPa"]),
        }
    write_json(VERIFICATION / "verification_summary.json", verification)
    return verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--mesh", type=int, default=32)
    parser.add_argument("--target-tolerance", type=float, default=0.01)
    parser.add_argument("--high-fidelity", action="store_true")
    parser.add_argument("--mpi-ranks", type=int, default=4)
    args = parser.parse_args()
    target = json.loads(TARGET_PROPERTIES.read_text(encoding="utf-8"))
    result = verify_design(
        args.eta,
        target,
        mesh=args.mesh,
        tolerance=args.target_tolerance,
        high_fidelity=args.high_fidelity,
        mpi_ranks=args.mpi_ranks,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
