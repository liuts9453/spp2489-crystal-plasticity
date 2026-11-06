#!/usr/bin/env python3
"""Generate the private 2D reference and a non-leaking public target file."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from common import (
    CONVERGENCE,
    DEFAULT_PROTOCOL,
    MATERIAL_PARAMETERS,
    MESH_FILES,
    TARGET,
    TARGET_PROPERTIES,
    ensure_directories,
    file_sha256,
    write_json,
)
from cp_forward_solver import run_forward


# These values are deliberately confined to the target generator and private
# manifest. Candidate generation and Bayesian optimization read only the public
# property file written below.
REFERENCE_ETA = 0.84
REFERENCE_THETA_DEG = -8.5

# The hidden point is selected before optimization from a post-hoc GP fit to
# the nine fixed DOE responses.  The selection maximizes the predicted
# Chebyshev relative property distance from every DOE point on an independent
# grid, subject to a minimum normalized design-space separation of 0.075.
# Only this private manifest records the selection; the optimizer receives the
# two target properties and cannot access the hidden design.
TARGET_SELECTION = {
    "method": "maximize_predicted_minimum_Chebyshev_property_distance_from_initial_DOE",
    "predicted_proof_stress_MPa": 366.26,
    "predicted_flow_stress_2pct_MPa": 424.56,
    "predicted_minimum_maximum_relative_property_distance": 0.0594,
    "minimum_normalized_design_distance": 0.075,
    "selection_grid": {"eta_step": 0.01, "theta_step_deg": 0.5},
}


def generate_target(reuse_cache: bool = True, mpi_ranks: int = 8) -> dict[str, object]:
    ensure_directories()
    result = run_forward(
        REFERENCE_ETA,
        REFERENCE_THETA_DEG,
        mesh_name="coarse",
        protocol=DEFAULT_PROTOCOL,
        label="hidden_reference_coarse",
        run_root=TARGET / "private_runs",
        reuse_cache=reuse_cache,
        mpi_ranks=mpi_ranks,
    )
    if not result.get("converged"):
        raise RuntimeError("Hidden-reference MOOSE FEM did not pass validation")

    target = {
        "schema": "spp2489_cp_inverse_design_2d_target_v1",
        "target_proof_stress_MPa": result["proof_stress_MPa"],
        "target_flow_stress_2pct_MPa": result["flow_stress_2pct_MPa"],
        "proof_stress_definition": result["proof_extraction"]["definition"],
        "physical_loading_time_s": 1.0,
        "loading_rate_per_s": 0.02,
        "mesh_name": "coarse",
        "mesh_sha256": file_sha256(MESH_FILES["coarse"]),
        "mpi_ranks": mpi_ranks,
        "time_protocol": asdict(DEFAULT_PROTOCOL),
        "material_parameters": MATERIAL_PARAMETERS,
        "convergence_tolerances": CONVERGENCE,
        "application_commit": result["configuration"]["application_commit"],
        "moose_commit": result["configuration"]["moose_commit"],
        "converged": True,
        "accepted_increments": result["accepted_increments"],
        "failed_attempts": result["failed_attempts"],
        "total_newton_iterations": result["total_newton_iterations"],
        "wall_time_s": result["wall_time_s"],
        "physical_checks": result["checks"],
    }
    forbidden = {"eta", "theta", "orientation_path", "run_directory", "cache_key"}
    if forbidden.intersection(target):
        raise RuntimeError("Public target payload contains hidden design information")
    write_json(TARGET_PROPERTIES, target)
    write_json(
        TARGET / "private_hidden_reference_manifest.json",
        {
            "eta_reference": REFERENCE_ETA,
            "theta_reference_deg": REFERENCE_THETA_DEG,
            "selection": TARGET_SELECTION,
            "result": result,
        },
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--mpi-ranks", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(generate_target(not args.no_reuse, args.mpi_ranks), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
