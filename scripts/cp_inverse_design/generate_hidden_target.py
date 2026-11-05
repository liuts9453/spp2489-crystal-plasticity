#!/usr/bin/env python3
"""Generate the hidden CP target; candidate optimization receives properties only."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from common import (
    CONVERGENCE,
    DEFAULT_PROTOCOL,
    MATERIAL_PARAMETERS,
    TARGET,
    TARGET_PROPERTIES,
    ensure_directories,
    write_json,
)
from cp_forward_solver import run_forward


# Deliberately confined to this target-generation module. The optimizer receives only a symbolic
# challenge name and the public target stresses; it never imports this mapping.
_REFERENCE_PROFILES = {"baseline": 0.65, "challenge_02": 0.90, "challenge_03": 0.70}


def _reference_design() -> tuple[str, float]:
    profile = os.environ.get("SPP2489_CP_TARGET_PROFILE", "baseline")
    if profile not in _REFERENCE_PROFILES:
        raise ValueError(f"Unknown hidden-target profile: {profile}")
    return profile, _REFERENCE_PROFILES[profile]


def generate_target(mesh: int = 32, reuse_cache: bool = True, mpi_ranks: int = 4) -> dict[str, object]:
    ensure_directories()
    profile, reference_eta = _reference_design()
    result = run_forward(
        reference_eta,
        mesh=mesh,
        protocol=DEFAULT_PROTOCOL,
        label=f"hidden_reference_n{mesh}",
        run_root=TARGET,
        reuse_cache=reuse_cache,
        mpi_ranks=mpi_ranks,
    )
    if not result.get("converged"):
        raise RuntimeError("Hidden reference FEM did not converge")
    # This public file is the only target artifact read by the optimizer. It contains no eta,
    # orientation path, run directory, cache key, or other route to the hidden design.
    target = {
        "schema": "spp2489_cp_target_properties_v1",
        "target_P11_at_F11_1p01_MPa": result["P11_at_F11_1p01_MPa"],
        "target_P11_at_F11_1p02_MPa": result["P11_at_F11_1p02_MPa"],
        "target_H_sec_MPa": result["H_sec_MPa"],
        "physical_loading_time_s": 1.0,
        "mesh": mesh,
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
    write_json(TARGET_PROPERTIES, target)
    write_json(
        TARGET / "private_hidden_reference_manifest.json",
        {"target_profile": profile, "eta_reference": reference_eta, "result": result},
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, default=32)
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--mpi-ranks", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(generate_target(args.mesh, not args.no_reuse, args.mpi_ranks), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
