#!/usr/bin/env python3
"""Shared constants and configuration helpers for CP inverse design."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results" / "cp_inverse_design"
RESULTS = Path(os.environ.get("SPP2489_CP_ID_RESULTS", DEFAULT_RESULTS)).resolve()
# Candidate FEM values depend on eta and the complete numerical configuration, not on which
# hidden-target challenge consumes them. Share the content-addressed cache across challenges.
RUNS = Path(os.environ.get("SPP2489_CP_RUN_CACHE", DEFAULT_RESULTS / "runs")).resolve()
TARGET = RESULTS / "target"
BO = RESULTS / "bo"
RANDOM = RESULTS / "random_search"
VERIFICATION = RESULTS / "verification"
EXODUS = RESULTS / "exodus"
FIGURES = RESULTS / "figures"
ANIMATIONS = RESULTS / "animations"

EXECUTABLE = ROOT / "spp2489-opt"
BASE_INPUT = ROOT / "examples" / "crystal_plasticity" / "polycrystal_20_cp.i"
CENTERS_INPUT = ROOT / "examples" / "polycrystal_20_centers.i"
RANDOM_MATERIALS = ROOT / "examples" / "crystal_plasticity" / "cp_20_random_materials.i"
SLIP_SYSTEM_FILE = ROOT / "examples" / "crystal_plasticity" / "fcc_12_slip_systems.txt"

TARGET_PROPERTIES = RESULTS / "target_properties.json"
OPTIMIZATION_HISTORY = RESULTS / "optimization_history.csv"
RANDOM_HISTORY = RESULTS / "random_search_history.csv"
BEST_DESIGN = RESULTS / "best_design.json"
SUMMARY = RESULTS / "cp_inverse_design_summary.json"
REPORT = Path(os.environ.get("SPP2489_CP_ID_REPORT", ROOT / "CP_INVERSE_DESIGN_REPORT.md")).resolve()

PREFERRED_EULER_DEG = (37.0, 52.0, 11.0)
INITIAL_ETA = (0.0, 0.2, 0.35, 0.5, 0.8, 1.0)
MATERIAL_PARAMETERS = {
    "C_ijkl_MPa": [200000.0, 120000.0, 120000.0, 200000.0, 120000.0,
                    200000.0, 100000.0, 100000.0, 100000.0],
    "reference_slip_rate_per_s": 0.001,
    "rate_sensitivity": 0.10,
    "initial_slip_resistance_MPa": 80.0,
    "hardening_modulus_MPa": 400.0,
    "saturation_resistance_MPa": 160.0,
    "hardening_exponent": 1.0,
    "latent_hardening_ratio": 1.4,
}
A4_SOLVER = {
    "local_solver_mode": "grouped",
    "reuse_tangent_factorization": True,
    "optimized_tensor_evaluation": True,
    "specialize_scalar_operations": True,
    "local_predictor": "previous",
}
CONVERGENCE = {
    "nl_max_its": 25,
    "nl_rel_tol": 1.0e-6,
    "nl_abs_tol": 1.0e-10,
    "linear_tolerance": 1.0e-8,
    "local_absolute_tolerance": 1.0e-10,
    "local_relative_tolerance": 1.0e-9,
    "local_max_iterations": 60,
}


@dataclass(frozen=True)
class TimeProtocol:
    name: str
    mode: str
    steps: int
    sync_times: tuple[float, ...] = (0.5, 1.0)
    nl_max_its: int = 25

    def command_arguments(self) -> list[str]:
        args = [f"steps={self.steps}", "Outputs/sync_times='0.5 1.0'"]
        if self.mode == "fixed":
            args.extend(
                [
                    "Executioner/TimeStepper/type=ConstantDT",
                    f"Executioner/TimeStepper/dt={1.0 / self.steps:.17g}",
                    f"Executioner/nl_max_its={self.nl_max_its}",
                    "--allow-unused",
                ]
            )
        elif self.mode != "adaptive":
            raise ValueError(f"Unknown time-integration mode: {self.mode}")
        return args


PROTOCOLS = {
    "adaptive_current": TimeProtocol("adaptive_current", "adaptive", 20),
    "adaptive_strict": TimeProtocol("adaptive_strict", "adaptive", 40),
    "fixed_200": TimeProtocol("fixed_200", "fixed", 200, nl_max_its=80),
}
DEFAULT_PROTOCOL = PROTOCOLS["adaptive_strict"]


def ensure_directories() -> None:
    for path in (RESULTS, RUNS, TARGET, BO, RANDOM, VERIFICATION, EXODUS, FIGURES, ANIMATIONS):
        path.mkdir(parents=True, exist_ok=True)


def git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True, capture_output=True, check=True
    )
    return completed.stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def simulation_configuration(
    eta: float, mesh: int, protocol: TimeProtocol, mpi_ranks: int = 4
) -> dict[str, Any]:
    protocol_data = asdict(protocol)
    # JSON has arrays but no tuples. Canonicalize before comparison so a result reloaded from
    # result.json is byte-for-byte equivalent to the requested configuration.
    protocol_data["sync_times"] = list(protocol.sync_times)
    return {
        "eta": float(f"{eta:.15g}"),
        "mesh": int(mesh),
        "mpi_ranks": int(mpi_ranks),
        "physical_end_time_s": 1.0,
        "maximum_F11": 1.02,
        "time_protocol": protocol_data,
        "material_parameters": MATERIAL_PARAMETERS,
        "solver": A4_SOLVER,
        "convergence": CONVERGENCE,
        "application_commit": git_commit(ROOT),
        "moose_commit": git_commit(Path("/home/liu/projects/moose")),
        "centers_sha256": file_sha256(CENTERS_INPUT),
        "random_orientation_sha256": file_sha256(RANDOM_MATERIALS),
        "slip_system_sha256": file_sha256(SLIP_SYSTEM_FILE),
    }


def eta_tag(eta: float) -> str:
    return f"{eta:.10f}".replace(".", "p")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
