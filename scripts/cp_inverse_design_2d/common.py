#!/usr/bin/env python3
"""Shared paths and immutable numerical configuration for the 2D workflow."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(
    os.environ.get(
        "SPP2489_CP_ID_2D_RESULTS", ROOT / "results" / "cp_inverse_design_2d"
    )
).resolve()
NEPER = RESULTS / "neper"
MESHES = RESULTS / "meshes"
RUNS = RESULTS / "runs"
TARGET = RESULTS / "target"
VERIFICATION = RESULTS / "verification"
FIGURES = RESULTS / "figures"
EXODUS = RESULTS / "exodus"
INTERACTIVE = RESULTS / "interactive"

EXECUTABLE = ROOT / "spp2489-opt"
BASE_INPUT = ROOT / "examples" / "crystal_plasticity" / "polycrystal_20_cp.i"
RANDOM_MATERIALS = ROOT / "examples" / "crystal_plasticity" / "cp_20_random_materials.i"
SLIP_SYSTEM_FILE = ROOT / "examples" / "crystal_plasticity" / "fcc_12_slip_systems.txt"

NEPER_EXECUTABLE = Path(
    os.environ.get(
        "SPP2489_NEPER",
        "/home/liu/miniforge/envs/spp2489-neper/bin/neper",
    )
)
GMSH_EXECUTABLE = Path(os.environ.get("SPP2489_GMSH", "/usr/bin/gmsh"))
CONDA_EXECUTABLE = Path(
    os.environ.get("SPP2489_CONDA", "/home/liu/miniforge/bin/conda")
)

TESS_FILE = NEPER / "rve20_periodic.tess"
MESH_FILES = {
    "coarse": MESHES / "rve20_coarse_tri6.msh",
    "fine": MESHES / "rve20_fine_tri6.msh",
}
MESH_CHARACTERISTIC_LENGTH_UM = {"coarse": 6.0, "fine": 4.0}

TARGET_PROPERTIES = RESULTS / "target_properties.json"
OPTIMIZATION_HISTORY = RESULTS / "optimization_history.csv"
BEST_DESIGN = RESULTS / "best_design.json"
SUMMARY = RESULTS / "cp_inverse_design_2d_summary.json"
REPORT = ROOT / "CP_INVERSE_DESIGN_2D_REPORT.md"

PREFERRED_EULER_DEG = (37.0, 52.0, 11.0)
ETA_BOUNDS = (0.1, 0.9)
THETA_BOUNDS_DEG = (-45.0, 45.0)
INITIAL_DESIGNS = tuple(
    (eta, theta) for eta in (0.15, 0.50, 0.85) for theta in (-30.0, 0.0, 30.0)
)

MATERIAL_PARAMETERS = {
    "C_ijkl_MPa": [
        200000.0,
        120000.0,
        120000.0,
        200000.0,
        120000.0,
        200000.0,
        100000.0,
        100000.0,
        100000.0,
    ],
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
    # The scalar stress constraint is an integral over the 200 x 200 um cell.
    # A tighter global tolerance is required on the Neper mesh to prevent the
    # automatically scaled scalar equation from terminating with a visible
    # nonzero macroscopic P22.
    "nl_rel_tol": 1.0e-10,
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
                ]
            )
        elif self.mode != "adaptive":
            raise ValueError(f"Unknown time-integration mode: {self.mode}")
        return args


PROTOCOLS = {
    "adaptive_strict": TimeProtocol("adaptive_strict", "adaptive", 40),
    "adaptive_refined": TimeProtocol("adaptive_refined", "adaptive", 80),
    "fixed_200": TimeProtocol("fixed_200", "fixed", 200, nl_max_its=80),
}
DEFAULT_PROTOCOL = PROTOCOLS["adaptive_strict"]


def ensure_directories() -> None:
    for path in (
        RESULTS,
        NEPER,
        MESHES,
        RUNS,
        TARGET,
        VERIFICATION,
        FIGURES,
        EXODUS,
        INTERACTIVE,
    ):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def simulation_configuration(
    eta: float,
    theta_deg: float,
    mesh_name: str,
    protocol: TimeProtocol,
    mpi_ranks: int,
) -> dict[str, Any]:
    if mesh_name not in MESH_FILES:
        raise ValueError(f"Unknown mesh: {mesh_name}")
    mesh = MESH_FILES[mesh_name]
    protocol_data = asdict(protocol)
    protocol_data["sync_times"] = list(protocol.sync_times)
    return {
        "eta": float(f"{eta:.15g}"),
        "theta_deg": float(f"{theta_deg:.15g}"),
        "mesh_name": mesh_name,
        "mesh_sha256": file_sha256(mesh),
        "mesh_manifest_sha256": file_sha256(MESHES / f"{mesh_name}_mesh_metrics.json"),
        "mpi_ranks": int(mpi_ranks),
        "physical_end_time_s": 1.0,
        "maximum_F11": 1.02,
        "loading_rate_per_s": 0.02,
        "macroscopic_constraints": {"P22_MPa": 0.0, "F12": 0.0, "F21": 0.0, "F33": 1.0},
        "fluctuation_translation_gauge": {
            "method": "closest_interior_mesh_node",
            "coordinate_um": [100.0, 100.0, 0.0],
        },
        "time_protocol": protocol_data,
        "material_parameters": MATERIAL_PARAMETERS,
        "solver": A4_SOLVER,
        "convergence": CONVERGENCE,
        "application_commit": git_commit(ROOT),
        "moose_commit": git_commit(Path("/home/liu/projects/moose")),
        "tessellation_sha256": file_sha256(TESS_FILE),
        "random_orientation_sha256": file_sha256(RANDOM_MATERIALS),
        "slip_system_sha256": file_sha256(SLIP_SYSTEM_FILE),
    }


def design_tag(eta: float, theta_deg: float) -> str:
    eta_text = f"{eta:.9f}".replace(".", "p")
    theta_text = (
        f"{theta_deg:+.7f}".replace("+", "p").replace("-", "m").replace(".", "p")
    )
    return f"eta_{eta_text}_theta_{theta_text}"
