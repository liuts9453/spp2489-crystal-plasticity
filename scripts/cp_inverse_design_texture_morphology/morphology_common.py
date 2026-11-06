#!/usr/bin/env python3
"""Paths and immutable configuration for morphology--texture inverse design."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ID2_SCRIPT_DIR = SCRIPT_DIR.parent / "cp_inverse_design_2d"
if str(ID2_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(ID2_SCRIPT_DIR))

from common import (  # noqa: E402
    GMSH_EXECUTABLE,
    NEPER_EXECUTABLE,
    PROTOCOLS,
    ROOT,
    file_sha256,
    write_json,
)

RESULTS = ROOT / "results" / "cp_inverse_design_texture_morphology"
MORPHOLOGY = RESULTS / "morphology"
MESHES = RESULTS / "meshes"
SENSITIVITY = RESULTS / "sensitivity"
RUNS = RESULTS / "runs"
OPTIMIZATION = RESULTS / "optimization"
VERIFICATION = RESULTS / "verification"
FIGURES = RESULTS / "figures"
ANIMATIONS = RESULTS / "animations"

OLD_RESULTS = ROOT / "results" / "cp_inverse_design_2d"
OLD_TESSELLATION = OLD_RESULTS / "neper" / "rve20_periodic.tess"
OLD_MESHES = {
    "coarse": OLD_RESULTS / "meshes" / "rve20_coarse_tri6.msh",
    "fine": OLD_RESULTS / "meshes" / "rve20_fine_tri6.msh",
}
OLD_BEST_DESIGN = OLD_RESULTS / "best_design.json"
OLD_MESH_METRICS = {
    name: OLD_RESULTS / "meshes" / f"{name}_mesh_metrics.json"
    for name in ("coarse", "fine")
}

MORPHOLOGY_LEVELS = (0.70, 0.85, 1.00, 1.25, 1.50)
MORPHOLOGY_BOUNDS = (0.70, 1.60)
BASE_GEOMETRY_SEED = 2026
ROBUSTNESS_GEOMETRY_SEEDS = (2027, 2028)
MESH_CHARACTERISTIC_LENGTH_UM = {"coarse": 6.0, "fine": 4.0}
MINIMUM_SCREENING_QUALITY = 0.025


def ensure_directories() -> None:
    for path in (
        RESULTS,
        MORPHOLOGY,
        MESHES,
        SENSITIVITY,
        RUNS,
        OPTIMIZATION,
        VERIFICATION,
        FIGURES,
        ANIMATIONS,
    ):
        path.mkdir(parents=True, exist_ok=True)


def morphology_tag(a_value: float) -> str:
    return f"a_{a_value:.6f}".replace(".", "p")


def geometry_directory(seed: int, a_value: float) -> Path:
    return MORPHOLOGY / f"seed_{seed}" / morphology_tag(a_value)


def tessellation_path(seed: int, a_value: float) -> Path:
    return geometry_directory(seed, a_value) / "rve20_periodic.tess"


def mesh_directory(seed: int, a_value: float) -> Path:
    return MESHES / f"seed_{seed}" / morphology_tag(a_value)


def mesh_path(seed: int, a_value: float, mesh_name: str) -> Path:
    if mesh_name not in MESH_CHARACTERISTIC_LENGTH_UM:
        raise ValueError(f"Unknown mesh fidelity: {mesh_name}")
    return mesh_directory(seed, a_value) / f"rve20_{mesh_name}_tri6.msh"


def mesh_metrics_path(seed: int, a_value: float, mesh_name: str) -> Path:
    return mesh_directory(seed, a_value) / f"{mesh_name}_mesh_metrics.json"


def baseline_texture() -> tuple[float, float]:
    data = json.loads(OLD_BEST_DESIGN.read_text(encoding="utf-8"))
    eta = float(data["eta"])
    theta = float(data["theta_deg"])
    if abs(eta - 0.7729026253348713) > 1.0e-12 or abs(
        theta - 4.242186925858952
    ) > 1.0e-11:
        raise RuntimeError("Existing optimized texture does not match commit be198402")
    return eta, theta


def geometry_identity(seed: int, a_value: float, mesh_name: str) -> dict[str, object]:
    tessellation = tessellation_path(seed, a_value)
    mesh = mesh_path(seed, a_value, mesh_name)
    metrics = mesh_metrics_path(seed, a_value, mesh_name)
    return {
        "morphology_input_a": float(a_value),
        "geometry_seed": int(seed),
        "tessellation_sha256": file_sha256(tessellation),
        "mesh_sha256": file_sha256(mesh),
        "mesh_metrics_sha256": file_sha256(metrics),
        "grain_id_mapping": list(range(1, 21)),
        "generator": "Neper 5.0.0 voronoi,aspratio(a,1)",
    }


def write_result(path: Path, payload: object) -> None:
    write_json(path, payload)


DEFAULT_PROTOCOL = PROTOCOLS["adaptive_strict"]
