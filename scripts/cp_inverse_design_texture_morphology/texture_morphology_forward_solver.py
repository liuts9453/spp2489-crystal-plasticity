#!/usr/bin/env python3
"""Run one cache-safe FCC-CP texture--morphology forward evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ID2_SCRIPT_DIR = SCRIPT_DIR.parent / "cp_inverse_design_2d"
for directory in (SCRIPT_DIR, ID2_SCRIPT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common import PROTOCOLS, TimeProtocol, file_sha256, write_json  # noqa: E402
from cp_forward_solver import run_forward  # noqa: E402
from generate_morphology import generate_design  # noqa: E402
from morphology_common import (  # noqa: E402
    BASE_GEOMETRY_SEED,
    DEFAULT_PROTOCOL,
    RUNS,
    baseline_texture,
    geometry_directory,
    geometry_identity,
    mesh_metrics_path,
    mesh_path,
    morphology_tag,
    tessellation_path,
)
from texture_parameterization import orientations_for_design  # noqa: E402


def _orientation_identity(eta: float, theta_deg: float) -> dict[str, object]:
    eulers, quaternions, metrics = orientations_for_design(eta, theta_deg)
    return {
        "grain_ids": list(range(1, 21)),
        "euler_angles_deg": eulers.tolist(),
        "quaternions": quaternions.tolist(),
        "so3_validation": metrics,
    }


def run_texture_morphology_forward(
    eta: float,
    theta_deg: float,
    a_value: float,
    *,
    geometry_seed: int = BASE_GEOMETRY_SEED,
    mesh_name: str = "coarse",
    protocol: TimeProtocol = DEFAULT_PROTOCOL,
    mpi_ranks: int = 8,
    label_prefix: str = "morphology",
    reuse_cache: bool = True,
) -> dict[str, object]:
    mesh = mesh_path(geometry_seed, a_value, mesh_name)
    metrics_path = mesh_metrics_path(geometry_seed, a_value, mesh_name)
    tessellation = tessellation_path(geometry_seed, a_value)
    if not mesh.exists() or not metrics_path.exists() or not tessellation.exists():
        generate_design(a_value, geometry_seed, (mesh_name,))
    geometry = geometry_identity(geometry_seed, a_value, mesh_name)
    orientation = _orientation_identity(eta, theta_deg)
    geometry["orientation_mapping_sha256"] = file_sha256(
        SCRIPT_DIR.parent
        / "cp_inverse_design_2d"
        / "texture_parameterization.py"
    )
    geometry["orientation_mapping"] = orientation
    label = (
        f"{label_prefix}_seed_{geometry_seed}_{morphology_tag(a_value)}_"
        f"eta_{eta:.9f}_theta_{theta_deg:+.7f}_{mesh_name}_{protocol.name}"
    )
    result = run_forward(
        eta,
        theta_deg,
        mesh_name=mesh_name,
        protocol=protocol,
        label=label,
        run_root=RUNS,
        reuse_cache=reuse_cache,
        mpi_ranks=mpi_ranks,
        mesh_path=mesh,
        mesh_manifest_path=metrics_path,
        tessellation_path=tessellation,
        configuration_extra=geometry,
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    statistics = json.loads(
        (geometry_directory(geometry_seed, a_value) / "morphology_statistics.json").read_text(
            encoding="utf-8"
        )
    )
    augmented = {
        **result,
        "morphology_input_a": float(a_value),
        "geometry_seed": int(geometry_seed),
        "achieved_directional_extent_ratio_x_over_y": statistics[
            "area_weighted_directional_extent_ratio_x_over_y"
        ],
        "area_coefficient_of_variation": statistics[
            "area_coefficient_of_variation"
        ],
        "area_weighted_orientation_statistics": statistics[
            "orientation_statistics"
        ],
        "mesh_metrics": metrics,
    }
    run_directory = Path(str(result["run_directory"]))
    write_json(run_directory / "texture_morphology_result.json", augmented)
    return augmented


def main() -> int:
    eta_default, theta_default = baseline_texture()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eta", type=float, default=eta_default)
    parser.add_argument("--theta", type=float, default=theta_default)
    parser.add_argument("--a", type=float, required=True)
    parser.add_argument("--seed", type=int, default=BASE_GEOMETRY_SEED)
    parser.add_argument("--mesh", choices=("coarse", "fine"), default="coarse")
    parser.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL.name)
    parser.add_argument("--mpi-ranks", type=int, default=8)
    parser.add_argument("--label-prefix", default="morphology")
    parser.add_argument("--no-reuse", action="store_true")
    args = parser.parse_args()
    result = run_texture_morphology_forward(
        args.eta,
        args.theta,
        args.a,
        geometry_seed=args.seed,
        mesh_name=args.mesh,
        protocol=PROTOCOLS[args.protocol],
        mpi_ranks=args.mpi_ranks,
        label_prefix=args.label_prefix,
        reuse_cache=not args.no_reuse,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("converged") else 1


if __name__ == "__main__":
    raise SystemExit(main())
