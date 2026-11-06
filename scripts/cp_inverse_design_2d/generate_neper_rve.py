#!/usr/bin/env python3
"""Generate one fixed Neper tessellation and coarse/fine periodic TRI6 MOOSE meshes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from common import (
    GMSH_EXECUTABLE,
    MESHES,
    MESH_CHARACTERISTIC_LENGTH_UM,
    MESH_FILES,
    NEPER,
    NEPER_EXECUTABLE,
    TESS_FILE,
    ensure_directories,
    file_sha256,
    write_json,
)
from neper_to_moose import convert
from validate_periodic_mesh import validate_mesh


def _version(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "--version"], text=True, capture_output=True, check=True
    )
    return (completed.stdout or completed.stderr).strip().splitlines()[-1]


def _run(command: list[str], cwd: Path, log_path: Path) -> float:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    start = time.perf_counter()
    completed = subprocess.run(
        command, cwd=cwd, env=environment, text=True, capture_output=True, check=False
    )
    elapsed = time.perf_counter() - start
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}); see {log_path}")
    return elapsed


def generate(*, force: bool = False) -> dict[str, object]:
    ensure_directories()
    if not NEPER_EXECUTABLE.exists():
        raise FileNotFoundError(f"Neper executable not found: {NEPER_EXECUTABLE}")
    if not GMSH_EXECUTABLE.exists():
        raise FileNotFoundError(f"Gmsh executable not found: {GMSH_EXECUTABLE}")

    commands: list[dict[str, object]] = []
    tess_command = [
        str(NEPER_EXECUTABLE),
        "-T",
        "-dim",
        "2",
        "-n",
        "20",
        "-id",
        "2026",
        "-domain",
        "square(200,200)",
        "-periodicity",
        "all",
        "-morpho",
        "voronoi",
        "-crysym",
        "cubic",
        "-format",
        "tess,geo",
        "-o",
        "rve20_periodic",
    ]
    if force or not TESS_FILE.exists():
        elapsed = _run(tess_command, NEPER, NEPER / "neper_tessellation.log")
    else:
        elapsed = 0.0
    commands.append(
        {
            "stage": "periodic_tessellation",
            "command": tess_command,
            "wall_time_s": elapsed,
        }
    )

    mesh_results: dict[str, object] = {}
    for mesh_name, mesh_path in MESH_FILES.items():
        characteristic_length = MESH_CHARACTERISTIC_LENGTH_UM[mesh_name]
        raw_base = f"rve20_neper_raw_{mesh_name}"
        raw_mesh = NEPER / f"{raw_base}.msh"
        neper_mesh_command = [
            str(NEPER_EXECUTABLE),
            "-M",
            TESS_FILE.name,
            "-elttype",
            "tri",
            "-order",
            "2",
            "-cl",
            f"{characteristic_length:g}",
            "-interface",
            "continuous",
            "-format",
            "msh,per",
            "-o",
            raw_base,
        ]
        if force or not raw_mesh.exists():
            raw_elapsed = _run(
                neper_mesh_command,
                NEPER,
                NEPER / f"neper_mesh_{mesh_name}.log",
            )
        else:
            raw_elapsed = 0.0
        commands.append(
            {
                "stage": f"neper_raw_mesh_{mesh_name}",
                "command": neper_mesh_command,
                "wall_time_s": raw_elapsed,
            }
        )
        start = time.perf_counter()
        if force or not mesh_path.exists():
            conversion = convert(TESS_FILE, mesh_path, characteristic_length)
        else:
            conversion = json.loads(
                mesh_path.with_suffix(".conversion.json").read_text(encoding="utf-8")
            )
        conversion_elapsed = time.perf_counter() - start
        metrics = validate_mesh(mesh_path)
        metrics["generation_wall_time_s"] = raw_elapsed + conversion_elapsed
        metrics["neper_raw_mesh"] = str(raw_mesh)
        metrics["neper_raw_mesh_sha256"] = file_sha256(raw_mesh)
        metrics["moose_mesh_sha256"] = file_sha256(mesh_path)
        write_json(MESHES / f"{mesh_name}_mesh_metrics.json", metrics)
        mesh_results[mesh_name] = {"conversion": conversion, "metrics": metrics}

    manifest = {
        "workflow": "fixed 20-grain periodic Neper tessellation with MOOSE-compatible TRI6 meshes",
        "neper_version": _version(NEPER_EXECUTABLE),
        "gmsh_version": _version(GMSH_EXECUTABLE),
        "seed": 2026,
        "dimension": 2,
        "domain_um": [200.0, 200.0],
        "grain_count": 20,
        "periodicity": ["x", "y"],
        "tessellation": str(TESS_FILE),
        "tessellation_sha256": file_sha256(TESS_FILE),
        "commands": commands,
        "meshes": mesh_results,
    }
    write_json(RESULTS_MANIFEST, manifest)
    return manifest


RESULTS_MANIFEST = NEPER / "neper_generation_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(generate(force=args.force), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
