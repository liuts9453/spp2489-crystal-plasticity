#!/usr/bin/env python3
"""Generate deterministic periodic Neper morphology families and TRI6 meshes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ID2_SCRIPT_DIR = SCRIPT_DIR.parent / "cp_inverse_design_2d"
for directory in (SCRIPT_DIR, ID2_SCRIPT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common import file_sha256, write_json  # noqa: E402
from measure_morphology import measure  # noqa: E402
from morphology_common import (  # noqa: E402
    BASE_GEOMETRY_SEED,
    GMSH_EXECUTABLE,
    MESH_CHARACTERISTIC_LENGTH_UM,
    MORPHOLOGY_BOUNDS,
    MORPHOLOGY_LEVELS,
    NEPER_EXECUTABLE,
    OLD_MESHES,
    OLD_TESSELLATION,
    ensure_directories,
    geometry_directory,
    mesh_directory,
    mesh_metrics_path,
    mesh_path,
    tessellation_path,
)
from neper_to_moose import convert  # noqa: E402
from validate_morphology_mesh import validate_morphology_mesh  # noqa: E402


def _version(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "--version"], text=True, capture_output=True, check=True
    )
    return (completed.stdout or completed.stderr).strip().splitlines()[-1]


def _run(command: list[str], cwd: Path, log_path: Path) -> dict[str, object]:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    wall_time = time.perf_counter() - start
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    record = {
        "command": command,
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "wall_time_s": wall_time,
        "log": str(log_path),
    }
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}); see {log_path}")
    return record


def generate_tessellation(
    a_value: float, seed: int, *, force: bool = False
) -> dict[str, object]:
    if not MORPHOLOGY_BOUNDS[0] <= a_value <= MORPHOLOGY_BOUNDS[1]:
        raise ValueError(f"a must lie in {MORPHOLOGY_BOUNDS}")
    directory = geometry_directory(seed, a_value)
    directory.mkdir(parents=True, exist_ok=True)
    tessellation = tessellation_path(seed, a_value)
    command = [
        str(NEPER_EXECUTABLE),
        "-T",
        "-dim",
        "2",
        "-n",
        "20",
        "-id",
        str(seed),
        "-domain",
        "square(200,200)",
        "-periodicity",
        "all",
        "-morpho",
        f"voronoi,aspratio({a_value:.15g},1)",
        "-crysym",
        "cubic",
        "-format",
        "tess,geo",
        "-o",
        "rve20_periodic",
    ]
    if force or not tessellation.exists():
        command_record = _run(command, directory, directory / "neper_tessellation.log")
    else:
        command_record = {
            "command": command,
            "cwd": str(directory),
            "returncode": 0,
            "wall_time_s": 0.0,
            "log": str(directory / "neper_tessellation.log"),
            "reused": True,
        }
    statistics = measure(
        tessellation,
        input_a=a_value,
        seed=seed,
        csv_path=directory / "grain_morphology.csv",
    )
    write_json(directory / "morphology_statistics.json", statistics)
    exact_old_baseline = False
    if seed == BASE_GEOMETRY_SEED and abs(a_value - 1.0) <= 1.0e-14:
        exact_old_baseline = file_sha256(tessellation) == file_sha256(OLD_TESSELLATION)
        if not exact_old_baseline:
            raise RuntimeError("a=1, seed=2026 did not reproduce the original tessellation")
    return {
        "input_a": float(a_value),
        "seed": int(seed),
        "tessellation": str(tessellation),
        "tessellation_sha256": file_sha256(tessellation),
        "command": command_record,
        "statistics": statistics,
        "exact_old_baseline": exact_old_baseline,
    }


def generate_mesh(
    a_value: float,
    seed: int,
    mesh_name: str,
    *,
    force: bool = False,
) -> dict[str, object]:
    tessellation = tessellation_path(seed, a_value)
    if not tessellation.exists():
        generate_tessellation(a_value, seed, force=force)
    directory = mesh_directory(seed, a_value)
    directory.mkdir(parents=True, exist_ok=True)
    characteristic_length = MESH_CHARACTERISTIC_LENGTH_UM[mesh_name]
    raw_base = f"rve20_neper_raw_{mesh_name}"
    raw_mesh = directory / f"{raw_base}.msh"
    raw_command = [
        str(NEPER_EXECUTABLE),
        "-M",
        str(tessellation),
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
        raw_record = _run(
            raw_command, directory, directory / f"neper_mesh_{mesh_name}.log"
        )
    else:
        raw_record = {
            "command": raw_command,
            "cwd": str(directory),
            "returncode": 0,
            "wall_time_s": 0.0,
            "log": str(directory / f"neper_mesh_{mesh_name}.log"),
            "reused": True,
        }
    output = mesh_path(seed, a_value, mesh_name)
    if force or not output.exists():
        conversion = convert(tessellation, output, characteristic_length)
    else:
        conversion = json.loads(
            output.with_suffix(".conversion.json").read_text(encoding="utf-8")
        )
    metrics = validate_morphology_mesh(output, mesh_name)
    metrics.update(
        {
            "input_a": float(a_value),
            "geometry_seed": int(seed),
            "tessellation": str(tessellation),
            "tessellation_sha256": file_sha256(tessellation),
            "mesh_sha256": file_sha256(output),
            "raw_neper_mesh": str(raw_mesh),
            "raw_neper_mesh_sha256": file_sha256(raw_mesh),
        }
    )
    write_json(mesh_metrics_path(seed, a_value, mesh_name), metrics)
    exact_old_mesh = False
    if seed == BASE_GEOMETRY_SEED and abs(a_value - 1.0) <= 1.0e-14:
        exact_old_mesh = file_sha256(output) == file_sha256(OLD_MESHES[mesh_name])
        if not exact_old_mesh:
            raise RuntimeError(f"a=1 {mesh_name} mesh did not reproduce old mesh")
    return {
        "mesh": str(output),
        "mesh_sha256": file_sha256(output),
        "raw_neper_command": raw_record,
        "conversion": conversion,
        "metrics": metrics,
        "exact_old_mesh": exact_old_mesh,
    }


def generate_design(
    a_value: float,
    seed: int,
    mesh_names: tuple[str, ...],
    *,
    force: bool = False,
) -> dict[str, object]:
    geometry = generate_tessellation(a_value, seed, force=force)
    meshes = {
        name: generate_mesh(a_value, seed, name, force=force) for name in mesh_names
    }
    manifest_path = geometry_directory(seed, a_value) / "generation_manifest.json"
    if manifest_path.exists() and not force:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        meshes = {**previous.get("meshes", {}), **meshes}
    manifest = {
        "workflow": "periodic Neper morphology family for texture--morphology FCC-CP design",
        "neper_version": _version(NEPER_EXECUTABLE),
        "gmsh_version": _version(GMSH_EXECUTABLE),
        "parameter_definition": "a = requested global characteristic extent x/y",
        "generator_morphology": f"voronoi,aspratio({a_value:.15g},1)",
        "geometry": geometry,
        "meshes": meshes,
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=float, action="append")
    parser.add_argument("--seed", type=int, default=BASE_GEOMETRY_SEED)
    parser.add_argument(
        "--mesh", choices=MESH_CHARACTERISTIC_LENGTH_UM, action="append"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_directories()
    a_values = tuple(args.a) if args.a else MORPHOLOGY_LEVELS
    mesh_names = tuple(args.mesh) if args.mesh else ("coarse",)
    results = [
        generate_design(value, args.seed, mesh_names, force=args.force)
        for value in a_values
    ]
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
