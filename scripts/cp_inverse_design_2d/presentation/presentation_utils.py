#!/usr/bin/env python3
"""Shared, read-only data access and plotting helpers for presentation visuals."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PolyCollection
from scipy.io import netcdf_file

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_SCRIPT_DIR = SCRIPT_DIR.parent
if str(PARENT_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_SCRIPT_DIR))

from common import (  # noqa: E402
    BEST_DESIGN,
    ETA_BOUNDS,
    MESH_FILES,
    OPTIMIZATION_HISTORY,
    RESULTS,
    ROOT,
    RUNS,
    SUMMARY,
    TARGET_PROPERTIES,
    THETA_BOUNDS_DEG,
    VERIFICATION,
)
from extract_proof_stress import (  # noqa: E402
    extract_offset_proof,
    physical_history,
    read_history,
)
from optimize_texture_2d import fit_objective_gp, normalize_design  # noqa: E402
from texture_parameterization import (  # noqa: E402
    moose_bunge_rotation,
    orientations_for_design,
)
from validate_periodic_mesh import read_msh  # noqa: E402


PRESENTATION = RESULTS / "presentation"
MANIFEST = PRESENTATION / "visualization_manifest.json"
REPORT = PRESENTATION / "PRESENTATION_VISUALIZATION_REPORT.md"

GP_OBJECTIVE_FLOOR = 1.0e-8
COLOR_TARGET = "#202124"
COLOR_CURRENT = "#0072B2"
COLOR_BEST = "#E69F00"
COLOR_OPTIMIZED = "#D55E00"
COLOR_DOE = "#009E73"
COLOR_BO = "#56B4E9"
COLOR_REFERENCE = "#CC79A7"


def configure_style() -> None:
    """Apply one colorblind-safe, presentation-oriented Matplotlib style."""

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def read_optimization_history() -> list[dict[str, str]]:
    with OPTIMIZATION_HISTORY.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _relocated_candidates(path: Path) -> list[Path]:
    """Relocate an old absolute result path into this or a sibling worktree."""

    candidates = [path]
    parts = path.parts
    if "results" in parts:
        suffix = Path(*parts[parts.index("results") :])
        candidates.append(ROOT / suffix)
        for worktree in ROOT.parent.glob("spp2489*"):
            candidates.append(worktree / suffix)
    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            result.append(resolved)
            seen.add(resolved)
    return result


def resolve_stored_path(value: str | Path) -> Path:
    for candidate in _relocated_candidates(Path(value).expanduser()):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not recover stored result path: {value}")


def resolve_history_artifacts(row: dict[str, str]) -> dict[str, Path]:
    run_id = Path(row["run_directory"]).name
    directories = _relocated_candidates(Path(row["run_directory"]))
    directories.append((RUNS / run_id).resolve())
    for worktree in ROOT.parent.glob("spp2489*"):
        directories.append(
            (worktree / "results" / "cp_inverse_design_2d" / "runs" / run_id).resolve()
        )

    resolved: dict[str, Path] = {}
    for key, filename in (
        ("csv", "solution.csv"),
        ("exodus", "solution.e"),
        ("result", "result.json"),
    ):
        stored_key = {"csv": "csv_path", "exodus": "exodus_path"}.get(key)
        candidates: list[Path] = []
        if stored_key and row.get(stored_key):
            candidates.extend(_relocated_candidates(Path(row[stored_key])))
        candidates.extend(directory / filename for directory in directories)
        match = next(
            (candidate for candidate in candidates if candidate.exists()), None
        )
        if match is None:
            raise FileNotFoundError(
                f"Call {row['call']} is missing {filename}; searched run id {run_id}"
            )
        resolved[key] = match.resolve()
    resolved["run_directory"] = resolved["result"].parent
    return resolved


def load_real_curve(csv_path: Path) -> list[dict[str, float]]:
    """Load an unsmoothed real FEM curve and replace only the known uninitialized t=0 row."""

    return physical_history(read_history(csv_path))


def curve_arrays(curve: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([row["F11_bar"] - 1.0 for row in curve]),
        np.asarray([row["P11_bar"] for row in curve]),
    )


def _relative_difference(first: float, second: float) -> float:
    return abs(first - second) / max(abs(second), 1.0e-30)


def audit_history() -> dict[str, Any]:
    """Validate all 28 executed FEM records and recover their local raw artifacts."""

    rows = read_optimization_history()
    calls = [int(row["call"]) for row in rows]
    if calls != list(range(1, 29)):
        raise RuntimeError(f"Expected calls 1..28, found {calls}")
    if [row["stage"] for row in rows[:9]] != ["initial"] * 9:
        raise RuntimeError("The first nine evaluations are not all initial DOE points")
    if [row["stage"] for row in rows[9:]] != ["sequential"] * 19:
        raise RuntimeError("Evaluations 10..28 are not all sequential BO points")
    if any(row["status"] != "converged" for row in rows):
        raise RuntimeError(
            "The presentation workflow requires all 28 converged records"
        )

    target = load_json(TARGET_PROPERTIES)
    target_proof = float(target["target_proof_stress_MPa"])
    target_flow = float(target["target_flow_stress_2pct_MPa"])
    records: list[dict[str, Any]] = []
    best_objective = float("inf")
    best_call = 0
    best_updates: list[int] = []

    for row in rows:
        call = int(row["call"])
        artifacts = resolve_history_artifacts(row)
        result = load_json(artifacts["result"])
        curve = load_real_curve(artifacts["csv"])
        proof = extract_offset_proof(curve)
        proof_value = float(proof["intersection"]["proof_stress_MPa"])
        flow_rows = [point for point in curve if abs(point["time"] - 1.0) <= 2.0e-11]
        if len(flow_rows) != 1:
            raise RuntimeError(f"Call {call} lacks one synchronized t=1 curve point")
        flow_value = float(flow_rows[0]["P11_bar"])

        eta = float(row["eta"])
        theta = float(row["theta_deg"])
        objective = float(row["objective_J"])
        recomputed = 0.5 * (
            ((proof_value - target_proof) / target_proof) ** 2
            + ((flow_value - target_flow) / target_flow) ** 2
        )
        checks = {
            "eta_matches_result": abs(eta - float(result["eta"])) <= 1.0e-12,
            "theta_matches_result": abs(theta - float(result["theta_deg"])) <= 1.0e-11,
            "proof_matches_history": _relative_difference(
                proof_value, float(row["proof_stress_MPa"])
            )
            <= 2.0e-10,
            "flow_matches_history": _relative_difference(
                flow_value, float(row["flow_stress_2pct_MPa"])
            )
            <= 2.0e-10,
            "objective_matches_history": abs(objective - recomputed) <= 2.0e-12,
            "exodus_exists": artifacts["exodus"].is_file(),
            "result_converged": result.get("converged") is True,
        }
        if not all(checks.values()):
            raise RuntimeError(f"Call {call} audit failed: {checks}")

        if objective < best_objective:
            best_objective = objective
            best_call = call
            best_updates.append(call)
        if abs(float(row["best_objective_J"]) - best_objective) > 2.0e-12:
            raise RuntimeError(f"Call {call} has an inconsistent best-so-far value")

        records.append(
            {
                "call": call,
                "stage": row["stage"],
                "eta": eta,
                "theta_deg": theta,
                "objective_J": objective,
                "relative_error_proof": float(row["relative_error_proof"]),
                "relative_error_flow": float(row["relative_error_flow"]),
                "proof_stress_MPa": float(row["proof_stress_MPa"]),
                "flow_stress_2pct_MPa": float(row["flow_stress_2pct_MPa"]),
                "best_objective_J": best_objective,
                "best_call": best_call,
                "is_best_update": call in best_updates,
                "run_id": str(result.get("cache_key", artifacts["run_directory"].name)),
                "curve_point_count": len(curve),
                "curve": curve,
                "proof_extraction": proof,
                "paths": artifacts,
                "checks": checks,
            }
        )

    best = load_json(BEST_DESIGN)
    if int(records[-1]["best_call"]) != 28:
        raise RuntimeError(
            "The executed history does not end with call 28 as the best design"
        )
    if (
        abs(float(best["eta"]) - records[-1]["eta"]) > 1.0e-12
        or abs(float(best["theta_deg"]) - records[-1]["theta_deg"]) > 1.0e-11
    ):
        raise RuntimeError("best_design.json does not match the audited history")

    return {
        "records": records,
        "summary": {
            "history_record_count": len(records),
            "initial_doe_count": 9,
            "sequential_bo_count": 19,
            "converged_count": 28,
            "solution_csv_count": sum(
                record["paths"]["csv"].is_file() for record in records
            ),
            "exodus_count": sum(
                record["paths"]["exodus"].is_file() for record in records
            ),
            "result_json_count": sum(
                record["paths"]["result"].is_file() for record in records
            ),
            "best_update_calls": best_updates,
            "all_checks_passed": True,
            "history_source": source_path(OPTIMIZATION_HISTORY),
        },
    }


def manifest_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "call": record["call"],
        "stage": record["stage"],
        "eta": record["eta"],
        "theta_deg": record["theta_deg"],
        "objective_J": record["objective_J"],
        "relative_error_proof": record["relative_error_proof"],
        "relative_error_flow": record["relative_error_flow"],
        "best_call_so_far": record["best_call"],
        "best_objective_J_so_far": record["best_objective_J"],
        "is_best_update": record["is_best_update"],
        "fem_run_identifier": record["run_id"],
        "curve_point_count": record["curve_point_count"],
        "solution_csv": source_path(record["paths"]["csv"]),
        "exodus": source_path(record["paths"]["exodus"]),
        "result_json": source_path(record["paths"]["result"]),
        "data_classification": "real MOOSE FCC CP FEM evaluation",
    }


def surface_grid(
    eta_count: int = 121, theta_count: int = 121
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    eta = np.linspace(ETA_BOUNDS[0], ETA_BOUNDS[1], eta_count)
    theta = np.linspace(THETA_BOUNDS_DEG[0], THETA_BOUNDS_DEG[1], theta_count)
    eta_mesh, theta_mesh = np.meshgrid(eta, theta)
    normalized = torch.tensor(
        [
            normalize_design(float(eta_value), float(theta_value))
            for eta_value, theta_value in zip(eta_mesh.ravel(), theta_mesh.ravel())
        ],
        dtype=torch.double,
    )
    return eta_mesh, theta_mesh, normalized


def objective_gp_surface(
    records: list[dict[str, Any]],
    through_call: int,
    eta_count: int = 121,
    theta_count: int = 121,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    available = [record for record in records if record["call"] <= through_call]
    if len(available) < 3:
        return None
    model = fit_objective_gp(
        [(record["eta"], record["theta_deg"]) for record in available],
        [record["objective_J"] for record in available],
    )
    eta_mesh, theta_mesh, normalized = surface_grid(eta_count, theta_count)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predicted = (
            -model.posterior(normalized).mean.cpu().numpy().reshape(eta_mesh.shape)
        )
    return eta_mesh, theta_mesh, np.maximum(predicted, GP_OBJECTIVE_FLOOR)


def mesh_faces(path: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, elements = read_msh(path or MESH_FILES["coarse"])
    ids = sorted(nodes)
    index = {node: position for position, node in enumerate(ids)}
    coordinates = np.asarray([nodes[node][:2] for node in ids])
    triangles = [element for element in elements if element["type"] == 9]
    connectivity = np.asarray(
        [[index[node] for node in element["nodes"][:3]] for element in triangles],
        dtype=int,
    )
    grains = np.asarray([int(element["tags"][0]) for element in triangles], dtype=int)
    return coordinates, connectivity, grains


def cubic_ipf_x_color(rotation: np.ndarray) -> np.ndarray:
    """Return a fixed cubic IPF-X color using the exact MOOSE `crysrot` convention.

    `rotation` maps crystal vectors into the sample frame.  The sample X axis is
    transformed into the crystal frame with Q^T, reduced to the cubic
    fundamental sector 0 <= u <= v <= w, and colored by the component-difference
    key whose vertices are [001]=red, [101]=green, and [111]=blue.
    """

    sample_x_in_crystal = np.asarray(rotation, dtype=float).T @ np.array(
        [1.0, 0.0, 0.0]
    )
    reduced = np.sort(np.abs(sample_x_in_crystal / np.linalg.norm(sample_x_in_crystal)))
    u_value, v_value, w_value = reduced
    weights = np.array(
        [w_value - v_value, np.sqrt(2.0) * (v_value - u_value), np.sqrt(3.0) * u_value]
    )
    weights = np.sqrt(np.clip(weights, 0.0, None))
    if float(weights.max()) <= 1.0e-15:
        return np.zeros(3)
    return weights / weights.max()


def orientation_colors(eta: float, theta_deg: float) -> np.ndarray:
    eulers = orientations_for_design(eta, theta_deg)[0]
    return np.asarray(
        [cubic_ipf_x_color(moose_bunge_rotation(*euler)) for euler in eulers]
    )


def draw_texture(
    axis: plt.Axes,
    eta: float,
    theta_deg: float,
    title: str,
    mesh: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    coordinates, connectivity, grains = mesh or mesh_faces()
    polygons = coordinates[connectivity]
    colors = orientation_colors(eta, theta_deg)[grains - 1]
    axis.add_collection(
        PolyCollection(
            polygons,
            facecolors=colors,
            edgecolors=(0.1, 0.1, 0.1, 0.12),
            linewidths=0.12,
        )
    )
    axis.set(
        xlim=(0.0, 200.0),
        ylim=(0.0, 200.0),
        aspect="equal",
        xlabel="x (µm)",
        ylabel="y (µm)",
        title=title,
    )


def exodus_field(path: Path, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with netcdf_file(path, "r", mmap=False) as database:
        x_values = np.asarray(database.variables["coordx"].data).copy()
        y_values = np.asarray(database.variables["coordy"].data).copy()
        names = [
            b"".join(row).decode().rstrip("\x00 ")
            for row in np.asarray(database.variables["name_elem_var"].data)
        ]
        field_index = names.index(field) + 1
        connectivity: list[np.ndarray] = []
        values: list[np.ndarray] = []
        block_count = len(np.asarray(database.variables["eb_prop1"].data))
        for block in range(1, block_count + 1):
            connectivity.append(
                np.asarray(database.variables[f"connect{block}"].data)[:, :3] - 1
            )
            values.append(
                np.asarray(
                    database.variables[f"vals_elem_var{field_index}eb{block}"].data[-1]
                )
            )
    return (
        np.column_stack((x_values, y_values)),
        np.vstack(connectivity).astype(int),
        np.concatenate(values),
    )


def verification_entry_sources(entry: dict[str, Any]) -> dict[str, Path]:
    return {
        "csv": resolve_stored_path(entry["csv_path"]),
        "exodus": resolve_stored_path(entry["exodus_path"]),
        "run_directory": resolve_stored_path(entry["run_directory"]),
    }


def save_static_figure(fig: plt.Figure, stem: str) -> dict[str, Path]:
    PRESENTATION.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for suffix in ("png", "pdf", "svg"):
        path = PRESENTATION / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        if suffix == "svg":
            clean_lines = [
                line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
            ]
            path.write_text("\n".join(clean_lines) + "\n", encoding="utf-8")
        outputs[suffix] = path
    plt.close(fig)
    return outputs


def output_metadata(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        suffix: {
            "path": source_path(path),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for suffix, path in paths.items()
    }


def common_inputs() -> dict[str, Any]:
    return {
        "summary": load_json(SUMMARY),
        "best": load_json(BEST_DESIGN),
        "target": load_json(TARGET_PROPERTIES),
        "verification": load_json(VERIFICATION / "verification_summary.json"),
    }
