#!/usr/bin/env python3
"""Generate the eight requested publication figures from executed FEM data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from matplotlib.collections import PolyCollection
from scipy.io import netcdf_file

from common import (
    BEST_DESIGN,
    ETA_BOUNDS,
    FIGURES,
    MESH_FILES,
    OPTIMIZATION_HISTORY,
    SUMMARY,
    THETA_BOUNDS_DEG,
    VERIFICATION,
)
from optimize_texture_2d import normalize_design
from texture_parameterization import moose_bunge_rotation, orientations_for_design
from validate_periodic_mesh import read_msh


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "font.size": 9.5,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def _history() -> list[dict[str, str]]:
    with OPTIMIZATION_HISTORY.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _mesh_faces(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, elements = read_msh(path)
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


def _orientation_colors(eta: float, theta: float) -> np.ndarray:
    eulers = orientations_for_design(eta, theta)[0]
    colors = []
    for euler in eulers:
        direction = np.abs(moose_bunge_rotation(*euler) @ np.array([1.0, 0.0, 0.0]))
        rgb = direction * direction
        colors.append(0.08 + 0.92 * rgb / max(float(rgb.max()), 1.0e-15))
    return np.clip(colors, 0.0, 1.0)


def _draw_texture(axis: plt.Axes, eta: float, theta: float, title: str) -> None:
    coordinates, connectivity, grains = _mesh_faces(MESH_FILES["coarse"])
    polygons = coordinates[connectivity]
    colors = _orientation_colors(eta, theta)[grains - 1]
    axis.add_collection(
        PolyCollection(
            polygons,
            facecolors=colors,
            edgecolors=(0.1, 0.1, 0.1, 0.12),
            linewidths=0.12,
        )
    )
    axis.set(xlim=(0, 200), ylim=(0, 200), aspect="equal", title=title, xlabel="x (µm)")


def plot_neper_rve() -> None:
    coordinates, connectivity, grains = _mesh_faces(MESH_FILES["coarse"])
    polygons = coordinates[connectivity]
    cmap = plt.get_cmap("tab20")
    fig, axis = plt.subplots(figsize=(7.1, 6.2), constrained_layout=True)
    axis.add_collection(
        PolyCollection(
            polygons,
            facecolors=cmap((grains - 1) / 19.0),
            edgecolors=(0.0, 0.0, 0.0, 0.22),
            linewidths=0.18,
        )
    )
    centroids = np.zeros((20, 2))
    for grain in range(1, 21):
        centroids[grain - 1] = polygons[grains == grain].mean(axis=(0, 1))
        axis.text(
            *centroids[grain - 1], str(grain), ha="center", va="center", fontsize=7
        )
    axis.annotate(
        "x periodic", (196, 195), (4, 195), arrowprops={"arrowstyle": "<->", "lw": 1.5}
    )
    axis.annotate(
        "y periodic",
        (195, 196),
        (195, 4),
        arrowprops={"arrowstyle": "<->", "lw": 1.5},
        rotation=90,
    )
    axis.set(
        xlim=(0, 200), ylim=(0, 200), aspect="equal", xlabel="x (µm)", ylabel="y (µm)"
    )
    axis.set_title(f"Neper periodic 20-grain RVE — {len(connectivity)} TRI6 elements")
    _save(fig, "fig01_neper_polycrystalline_rve")


def plot_texture_space(best: dict[str, object]) -> None:
    designs = [
        (0.15, -30.0),
        (0.15, 30.0),
        (0.50, 0.0),
        (0.85, -30.0),
        (0.85, 30.0),
        (float(best["eta"]), float(best["theta_deg"])),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.1), constrained_layout=True)
    for index, (axis, (eta, theta)) in enumerate(zip(axes.flat, designs)):
        suffix = " — optimized" if index == len(designs) - 1 else ""
        _draw_texture(axis, eta, theta, f"η={eta:.3f}, θ={theta:+.1f}°{suffix}")
    axes[0, 0].set_ylabel("y (µm)")
    axes[1, 0].set_ylabel("y (µm)")
    fig.suptitle("Fixed Neper geometry; RGB encodes full 3D crystal orientation")
    _save(fig, "fig02_two_parameter_texture_space")


def _fit_property_gp(rows: list[dict[str, str]], key: str) -> SingleTaskGP:
    valid = [row for row in rows if row["status"] == "converged"]
    train_x = torch.tensor(
        [normalize_design(float(row["eta"]), float(row["theta_deg"])) for row in valid],
        dtype=torch.double,
    )
    train_y = torch.tensor(
        [float(row[key]) for row in valid], dtype=torch.double
    ).reshape(-1, 1)
    train_yvar = torch.full_like(train_y, 1.0e-8)
    model = SingleTaskGP(
        train_x, train_y, train_Yvar=train_yvar, outcome_transform=Standardize(m=1)
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def plot_structure_property(
    rows: list[dict[str, str]], target: dict[str, object]
) -> None:
    valid = [row for row in rows if row["status"] == "converged"]
    eta_grid = np.linspace(*ETA_BOUNDS, 61)
    theta_grid = np.linspace(*THETA_BOUNDS_DEG, 71)
    eta_mesh, theta_mesh = np.meshgrid(eta_grid, theta_grid)
    normalized = torch.tensor(
        [normalize_design(e, t) for e, t in zip(eta_mesh.ravel(), theta_mesh.ravel())],
        dtype=torch.double,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.4), constrained_layout=True)
    definitions = (
        ("proof_stress_MPa", "target_proof_stress_MPa", "0.2% proof stress (MPa)"),
        ("flow_stress_2pct_MPa", "target_flow_stress_2pct_MPa", "2% flow stress (MPa)"),
    )
    for axis, (key, target_key, title) in zip(axes, definitions):
        model = _fit_property_gp(rows, key)
        with torch.no_grad():
            mean = (
                model.posterior(normalized).mean.reshape(theta_mesh.shape).cpu().numpy()
            )
        image = axis.contourf(eta_mesh, theta_mesh, mean, levels=18, cmap="viridis")
        axis.scatter(
            [float(row["eta"]) for row in valid],
            [float(row["theta_deg"]) for row in valid],
            c=[float(row[key]) for row in valid],
            cmap="viridis",
            edgecolor="white",
            linewidth=0.7,
            label="Real FEM",
        )
        target_contour = axis.contour(
            eta_mesh,
            theta_mesh,
            mean,
            levels=[float(target[target_key])],
            colors="white",
            linewidths=1.5,
        )
        axis.clabel(
            target_contour, fmt={float(target[target_key]): "target"}, fontsize=8
        )
        fig.colorbar(image, ax=axis, label=title)
        axis.set(
            xlabel="Texture concentration η",
            ylabel="Preferred rotation θ (deg)",
            title=title,
        )
        axis.legend(loc="best", fontsize=8)
    fig.suptitle(
        "Post-hoc GP property surfaces; markers are executed MOOSE FEM evaluations"
    )
    _save(fig, "fig03_elastoplastic_structure_property")


def plot_convergence(rows: list[dict[str, str]], tolerance: float) -> None:
    valid = [row for row in rows if row["status"] == "converged"]
    calls = np.asarray([int(row["call"]) for row in valid])
    objective = np.asarray([float(row["objective_J"]) for row in valid])
    errors = np.asarray(
        [
            max(float(row["relative_error_proof"]), float(row["relative_error_flow"]))
            for row in valid
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    axes[0].semilogy(
        calls, np.minimum.accumulate(objective), "o-", label="Best observed J"
    )
    axes[1].semilogy(
        calls, np.minimum.accumulate(errors), "o-", label="Best max property error"
    )
    axes[1].axhline(
        tolerance, color="black", ls="--", label=f"{100*tolerance:g}% stop threshold"
    )
    initial_end = sum(row["stage"] == "initial" for row in rows)
    for axis in axes:
        if initial_end < len(rows):
            axis.axvline(
                initial_end + 0.5, color="0.45", ls=":", label="DOE / BO boundary"
            )
        axis.set_xlabel("Real FEM evaluation number")
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Best objective J")
    axes[1].set_ylabel("Best max(relative property error)")
    fig.suptitle("Bayesian optimization convergence (no random-search comparison)")
    _save(fig, "fig04_bayesian_optimization_convergence")


def plot_response(verification: dict[str, object]) -> None:
    target = verification["coarse_target"]
    candidate = verification["independent_coarse_candidate"]
    fig, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for data, style, label in (
        (target, "-", "Hidden target — real FEM"),
        (candidate, "--", "Optimized — independent real FEM"),
    ):
        curve = data["full_curve"]
        axis.plot(
            [point["strain"] for point in curve],
            [point["P11_MPa"] for point in curve],
            style,
            lw=2.0,
            label=label,
        )
        intersection = data["proof_extraction"]["intersection"]
        axis.scatter(
            intersection["proof_strain"], intersection["proof_stress_MPa"], s=35
        )
        axis.scatter(0.02, data["flow_stress_2pct_MPa"], marker="s", s=32)
    axis.set(
        xlabel=r"Macroscopic nominal strain $\bar F_{11}-1$",
        ylabel=r"Macroscopic PK1 $\bar P_{11}$ (MPa)",
        title="Target versus optimized elastoplastic response",
    )
    axis.legend(fontsize=8)
    _save(fig, "fig05_target_vs_optimized_response")


def _exodus_field(path: Path, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with netcdf_file(path, "r", mmap=False) as database:
        x = np.asarray(database.variables["coordx"].data).copy()
        y = np.asarray(database.variables["coordy"].data).copy()
        names = [
            b"".join(row).decode().rstrip("\x00 ")
            for row in np.asarray(database.variables["name_elem_var"].data)
        ]
        field_index = names.index(field) + 1
        connectivity: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for block in range(1, len(np.asarray(database.variables["eb_prop1"].data)) + 1):
            connectivity.append(
                np.asarray(database.variables[f"connect{block}"].data)[:, :3] - 1
            )
            values.append(
                np.asarray(
                    database.variables[f"vals_elem_var{field_index}eb{block}"].data[-1]
                )
            )
    return (
        np.column_stack((x, y)),
        np.vstack(connectivity).astype(int),
        np.concatenate(values),
    )


def plot_local_slip(verification: dict[str, object]) -> None:
    target = _exodus_field(
        Path(verification["coarse_target"]["exodus_path"]), "accumulated_slip"
    )
    candidate = _exodus_field(
        Path(verification["independent_coarse_candidate"]["exodus_path"]),
        "accumulated_slip",
    )
    lower = min(float(target[2].min()), float(candidate[2].min()))
    upper = max(float(target[2].max()), float(candidate[2].max()))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3), constrained_layout=True)
    image = None
    for axis, data, title in zip(
        axes, (target, candidate), ("Hidden target", "Optimized candidate")
    ):
        triangulation = mtri.Triangulation(data[0][:, 0], data[0][:, 1], data[1])
        image = axis.tripcolor(
            triangulation,
            facecolors=data[2],
            shading="flat",
            cmap="magma",
            vmin=lower,
            vmax=upper,
        )
        axis.set(xlabel="x (µm)", ylabel="y (µm)", title=title, aspect="equal")
    fig.colorbar(image, ax=axes, label="Accumulated slip at F11=1.02")
    fig.suptitle(
        "Local plastic slip on identical Neper geometry and a common color scale"
    )
    _save(fig, "fig06_local_plastic_slip_distribution")


def plot_mesh_verification(verification: dict[str, object]) -> None:
    labels = ["Coarse", "Fine"]
    target = [verification["coarse_target"], verification["fine_target"]]
    candidate = [
        verification["independent_coarse_candidate"],
        verification["fine_candidate"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.2), constrained_layout=True)
    x = np.arange(2)
    width = 0.35
    for axis, key, title in (
        (axes[0], "proof_stress_MPa", "0.2% proof stress"),
        (axes[1], "flow_stress_2pct_MPa", "2% flow stress"),
    ):
        axis.bar(x - width / 2, [item[key] for item in target], width, label="Target")
        axis.bar(
            x + width / 2, [item[key] for item in candidate], width, label="Optimized"
        )
        axis.set_xticks(x, labels)
        axis.set(ylabel="PK1 stress (MPa)", title=title)
        axis.legend()
    fig.suptitle("Same-Neper-geometry coarse/fine verification")
    _save(fig, "fig07_coarse_fine_mesh_verification")


def plot_summary(summary: dict[str, object], verification: dict[str, object]) -> None:
    target = verification["coarse_target"]
    candidate = verification["independent_coarse_candidate"]
    best = summary["best_observed_design"]
    verification_runs = [verification["independent_coarse_candidate"]]
    for key in ("fine_target", "fine_candidate"):
        if key in verification:
            verification_runs.append(verification[key])
    if "time_integration" in verification:
        verification_runs.extend(
            [
                verification["time_integration"]["target"],
                verification["time_integration"]["candidate"],
            ]
        )
    total_calls = 1 + int(summary["bo_fem_calls"]) + len(verification_runs)
    total_wall = (
        float(summary["target_properties"]["wall_time_s"])
        + float(summary["bo_fem_wall_time_s"])
        + sum(float(item["wall_time_s"]) for item in verification_runs)
    )
    fig, axis = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    axis.axis("off")
    rows = [
        [
            "Proof stress (MPa)",
            f"{target['proof_stress_MPa']:.4f}",
            f"{candidate['proof_stress_MPa']:.4f}",
        ],
        [
            "2% flow stress (MPa)",
            f"{target['flow_stress_2pct_MPa']:.4f}",
            f"{candidate['flow_stress_2pct_MPa']:.4f}",
        ],
        ["Relative proof error", "—", f"{100*candidate['relative_error_proof']:.4f}%"],
        ["Relative flow error", "—", f"{100*candidate['relative_error_flow']:.4f}%"],
        ["Texture η", "hidden during BO", f"{float(best['eta']):.7f}"],
        ["Texture θ", "hidden during BO", f"{float(best['theta_deg']):+.5f}°"],
        ["Real inverse-loop FEM calls", "—", str(total_calls)],
        ["Target + BO + verification wall", "—", f"{total_wall:.1f} s"],
    ]
    table = axis.table(
        cellText=rows,
        colLabels=["Quantity", "Target", "Optimized / workflow"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.65)
    axis.set_title("SPP2489 two-parameter inverse-design summary", pad=18, fontsize=13)
    _save(fig, "fig08_design_summary")


def make_all_figures() -> None:
    _style()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    verification_path = VERIFICATION / "verification_summary.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    best = json.loads(BEST_DESIGN.read_text(encoding="utf-8"))
    rows = _history()
    plot_neper_rve()
    plot_texture_space(best)
    plot_structure_property(rows, summary["target_properties"])
    plot_convergence(rows, float(summary["target_tolerance"]))
    plot_response(verification)
    plot_local_slip(verification)
    plot_mesh_verification(verification)
    plot_summary(summary, verification)


def main() -> int:
    make_all_figures()
    print(f"Wrote figures to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
