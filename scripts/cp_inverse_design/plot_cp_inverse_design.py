#!/usr/bin/env python3
"""Generate eight publication-style figures from executed CP FEM data."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf_file

from common import CENTERS_INPUT, FIGURES, OPTIMIZATION_HISTORY, RANDOM_HISTORY, SUMMARY
from texture_generator import moose_bunge_rotation, orientations_for_eta


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 190,
            "font.size": 9.5,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _centers() -> np.ndarray:
    match = re.search(r"voronoi_centers\s*=\s*'([^']+)'", CENTERS_INPUT.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError("Cannot parse periodic Voronoi centers")
    return np.fromstring(match.group(1), sep=" ").reshape(-1, 2)


def _grain_map(resolution: int = 420) -> np.ndarray:
    coordinates = (np.arange(resolution) + 0.5) * 200.0 / resolution
    x, y = np.meshgrid(coordinates, coordinates)
    distances = []
    for cx, cy in _centers():
        dx = np.minimum(abs(x - cx), 200.0 - abs(x - cx))
        dy = np.minimum(abs(y - cy), 200.0 - abs(y - cy))
        distances.append(dx * dx + dy * dy)
    return np.argmin(np.stack(distances), axis=0)


def _orientation_colors(eta: float) -> np.ndarray:
    eulers, _, _ = orientations_for_eta(eta)
    colors = []
    for euler in eulers:
        direction = abs(moose_bunge_rotation(*euler) @ np.array([1.0, 0.0, 0.0]))
        rgb = direction * direction
        colors.append(0.10 + 0.90 * rgb / max(float(rgb.max()), 1.0e-15))
    return np.clip(colors, 0.0, 1.0)


def _element_grid(path: Path, field: str) -> np.ndarray:
    with netcdf_file(path, "r", mmap=False) as db:
        x = np.asarray(db.variables["coordx"].data).copy()
        y = np.asarray(db.variables["coordy"].data).copy()
        names = [b"".join(row).decode().rstrip("\x00 ") for row in db.variables["name_elem_var"].data]
        field_index = names.index(field) + 1
        points: list[tuple[float, float, float]] = []
        blocks = len(np.asarray(db.variables["eb_prop1"].data))
        for block in range(1, blocks + 1):
            values_name = f"vals_elem_var{field_index}eb{block}"
            connect_name = f"connect{block}"
            if values_name not in db.variables or connect_name not in db.variables:
                continue
            connectivity = np.asarray(db.variables[connect_name].data).copy().astype(int) - 1
            values = np.asarray(db.variables[values_name].data[-1]).copy()
            for nodes, value in zip(connectivity, values):
                points.append((float(x[nodes].mean()), float(y[nodes].mean()), float(value)))
    array = np.asarray(points)
    xs, ys = np.unique(np.round(array[:, 0], 10)), np.unique(np.round(array[:, 1], 10))
    grid = np.full((len(ys), len(xs)), np.nan)
    x_index = {value: index for index, value in enumerate(xs)}
    y_index = {value: index for index, value in enumerate(ys)}
    for px, py, value in array:
        grid[y_index[round(py, 10)], x_index[round(px, 10)]] = value
    if np.isnan(grid).any():
        raise RuntimeError(f"Incomplete Exodus element grid for {field}: {path}")
    return grid


def plot_workflow(output: Path) -> None:
    labels = [
        ("Texture parameter\nη ∈ [0,1]", "#8ecae6"),
        ("20-grain periodic\nRVE + SO(3) SLERP", "#bde0fe"),
        ("FCC crystal\nplasticity", "#ffb703"),
        ("Real MOOSE TL\nhomogenization", "#fb8500"),
        ("P11 at 1% & 2%\nreal FEM objectives", "#90be6d"),
        ("GP surrogate of −J\nBayesian optimization", "#cdb4db"),
    ]
    fig, axis = plt.subplots(figsize=(13.5, 2.8), constrained_layout=True)
    axis.set_xlim(-0.4, len(labels) - 0.6)
    axis.set_ylim(-0.55, 0.55)
    axis.axis("off")
    for index, (label, color) in enumerate(labels):
        axis.text(index, 0, label, ha="center", va="center", fontsize=10, weight="bold",
                  bbox={"boxstyle": "round,pad=0.65", "fc": color, "ec": "#264653", "lw": 1.2})
        if index + 1 < len(labels):
            axis.annotate("", (index + 0.63, 0), (index + 0.37, 0),
                          arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#264653"})
    axis.set_title("Finite-strain crystal-plasticity inverse-design workflow", fontsize=13, pad=12)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_textures(best_eta: float, output: Path) -> None:
    grain_map = _grain_map()
    designs = [(0.0, "η=0.0"), (0.2, "η=0.2"), (0.5, "η=0.5"), (0.8, "η=0.8"), (1.0, "η=1.0")]
    fig, axes = plt.subplots(1, 5, figsize=(15.2, 3.2), constrained_layout=True)
    for axis, (eta, title) in zip(axes, designs):
        axis.imshow(_orientation_colors(eta)[grain_map], origin="lower", extent=(0, 200, 0, 200), interpolation="nearest")
        suffix = "\n(best)" if abs(eta - best_eta) < 1.0e-8 else ""
        axis.set_title(title + suffix)
        axis.set_xlabel("x (µm)")
        axis.set_aspect("equal")
    axes[0].set_ylabel("y (µm)")
    fig.suptitle("Fixed grain geometry; RGB encodes full 3D crystal orientation")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_properties(rows: list[dict[str, str]], summary: dict[str, object], output: Path) -> None:
    valid = [row for row in rows if row["status"] == "converged"]
    eta = np.asarray([float(row["eta"]) for row in valid])
    order = np.argsort(eta)
    best = summary["best_observed_design"]
    target = summary["target_properties"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for axis, key, target_key, title in (
        (axes[0], "P11_1pct_MPa", "target_P11_at_F11_1p01_MPa", "F11=1.01"),
        (axes[1], "P11_2pct_MPa", "target_P11_at_F11_1p02_MPa", "F11=1.02"),
    ):
        values = np.asarray([float(row[key]) for row in valid])
        axis.plot(eta[order], values[order], color="#4c78a8", alpha=0.65, lw=1.2)
        axis.scatter(eta, values, color="#4c78a8", label="Real CP FEM", zorder=3)
        axis.axhline(float(target[target_key]), color="black", ls="--", label="Hidden target")
        axis.scatter([float(best["eta"])], [float(best[key])], marker="*", s=160, color="#e45756", label="Best candidate", zorder=4)
        axis.set(xlabel="Texture parameter η", ylabel="Macroscopic PK1 P11 (MPa)", title=title)
        axis.legend(fontsize=8)
    fig.suptitle("Plastic-response objectives versus texture parameter")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(bo_rows: list[dict[str, str]], random_rows: list[dict[str, str]], tolerance: float, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.3, 4.6), constrained_layout=True)
    for rows, color, label in ((bo_rows, "#4c78a8", "Bayesian optimization"), (random_rows, "#f58518", "Random search")):
        errors = [max(float(row["relative_error_1pct"]), float(row["relative_error_2pct"])) for row in rows if row["status"] == "converged"]
        calls = [int(row["call"]) for row in rows if row["status"] == "converged"]
        if errors:
            axis.semilogy(calls, np.minimum.accumulate(errors), "o-", color=color, label=label)
    axis.axhline(tolerance, color="black", ls="--", label=f"{100*tolerance:g}% dual-target threshold")
    axis.set(xlabel="Real candidate FEM evaluations", ylabel="Best max(e1,e2)", title="Tolerance-driven optimization convergence")
    axis.legend()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_response(summary: dict[str, object], output: Path) -> None:
    verification = summary["verification"]
    target = verification["low_fidelity_target"]
    candidate = verification["independent_low_fidelity"]
    fig, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for data, style, label in ((target, "-", "Hidden target"), (candidate, "--", "Optimized, independent")):
        curve = data["full_curve"]
        axis.plot([point["F11"] for point in curve], [point["P11_MPa"] for point in curve], style, lw=2, label=label)
    axis.scatter([1.01, 1.02], [target["P11_1pct_MPa"], target["P11_2pct_MPa"]], color="black", zorder=4)
    text = (
        f"η*={candidate['eta']:.7f}\n"
        f"e1={100*candidate['relative_error_1pct']:.4f}%\n"
        f"e2={100*candidate['relative_error_2pct']:.4f}%"
    )
    axis.text(0.03, 0.95, text, transform=axis.transAxes, va="top", bbox={"fc": "white", "alpha": 0.85})
    axis.set(xlabel="Macroscopic F11", ylabel="Macroscopic PK1 P11 (MPa)", title="Target versus optimized elastoplastic response")
    axis.legend()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_slip(summary: dict[str, object], output: Path) -> None:
    verification = summary["verification"]
    target_path = Path(verification["low_fidelity_target"]["exodus_path"])
    candidate_path = Path(verification["independent_low_fidelity"]["exodus_path"])
    grids = [_element_grid(target_path, "accumulated_slip"), _element_grid(candidate_path, "accumulated_slip")]
    lower, upper = min(grid.min() for grid in grids), max(grid.max() for grid in grids)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), constrained_layout=True)
    image = None
    for axis, grid, title in zip(axes, grids, ("Hidden target", "Optimized candidate")):
        image = axis.imshow(grid, origin="lower", extent=(0, 200, 0, 200), vmin=lower, vmax=upper, cmap="magma", interpolation="nearest")
        axis.set(xlabel="x (µm)", ylabel="y (µm)", title=title, aspect="equal")
    fig.colorbar(image, ax=axes, label="Accumulated slip at F11=1.02")
    fig.suptitle("Local plastic-slip distribution (common color scale)")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_fidelity(summary: dict[str, object], output: Path) -> None:
    verification = summary["verification"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    low_t, low_c = verification["low_fidelity_target"], verification["independent_low_fidelity"]
    meshes = ["32×32"]
    target1, candidate1 = [low_t["P11_1pct_MPa"]], [low_c["P11_1pct_MPa"]]
    target2, candidate2 = [low_t["P11_2pct_MPa"]], [low_c["P11_2pct_MPa"]]
    if "high_fidelity_target" in verification:
        high_t, high_c = verification["high_fidelity_target"], verification["high_fidelity_candidate"]
        meshes.append("64×64")
        target1.append(high_t["P11_1pct_MPa"]); candidate1.append(high_c["P11_1pct_MPa"])
        target2.append(high_t["P11_2pct_MPa"]); candidate2.append(high_c["P11_2pct_MPa"])
    x = np.arange(len(meshes)); width = 0.35
    for axis, target_values, candidate_values, title in ((axes[0], target1, candidate1, "F11=1.01"), (axes[1], target2, candidate2, "F11=1.02")):
        axis.bar(x - width/2, target_values, width, label="Target")
        axis.bar(x + width/2, candidate_values, width, label="Optimized")
        axis.set_xticks(x, meshes)
        axis.set(ylabel="PK1 P11 (MPa)", title=title)
        axis.legend()
    fig.suptitle("Same-mesh low- and high-fidelity verification")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_cost(bo_rows: list[dict[str, str]], random_rows: list[dict[str, str]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.2), constrained_layout=True)
    for rows, color, label in ((bo_rows, "#4c78a8", "BO FEM"), (random_rows, "#f58518", "Random FEM")):
        calls = [int(row["call"]) for row in rows]
        wall = np.asarray([float(row["wall_time_s"]) for row in rows])
        axes[0].plot(calls, wall, "o-", color=color, label=label)
        axes[1].plot(calls, np.cumsum(wall), "o-", color=color, label=label)
    axes[0].set(xlabel="Evaluation", ylabel="Wall time (s)", title="Per-FEM cost")
    axes[1].set(xlabel="Evaluation", ylabel="Cumulative FEM wall time (s)", title="Cumulative cost")
    for axis in axes:
        axis.legend()
    fig.suptitle("Crystal-plasticity computational cost (GP time excluded)")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def make_all_figures(summary: dict[str, object]) -> None:
    _style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    bo_rows = _csv_rows(OPTIMIZATION_HISTORY)
    random_rows = _csv_rows(RANDOM_HISTORY)
    plot_workflow(FIGURES / "fig01_cp_inverse_design_workflow.png")
    plot_textures(float(summary["best_observed_design"]["eta"]), FIGURES / "fig02_initial_texture_candidates.png")
    plot_properties(bo_rows, summary, FIGURES / "fig03_plastic_properties_vs_eta.png")
    plot_convergence(bo_rows, random_rows, float(summary["target_tolerance"]), FIGURES / "fig04_bo_convergence.png")
    plot_response(summary, FIGURES / "fig05_target_vs_optimized_response.png")
    plot_slip(summary, FIGURES / "fig06_plastic_slip_distribution.png")
    plot_fidelity(summary, FIGURES / "fig07_low_high_fidelity_verification.png")
    plot_cost(bo_rows, random_rows, FIGURES / "fig08_computational_cost.png")


def main() -> int:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    make_all_figures(summary)
    print(f"Wrote figures to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
