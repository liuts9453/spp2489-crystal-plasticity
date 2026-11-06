#!/usr/bin/env python3
"""Create publication figures from the executed morphology FCC-CP data."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import PolyCollection
from scipy.io import netcdf_file

SCRIPT_DIR = Path(__file__).resolve().parent
ID2_SCRIPT_DIR = SCRIPT_DIR.parent / "cp_inverse_design_2d"
for directory in (SCRIPT_DIR, ID2_SCRIPT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from morphology_common import (  # noqa: E402
    BASE_GEOMETRY_SEED,
    FIGURES,
    MORPHOLOGY_LEVELS,
    RESULTS,
    mesh_path,
)
from validate_periodic_mesh import read_msh  # noqa: E402


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#6B7280"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 260,
            "font.size": 9.5,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def _load() -> tuple[list[dict[str, str]], dict[str, object], dict[str, object]]:
    with (RESULTS / "morphology_sensitivity.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    sensitivity = json.loads(
        (RESULTS / "morphology_sensitivity_summary.json").read_text(encoding="utf-8")
    )
    verification = json.loads(
        (RESULTS / "verification" / "verification_summary.json").read_text(
            encoding="utf-8"
        )
    )
    return rows, sensitivity, verification


def _mesh_faces(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, elements = read_msh(path)
    node_ids = sorted(nodes)
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    coordinates = np.asarray([nodes[node_id][:2] for node_id in node_ids])
    elements = [element for element in elements if element["type"] == 9]
    connectivity = np.asarray(
        [[index[node] for node in element["nodes"][:3]] for element in elements]
    )
    grains = np.asarray([int(element["tags"][0]) for element in elements])
    return coordinates, connectivity, grains


def _exodus_field(path: Path, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with netcdf_file(path, "r", mmap=False) as database:
        coordinates = np.column_stack(
            (
                np.asarray(database.variables["coordx"].data).copy(),
                np.asarray(database.variables["coordy"].data).copy(),
            )
        )
        names = [
            b"".join(row).decode().rstrip("\x00 ")
            for row in np.asarray(database.variables["name_elem_var"].data)
        ]
        index = names.index(field) + 1
        connectivity = []
        values = []
        block_count = len(np.asarray(database.variables["eb_prop1"].data))
        for block in range(1, block_count + 1):
            connectivity.append(
                np.asarray(database.variables[f"connect{block}"].data)[:, :3] - 1
            )
            values.append(
                np.asarray(
                    database.variables[f"vals_elem_var{index}eb{block}"].data[-1]
                )
            )
    return coordinates, np.vstack(connectivity).astype(int), np.concatenate(values)


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_1(rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.5), constrained_layout=True)
    cmap = plt.get_cmap("tab20")
    for axis, row in zip(axes, rows):
        value = float(row["input_a"])
        coordinates, connectivity, grains = _mesh_faces(
            mesh_path(BASE_GEOMETRY_SEED, value, "coarse")
        )
        axis.add_collection(
            PolyCollection(
                coordinates[connectivity],
                facecolors=cmap((grains - 1) / 19.0),
                edgecolors=(0.0, 0.0, 0.0, 0.16),
                linewidths=0.13,
            )
        )
        axis.set(
            xlim=(0, 200),
            ylim=(0, 200),
            aspect="equal",
            title=(
                rf"input $a={value:.2f}$" + "\n" +
                rf"achieved $a_m={float(row['achieved_a']):.3f}$"
            ),
            xlabel="x (µm)",
        )
        axis.grid(False)
    axes[0].set_ylabel("y (µm)")
    fig.suptitle("Periodic 20-grain morphology family — stable grain IDs, coarse TRI6 meshes")
    _save(fig, "fig01_grain_morphology_design_space")


def figure_2(
    rows: list[dict[str, str]], verification: dict[str, object]
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    definitions = (
        ("proof_stress_MPa", "0.2% proof stress (MPa)"),
        ("flow_stress_2pct_MPa", "2% flow stress (MPa)"),
    )
    x_value = np.asarray([float(row["achieved_a"]) for row in rows])
    fine_by_a = {float(row["input_a"]): row for row in verification["fine_results"]}
    time_by_a = {
        float(row["input_a"]): row for row in verification["time_refined_results"]
    }
    paired = verification["realization_comparisons"]
    for axis, (key, label) in zip(axes, definitions):
        y_value = np.asarray([float(row[key]) for row in rows])
        axis.plot(x_value, y_value, "o-", color=BLUE, label="seed 2026 coarse")
        for value, fine in fine_by_a.items():
            achieved = next(
                float(row["achieved_a"])
                for row in rows
                if abs(float(row["input_a"]) - value) < 1.0e-12
            )
            axis.scatter(
                achieved,
                float(fine[key]),
                marker="s",
                facecolors="none",
                edgecolors=ORANGE,
                s=48,
                label="fine mesh" if value == 0.70 else None,
                zorder=4,
            )
        for value, refined in time_by_a.items():
            achieved = next(
                float(row["achieved_a"])
                for row in rows
                if abs(float(row["input_a"]) - value) < 1.0e-12
            )
            axis.scatter(
                achieved,
                float(refined[key]),
                marker="^",
                color=GREEN,
                s=38,
                label="refined time" if value == 0.70 else None,
                zorder=4,
            )
        for index, pair in enumerate(paired):
            axis.plot(
                [1.0, 1.5],
                [float(pair[f"a1_{key}"]), float(pair[f"a1p5_{key}"])],
                marker="x",
                color=GRAY,
                alpha=0.55,
                label="paired geometry seeds" if index == 0 else None,
            )
        axis.set(xlabel=r"Achieved directional morphology $a_m$", ylabel=label)
        axis.legend(fontsize=8)
    fig.suptitle("Morphology sensitivity and numerical/realization variation — real FCC-CP FEM")
    _save(fig, "fig02_morphology_sensitivity")


def figure_3(rows: list[dict[str, str]]) -> None:
    selected = [
        next(row for row in rows if abs(float(row["input_a"]) - value) < 1.0e-12)
        for value in (0.70, 1.00, 1.50)
    ]
    fields = [
        _exodus_field(Path(row["exodus_path"]), "accumulated_slip")
        for row in selected
    ]
    lower = min(float(values.min()) for _, _, values in fields)
    upper = max(float(values.max()) for _, _, values in fields)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), constrained_layout=True)
    image = None
    for axis, row, (coordinates, connectivity, values) in zip(axes, selected, fields):
        triangulation = mtri.Triangulation(
            coordinates[:, 0], coordinates[:, 1], connectivity
        )
        image = axis.tripcolor(
            triangulation,
            facecolors=values,
            shading="flat",
            cmap="magma",
            vmin=lower,
            vmax=upper,
        )
        axis.set(
            xlim=(0, 200),
            ylim=(0, 200),
            aspect="equal",
            title=(
                rf"$a={float(row['input_a']):.2f}$, "
                rf"$a_m={float(row['achieved_a']):.3f}$"
            ),
            xlabel="x (µm)",
        )
        axis.grid(False)
    axes[0].set_ylabel("y (µm)")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.02)
    colorbar.set_label(r"Accumulated slip at $\bar F_{11}=1.02$")
    fig.suptitle("Local plastic-slip redistribution under identical texture and loading")
    _save(fig, "fig03_plastic_slip_fields")


def figure_5(rows: list[dict[str, str]]) -> None:
    selected = [
        next(row for row in rows if abs(float(row["input_a"]) - value) < 1.0e-12)
        for value in (0.70, 1.00, 1.50)
    ]
    fig, axis = plt.subplots(figsize=(7.3, 4.8), constrained_layout=True)
    for row, color in zip(selected, (BLUE, GREEN, RED)):
        result = json.loads(
            (Path(row["run_directory"]) / "result.json").read_text(encoding="utf-8")
        )
        curve = result["full_curve"]
        axis.plot(
            [100.0 * float(point["strain"]) for point in curve],
            [float(point["P11_MPa"]) for point in curve],
            color=color,
            linewidth=1.8,
            label=(
                rf"$a={float(row['input_a']):.2f}$ "
                rf"($P_{{0.2}}={float(row['proof_stress_MPa']):.2f}$ MPa)"
            ),
        )
    axis.set(
        xlabel=r"Macroscopic strain $\bar F_{11}-1$ (%)",
        ylabel=r"Macroscopic PK1 stress $\bar P_{11}$ (MPa)",
        title="Representative morphology responses — real MOOSE FEM",
    )
    axis.legend()
    _save(fig, "fig05_forward_response_comparison")


def figure_7(verification: dict[str, object]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    definitions = (
        ("proof_stress_MPa", "Proof-stress change (MPa)"),
        ("flow_stress_2pct_MPa", "Flow-stress change (MPa)"),
    )
    for axis, (key, label) in zip(axes, definitions):
        mesh = verification["mesh_comparisons"]
        time = verification["time_comparisons"]
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.plot(
            [float(row["input_a"]) for row in mesh],
            [
                float(row[f"fine_{key}"]) - float(row[f"coarse_{key}"])
                for row in mesh
            ],
            "s-",
            color=ORANGE,
            label="fine − coarse",
        )
        axis.plot(
            [float(row["input_a"]) for row in time],
            [
                float(row[f"refined_{key}"]) - float(row[f"default_{key}"])
                for row in time
            ],
            "^-",
            color=GREEN,
            label="refined − default time",
        )
        axis.set(xlabel="Input morphology a", ylabel=label)
        axis.legend(fontsize=8)
    fig.suptitle("Coarse/fine and time-integration verification on identical tessellations")
    _save(fig, "fig07_coarse_fine_time_verification")


def figure_8(
    rows: list[dict[str, str]], verification: dict[str, object]
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True)
    properties = ("proof_stress_MPa", "flow_stress_2pct_MPa")
    labels = ("Proof", "Flow")
    signal = [
        float(verification["uncertainty"][key]["five_level_morphology_range_MPa"])
        for key in properties
    ]
    numerical = [
        float(verification["uncertainty"][key]["numerical_envelope_MPa"])
        for key in properties
    ]
    realization = [
        float(verification["uncertainty"][key]["a1_realization_span_MPa"])
        for key in properties
    ]
    positions = np.arange(2)
    width = 0.25
    axes[0, 0].bar(positions - width, signal, width, color=BLUE, label="5-level morphology")
    axes[0, 0].bar(positions, numerical, width, color=ORANGE, label="numerical envelope")
    axes[0, 0].bar(positions + width, realization, width, color=GRAY, label="seed span")
    axes[0, 0].set(xticks=positions, xticklabels=labels, ylabel="Stress range (MPa)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("Signal versus uncertainty")

    paired = verification["realization_comparisons"]
    seeds = [str(row["geometry_seed"]) for row in paired]
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].plot(
        seeds,
        [float(row["paired_contrast_proof_stress_MPa"]) for row in paired],
        "o-",
        color=BLUE,
        label="Proof",
    )
    axes[0, 1].plot(
        seeds,
        [float(row["paired_contrast_flow_stress_2pct_MPa"]) for row in paired],
        "s-",
        color=RED,
        label="Flow",
    )
    axes[0, 1].set(
        xlabel="Geometry seed",
        ylabel=r"Paired $a=1.5-a=1$ contrast (MPa)",
        title="Morphology contrast reverses sign",
    )
    axes[0, 1].legend(fontsize=8)

    achieved = [float(row["achieved_a"]) for row in rows]
    axes[1, 0].plot(
        achieved,
        [float(row["mean_accumulated_slip"]) for row in rows],
        "o-",
        color=GREEN,
        label="Mean",
    )
    axes[1, 0].plot(
        achieved,
        [float(row["maximum_accumulated_slip"]) for row in rows],
        "s-",
        color=PURPLE,
        label="Maximum",
    )
    axes[1, 0].set(
        xlabel=r"Achieved morphology $a_m$",
        ylabel="Accumulated slip",
        title="Local slip remains morphology-sensitive",
    )
    axes[1, 0].legend(fontsize=8)

    for row, marker, color in zip(paired, ("o", "s", "^"), (BLUE, ORANGE, GREEN)):
        distance = row["area_weighted_texture_dyad_distance_from_seed2026_a1"]
        axes[1, 1].plot(
            [1.0, 1.5],
            [float(distance["1.0"]), float(distance["1.5"])],
            marker=marker,
            color=color,
            label=f"seed {row['geometry_seed']}",
        )
    axes[1, 1].set(
        xlabel="Input morphology a",
        ylabel="Area-weighted texture-dyad distance",
        title="Area weighting is a measured confounder",
    )
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Morphology–texture–property decision summary: three-variable BO rejected")
    _save(fig, "fig08_morphology_texture_property_summary")


def main() -> int:
    _style()
    rows, _, verification = _load()
    figure_1(rows)
    figure_2(rows, verification)
    figure_3(rows)
    figure_5(rows)
    figure_7(verification)
    figure_8(rows, verification)
    print(f"Wrote 6 figures to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
