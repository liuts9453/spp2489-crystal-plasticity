#!/usr/bin/env python3
"""Build Figure A: post-hoc BO objective landscape and executed sampling path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_SCRIPT_DIR = SCRIPT_DIR.parent
for directory in (SCRIPT_DIR, PARENT_SCRIPT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from plot_optimization import _fit_property_gp  # noqa: E402
from presentation_utils import (  # noqa: E402
    COLOR_BO,
    COLOR_DOE,
    COLOR_OPTIMIZED,
    COLOR_REFERENCE,
    GP_OBJECTIVE_FLOOR,
    PRESENTATION,
    audit_history,
    common_inputs,
    configure_style,
    objective_gp_surface,
    output_metadata,
    read_optimization_history,
    save_static_figure,
    surface_grid,
)


def build_design_landscape(audit: dict[str, object] | None = None) -> dict[str, object]:
    configure_style()
    audit = audit or audit_history()
    records = audit["records"]
    inputs = common_inputs()
    target = inputs["target"]
    hidden = inputs["verification"]["hidden_design_disclosed_after_optimization"]

    surface = objective_gp_surface(
        records, through_call=28, eta_count=161, theta_count=181
    )
    if surface is None:
        raise RuntimeError(
            "The final objective surrogate requires at least three FEM points"
        )
    eta_mesh, theta_mesh, predicted_objective = surface
    log_objective = np.log10(np.maximum(predicted_objective, GP_OBJECTIVE_FLOOR))

    raw_rows = read_optimization_history()
    _, _, normalized = surface_grid(eta_mesh.shape[1], eta_mesh.shape[0])
    property_surfaces: dict[str, np.ndarray] = {}
    for key in ("proof_stress_MPa", "flow_stress_2pct_MPa"):
        model = _fit_property_gp(raw_rows, key)
        with torch.no_grad():
            property_surfaces[key] = (
                model.posterior(normalized).mean.cpu().numpy().reshape(eta_mesh.shape)
            )

    eta_values = np.asarray([record["eta"] for record in records])
    theta_values = np.asarray([record["theta_deg"] for record in records])
    initial = np.arange(len(records)) < 9
    updates = [record for record in records if record["is_best_update"]]
    final = records[-1]

    fig, axis = plt.subplots(figsize=(9.4, 7.0))
    fig.subplots_adjust(left=0.10, right=0.84, bottom=0.16, top=0.84)
    level_min = min(float(log_objective.min()), -5.0)
    level_max = max(float(log_objective.max()), -1.0)
    levels = np.linspace(level_min, level_max, 28)
    image = axis.contourf(
        eta_mesh,
        theta_mesh,
        log_objective,
        levels=levels,
        cmap="magma_r",
        extend="both",
    )
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(
        rf"Post-hoc GP surrogate $\log_{{10}}(J)$ (floor $10^{{{int(np.log10(GP_OBJECTIVE_FLOOR))}}}$)"
    )

    proof_contour = axis.contour(
        eta_mesh,
        theta_mesh,
        property_surfaces["proof_stress_MPa"],
        levels=[float(target["target_proof_stress_MPa"])],
        colors="white",
        linewidths=2.0,
        linestyles="-",
    )
    flow_contour = axis.contour(
        eta_mesh,
        theta_mesh,
        property_surfaces["flow_stress_2pct_MPa"],
        levels=[float(target["target_flow_stress_2pct_MPa"])],
        colors="white",
        linewidths=2.0,
        linestyles="--",
    )
    axis.clabel(
        proof_contour, fmt={float(target["target_proof_stress_MPa"]): "P0.2 target"}
    )
    axis.clabel(
        flow_contour, fmt={float(target["target_flow_stress_2pct_MPa"]): "Pflow target"}
    )

    axis.plot(
        eta_values,
        theta_values,
        color="white",
        alpha=0.60,
        linewidth=1.0,
        zorder=3,
    )
    axis.scatter(
        eta_values[initial],
        theta_values[initial],
        marker="s",
        s=54,
        facecolor=COLOR_DOE,
        edgecolor="white",
        linewidth=1.0,
        zorder=4,
    )
    axis.scatter(
        eta_values[~initial],
        theta_values[~initial],
        marker="o",
        s=48,
        facecolor=COLOR_BO,
        edgecolor="white",
        linewidth=1.0,
        zorder=4,
    )
    axis.scatter(
        [record["eta"] for record in updates],
        [record["theta_deg"] for record in updates],
        marker="*",
        s=150,
        facecolor="#F0E442",
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
    )
    for record in updates:
        axis.annotate(
            str(record["call"]),
            (record["eta"], record["theta_deg"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "alpha": 0.78,
                "edgecolor": "none",
            },
            zorder=6,
        )
    axis.scatter(
        final["eta"],
        final["theta_deg"],
        marker="*",
        s=270,
        facecolor=COLOR_OPTIMIZED,
        edgecolor="black",
        linewidth=1.0,
        zorder=7,
    )
    axis.scatter(
        float(hidden["eta"]),
        float(hidden["theta_deg"]),
        marker="X",
        s=120,
        facecolor=COLOR_REFERENCE,
        edgecolor="black",
        linewidth=0.9,
        zorder=6,
    )

    legend = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=COLOR_DOE,
            markeredgecolor="white",
            markersize=8,
            label="Initial DOE (real FEM)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_BO,
            markeredgecolor="white",
            markersize=8,
            label="Sequential BO (real FEM)",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#F0E442",
            markeredgecolor="black",
            markersize=12,
            label="Best-so-far update",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor=COLOR_OPTIMIZED,
            markeredgecolor="black",
            markersize=13,
            label="Final optimized design",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="none",
            markerfacecolor=COLOR_REFERENCE,
            markeredgecolor="black",
            markersize=9,
            label="Hidden reference (post-analysis)",
        ),
        Line2D(
            [0],
            [0],
            color="white",
            linewidth=2,
            linestyle="-",
            label="GP P0.2 target contour",
        ),
        Line2D(
            [0],
            [0],
            color="white",
            linewidth=2,
            linestyle="--",
            label="GP Pflow target contour",
        ),
    ]
    axis.legend(
        handles=legend,
        loc="lower left",
        fontsize=8.3,
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="0.75",
        framealpha=0.92,
    )
    axis.set(
        xlim=(0.1, 0.9),
        ylim=(-45.0, 45.0),
        xlabel=r"Texture concentration $\eta$",
        ylabel=r"Preferred sample-frame rotation $\theta$ (deg)",
        title="Objective landscape and executed sampling path",
    )
    fig.suptitle(
        "Post-hoc final GP surrogate with 28 executed MOOSE FCC-CP evaluations",
        fontsize=14,
        y=0.96,
    )
    fig.text(
        0.5,
        0.045,
        "Background and white contours are surrogate predictions, not continuous FEM results. "
        "The connecting line shows evaluation order only.",
        ha="center",
        fontsize=9,
    )

    paths = save_static_figure(fig, "figA_bo_design_landscape")
    return {
        "name": "Figure A — Bayesian Optimization Design Landscape",
        "outputs": output_metadata(paths),
        "data_sources": {
            "real_fem_history": {
                "classification": "28 real MOOSE FCC CP FEM evaluations",
                "history_csv": "results/cp_inverse_design_2d/optimization_history.csv",
                "calls": [int(record["call"]) for record in records],
                "run_identifiers": [record["run_id"] for record in records],
            },
            "objective_background": {
                "classification": "post-hoc surrogate prediction, not continuous FEM",
                "model": "SingleTaskGP fit to -J using calls 1..28",
            },
            "property_contours": {
                "classification": "post-hoc surrogate predictions, not continuous FEM",
                "models": "SingleTaskGP fits to real proof and flow stresses using calls 1..28",
            },
            "hidden_reference": {
                "classification": "executed MOOSE FCC CP FEM reference disclosed post optimization",
                "summary": "results/cp_inverse_design_2d/verification/verification_summary.json",
            },
        },
        "prediction_disclaimer": "Background and target contours are GP surrogate predictions; markers are real FEM evaluations.",
        "best_update_calls": [record["call"] for record in updates],
        "optimized_call": int(final["call"]),
    }


def main() -> int:
    metadata = build_design_landscape()
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
