#!/usr/bin/env python3
"""Build Figure B: PI-ready target versus optimized macroscopic response."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_SCRIPT_DIR = SCRIPT_DIR.parent
for directory in (SCRIPT_DIR, PARENT_SCRIPT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from presentation_utils import (  # noqa: E402
    COLOR_OPTIMIZED,
    COLOR_TARGET,
    audit_history,
    common_inputs,
    configure_style,
    curve_arrays,
    load_real_curve,
    output_metadata,
    save_static_figure,
    source_path,
    verification_entry_sources,
)


def _proof_point(entry: dict[str, object]) -> tuple[float, float]:
    intersection = entry["proof_extraction"]["intersection"]
    return float(intersection["proof_strain"]), float(intersection["proof_stress_MPa"])


def build_final_comparison(audit: dict[str, object] | None = None) -> dict[str, object]:
    configure_style()
    audit = audit or audit_history()
    inputs = common_inputs()
    verification = inputs["verification"]
    best = inputs["best"]

    target = verification["coarse_target"]
    candidate = verification["independent_coarse_candidate"]
    target_sources = verification_entry_sources(target)
    candidate_sources = verification_entry_sources(candidate)
    target_curve = load_real_curve(target_sources["csv"])
    candidate_curve = load_real_curve(candidate_sources["csv"])
    target_x, target_y = curve_arrays(target_curve)
    candidate_x, candidate_y = curve_arrays(candidate_curve)

    fig, axis = plt.subplots(figsize=(9.2, 5.8))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.18, top=0.82)
    axis.plot(
        target_x,
        target_y,
        color=COLOR_TARGET,
        linewidth=2.8,
        label="Hidden target — real coarse FEM",
        zorder=3,
    )
    axis.plot(
        candidate_x,
        candidate_y,
        color=COLOR_OPTIMIZED,
        linewidth=2.6,
        linestyle="--",
        label="Optimized — real independent coarse FEM",
        zorder=3,
    )

    target_proof = _proof_point(target)
    candidate_proof = _proof_point(candidate)
    axis.scatter(
        *target_proof, s=58, marker="o", color=COLOR_TARGET, edgecolor="white", zorder=5
    )
    axis.scatter(
        *candidate_proof,
        s=58,
        marker="o",
        color=COLOR_OPTIMIZED,
        edgecolor="white",
        zorder=5,
    )
    axis.scatter(
        0.02,
        float(target["flow_stress_2pct_MPa"]),
        s=62,
        marker="s",
        color=COLOR_TARGET,
        edgecolor="white",
        zorder=5,
    )
    axis.scatter(
        0.02,
        float(candidate["flow_stress_2pct_MPa"]),
        s=62,
        marker="s",
        color=COLOR_OPTIMIZED,
        edgecolor="white",
        zorder=5,
    )

    axis.annotate(
        f"Target P0.2 = {target_proof[1]:.2f} MPa",
        target_proof,
        xytext=(45, 22),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": COLOR_TARGET, "lw": 1.0},
        fontsize=9,
    )
    axis.annotate(
        f"Optimized P0.2 = {candidate_proof[1]:.2f} MPa",
        candidate_proof,
        xytext=(20, -30),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": COLOR_OPTIMIZED, "lw": 1.0},
        fontsize=9,
        color=COLOR_OPTIMIZED,
    )
    axis.annotate(
        f"2% flow: {float(target['flow_stress_2pct_MPa']):.2f} / "
        f"{float(candidate['flow_stress_2pct_MPa']):.2f} MPa",
        (
            0.02,
            0.5
            * (
                float(target["flow_stress_2pct_MPa"])
                + float(candidate["flow_stress_2pct_MPa"])
            ),
        ),
        xytext=(-165, 18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": "0.35", "lw": 1.0},
        fontsize=9,
    )

    axis.text(
        0.985,
        0.54,
        "Independent coarse errors\n"
        f"Proof: {100*float(candidate['relative_error_proof']):.3f}%\n"
        f"Flow:  {100*float(candidate['relative_error_flow']):.3f}%",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.94,
        },
    )

    fine_candidate = verification["fine_candidate"]
    inset = inset_axes(
        axis, width="27%", height="31%", loc="lower right", borderpad=1.2
    )
    properties = np.arange(2)
    width = 0.34
    coarse_errors = np.array(
        [
            100.0 * float(candidate["relative_error_proof"]),
            100.0 * float(candidate["relative_error_flow"]),
        ]
    )
    fine_errors = np.array(
        [
            100.0 * float(fine_candidate["relative_error_proof"]),
            100.0 * float(fine_candidate["relative_error_flow"]),
        ]
    )
    inset.bar(
        properties - width / 2, coarse_errors, width, color="#7F7F7F", label="Coarse"
    )
    inset.bar(
        properties + width / 2, fine_errors, width, color=COLOR_OPTIMIZED, label="Fine"
    )
    inset.axhline(1.0, color="0.2", linestyle=":", linewidth=1.0)
    inset.set_xticks(properties, ["P0.2", "Pflow"])
    inset.set_ylim(
        0.0, max(1.05, 1.25 * float(max(coarse_errors.max(), fine_errors.max())))
    )
    inset.set_ylabel("Error (%)", fontsize=8)
    inset.set_title("Same-tessellation verification", fontsize=8.5)
    inset.tick_params(labelsize=8)
    inset.legend(fontsize=7.5, loc="upper right")
    inset.grid(axis="y", alpha=0.2)

    y_upper = 1.08 * max(float(target_y.max()), float(candidate_y.max()))
    axis.set(
        xlim=(-0.00035, 0.0207),
        ylim=(-0.03 * y_upper, y_upper),
        xlabel=r"Macroscopic nominal strain $\bar F_{11}-1$",
        ylabel=r"Macroscopic PK1 $\bar P_{11}$ (MPa)",
    )
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.22, 0.055),
        fontsize=9.2,
        frameon=True,
        facecolor="white",
        edgecolor="0.78",
        framealpha=0.94,
    )
    fig.suptitle(
        "Target versus optimized FCC crystal-plasticity RVE response",
        fontsize=14,
        y=0.965,
    )
    fig.text(
        0.5,
        0.885,
        rf"Best texture: $\eta$={float(best['eta']):.6f}, $\theta$={float(best['theta_deg']):+.4f}°",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.048,
        "P0.2 is the 0.2% offset proof stress under the prescribed plane-strain loading condition.\n"
        "Main curves are unsmoothed coarse FEM histories; fine verification is shown in the inset.",
        ha="center",
        fontsize=9,
    )

    paths = save_static_figure(fig, "figB_final_macroscopic_response")
    return {
        "name": "Figure B — Final Macroscopic Response",
        "outputs": output_metadata(paths),
        "data_sources": {
            "target_curve": {
                "classification": "real MOOSE FCC CP FEM curve",
                "solution_csv": source_path(target_sources["csv"]),
                "run_identifier": target_sources["run_directory"].name,
                "curve_points": len(target_curve),
            },
            "optimized_curve": {
                "classification": "real independent MOOSE FCC CP FEM curve",
                "solution_csv": source_path(candidate_sources["csv"]),
                "run_identifier": candidate_sources["run_directory"].name,
                "eta": float(best["eta"]),
                "theta_deg": float(best["theta_deg"]),
                "curve_points": len(candidate_curve),
            },
            "fine_verification": "results/cp_inverse_design_2d/verification/verification_summary.json",
        },
        "coarse_relative_errors": {
            "proof": float(candidate["relative_error_proof"]),
            "flow": float(candidate["relative_error_flow"]),
        },
        "fine_relative_errors": {
            "proof": float(fine_candidate["relative_error_proof"]),
            "flow": float(fine_candidate["relative_error_flow"]),
        },
    }


def main() -> int:
    print(json.dumps(build_final_comparison(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
