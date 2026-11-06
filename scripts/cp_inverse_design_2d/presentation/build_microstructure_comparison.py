#!/usr/bin/env python3
"""Build Figure C: target/optimized texture and accumulated-slip comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_SCRIPT_DIR = SCRIPT_DIR.parent
for directory in (SCRIPT_DIR, PARENT_SCRIPT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from presentation_utils import (  # noqa: E402
    audit_history,
    common_inputs,
    configure_style,
    draw_texture,
    exodus_field,
    mesh_faces,
    output_metadata,
    save_static_figure,
    source_path,
    verification_entry_sources,
)


def build_microstructure_comparison(
    audit: dict[str, object] | None = None,
) -> dict[str, object]:
    configure_style()
    audit = audit or audit_history()
    inputs = common_inputs()
    verification = inputs["verification"]
    best = inputs["best"]
    hidden = verification["hidden_design_disclosed_after_optimization"]
    target = verification["coarse_target"]
    candidate = verification["independent_coarse_candidate"]
    target_sources = verification_entry_sources(target)
    candidate_sources = verification_entry_sources(candidate)

    target_slip = exodus_field(target_sources["exodus"], "accumulated_slip")
    candidate_slip = exodus_field(candidate_sources["exodus"], "accumulated_slip")
    slip_min = min(float(target_slip[2].min()), float(candidate_slip[2].min()))
    slip_max = max(float(target_slip[2].max()), float(candidate_slip[2].max()))
    mesh = mesh_faces()

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 9.1))
    fig.subplots_adjust(
        left=0.08, right=0.90, bottom=0.10, top=0.87, wspace=0.22, hspace=0.26
    )
    draw_texture(
        axes[0, 0],
        float(hidden["eta"]),
        float(hidden["theta_deg"]),
        rf"Target texture ($\eta$={float(hidden['eta']):.3f}, $\theta$={float(hidden['theta_deg']):+.1f}°)",
        mesh,
    )
    draw_texture(
        axes[0, 1],
        float(best["eta"]),
        float(best["theta_deg"]),
        rf"Optimized texture ($\eta$={float(best['eta']):.3f}, $\theta$={float(best['theta_deg']):+.2f}°)",
        mesh,
    )

    image = None
    for axis, data, title in (
        (axes[1, 0], target_slip, "Target accumulated slip — real FEM"),
        (axes[1, 1], candidate_slip, "Optimized accumulated slip — real FEM"),
    ):
        triangulation = mtri.Triangulation(data[0][:, 0], data[0][:, 1], data[1])
        image = axis.tripcolor(
            triangulation,
            facecolors=data[2],
            shading="flat",
            cmap="magma",
            vmin=slip_min,
            vmax=slip_max,
        )
        axis.set(
            xlim=(0.0, 200.0),
            ylim=(0.0, 200.0),
            aspect="equal",
            xlabel="x (µm)",
            ylabel="y (µm)",
            title=title,
        )
    colorbar = fig.colorbar(image, ax=axes[1, :], fraction=0.045, pad=0.04)
    colorbar.set_label(r"Accumulated slip at $\bar F_{11}=1.02$")

    for axis in axes.flat:
        axis.grid(False)
    key = [
        Patch(facecolor=(1.0, 0.0, 0.0), label="[001]"),
        Patch(facecolor=(0.0, 1.0, 0.0), label="[101]"),
        Patch(facecolor=(0.0, 0.0, 1.0), label="[111]"),
    ]
    fig.legend(
        handles=key,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=3,
        title="Fixed cubic IPF-X key",
        fontsize=9,
        title_fontsize=9,
    )
    fig.suptitle(
        "Target and optimized microstructure on the identical Neper RVE",
        fontsize=14,
        y=0.97,
    )
    fig.text(
        0.5,
        0.025,
        r"Texture colors use $Q^T e_X$ with the MOOSE crystal-to-sample rotation, cubic-symmetry reduction, "
        "and one fixed [001]–[101]–[111] component-difference key. Slip panels share one numerical scale.",
        ha="center",
        fontsize=9,
    )

    paths = save_static_figure(fig, "figC_target_optimized_microstructure")
    return {
        "name": "Figure C — Target and Optimized Microstructure",
        "outputs": output_metadata(paths),
        "data_sources": {
            "geometry": {
                "classification": "fixed Neper 20-grain periodic TRI6 geometry",
                "mesh": "results/cp_inverse_design_2d/meshes/rve20_coarse_tri6.msh",
            },
            "target_texture": {
                "classification": "SO(3) texture regenerated from the executed hidden design",
                "eta": float(hidden["eta"]),
                "theta_deg": float(hidden["theta_deg"]),
            },
            "optimized_texture": {
                "classification": "SO(3) texture regenerated from the executed optimized design",
                "eta": float(best["eta"]),
                "theta_deg": float(best["theta_deg"]),
            },
            "target_slip": {
                "classification": "real MOOSE FCC CP Exodus element field",
                "exodus": source_path(target_sources["exodus"]),
                "run_identifier": target_sources["run_directory"].name,
            },
            "optimized_slip": {
                "classification": "real independent MOOSE FCC CP Exodus element field",
                "exodus": source_path(candidate_sources["exodus"]),
                "run_identifier": candidate_sources["run_directory"].name,
            },
        },
        "orientation_color_rule": {
            "name": "fixed cubic IPF-X component-difference key",
            "rotation_convention": "sample X in crystal frame = Q^T e_X, where Q is MOOSE crysrot",
            "fundamental_sector": "0 <= u <= v <= w after cubic sign/permutation reduction",
            "vertices": {"[001]": "red", "[101]": "green", "[111]": "blue"},
            "frame_normalization": False,
        },
        "shared_slip_range": [slip_min, slip_max],
    }


def main() -> int:
    print(json.dumps(build_microstructure_comparison(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
