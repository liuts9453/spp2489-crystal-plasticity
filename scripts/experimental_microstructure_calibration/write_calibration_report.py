#!/usr/bin/env python3
"""Write a traceable report that never upgrades missing or failed data to success."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_report(results_root: Path, manifest: dict[str, object]) -> Path:
    statistics = pd.read_csv(results_root / "experimental_grain_statistics.csv")
    distributions = json.loads(
        (results_root / "experimental_distribution_parameters.json").read_text(
            encoding="utf-8"
        )
    )
    lines = [
        "# Experimental EBSD → Neper Statistical Calibration",
        "",
        "## Scope",
        "",
        "This workflow calibrates periodic two-dimensional Neper tessellations to experimental EBSD grain statistics. It does not run MOOSE, crystal plasticity, constitutive calibration, or mechanical inverse design.",
        "",
        "The four sections are analyzed independently. No unique three-dimensional reconstruction or oBD/pBD versus 00/90 coordinate equivalence is claimed without the original acquisition metadata.",
        "",
        "## Data provenance and cleaning",
        "",
        "Raw private CSV files are read through `SPP2489_EBSD_DATA_ROOT`; absolute private paths and raw records are not included in the public repository. Source filenames and SHA-256 hashes are retained locally.",
        "",
        "The cleaned population removes invalid/duplicate records and grains below the explicit 2.8 μm equivalent-diameter threshold. It does not silently remove the upper grain-size or aspect-ratio tails.",
        "",
        "The private-source MTEX script explicitly selects `fcc` grains before applying the 2.8 μm threshold. The exported CSVs contain only numeric phase label `0`; the workflow preserves that label because no explicit numeric phase-index mapping was found. Separate DRAGen preprocessing applies `b ≥ 0.5 μm` for geometry/meshing, which is documented but not substituted for the experimental-statistics filter.",
        "",
        "The CSV `alpha` field is constant by section (0° or 90°), and the Al1 export code explicitly assigns it from the filename. It is section metadata, not a measured per-grain principal-axis direction. The direction discrepancy is therefore reported diagnostically with zero calibration weight; a true direction distribution cannot be recovered from these four tables.",
        "",
        "Because the direction is unavailable, `r_x/r_y` and its reciprocal cannot be distinguished from AR alone. Calibration therefore constrains the reported ratio to ≥1 as an unsigned major/minor elongation convention; it must not be interpreted as a measured preference for the global x direction.",
        "",
        "| Dataset | Population | Grains | Mean area (μm²) | Mean ECD (μm) | CV ECD | Mean AR | Spearman size–AR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in statistics.itertuples():
        lines.append(
            f"| {row.dataset} | {row.stage} | {int(row.grain_count)} | {row.mean_area_um2:.4f} | "
            f"{row.mean_ecd_um:.4f} | {row.cv_ecd:.4f} | {row.mean_aspect_ratio:.4f} | "
            f"{row.size_ar_spearman:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Statistical model",
            "",
            "ECD candidates are lognormal, Weibull, gamma, and a two-component lognormal mixture, ranked by BIC and checked against empirical CDF and q95/q99 tail error. Aspect ratio is fitted as the positive excess `AR − 1`, preserving `AR ≥ 1`.",
            "",
            "Texture is represented by complete Bunge rotations using the same crystal-to-sample convention as SPP2489. The pipeline uses all 24 proper cubic symmetry rotations and empirical SO(3) sampling; Euler components are never sampled independently. Texture statistics remain phase-separated.",
            "",
            "| Dataset | ECD model selected by BIC | ECD BIC | ECD CDF sup. error | AR−1 model selected by BIC | AR−1 BIC |",
            "|---|---|---:|---:|---|---:|",
        ]
    )
    for key, payload in distributions.items():
        ecd = payload["equivalent_diameter_um"]["selected_by_bic"]
        ar = payload["aspect_ratio"]["excess_over_one"]["selected_by_bic"]
        ecd_cdf = ecd.get("cdf_supremum_discrepancy", ecd.get("ks_statistic"))
        lines.append(
            f"| {key} | {ecd['model']} | {ecd['bic']:.3f} | {ecd_cdf:.4f} | "
            f"{ar['model']} | {ar['bic']:.3f} |"
        )
    lines.extend(
        [
            "",
            "BIC is not treated as sufficient by itself: empirical CDF, q95/q99 tail error, and the achieved Neper distributions are retained for review.",
            "",
        ]
    )
    if manifest["calibration_complete"]:
        validation = json.loads(
            (results_root / "neper_statistical_validation.json").read_text(
                encoding="utf-8"
            )
        )
        lines.extend(
            [
                "## Neper calibration and independent validation",
                "",
                "Neper uses paired cell-wise `diameq:file(...)` and `1-circularity:file(...)` targets so wide tails and size--shape dependence are not forced into independent one-parameter distributions. The circularity target is a documented fitted-ellipse proxy because the CSVs do not contain measured grain perimeter. A deterministic candidate matrix calibrates ECD spread and global aspect ratio; an explicit input copula parameter compensates the Laguerre generator's intrinsic size--shape bias without changing either target marginal. Five disjoint validation seeds quantify realization variability.",
                "",
                "All final tessellations use the same 60 s native Neper morphology-optimization budget. The outer process timeout is only a fault guard; it is not used as a surrogate convergence result.",
                "",
                "| Dataset | Recommended side (μm) | Cells | Spread | x/y ratio | Input size–shape ρ | Evaluations | Validation objective mean ± SD (max) | Experimental / synthetic size–AR ρ |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for key, item in validation.items():
            parameters = item["best_parameters"]
            lines.append(
                f"| {key} | {item['side_length_um']:.0f} | {item['grain_count']} | "
                f"{parameters['diameter_log_spread_scale']:.6f} | "
                f"{parameters['directional_ratio_x_over_y']:.6f} | "
                f"{parameters['input_size_shape_rank_correlation']:.3f} | "
                f"{item['evaluation_count']} | "
                f"{item['validation_objective_mean']:.6f} ± {item['validation_objective_standard_deviation']:.6f} "
                f"({item['validation_objective_maximum']:.6f}) | "
                f"{item['experimental_size_ar_spearman']:.4f} / "
                f"{item['synthetic_size_ar_spearman_mean']:.4f} |"
            )
        lines.extend(
            [
                "",
                "The discrepancy combines normalized ECD and AR Wasserstein distances, size–AR rank-correlation error, and a cubic-IPF texture histogram distance. The section-label `alpha` direction term has zero weight. The complete component values and weights are stored in the validation JSON.",
                "",
                "The final output correlations are substantially closer to experiment than the uncorrected generator, but they are not exact. In particular, Al1_pBD retains a lower output size–AR correlation than the experiment. This is reported as a capability limit of the present Laguerre + global-`aspratio` parameterization, not hidden by selecting a favorable realization.",
                "",
            ]
        )
        representativeness_path = results_root / "rve_representativeness.json"
        if representativeness_path.is_file():
            representativeness = json.loads(
                representativeness_path.read_text(encoding="utf-8")
            )
            lines.extend(
                [
                    "## RVE-size representativeness",
                    "",
                    "| Dataset | Side (μm) | Grains | Objective mean ± SD | Worst realization | Maximum area error |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for key, payload in representativeness.items():
                for item in payload["side_length_summaries"]:
                    lines.append(
                        f"| {key} | {item['side_length_um']:.0f} | {item['grain_count']} | "
                        f"{item['objective_mean']:.6f} ± {item['objective_standard_deviation']:.6f} | "
                        f"{item['objective_maximum']:.6f} | {item['maximum_area_relative_error']:.3e} |"
                    )
                for failed in payload.get("failed_tasks", []):
                    lines.append(
                        f"| {key} | {failed['side_length_um']:.0f} | — | failed (seed {failed['seed']}) | — | — |"
                    )
            lines.extend(
                [
                    "",
                    "The RVE assessment uses the ensemble, not the single closest seed. A size is not declared representative solely because its mean objective is low; the change in ensemble spread and tail coverage is also inspected.",
                    "",
                    "All four datasets recommend a 400 μm periodic RVE. The 200 μm Al1_pBD ensemble contains one invalid-polygon failure (seed 8304), while its 400 μm ensemble completes 5/5 realizations. The attempted 800 μm study was stopped after exceeding the declared 30-minute wall-time cap and is not reported as completed.",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "## Calibration status",
                "",
                "Only the experimental audit/statistics stage was requested. No Neper calibration result is reported.",
                "",
            ]
        )
    lines.extend(
        [
            "## Reproduction",
            "",
            "```bash",
            "export SPP2489_EBSD_DATA_ROOT=/read-only/path/to/damic_spp2489",
            "python scripts/experimental_microstructure_calibration/run_calibration.py --statistics-only --require-reference-count",
            "# Run the reviewed Neper candidate/validation commands documented in the JSON history.",
            "python scripts/experimental_microstructure_calibration/finalize_calibration_results.py",
            "python scripts/experimental_microstructure_calibration/export_public_results.py",
            "```",
            "",
            "All generated experimental derivatives remain under `results/experimental_microstructure_calibration/`, which is intentionally excluded from public Git tracking pending data-sharing approval.",
        ]
    )
    report = results_root / "EXPERIMENTAL_NEPER_CALIBRATION_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
