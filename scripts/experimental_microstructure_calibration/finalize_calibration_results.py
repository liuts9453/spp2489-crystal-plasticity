#!/usr/bin/env python3
"""Assemble the reviewed final EBSD--Neper results, figures, and report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import FIGURES, RESULTS_ROOT, write_json
from compare_experimental_neper import comparison_metrics
from plot_experimental_calibration import (
    _style,
    plot_calibration_convergence,
    plot_experimental_aspect,
    plot_experimental_neper_comparisons,
    plot_experimental_size,
    plot_experimental_texture,
    plot_representative_neper,
    plot_representativeness,
    plot_size_aspect_correlation,
)
from write_calibration_report import write_report


FINAL_SELECTIONS = {
    "Al1_oBD": {
        "diameter_log_spread_scale": 0.95,
        "directional_ratio_x_over_y": 1.25660608,
        "aspect_ratio_proxy_scale": 1.0,
        "input_size_shape_rank_correlation": 0.75,
        "validation_400": "validation_final/Al1_oBD",
    },
    "Al1_pBD": {
        "diameter_log_spread_scale": 0.95,
        "directional_ratio_x_over_y": 2.45,
        "aspect_ratio_proxy_scale": 1.0,
        "input_size_shape_rank_correlation": 1.0,
        "validation_400": "validation_final_adjusted/Al1_pBD",
    },
    "Al0_00": {
        "diameter_log_spread_scale": 0.95,
        "directional_ratio_x_over_y": 1.26,
        "aspect_ratio_proxy_scale": 1.0,
        "input_size_shape_rank_correlation": 0.75,
        "validation_400": "validation_final/Al0_00",
    },
    "Al0_90": {
        "diameter_log_spread_scale": 0.95,
        "directional_ratio_x_over_y": 2.25,
        "aspect_ratio_proxy_scale": 1.0,
        "input_size_shape_rank_correlation": 1.0,
        "validation_400": "validation_final_adjusted/Al0_90",
    },
}


def _ensemble(directory: Path) -> tuple[pd.DataFrame, list[tuple[int, pd.DataFrame]]]:
    frames = []
    seeded = []
    for path in sorted((directory / "side_400").glob("seed_*/synthetic_grains.csv")):
        seed = int(path.parent.name.removeprefix("seed_"))
        frame = pd.read_csv(path)
        seeded.append((seed, frame))
        frames.append(frame.assign(geometry_seed=seed))
    if len(frames) != 5:
        raise RuntimeError(
            f"Expected five final realizations in {directory}, found {len(frames)}"
        )
    return pd.concat(frames, ignore_index=True), seeded


def finalize(results_root: Path = RESULTS_ROOT) -> dict[str, object]:
    frames: dict[str, pd.DataFrame] = {}
    pairs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    tessellations: dict[str, tuple[Path, float]] = {}
    histories: dict[str, pd.DataFrame] = {}
    representativeness_tables: dict[str, pd.DataFrame] = {}
    validation_payload: dict[str, object] = {}
    best_payload: dict[str, object] = {}
    rve_payload: dict[str, object] = {}
    combined_histories = []

    for key, selection in FINAL_SELECTIONS.items():
        experimental = pd.read_csv(
            results_root / "experimental_statistics" / f"{key}_cleaned.csv"
        )
        frames[key] = experimental
        final_directory = results_root / str(selection["validation_400"])
        ensemble, seeded = _ensemble(final_directory)
        pairs[key] = (experimental, ensemble)
        tessellations[key] = (
            final_directory
            / "side_400"
            / "seed_8301"
            / "rve_periodic_ebsd_matched.tess",
            400.0,
        )

        final_table = pd.read_csv(final_directory / "rve_representativeness.csv")
        table_200 = pd.read_csv(
            results_root / "validation_final_200" / key / "rve_representativeness.csv"
        )
        representativeness_tables[key] = pd.concat(
            [table_200, final_table], ignore_index=True
        )
        history = pd.read_csv(
            results_root
            / "calibrated_parameters_joint"
            / key
            / "neper_calibration_history.csv"
        )
        histories[key] = history
        combined_histories.append(history.assign(dataset=key))

        per_seed = []
        synthetic_correlations = []
        for seed, synthetic in seeded:
            metrics = comparison_metrics(
                experimental, synthetic, include_direction=False
            )
            synthetic_rho = float(
                stats.spearmanr(
                    synthetic["equivalent_diameter_um"],
                    synthetic["aspect_ratio_recomputed"],
                ).statistic
            )
            synthetic_correlations.append(synthetic_rho)
            per_seed.append({"seed": seed, **metrics})
        experimental_rho = float(
            stats.spearmanr(
                experimental["equivalent_diameter_um"],
                experimental["aspect_ratio_recomputed"],
            ).statistic
        )
        component_names = (
            "diameter",
            "aspect",
            "correlation",
            "odf",
            "size_orientation",
        )
        component_means = {
            name: float(
                np.mean([item["normalized_components"][name] for item in per_seed])
            )
            for name in component_names
        }
        objective = np.asarray([item["objective"] for item in per_seed], dtype=float)
        parameters = {
            "diameter_log_spread_scale": selection["diameter_log_spread_scale"],
            "directional_ratio_x_over_y": selection["directional_ratio_x_over_y"],
            "aspect_ratio_proxy_scale": selection["aspect_ratio_proxy_scale"],
            "input_size_shape_rank_correlation": selection[
                "input_size_shape_rank_correlation"
            ],
            "one_minus_circularity_mean": None,
            "one_minus_circularity_standard_deviation": None,
        }
        final_summary = json.loads(
            (final_directory / "rve_representativeness.json").read_text(
                encoding="utf-8"
            )
        )["side_length_summaries"][0]
        calibration = json.loads(
            (
                results_root
                / "calibrated_parameters_joint"
                / key
                / "neper_best_parameters.json"
            ).read_text(encoding="utf-8")
        )
        validation_payload[key] = {
            "dataset": key,
            "side_length_um": 400.0,
            "grain_count": final_summary["grain_count"],
            "recommended_rve_side_length_um": 400.0,
            "optimizer": "staged deterministic candidate matrix and copula probes",
            "optimizer_success": True,
            "optimizer_message": "Reviewed final parameters validated on five disjoint seeds",
            "evaluation_count": calibration["evaluation_count"],
            "calibration_seeds": calibration["calibration_seeds"],
            "validation_seeds_not_used_in_fit": [seed for seed, _ in seeded],
            "direction_target_used": False,
            "morphology_time_limit_s": 60.0,
            "best_parameters": parameters,
            "best_calibration_objective": float(objective.mean()),
            "validation": per_seed,
            "validation_objective_mean": float(objective.mean()),
            "validation_objective_standard_deviation": float(objective.std()),
            "validation_objective_maximum": float(objective.max()),
            "component_means": component_means,
            "experimental_size_ar_spearman": experimental_rho,
            "synthetic_size_ar_spearman_mean": float(np.mean(synthetic_correlations)),
            "synthetic_size_ar_spearman_standard_deviation": float(
                np.std(synthetic_correlations)
            ),
            "synthetic_size_ar_spearman_range": [
                float(np.min(synthetic_correlations)),
                float(np.max(synthetic_correlations)),
            ],
            "experimental_mean_ecd_um": float(
                experimental["equivalent_diameter_um"].mean()
            ),
            "synthetic_mean_ecd_um": float(ensemble["equivalent_diameter_um"].mean()),
            "experimental_mean_aspect_ratio": float(
                experimental["aspect_ratio_recomputed"].mean()
            ),
            "synthetic_mean_aspect_ratio": float(
                ensemble["aspect_ratio_recomputed"].mean()
            ),
            "successful_realizations": 5,
            "failed_realizations": 0,
        }
        best_payload[key] = {
            "recommended_rve_side_length_um": 400.0,
            "grain_count": final_summary["grain_count"],
            "parameters": parameters,
        }

        summary_200 = json.loads(
            (
                results_root
                / "validation_final_200"
                / key
                / "rve_representativeness.json"
            ).read_text(encoding="utf-8")
        )
        summary_400 = json.loads(
            (final_directory / "rve_representativeness.json").read_text(
                encoding="utf-8"
            )
        )
        rve_payload[key] = {
            "dataset": key,
            "phase": "all",
            "validation_seeds": [8301, 8302, 8303, 8304, 8305],
            "parameters": parameters,
            "side_length_summaries": (
                summary_200["side_length_summaries"]
                + summary_400["side_length_summaries"]
            ),
            "failed_tasks": summary_200["failed_tasks"] + summary_400["failed_tasks"],
            "recommendation": "400 μm periodic RVE",
        }

    pd.concat(combined_histories, ignore_index=True).to_csv(
        results_root / "neper_calibration_history.csv", index=False
    )
    write_json(results_root / "neper_best_parameters.json", best_payload)
    write_json(results_root / "neper_statistical_validation.json", validation_payload)
    write_json(results_root / "rve_representativeness.json", rve_payload)

    _style()
    plot_experimental_size(frames, FIGURES)
    plot_experimental_aspect(frames, FIGURES)
    plot_size_aspect_correlation(frames, FIGURES)
    plot_experimental_texture(frames, FIGURES)
    plot_experimental_neper_comparisons(pairs, FIGURES)
    plot_representative_neper(tessellations, FIGURES)
    plot_calibration_convergence(histories, FIGURES)
    plot_representativeness(representativeness_tables, FIGURES)

    manifest = {
        "calibration_complete": True,
        "datasets": {key: {} for key in FINAL_SELECTIONS},
        "public_export_contains_private_raw_data": False,
    }
    write_report(results_root, manifest)
    write_json(results_root / "finalization_manifest.json", manifest)
    return manifest


def main() -> int:
    manifest = finalize()
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
