#!/usr/bin/env python3
"""Run the complete private EBSD audit, statistics, Neper calibration, and validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from analyze_grain_correlations import analyze_joint_statistics
from audit_coordinate_metadata import audit_coordinate_metadata
from calibrate_neper_parameters import calibrate
from clean_ebsd_data import CleaningPolicy
from common import (
    CALIBRATED_PARAMETERS,
    DATASETS,
    DISTRIBUTION_FITS,
    EXPERIMENTAL_STATISTICS,
    FIGURES,
    RESULTS_ROOT,
    VALIDATION,
    configured_data_root,
    ensure_result_directories,
    find_dataset_path,
    write_json,
)
from extract_experimental_statistics import raw_and_cleaned_statistics
from extract_texture_statistics import extract_texture_statistics
from fit_grain_distributions import fit_grain_distributions
from generate_ebsd_matched_neper import NeperMorphologyParameters
from load_ebsd_grains import load_dataset
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
from verify_rve_statistics import verify_representativeness
from write_calibration_report import write_report


def _summary_row(key: str, stage: str, summary: dict[str, object]) -> dict[str, object]:
    values = summary[stage]
    ecd = values["equivalent_diameter_um"]["number_weighted"]
    aspect = values["aspect_ratio"]["number_weighted"]
    return {
        "dataset": key,
        "stage": stage,
        "grain_count": values["grain_count"],
        "mean_area_um2": values["grain_area_um2"]["number_weighted"]["mean"],
        "mean_ecd_um": ecd["mean"],
        "median_ecd_um": ecd["q50"],
        "cv_ecd": ecd["coefficient_of_variation"],
        "q95_ecd_um": ecd["q95"],
        "q99_ecd_um": ecd["q99"],
        "maximum_ecd_um": ecd["maximum"],
        "mean_aspect_ratio": aspect["mean"],
        "median_aspect_ratio": aspect["q50"],
        "cv_aspect_ratio": aspect["coefficient_of_variation"],
        "q95_aspect_ratio": aspect["q95"],
        "size_ar_spearman": values["size_aspect_joint"]["spearman_rho"],
    }


def _combine_validation_realizations(directory: Path, seeds: list[int]) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        frame = pd.read_csv(
            directory
            / "validation_realizations"
            / f"seed_{seed}"
            / "synthetic_grains.csv"
        )
        frame.insert(0, "geometry_seed", seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run(args: argparse.Namespace) -> dict[str, object]:
    ensure_result_directories()
    try:
        data_root = (
            args.data_root.resolve()
            if args.data_root
            else configured_data_root(required=True)
        )
    except FileNotFoundError as error:
        status = {
            "status": "blocked_missing_private_ebsd_data",
            "error": str(error),
            "required_filenames": [spec.filename for spec in DATASETS.values()],
            "resolution": (
                "Set SPP2489_EBSD_DATA_ROOT to a read-only local checkout of the private "
                "GitLab repository or to a directory containing the four CSV files."
            ),
            "no_synthetic_experimental_results_generated": True,
        }
        write_json(RESULTS_ROOT / "data_access_status.json", status)
        raise

    selected_keys = args.dataset or list(DATASETS)
    manifest: dict[str, object] = {
        "workflow": "Experimental EBSD statistics to statistically equivalent periodic Neper RVEs",
        "private_source_root_is_not_recorded": True,
        "datasets": {},
        "figures_directory": str(FIGURES),
        "calibration_complete": not args.statistics_only,
        "coordinate_convention_status": (
            "Each section is analyzed independently. Cross-section or 3D coordinate mapping "
            "is not claimed until original EBSD metadata are audited."
        ),
    }
    coordinate_audit = audit_coordinate_metadata(data_root)
    write_json(
        EXPERIMENTAL_STATISTICS / "coordinate_metadata_evidence.json",
        coordinate_audit,
    )
    manifest["coordinate_metadata_evidence"] = str(
        EXPERIMENTAL_STATISTICS / "coordinate_metadata_evidence.json"
    )
    summary_rows: list[dict[str, object]] = []
    cleaned_frames: dict[str, pd.DataFrame] = {}
    distribution_payload: dict[str, object] = {}
    texture_payload: dict[str, object] = {}
    calibration_histories: dict[str, pd.DataFrame] = {}
    representativeness_tables: dict[str, pd.DataFrame] = {}
    experimental_synthetic_pairs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    representative_tessellations: dict[str, tuple[Path, float]] = {}
    best_payload: dict[str, object] = {}
    validation_payload: dict[str, object] = {}
    rve_payload: dict[str, object] = {}

    for key in selected_keys:
        source_path = find_dataset_path(DATASETS[key], data_root)
        frame, source = load_dataset(
            key, data_root=data_root, angle_unit=args.angle_unit
        )
        if args.require_reference_count and not source["raw_count_matches_reference"]:
            raise RuntimeError(
                f"{key} has {source['raw_row_count']} rows, expected reference "
                f"{DATASETS[key].expected_raw_count}; audit the source before calibration"
            )
        statistics, cleaned, audited = raw_and_cleaned_statistics(
            frame,
            CleaningPolicy(minimum_ecd_um=args.minimum_ecd),
            alpha_is_grain_axis_direction=DATASETS[key].alpha_is_grain_axis_direction,
            alpha_semantics=DATASETS[key].alpha_semantics,
        )
        cleaned_path = EXPERIMENTAL_STATISTICS / f"{key}_cleaned.csv"
        audit_path = EXPERIMENTAL_STATISTICS / f"{key}_cleaning_audit.csv"
        statistics_path = EXPERIMENTAL_STATISTICS / f"{key}_statistics.json"
        cleaned.to_csv(cleaned_path, index=False)
        audited.to_csv(audit_path, index=False)
        write_json(statistics_path, {"source": source, **statistics})
        cleaned_frames[key] = cleaned
        summary_rows.extend(
            [
                _summary_row(key, "raw_valid_geometry", statistics),
                _summary_row(key, "cleaned", statistics),
            ]
        )
        fits = fit_grain_distributions(cleaned)
        correlations = analyze_joint_statistics(cleaned)
        texture = extract_texture_statistics(cleaned)
        write_json(DISTRIBUTION_FITS / f"{key}_distribution_fits.json", fits)
        write_json(
            EXPERIMENTAL_STATISTICS / f"{key}_joint_statistics.json", correlations
        )
        write_json(EXPERIMENTAL_STATISTICS / f"{key}_texture_statistics.json", texture)
        distribution_payload[key] = fits
        texture_payload[key] = texture
        dataset_manifest: dict[str, object] = {
            "source_filename": source_path.name,
            "source_sha256": source["source_sha256"],
            "cleaned_csv": str(cleaned_path),
            "cleaning_audit_csv": str(audit_path),
            "statistics_json": str(statistics_path),
        }
        if not args.statistics_only:
            calibration_directory = CALIBRATED_PARAMETERS / key
            calibration = calibrate(
                cleaned,
                calibration_directory,
                dataset_key=key,
                phase="all",
                side_length_um=args.calibration_side,
                maximum_evaluations=args.maximum_evaluations,
                include_direction=DATASETS[key].alpha_is_grain_axis_direction,
                aspect_ratio_proxy_scale=args.aspect_ratio_proxy_scale,
            )
            best_payload[key] = calibration["best_parameters"]
            validation_payload[key] = calibration
            calibration_history = pd.read_csv(
                calibration_directory / "neper_calibration_history.csv"
            )
            calibration_histories[key] = calibration_history
            validation_seeds = list(calibration["validation_seeds_not_used_in_fit"])
            ensemble = _combine_validation_realizations(
                calibration_directory, validation_seeds
            )
            ensemble_path = VALIDATION / f"{key}_validation_ensemble.csv"
            ensemble.to_csv(ensemble_path, index=False)
            experimental_synthetic_pairs[key] = (cleaned, ensemble)
            representative = (
                calibration_directory
                / "validation_realizations"
                / f"seed_{validation_seeds[0]}"
                / "rve_periodic_ebsd_matched.tess"
            )
            representative_tessellations[key] = (representative, args.calibration_side)
            parameters = NeperMorphologyParameters(**calibration["best_parameters"])
            representativeness_directory = VALIDATION / key / "representativeness"
            representativeness = verify_representativeness(
                cleaned,
                representativeness_directory,
                dataset_key=key,
                phase="all",
                parameters=parameters,
                side_lengths_um=tuple(args.rve_side or (200.0, 400.0)),
                include_direction=DATASETS[key].alpha_is_grain_axis_direction,
            )
            rve_payload[key] = representativeness
            representativeness_table = pd.read_csv(
                representativeness_directory / "rve_representativeness.csv"
            )
            representativeness_tables[key] = representativeness_table
            dataset_manifest.update(
                {
                    "side_length_um": args.calibration_side,
                    "calibration_history_csv": str(
                        calibration_directory / "neper_calibration_history.csv"
                    ),
                    "validation_ensemble_csv": str(ensemble_path),
                    "representative_tessellation": str(representative),
                    "representativeness_csv": str(
                        representativeness_directory / "rve_representativeness.csv"
                    ),
                }
            )
        manifest["datasets"][key] = dataset_manifest

    pd.DataFrame(summary_rows).to_csv(
        RESULTS_ROOT / "experimental_grain_statistics.csv", index=False
    )
    write_json(
        RESULTS_ROOT / "experimental_distribution_parameters.json", distribution_payload
    )
    write_json(RESULTS_ROOT / "experimental_texture_statistics.json", texture_payload)
    write_json(RESULTS_ROOT / "workflow_manifest.json", manifest)
    _style()
    plot_experimental_size(cleaned_frames, FIGURES)
    plot_experimental_aspect(cleaned_frames, FIGURES)
    plot_size_aspect_correlation(cleaned_frames, FIGURES)
    plot_experimental_texture(cleaned_frames, FIGURES)
    if not args.statistics_only:
        write_json(RESULTS_ROOT / "neper_best_parameters.json", best_payload)
        write_json(
            RESULTS_ROOT / "neper_statistical_validation.json", validation_payload
        )
        write_json(RESULTS_ROOT / "rve_representativeness.json", rve_payload)
        combined_history = pd.concat(
            [frame.assign(dataset=key) for key, frame in calibration_histories.items()],
            ignore_index=True,
        )
        combined_history.to_csv(
            RESULTS_ROOT / "neper_calibration_history.csv", index=False
        )
        plot_experimental_neper_comparisons(experimental_synthetic_pairs, FIGURES)
        plot_representative_neper(representative_tessellations, FIGURES)
        plot_calibration_convergence(calibration_histories, FIGURES)
        plot_representativeness(representativeness_tables, FIGURES)
    write_report(RESULTS_ROOT, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument(
        "--angle-unit", choices=("auto", "degree", "radian"), default="auto"
    )
    parser.add_argument("--minimum-ecd", type=float, default=2.8)
    parser.add_argument("--require-reference-count", action="store_true")
    parser.add_argument("--statistics-only", action="store_true")
    parser.add_argument("--calibration-side", type=float, default=200.0)
    parser.add_argument("--rve-side", type=float, action="append")
    parser.add_argument("--maximum-evaluations", type=int, default=24)
    parser.add_argument("--aspect-ratio-proxy-scale", type=float, default=1.0)
    args = parser.parse_args()
    try:
        manifest = run(args)
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
