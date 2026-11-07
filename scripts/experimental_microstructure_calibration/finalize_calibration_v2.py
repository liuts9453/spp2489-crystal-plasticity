#!/usr/bin/env python3
"""Aggregate v2 calibration, mechanical pilots, figures, and a traceable report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from scipy.io import netcdf_file

from common import RESULTS_ROOT, V2_RESULTS_ROOT, write_json
from neper_statistics import read_neper_faces


DATASETS = ("Al1_oBD", "Al1_pBD", "Al0_00", "Al0_90")
CALIBRATION_DIRECTORIES = {
    "Al1_oBD": "Al1_oBD",
    "Al1_pBD": "Al1_pBD_robust",
    "Al0_00": "Al0_00",
    "Al0_90": "Al0_90",
}
HISTORY_FILES = {
    "Al1_oBD": "calibration_history.csv",
    "Al1_pBD": "robust_calibration_history.csv",
    "Al0_00": "calibration_history.csv",
    "Al0_90": "calibration_history.csv",
}
METRICS = (
    "ecd_wasserstein_normalized",
    "ar_wasserstein_normalized",
    "mean_ecd_relative_error",
    "mean_ar_relative_error",
    "size_ar_spearman_absolute_error",
    "size_conditioned_ar_error",
)
COLORS = {"baseline": "#777777", "improved": "#0072B2"}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _calibration_directory(dataset: str) -> Path:
    return V2_RESULTS_ROOT / "calibration" / CALIBRATION_DIRECTORIES[dataset]


def _history(dataset: str) -> pd.DataFrame:
    return pd.read_csv(_calibration_directory(dataset) / HISTORY_FILES[dataset])


def _holdout(history: pd.DataFrame) -> pd.DataFrame:
    stage = (
        "second_independent_holdout"
        if "second_independent_holdout" in set(history["stage"])
        else "independent_holdout"
    )
    return history.loc[(history["stage"] == stage) & (history["status"] == "completed")]


def _validation_holdout(dataset: str) -> tuple[pd.DataFrame, bool]:
    fixed = (
        V2_RESULTS_ROOT
        / "fixed_iteration_confirmation"
        / dataset
        / "fixed_iteration_history.csv"
    )
    if fixed.is_file():
        history = pd.read_csv(fixed)
        return history.loc[history["status"] == "completed"], True
    return _holdout(_history(dataset)), False


def _run_directory(dataset: str, family: str, candidate: int, seed: int) -> Path:
    return (
        _calibration_directory(dataset)
        / "runs"
        / family
        / f"candidate_{candidate:03d}"
        / f"seed_{seed}"
    )


def _ensemble(dataset: str, family: str) -> pd.DataFrame:
    selected, fixed_iteration = _validation_holdout(dataset)
    rows = selected.loc[selected["family"] == family]
    frames = []
    for row in rows.itertuples():
        if fixed_iteration:
            path = (
                V2_RESULTS_ROOT
                / "fixed_iteration_confirmation"
                / dataset
                / "runs"
                / family
                / f"seed_{int(row.seed)}"
            )
            if not path.is_dir():
                path = path.parent.parent / "baseline" / path.name
        else:
            path = _run_directory(dataset, family, int(row.candidate_id), int(row.seed))
        frames.append(pd.read_csv(path / "synthetic_grains.csv"))
    if not frames:
        raise RuntimeError(f"No completed {family} holdouts for {dataset}")
    return pd.concat(frames, ignore_index=True)


def _representative_tessellation(dataset: str, family: str) -> Path:
    selected, fixed_iteration = _validation_holdout(dataset)
    row = selected.loc[selected["family"] == family].iloc[0]
    if fixed_iteration:
        path = (
            V2_RESULTS_ROOT
            / "fixed_iteration_confirmation"
            / dataset
            / "runs"
            / family
            / f"seed_{int(row['seed'])}"
        )
        if not path.is_dir():
            path = path.parent.parent / "baseline" / path.name
        return path / "rve_periodic_ebsd_matched.tess"
    return (
        _run_directory(dataset, family, int(row["candidate_id"]), int(row["seed"]))
        / "rve_periodic_ebsd_matched.tess"
    )


def _metric_summary() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        holdout, _ = _validation_holdout(dataset)
        for family in ("baseline", "improved"):
            selected = holdout.loc[holdout["family"] == family]
            row: dict[str, object] = {
                "dataset": dataset,
                "family": family,
                "successful_realizations": len(selected),
                "objective_mean": float(selected["morphology_objective"].mean()),
                "objective_standard_deviation": float(
                    selected["morphology_objective"].std(ddof=0)
                ),
            }
            row.update({metric: float(selected[metric].mean()) for metric in METRICS})
            rows.append(row)
    return pd.DataFrame(rows)


def _figure_1(summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(
        2, 2, figsize=(11.0, 8.2), sharey=False, constrained_layout=True
    )
    x_values = np.arange(len(METRICS))
    labels = ("ECD W", "AR W", "mean ECD", "mean AR", "|Δρ|", "conditional AR")
    for axis, dataset in zip(axes.flat, DATASETS):
        selected = summary.loc[summary["dataset"] == dataset]
        for offset, family in ((-0.18, "baseline"), (0.18, "improved")):
            values = selected.loc[selected["family"] == family, list(METRICS)].iloc[0]
            axis.bar(
                x_values + offset,
                values,
                width=0.36,
                color=COLORS[family],
                label=family,
            )
        axis.set_title(dataset)
        axis.set_xticks(x_values, labels, rotation=30, ha="right")
        axis.set_ylabel("normalized discrepancy")
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        "Statistical discrepancy before and after four-parameter calibration"
    )
    _save(figure, output / "fig01_statistical_discrepancy_before_after.png")


def _cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values, dtype=float))
    return ordered, np.arange(1, len(ordered) + 1) / len(ordered)


def _distribution_figure(column: str, label: str, name: str, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.4))
    for axis, dataset in zip(axes.flat, DATASETS):
        experimental = pd.read_csv(
            RESULTS_ROOT / "experimental_statistics" / f"{dataset}_cleaned.csv"
        )
        for frame, style, title, color in (
            (experimental, "-", "Experimental", "#D55E00"),
            (_ensemble(dataset, "baseline"), "--", "Baseline Neper", "#777777"),
            (_ensemble(dataset, "improved"), "-.", "V2 Neper", "#0072B2"),
        ):
            x_value, y_value = _cdf(frame[column].to_numpy())
            axis.plot(x_value, y_value, style, color=color, label=title)
        axis.set(title=dataset, xlabel=label, ylabel="empirical CDF")
    axes[0, 0].legend(frameon=False)
    figure.suptitle(f"Experimental, baseline, and v2 Neper {label.lower()}")
    _save(figure, output / name)


def _figure_4(output: Path) -> None:
    figure, axes = plt.subplots(4, 3, figsize=(11.2, 13.0), sharex="row", sharey="row")
    for row, dataset in enumerate(DATASETS):
        experimental = pd.read_csv(
            RESULTS_ROOT / "experimental_statistics" / f"{dataset}_cleaned.csv"
        )
        frames = (
            experimental,
            _ensemble(dataset, "baseline"),
            _ensemble(dataset, "improved"),
        )
        for axis, frame, title in zip(
            axes[row], frames, ("Experimental", "Baseline", "V2")
        ):
            axis.hexbin(
                frame["equivalent_diameter_um"],
                frame["aspect_ratio_recomputed"],
                gridsize=30,
                mincnt=1,
                bins="log",
                cmap="viridis",
            )
            rho = (
                frame[["equivalent_diameter_um", "aspect_ratio_recomputed"]]
                .corr(method="spearman")
                .iloc[0, 1]
            )
            axis.set_title(f"{dataset} · {title} · ρ={rho:.3f}")
            axis.set(xlabel="ECD (µm)", ylabel="AR")
    figure.suptitle("Grain size–aspect ratio joint statistics")
    _save(figure, output / "fig04_size_aspect_ratio_correlation.png")


def _figure_5(output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    components = (
        "ecd_wasserstein_normalized",
        "ar_wasserstein_normalized",
        "size_ar_spearman_absolute_error",
    )
    for axis, dataset in zip(axes.flat, DATASETS):
        history = _history(dataset)
        screening = history.loc[
            (history["family"] == "improved")
            & history["status"].eq("completed")
            & history["stage"].isin(
                ("screening", "local_refinement", "robust_calibration")
            )
        ].reset_index(drop=True)
        for component in components:
            axis.plot(
                screening.index + 1,
                screening[component],
                marker=".",
                ms=3,
                label=component,
            )
        axis.set(
            title=dataset, xlabel="completed geometry evaluation", ylabel="component"
        )
    axes[0, 0].legend(frameon=False, fontsize=7)
    figure.suptitle("Recorded statistical objective components")
    _save(figure, output / "fig05_statistical_objective_components.png")


def _figure_6(summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    metrics = (
        "ecd_wasserstein_normalized",
        "ar_wasserstein_normalized",
        "size_ar_spearman_absolute_error",
    )
    for axis, dataset in zip(axes.flat, DATASETS):
        uncertainty = pd.read_csv(
            V2_RESULTS_ROOT
            / "uncertainty"
            / dataset
            / "matched_size_exp_exp_replicates.csv"
        )
        x_value = np.arange(len(metrics))
        low = [uncertainty[item].quantile(0.025) for item in metrics]
        high = [uncertainty[item].quantile(0.975) for item in metrics]
        axis.vlines(
            x_value, low, high, color="#999999", lw=8, alpha=0.45, label="exp–exp 95%"
        )
        selected = summary.loc[summary["dataset"] == dataset]
        for offset, family, marker in (
            (-0.08, "baseline", "o"),
            (0.08, "improved", "s"),
        ):
            values = selected.loc[selected["family"] == family, list(metrics)].iloc[0]
            axis.scatter(
                x_value + offset,
                values,
                marker=marker,
                color=COLORS[family],
                label=family,
            )
        axis.set_xticks(x_value, ("ECD W", "AR W", "|Δρ|"))
        axis.set(title=dataset, ylabel="discrepancy")
    axes[0, 0].legend(frameon=False, fontsize=7)
    figure.suptitle(
        "Neper discrepancies relative to matched-size experimental sampling"
    )
    _save(figure, output / "fig06_bootstrap_finite_size_uncertainty.png")


def _figure_7(output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for axis, dataset in zip(axes, ("Al1_oBD", "Al1_pBD")):
        for family, shift, color in (
            ("baseline", 0.0, "#BBBBBB"),
            ("improved", 420.0, "#0072B2"),
        ):
            faces = read_neper_faces(_representative_tessellation(dataset, family))
            polygons = [[(x + shift, y) for x, y in polygon] for _, polygon in faces]
            axis.add_collection(
                PolyCollection(
                    polygons, facecolors="none", edgecolors=color, linewidths=0.25
                )
            )
        axis.set(
            xlim=(0, 820),
            ylim=(0, 400),
            aspect="equal",
            title=dataset,
            xlabel="baseline | v2",
        )
        axis.grid(False)
    figure.suptitle("Representative real periodic Neper tessellations")
    _save(figure, output / "fig07_baseline_vs_improved_microstructures.png")


def _mechanical_results() -> dict[str, dict[str, object]]:
    directory = V2_RESULTS_ROOT / "mechanical" / "Al1_oBD"
    results = {}
    for family in ("baseline", "improved"):
        path = (
            directory / family / "seed_10303" / "h_6_smooth_30_a_0p012" / "result.json"
        )
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("fem_executed") and payload.get("returncode") == 0:
                results[family] = payload
    return results


def _mechanical_summary(
    results: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    keys = (
        "dataset",
        "family",
        "geometry_seed",
        "grain_count",
        "characteristic_length_um",
        "phase_assumption",
        "converged",
        "returncode",
        "wall_time_s",
        "mpi_ranks",
        "peak_rss_kib",
        "failed_global_attempts",
        "proof_stress_MPa",
        "flow_stress_2pct_MPa",
        "volume_weighted_q95_accumulated_slip",
        "accumulated_slip_q95_definition",
        "mean_accumulated_slip",
        "maximum_accumulated_slip",
        "accepted_increments",
        "accepted_newton_iterations",
        "hill_mandel_relative_residual_max",
        "maximum_abs_P22_MPa",
        "maximum_abs_macro_shear",
        "minimum_det_Fp",
        "maximum_det_Fp",
        "integrated_plastic_dissipation",
        "checks",
    )
    summary = {
        family: {key: payload[key] for key in keys if key in payload}
        for family, payload in results.items()
    }
    holdout = _holdout(_history("Al1_oBD"))
    for family, payload in summary.items():
        row = holdout.loc[(holdout["family"] == family) & (holdout["seed"] == 10303)]
        if row.empty:
            continue
        selected = row.iloc[-1]
        payload["input_statistical_discrepancy"] = {
            key: float(selected[key])
            for key in (
                *METRICS,
                "odf_number_total_variation",
                "odf_area_total_variation",
            )
        }
    return summary


def _exodus_field(path: Path, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with netcdf_file(path, "r", mmap=False) as database:
        coordinates = np.column_stack(
            (
                np.asarray(database.variables["coordx"].data),
                np.asarray(database.variables["coordy"].data),
            )
        )
        names = [
            b"".join(row).decode().rstrip("\x00 ")
            for row in np.asarray(database.variables["name_elem_var"].data)
        ]
        field_index = names.index(field) + 1
        blocks = len(np.asarray(database.variables["eb_prop1"].data))
        connectivity = np.vstack(
            [
                np.asarray(database.variables[f"connect{block}"].data)[:, :3] - 1
                for block in range(1, blocks + 1)
            ]
        ).astype(int)
        values = np.concatenate(
            [
                np.asarray(
                    database.variables[f"vals_elem_var{field_index}eb{block}"].data[-1]
                )
                for block in range(1, blocks + 1)
            ]
        )
    return coordinates, connectivity, values


def _mechanical_figures(results: dict[str, dict[str, object]], output: Path) -> None:
    if set(results) != {"baseline", "improved"}:
        return
    figure, axis = plt.subplots(figsize=(7.1, 4.6))
    for family in ("baseline", "improved"):
        curve = results[family]["full_curve"]
        axis.plot(
            [row["strain"] for row in curve],
            [row["P11_MPa"] for row in curve],
            label=family,
            color=COLORS[family],
        )
    axis.set(
        xlabel=r"$\bar F_{11}-1$",
        ylabel=r"$\bar P_{11}$ (MPa)",
        title="Real MOOSE FCC-CP pilot",
    )
    axis.legend(frameon=False)
    _save(figure, output / "fig08_macroscopic_plastic_response.png")

    fields = [
        _exodus_field(Path(results[item]["exodus_path"]), "accumulated_slip")
        for item in ("baseline", "improved")
    ]
    lower = min(float(item[2].min()) for item in fields)
    upper = max(float(item[2].max()) for item in fields)
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    image = None
    for axis, family, data in zip(axes, ("baseline", "improved"), fields):
        triangulation = mtri.Triangulation(data[0][:, 0], data[0][:, 1], data[1])
        image = axis.tripcolor(
            triangulation,
            facecolors=data[2],
            shading="flat",
            cmap="magma",
            vmin=lower,
            vmax=upper,
        )
        axis.set(title=family, xlabel="x (µm)", ylabel="y (µm)", aspect="equal")
        axis.grid(False)
    figure.colorbar(image, ax=axes, label="accumulated slip at 2%")
    _save(figure, output / "fig09_local_plastic_slip.png")

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    labels = ("proof stress", "flow stress", "element-field Q95 slip")
    baseline = results["baseline"]
    improved = results["improved"]
    differences = (
        100.0
        * abs(improved["proof_stress_MPa"] - baseline["proof_stress_MPa"])
        / baseline["proof_stress_MPa"],
        100.0
        * abs(improved["flow_stress_2pct_MPa"] - baseline["flow_stress_2pct_MPa"])
        / baseline["flow_stress_2pct_MPa"],
        100.0
        * abs(
            improved["volume_weighted_q95_accumulated_slip"]
            - baseline["volume_weighted_q95_accumulated_slip"]
        )
        / baseline["volume_weighted_q95_accumulated_slip"],
    )
    axis.bar(labels, differences, color=("#0072B2", "#009E73", "#D55E00"))
    axis.set(
        ylabel="paired relative difference (%)",
        title="Mechanical sensitivity; uncertainty bounds not yet available",
    )
    _save(figure, output / "fig10_numerical_statistical_uncertainty.png")


def finalize() -> dict[str, object]:
    _style()
    V2_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    figures = V2_RESULTS_ROOT / "figures"
    summary = _metric_summary()
    summary.to_csv(
        V2_RESULTS_ROOT / "statistical_discrepancy_comparison.csv", index=False
    )
    histories = []
    optimized: dict[str, object] = {}
    validation: dict[str, object] = {}
    uncertainty: dict[str, object] = {}
    for dataset in DATASETS:
        history = _history(dataset).copy()
        history["dataset"] = dataset
        histories.append(history)
        directory = _calibration_directory(dataset)
        optimized_name = (
            "optimized_parameters_robust.json"
            if dataset == "Al1_pBD"
            else "optimized_parameters.json"
        )
        validation_name = (
            "independent_validation_robust.json"
            if dataset == "Al1_pBD"
            else "independent_validation.json"
        )
        optimized[dataset] = json.loads(
            (directory / optimized_name).read_text(encoding="utf-8")
        )
        validation[dataset] = json.loads(
            (directory / validation_name).read_text(encoding="utf-8")
        )
        fixed_confirmation = (
            V2_RESULTS_ROOT
            / "fixed_iteration_confirmation"
            / dataset
            / "fixed_iteration_confirmation.json"
        )
        if fixed_confirmation.is_file():
            validation[dataset]["fixed_iteration_confirmation"] = json.loads(
                fixed_confirmation.read_text(encoding="utf-8")
            )
            validation[dataset][
                "primary_public_validation"
            ] = "fixed_iteration_confirmation"
        uncertainty[dataset] = json.loads(
            (
                V2_RESULTS_ROOT
                / "uncertainty"
                / dataset
                / "experimental_subsampling_uncertainty.json"
            ).read_text(encoding="utf-8")
        )
    pd.concat(histories, ignore_index=True).to_csv(
        V2_RESULTS_ROOT / "calibration_history.csv", index=False
    )
    write_json(V2_RESULTS_ROOT / "optimized_parameters.json", optimized)
    write_json(V2_RESULTS_ROOT / "independent_validation.json", validation)
    write_json(
        V2_RESULTS_ROOT / "experimental_subsampling_uncertainty.json", uncertainty
    )
    _figure_1(summary, figures)
    _distribution_figure(
        "equivalent_diameter_um",
        "Equivalent diameter (µm)",
        "fig02_grain_size_distribution.png",
        figures,
    )
    _distribution_figure(
        "aspect_ratio_recomputed",
        "Aspect ratio",
        "fig03_aspect_ratio_distribution.png",
        figures,
    )
    _figure_4(figures)
    _figure_5(figures)
    _figure_6(summary, figures)
    _figure_7(figures)
    mechanical = _mechanical_results()
    _mechanical_figures(mechanical, figures)
    mechanical_summary = _mechanical_summary(mechanical)
    geometry_quality = {}
    for family in ("baseline", "improved"):
        path = (
            V2_RESULTS_ROOT
            / "mechanical"
            / "Al1_oBD"
            / family
            / "seed_10303"
            / "h_6_smooth_30_a_0p012"
            / "result.json"
        )
        if path.is_file():
            item = json.loads(path.read_text(encoding="utf-8"))
            mesh = item["mesh_metrics"]
            geometry_quality[family] = {
                "mesh_metrics": {
                    key: mesh[key]
                    for key in (
                        "node_count",
                        "element_count",
                        "quadrature_point_count_TRI6_default_fifth_order",
                        "domain_area_um2",
                        "minimum_triangle_quality",
                        "mean_triangle_quality",
                        "maximum_aspect_ratio",
                        "maximum_periodic_coordinate_error_um",
                        "checks",
                        "all_checks_passed",
                    )
                },
                "geometry_level_correction": item["geometry_level_correction"],
            }
    write_json(V2_RESULTS_ROOT / "geometry_quality_summary.json", geometry_quality)
    payload = {
        "datasets": DATASETS,
        "metric_summary": summary.to_dict(orient="records"),
        "mechanical_pilots": mechanical_summary,
        "figure_count": len(list(figures.glob("fig*.png"))),
        "raw_experimental_data_included": False,
    }
    write_json(V2_RESULTS_ROOT / "mechanical_pilot_summary.json", mechanical_summary)
    write_json(V2_RESULTS_ROOT / "finalization_summary.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(finalize(), indent=2))
