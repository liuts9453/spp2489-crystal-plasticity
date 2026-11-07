#!/usr/bin/env python3
"""Create the ten requested experimental--Neper statistical figures from real data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection

from extract_texture_statistics import (
    ipf_coordinates,
    pole_figure_coordinates,
    rotations_from_frame,
)
from neper_statistics import read_neper_faces


DATASET_COLORS = {
    "Al1_oBD": "#0072B2",
    "Al1_pBD": "#D55E00",
    "Al0_00": "#009E73",
    "Al0_90": "#CC79A7",
}


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


def plot_experimental_size(frames: dict[str, pd.DataFrame], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    for key, frame in frames.items():
        values = frame["equivalent_diameter_um"].to_numpy(dtype=float)
        area_weights = np.array(frame["grain_area_um2"], dtype=float, copy=True)
        area_weights /= np.sum(area_weights)
        linear_bins = np.histogram_bin_edges(values, bins="fd")
        color = DATASET_COLORS[key]
        axes[0, 0].hist(
            values,
            bins=linear_bins,
            density=True,
            histtype="step",
            lw=1.4,
            label=key,
            color=color,
        )
        axes[0, 1].hist(
            values,
            bins=linear_bins,
            weights=area_weights,
            density=True,
            histtype="step",
            lw=1.4,
            label=key,
            color=color,
        )
        ordered = np.sort(values)
        order = np.argsort(values)
        axes[1, 0].plot(
            ordered,
            np.arange(1, len(values) + 1) / len(values),
            label=f"{key} number",
            color=color,
        )
        axes[1, 0].plot(
            ordered,
            np.cumsum(area_weights[order]),
            linestyle="--",
            label=f"{key} area",
            color=color,
        )
        axes[1, 1].hist(
            values,
            bins=np.geomspace(np.min(values), np.max(values), 24),
            density=True,
            histtype="step",
            lw=1.4,
            label=key,
            color=color,
        )
    axes[0, 0].set(
        xlabel="Equivalent grain diameter (μm)",
        ylabel="Number-weighted density",
    )
    axes[0, 1].set(
        xlabel="Equivalent grain diameter (μm)",
        ylabel="Area-weighted density",
    )
    axes[1, 0].set(xlabel="Equivalent grain diameter (μm)", ylabel="Empirical CDF")
    axes[1, 1].set(
        xlabel="Equivalent grain diameter (μm)",
        ylabel="Probability density",
        xscale="log",
        yscale="log",
    )
    axes[0, 0].legend(frameon=False)
    axes[1, 0].legend(frameon=False, ncol=2, fontsize=7)
    figure.suptitle("Experimental grain-size distributions: datasets remain separate")
    _save(figure, output / "fig01_experimental_grain_size_statistics.png")


def plot_experimental_aspect(frames: dict[str, pd.DataFrame], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    for key, frame in frames.items():
        values = frame["aspect_ratio_recomputed"].to_numpy(dtype=float)
        color = DATASET_COLORS[key]
        axes[0].hist(
            values,
            bins="fd",
            density=True,
            histtype="step",
            lw=1.4,
            label=key,
            color=color,
        )
        alpha = np.radians(
            2.0 * np.mod(frame["alpha_deg"].to_numpy(dtype=float), 180.0)
        )
        bins = np.linspace(0.0, 2.0 * np.pi, 37)
        histogram = np.histogram(alpha, bins=bins, density=True)[0]
        centers = 0.25 * np.degrees(bins[:-1] + bins[1:])
        axes[1].plot(centers, histogram, color=color, label=key)
    axes[0].set(xlabel="Principal aspect ratio (≥1)", ylabel="Probability density")
    axes[1].set(
        xlabel="Principal-axis direction (deg, modulo 180°)", ylabel="Axial density"
    )
    axes[0].legend(frameon=False)
    figure.suptitle("Experimental aspect-ratio and morphology-direction statistics")
    _save(figure, output / "fig02_experimental_aspect_ratio_statistics.png")


def plot_size_aspect_correlation(frames: dict[str, pd.DataFrame], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.8, 9.0), sharex=False, sharey=False)
    for axis, (key, frame) in zip(axes.flat, frames.items()):
        axis.hexbin(
            frame["equivalent_diameter_um"],
            frame["aspect_ratio_recomputed"],
            gridsize=35,
            mincnt=1,
            bins="log",
            cmap="viridis",
        )
        rho = (
            frame[["equivalent_diameter_um", "aspect_ratio_recomputed"]]
            .corr(method="spearman")
            .iloc[0, 1]
        )
        bins = pd.qcut(frame["equivalent_diameter_um"], q=5, duplicates="drop")
        grouped = frame.groupby(bins, observed=True)
        x_values = grouped["equivalent_diameter_um"].median().to_numpy()
        medians = grouped["aspect_ratio_recomputed"].median().to_numpy()
        low = grouped["aspect_ratio_recomputed"].quantile(0.25).to_numpy()
        high = grouped["aspect_ratio_recomputed"].quantile(0.75).to_numpy()
        axis.errorbar(
            x_values,
            medians,
            yerr=np.vstack((medians - low, high - medians)),
            color="white",
            marker="o",
            markeredgecolor="0.1",
            lw=1.2,
            capsize=2,
            label="size-bin median ± IQR",
        )
        axis.set_title(f"{key} · Spearman ρ={rho:.3f}")
        axis.set(xlabel="Equivalent diameter (μm)", ylabel="Aspect ratio")
        axis.legend(frameon=False, fontsize=7)
    figure.suptitle("Experimental grain size–aspect ratio joint statistics")
    figure.subplots_adjust(hspace=0.42, wspace=0.24, top=0.92, bottom=0.08)
    _save(figure, output / "fig03_size_aspect_ratio_correlation.png")


def _plot_ipf_density(axis: plt.Axes, frame: pd.DataFrame, title: str) -> None:
    coordinates = ipf_coordinates(
        rotations_from_frame(frame), np.array([1.0, 0.0, 0.0])
    )
    axis.hexbin(
        coordinates[:, 0], coordinates[:, 1], gridsize=24, mincnt=1, cmap="magma"
    )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="u/w", ylabel="v/w", title=title)
    axis.plot([0, 1], [0, 1], color="0.35", lw=0.7)


def plot_experimental_texture(frames: dict[str, pd.DataFrame], output: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(13.5, 7.6))
    for column, (key, frame) in enumerate(frames.items()):
        rotations = rotations_from_frame(frame)
        _plot_ipf_density(axes[0, column], frame, f"{key}\nIPF-X")
        poles = pole_figure_coordinates(rotations, "111").reshape(-1, 2)
        axes[1, column].hexbin(
            poles[:, 0],
            poles[:, 1],
            gridsize=24,
            mincnt=1,
            cmap="magma",
        )
        axes[1, column].add_patch(
            plt.Circle((0.0, 0.0), 1.0, fill=False, color="0.35", lw=0.7)
        )
        axes[1, column].set(
            xlim=(-1, 1),
            ylim=(-1, 1),
            aspect="equal",
            xlabel="stereographic x",
            ylabel="stereographic y",
            title="{111} pole figure",
        )
    figure.suptitle(
        "Experimental cubic IPF-X and {111} pole density (MOOSE rotation convention)"
    )
    figure.subplots_adjust(hspace=0.40, wspace=0.30, top=0.89, bottom=0.08)
    _save(figure, output / "fig04_experimental_texture.png")


def _distribution_comparison(
    pairs: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    column: str,
    xlabel: str,
    title: str,
    filename: str,
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        len(pairs), 2, figsize=(10.8, 3.45 * len(pairs)), squeeze=False
    )
    for row, (key, (experimental, synthetic)) in enumerate(pairs.items()):
        combined = np.concatenate(
            [experimental[column].to_numpy(), synthetic[column].to_numpy()]
        )
        bins = np.histogram_bin_edges(combined, bins="fd")
        for frame, label, style in (
            (experimental, "Experimental", "-"),
            (synthetic, "Neper ensemble", "--"),
        ):
            values = np.sort(frame[column].to_numpy(dtype=float))
            axes[row, 0].hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                lw=1.35,
                linestyle=style,
                label=label,
            )
            axes[row, 1].plot(
                values,
                np.arange(1, len(values) + 1) / len(values),
                style,
                lw=1.35,
                label=label,
            )
        axes[row, 0].set(title=key, xlabel=xlabel, ylabel="Probability density")
        axes[row, 1].set(title=key, xlabel=xlabel, ylabel="Empirical CDF")
        axes[row, 0].legend(frameon=False)
        axes[row, 1].legend(frameon=False)
    figure.suptitle(title)
    figure.subplots_adjust(hspace=0.72, wspace=0.24, top=0.94, bottom=0.06)
    _save(figure, output / filename)


def plot_experimental_neper_comparisons(
    pairs: dict[str, tuple[pd.DataFrame, pd.DataFrame]], output: Path
) -> None:
    _distribution_comparison(
        pairs,
        "equivalent_diameter_um",
        "Equivalent diameter (μm)",
        "Experimental versus Neper grain size",
        "fig05_experimental_vs_neper_grain_size.png",
        output,
    )
    _distribution_comparison(
        pairs,
        "aspect_ratio_recomputed",
        "Principal aspect ratio",
        "Experimental versus Neper aspect ratio",
        "fig06_experimental_vs_neper_aspect_ratio.png",
        output,
    )
    figure, axes = plt.subplots(2, 4, figsize=(13.0, 6.4), sharex=True, sharey=True)
    for column, (key, (experimental, synthetic)) in enumerate(pairs.items()):
        _plot_ipf_density(axes[0, column], experimental, f"{key}\nExperimental")
        _plot_ipf_density(axes[1, column], synthetic, "Synthetic")
    figure.suptitle("Experimental versus synthetic cubic IPF-X texture")
    _save(figure, output / "fig07_experimental_vs_synthetic_texture.png")


def plot_representative_neper(
    tessellations: dict[str, tuple[Path, float]], output: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 9.2))
    for axis, (key, (path, side)) in zip(axes.flat, tessellations.items()):
        polygons = []
        for _, polygon in read_neper_faces(path):
            base = np.asarray(polygon)
            for x_shift in (-side, 0.0, side):
                for y_shift in (-side, 0.0, side):
                    polygons.append(base + np.array([x_shift, y_shift]))
        axis.add_collection(
            PolyCollection(
                polygons, facecolors="none", edgecolors="0.15", linewidths=0.55
            )
        )
        axis.set(
            xlim=(0, side),
            ylim=(0, side),
            aspect="equal",
            title=key,
            xlabel="x (μm)",
            ylabel="y (μm)",
        )
    figure.suptitle("Representative periodic Neper RVEs")
    figure.subplots_adjust(hspace=0.36, wspace=0.28, top=0.93, bottom=0.07)
    _save(figure, output / "fig08_representative_neper_microstructures.png")


def plot_calibration_convergence(
    histories: dict[str, pd.DataFrame], output: Path
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    for key, history in histories.items():
        best = np.minimum.accumulate(history["objective_mean"].to_numpy(dtype=float))
        axis.plot(history["evaluation"], best, marker="o", ms=3, label=key)
    axis.set(
        xlabel="Neper parameter evaluations",
        ylabel="Best statistical discrepancy",
        yscale="log",
    )
    axis.legend(frameon=False, ncol=2)
    axis.set_title("Deterministic Neper calibration convergence")
    _save(figure, output / "fig09_calibration_convergence.png")


def plot_representativeness(tables: dict[str, pd.DataFrame], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    for key, table in tables.items():
        grouped = table.groupby("side_length_um")["objective"]
        mean, std = grouped.mean(), grouped.std(ddof=0)
        axis.errorbar(mean.index, mean, yerr=std, marker="o", capsize=3, label=key)
    axis.set(
        xlabel="Periodic RVE side length (μm)",
        ylabel="Statistical discrepancy (mean ± realization SD)",
    )
    axis.legend(frameon=False, ncol=2)
    axis.set_title("RVE statistical representativeness")
    _save(figure, output / "fig10_rve_statistical_representativeness.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest", type=Path, help="JSON manifest produced by run_calibration.py"
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = Path(manifest["figures_directory"])
    frames = {
        key: pd.read_csv(value["cleaned_csv"])
        for key, value in manifest["datasets"].items()
    }
    _style()
    plot_experimental_size(frames, output)
    plot_experimental_aspect(frames, output)
    plot_size_aspect_correlation(frames, output)
    plot_experimental_texture(frames, output)
    if manifest.get("calibration_complete"):
        pairs = {
            key: (frames[key], pd.read_csv(value["validation_ensemble_csv"]))
            for key, value in manifest["datasets"].items()
        }
        plot_experimental_neper_comparisons(pairs, output)
        plot_representative_neper(
            {
                key: (
                    Path(value["representative_tessellation"]),
                    float(value["side_length_um"]),
                )
                for key, value in manifest["datasets"].items()
            },
            output,
        )
        plot_calibration_convergence(
            {
                key: pd.read_csv(value["calibration_history_csv"])
                for key, value in manifest["datasets"].items()
            },
            output,
        )
        plot_representativeness(
            {
                key: pd.read_csv(value["representativeness_csv"])
                for key, value in manifest["datasets"].items()
            },
            output,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
