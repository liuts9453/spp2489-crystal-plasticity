#!/usr/bin/env python3
"""Generate the six requested figures from executed CP benchmark data only."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from scipy.io import netcdf_file


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/crystal_plasticity"
FIGURES = RESULTS / "figures"


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=230, bbox_inches="tight")
    plt.close(fig)


def physical_csv(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data[(data.time > 0) & (data.F11_bar > 0)].copy()


def decode_names(array) -> list[str]:
    return [b"".join(row).decode("utf-8").rstrip("\x00 ") for row in np.asarray(array)]


def exodus_elements(path: Path, fields: list[str]):
    polygons, output = [], {field: [] for field in fields}
    with netcdf_file(path, "r", mmap=False) as db:
        x = np.asarray(db.variables["coordx"].data).copy()
        y = np.asarray(db.variables["coordy"].data).copy()
        names = decode_names(db.variables["name_elem_var"].data)
        for block in range(1, len(np.asarray(db.variables["eb_prop1"].data)) + 1):
            connectivity = np.asarray(db.variables[f"connect{block}"].data).copy() - 1
            polygons.extend(np.column_stack((x[nodes], y[nodes])) for nodes in connectivity)
            for field in fields:
                index = names.index(field) + 1
                key = f"vals_elem_var{index}eb{block}"
                output[field].extend(np.asarray(db.variables[key].data[-1]).copy())
    return polygons, {key: np.asarray(value) for key, value in output.items()}


def largest_cp_case() -> tuple[int, Path]:
    for mesh in (64, 32, 16):
        path = RESULTS / f"analytic_cp_{mesh}_s40.e"
        if path.exists() and (RESULTS / f"analytic_cp_{mesh}_s40.csv").exists():
            data = pd.read_csv(RESULTS / f"analytic_cp_{mesh}_s40.csv")
            if np.isclose(data.time.iloc[-1], 1.0):
                return mesh, path
    raise FileNotFoundError("No completed analytic CP Exodus case")


def response_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    single = [
        (RESULTS / "substep_single_s20.csv", "single crystal (0°, 0°, 0°)"),
        (RESULTS / "analytic_single_o2_s20.csv", "single crystal (37°, 52°, 11°)"),
    ]
    for path, label in single:
        if path.exists():
            data = pd.read_csv(path)
            data = data[(data.time > 0) & (data.F11 > 0)]
            ax.plot(data.F11 - 1, data.P11, label=label)
    for mesh in (16, 32, 64):
        path = RESULTS / f"analytic_cp_{mesh}_s40.csv"
        if path.exists():
            data = physical_csv(path)
            if np.isclose(data.time.iloc[-1], 1.0):
                ax.plot(data.F11_bar - 1, data.P11_bar, label=f"20-grain CP {mesh}×{mesh}")
    hyper = RESULTS / "hyper_64_s40.csv"
    if hyper.exists():
        data = physical_csv(hyper)
        ax.plot(data.F11_bar - 1, data.P11_bar, "--", lw=2, label="anisotropic hyperelastic")
    ax.set(xlabel=r"$\bar F_{11}-1$", ylabel=r"$\bar P_{11}$ [MPa]", title="FCC crystal-plasticity response")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, "fig01_cp_stress_strain.png")


def slip_field_figure() -> None:
    mesh, path = largest_cp_case()
    polygons, fields = exodus_elements(path, ["grain_id", "accumulated_slip"])
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    collection = PolyCollection(polygons, array=fields["accumulated_slip"], cmap="magma", edgecolors="0.2", linewidths=0.03)
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set(xlabel=r"$X_1$ [$\mu$m]", ylabel=r"$X_2$ [$\mu$m]", title=f"Accumulated slip, {mesh}×{mesh} RVE")
    seeds = pd.read_csv(ROOT / "results/polycrystal/grain_data.csv")
    for row in seeds.itertuples(index=False):
        label = ax.text(
            row.seed_x_um,
            row.seed_y_um,
            str(int(row.grain_id)),
            color="white",
            fontsize=6.5,
            ha="center",
            va="center",
            weight="bold",
        )
        label.set_path_effects([path_effects.withStroke(linewidth=1.4, foreground="black")])
    bar = fig.colorbar(collection, ax=ax, pad=0.02)
    bar.set_label(r"$\sum_\alpha\int|\dot\gamma^\alpha|dt$")
    save(fig, "fig02_accumulated_slip.png")


def runtime_scaling_figure() -> None:
    data = pd.read_csv(RESULTS / "cp_runtime_summary.csv")
    # CP-1/2/3 are the nominal 20-increment performance matrix.  The 40-increment
    # 64x64 sensitivity case is deliberately excluded here rather than drawing two
    # different timings at the same element count.
    data = data[(data.completed == 1) & (data.requested_steps == 20)]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for model, label, marker in (("cp", "analytic-tangent CP", "o"), ("hyper", "hyperelastic", "s")):
        group = data[data.model == model].sort_values("mesh")
        if not group.empty:
            ax.plot(group.elements, group.wall_seconds, marker=marker, label=label)
    ax.set(xlabel="QUAD4 elements", ylabel="wall time [s]", title="Measured runtime scaling (4 MPI ranks)")
    ax.set_xscale("log", base=4)
    ax.set_yscale("log")
    ax.set_xticks([256, 1024, 4096], labels=["256", "1024", "4096"])
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    save(fig, "fig03_runtime_scaling.png")


def breakdown_figure() -> None:
    data = pd.read_csv(RESULTS / "cp_runtime_summary.csv")
    data = data[
        (data.model == "cp") & (data.completed == 1) & (data.requested_steps == 20)
    ].sort_values("mesh")
    material = data.cp_update_wall_equivalent_seconds.to_numpy()
    other = np.maximum(data.wall_seconds.to_numpy() - material, 0.0)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    labels = [f"{value}×{value}" for value in data.mesh]
    ax.bar(labels, material, label="local CP integration")
    ax.bar(labels, other, bottom=material, label="assembly, solves, I/O and other")
    ax.set(ylabel="wall-equivalent time [s]", title="Non-overlapping measured runtime breakdown")
    ax.legend(fontsize=8)
    ax.text(0.01, -0.18, "PerfGraph assembly regions include material calls; they are not double-counted as a separate bar.", transform=ax.transAxes, fontsize=8)
    save(fig, "fig04_runtime_breakdown.png")


def iterations_figure() -> None:
    steps = pd.read_csv(RESULTS / "cp_step_statistics.csv")
    summary = pd.read_csv(RESULTS / "cp_runtime_summary.csv")
    complete = set(summary[(summary.model == "cp") & (summary.completed == 1)].case)
    cases = [case for case in steps.case.unique() if case in complete]
    case = sorted(cases, key=lambda name: int(re.search(r"_(\d+)_", name).group(1)))[-1]
    data = steps[steps.case == case]
    fig, ax1 = plt.subplots(figsize=(7.3, 4.7))
    ax2 = ax1.twinx()
    line1 = ax1.plot(data.F11 - 1, data.newton_iterations, "o-", ms=3, label="global Newton iterations", color="C0")
    if data.step_wall_seconds.notna().any():
        line2 = ax2.plot(data.F11 - 1, data.step_wall_seconds, "s-", ms=3, label="step wall time", color="C1")
    else:
        line2 = []
    ax1.set(xlabel=r"$\bar F_{11}-1$", ylabel="Newton iterations", title=f"Load-step cost: {case}")
    ax2.set_ylabel("wall time [s]")
    ax1.grid(alpha=0.25)
    handles = line1 + line2
    ax1.legend(handles, [item.get_label() for item in handles], fontsize=8)
    save(fig, "fig05_newton_iterations.png")


def slip_activity_figure() -> None:
    mesh, path = largest_cp_case()
    fields = ["grain_id"] + [f"slip_rate_{index:02d}" for index in range(12)]
    _, values = exodus_elements(path, fields)
    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    x = np.arange(1, 13)
    for grain in (1, 10, 20):
        mask = np.rint(values["grain_id"]).astype(int) == grain
        activity = [np.mean(np.abs(values[f"slip_rate_{index:02d}"][mask])) for index in range(12)]
        ax.plot(x, activity, "o-", label=f"grain {grain}")
    ax.set(xlabel="FCC slip-system index", ylabel=r"mean $|\dot\gamma^\alpha|$ [s$^{-1}$]", title=f"Final slip activity ({mesh}×{mesh})")
    ax.set_xticks(x)
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, "fig06_slip_activity.png")


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    response_figure()
    slip_field_figure()
    runtime_scaling_figure()
    breakdown_figure()
    iterations_figure()
    slip_activity_figure()
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
