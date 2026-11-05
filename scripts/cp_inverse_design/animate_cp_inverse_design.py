#!/usr/bin/env python3
"""Animate the discrete history of real BO FEM evaluations (not morphology evolution)."""

from __future__ import annotations

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from common import ANIMATIONS, OPTIMIZATION_HISTORY, SUMMARY
from plot_cp_inverse_design import _grain_map, _orientation_colors


def make_animation(summary: dict[str, object]) -> str:
    with OPTIMIZATION_HISTORY.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["status"] == "converged"]
    if not rows:
        raise RuntimeError("No converged BO FEM evaluations are available for animation")
    ANIMATIONS.mkdir(parents=True, exist_ok=True)
    output = ANIMATIONS / "cp_inverse_design_history.gif"
    target = summary["target_properties"]
    t1, t2 = float(target["target_P11_at_F11_1p01_MPa"]), float(target["target_P11_at_F11_1p02_MPa"])
    grain_map = _grain_map(260)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), constrained_layout=True)

    def update(frame: int):
        for axis in axes:
            axis.clear()
        row = rows[frame]
        eta = float(row["eta"])
        axes[0].imshow(_orientation_colors(eta)[grain_map], origin="lower", extent=(0, 200, 0, 200), interpolation="nearest")
        axes[0].set(title=f"Real FEM #{row['call']}: η={eta:.5f}", xlabel="x (µm)", ylabel="y (µm)", aspect="equal")
        x = np.arange(2)
        axes[1].bar(x - 0.18, [t1, t2], 0.36, label="Target")
        axes[1].bar(x + 0.18, [float(row["P11_1pct_MPa"]), float(row["P11_2pct_MPa"])], 0.36, label="Candidate")
        axes[1].set_xticks(x, ["F11=1.01", "F11=1.02"])
        axes[1].set(ylabel="PK1 P11 (MPa)", title="Dual-property match")
        axes[1].legend(fontsize=8)
        previous = rows[: frame + 1]
        calls = [int(item["call"]) for item in previous]
        errors = [max(float(item["relative_error_1pct"]), float(item["relative_error_2pct"])) for item in previous]
        axes[2].semilogy(calls, np.minimum.accumulate(errors), "o-", color="#4c78a8")
        axes[2].axhline(float(summary["target_tolerance"]), color="black", ls="--")
        axes[2].set(xlim=(0.5, max(len(rows) + 0.5, 2)), ylim=(max(min(errors) * 0.4, 1e-6), max(errors) * 2.0),
                    xlabel="Real FEM evaluation", ylabel="Best max(e1,e2)", title="Best target error")
        fig.suptitle("Discrete Bayesian-optimization history; frames are executed FEM candidates")
        return axes

    animation = FuncAnimation(fig, update, frames=len(rows), interval=900, blit=False, repeat=True)
    animation.save(output, writer=PillowWriter(fps=1.1))
    plt.close(fig)
    return str(output)


def main() -> int:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    print(make_animation(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
