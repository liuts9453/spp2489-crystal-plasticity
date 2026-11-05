#!/usr/bin/env python3
"""Fixed-seed random-search baseline using the same real CP FEM interface."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from common import DEFAULT_PROTOCOL, RANDOM_HISTORY
from cp_forward_solver import run_forward


def errors_and_objective(result: dict[str, object], target: dict[str, object]) -> tuple[float, float, float]:
    p1 = float(result["P11_at_F11_1p01_MPa"])
    p2 = float(result["P11_at_F11_1p02_MPa"])
    t1 = float(target["target_P11_at_F11_1p01_MPa"])
    t2 = float(target["target_P11_at_F11_1p02_MPa"])
    signed1, signed2 = (p1 - t1) / t1, (p2 - t2) / t2
    return abs(signed1), abs(signed2), 0.5 * (signed1 * signed1 + signed2 * signed2)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_random_search(
    target: dict[str, object],
    *,
    budget: int,
    mesh: int,
    tolerance: float,
    seed: int,
    reuse_cache: bool,
    mpi_ranks: int = 4,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    best = float("inf")
    first_success: int | None = None
    for call, eta in enumerate(rng.uniform(0.0, 1.0, budget), start=1):
        result = run_forward(
            float(eta),
            mesh=mesh,
            protocol=DEFAULT_PROTOCOL,
            label=f"random_{call:02d}_eta_{eta:.8f}",
            reuse_cache=reuse_cache,
            mpi_ranks=mpi_ranks,
        )
        row: dict[str, object] = {
            "strategy": "random",
            "call": call,
            "eta": float(eta),
            "status": "converged" if result.get("converged") else "failed",
            "P11_1pct_MPa": "",
            "P11_2pct_MPa": "",
            "H_sec_MPa": "",
            "relative_error_1pct": "",
            "relative_error_2pct": "",
            "maximum_relative_error": "",
            "objective_J": "",
            "best_objective_J": best if np.isfinite(best) else "",
            "mean_accumulated_slip": "",
            "maximum_accumulated_slip": "",
            "mean_slip_resistance_MPa": "",
            "maximum_local_P11_MPa": "",
            "wall_time_s": result.get("wall_time_s", ""),
            "accepted_increments": result.get("accepted_increments", ""),
            "failed_attempts": result.get("failed_attempts", ""),
            "total_newton_iterations": result.get("total_newton_iterations", ""),
            "reused_cache": result.get("reused_cache", False),
            "run_directory": result.get("run_directory", ""),
            "csv_path": result.get("csv_path", ""),
            "exodus_path": result.get("exodus_path", ""),
            "failure_reason": result.get("failure_reason", ""),
        }
        if result.get("converged"):
            e1, e2, objective = errors_and_objective(result, target)
            best = min(best, objective)
            if e1 <= tolerance and e2 <= tolerance and first_success is None:
                first_success = call
            row.update(
                {
                    "P11_1pct_MPa": result["P11_at_F11_1p01_MPa"],
                    "P11_2pct_MPa": result["P11_at_F11_1p02_MPa"],
                    "H_sec_MPa": result["H_sec_MPa"],
                    "relative_error_1pct": e1,
                    "relative_error_2pct": e2,
                    "maximum_relative_error": max(e1, e2),
                    "objective_J": objective,
                    "best_objective_J": best,
                    "mean_accumulated_slip": result["mean_accumulated_slip"],
                    "maximum_accumulated_slip": result["maximum_accumulated_slip"],
                    "mean_slip_resistance_MPa": result["mean_slip_resistance_MPa"],
                    "maximum_local_P11_MPa": result["maximum_local_P11_MPa"],
                }
            )
        rows.append(row)
        _write(RANDOM_HISTORY, rows)
        if first_success is not None:
            break
    return rows
