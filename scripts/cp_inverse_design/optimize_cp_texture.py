#!/usr/bin/env python3
"""Run the complete real-MOOSE FCC crystal-plasticity inverse-design loop."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="`torch.jit.script` is deprecated.*", category=FutureWarning)

import botorch
import gpytorch
import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from common import (
    BEST_DESIGN,
    BO,
    DEFAULT_PROTOCOL,
    INITIAL_ETA,
    OPTIMIZATION_HISTORY,
    RESULTS,
    ROOT,
    SUMMARY,
    TARGET_PROPERTIES,
    git_commit,
    write_json,
)
from cp_forward_solver import run_forward
from random_search_baseline import errors_and_objective, run_random_search
from texture_generator import endpoint_validation, rotation_convention_validation


def objective_from_properties(p1: float, p2: float, target1: float, target2: float) -> float:
    """Dimensionless dual-stress objective from the task definition."""
    return 0.5 * (((p1 - target1) / target1) ** 2 + ((p2 - target2) / target2) ** 2)


def early_stop(e1: float, e2: float, tolerance: float) -> bool:
    return bool(e1 <= tolerance and e2 <= tolerance)


def valid_training_data(rows: list[dict[str, object]]) -> tuple[list[float], list[float]]:
    """Exclude failed FEM evaluations and any row without a physical objective."""
    valid = [row for row in rows if row.get("status") == "converged" and row.get("objective_J") != ""]
    return [float(row["eta"]) for row in valid], [float(row["objective_J"]) for row in valid]


def fit_objective_gp(etas: list[float], objectives: list[float]) -> SingleTaskGP:
    if len(etas) != len(objectives) or len(etas) < 2:
        raise ValueError("GP training requires matching eta/objective arrays with at least two rows")
    train_x = torch.tensor(etas, dtype=torch.double).reshape(-1, 1)
    train_y = -torch.tensor(objectives, dtype=torch.double).reshape(-1, 1)
    train_yvar = torch.full_like(train_y, 1.0e-12)
    with gpytorch.settings.min_fixed_noise(double_value=1.0e-12):
        model = SingleTaskGP(
            train_x,
            train_y,
            train_Yvar=train_yvar,
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
    return model


def propose_eta(
    etas: list[float], objectives: list[float], attempted: list[float], seed: int
) -> tuple[float, float, float]:
    """Return a nonduplicate candidate using only observed eta and objective values."""
    start = time.perf_counter()
    torch.manual_seed(seed)
    model = fit_objective_gp(etas, objectives)
    acquisition = LogExpectedImprovement(model=model, best_f=-min(objectives), maximize=True)
    candidate, score = optimize_acqf(
        acquisition,
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        q=1,
        num_restarts=16,
        raw_samples=256,
        options={"batch_limit": 8, "maxiter": 200},
    )
    eta = float(candidate.item())
    if min(abs(eta - old) for old in attempted) < 1.0e-5:
        grid = np.linspace(0.0, 1.0, 5001)
        allowed = np.asarray([min(abs(value - old) for old in attempted) >= 1.0e-4 for value in grid])
        if not allowed.any():
            raise RuntimeError("No numerically distinct BO point remains")
        points = torch.tensor(grid, dtype=torch.double).reshape(-1, 1, 1)
        with torch.no_grad():
            values = acquisition(points).reshape(-1).cpu().numpy()
        values[~allowed] = -np.inf
        index = int(np.argmax(values))
        eta, score = float(grid[index]), torch.tensor(values[index])
    return eta, float(score.item()), time.perf_counter() - start


def _write_history(rows: list[dict[str, object]]) -> None:
    OPTIMIZATION_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with OPTIMIZATION_HISTORY.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _history_row(
    *,
    call: int,
    stage: str,
    result: dict[str, object],
    target: dict[str, object],
    acquisition: float | None,
    gp_time: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "strategy": "BO",
        "stage": stage,
        "call": call,
        "eta": result.get("eta", ""),
        "status": "converged" if result.get("converged") else "failed",
        "P11_1pct_MPa": "",
        "P11_2pct_MPa": "",
        "H_sec_MPa": "",
        "relative_error_1pct": "",
        "relative_error_2pct": "",
        "maximum_relative_error": "",
        "objective_J": "",
        "best_objective_J": "",
        "acquisition_log_ei": "" if acquisition is None else acquisition,
        "gp_fit_and_acquisition_time_s": gp_time,
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
        row.update(
            {
                "P11_1pct_MPa": result["P11_at_F11_1p01_MPa"],
                "P11_2pct_MPa": result["P11_at_F11_1p02_MPa"],
                "H_sec_MPa": result["H_sec_MPa"],
                "relative_error_1pct": e1,
                "relative_error_2pct": e2,
                "maximum_relative_error": max(e1, e2),
                "objective_J": objective,
                "mean_accumulated_slip": result["mean_accumulated_slip"],
                "maximum_accumulated_slip": result["maximum_accumulated_slip"],
                "mean_slip_resistance_MPa": result["mean_slip_resistance_MPa"],
                "maximum_local_P11_MPa": result["maximum_local_P11_MPa"],
            }
        )
    return row


def _load_or_generate_target(mesh: int, reuse_cache: bool, mpi_ranks: int) -> dict[str, object]:
    valid = False
    if TARGET_PROPERTIES.exists():
        target = json.loads(TARGET_PROPERTIES.read_text(encoding="utf-8"))
        valid = (
            target.get("mesh") == mesh
            and target.get("mpi_ranks") == mpi_ranks
            and target.get("converged") is True
            and target.get("application_commit") == git_commit(ROOT)
            and target.get("time_protocol", {}).get("name") == DEFAULT_PROTOCOL.name
        )
    if not valid:
        command = [sys.executable, str(Path(__file__).with_name("generate_hidden_target.py")), "--mesh", str(mesh), "--mpi-ranks", str(mpi_ranks)]
        if not reuse_cache:
            command.append("--no-reuse")
        subprocess.run(command, cwd=ROOT, check=True)
        target = json.loads(TARGET_PROPERTIES.read_text(encoding="utf-8"))
    # Deliberately return only the public property file. This module never imports the hidden eta.
    forbidden = {"eta", "eta_reference", "orientation_path", "run_directory", "cache_key"}
    if forbidden.intersection(target):
        raise RuntimeError("Public target file leaks hidden design information")
    return target


def _ensure_time_study(mesh: int, reuse_cache: bool) -> dict[str, object]:
    path = RESULTS / "time_integration_study.json"
    shared_path = ROOT / "results" / "cp_inverse_design" / "time_integration_study.json"
    source = path if path.exists() else shared_path
    if source.exists():
        study = json.loads(source.read_text(encoding="utf-8"))
        if study.get("mesh") == mesh and study.get("selected_protocol") == DEFAULT_PROTOCOL.name:
            study.setdefault("mpi_ranks", 4)  # The original benchmark protocol used four ranks.
            return study
    command = [sys.executable, str(Path(__file__).with_name("time_integration_study.py")), "--mesh", str(mesh)]
    if not reuse_cache:
        command.append("--no-reuse")
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(path.read_text(encoding="utf-8"))


def run_optimization(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.double)
    BO.mkdir(parents=True, exist_ok=True)

    if not 1 <= args.initial_samples <= len(INITIAL_ETA):
        raise ValueError(f"initial_samples must be in [1,{len(INITIAL_ETA)}]")
    texture_checks = {
        "endpoints": endpoint_validation(),
        "rotation_convention": rotation_convention_validation(),
    }
    time_study = _ensure_time_study(args.mesh, args.reuse_cache)
    target = _load_or_generate_target(args.mesh, args.reuse_cache, args.mpi_ranks)

    rows: list[dict[str, object]] = []
    attempted: list[float] = []

    def evaluate(eta: float, stage: str, acquisition: float | None = None, gp_time: float = 0.0) -> None:
        call = len(rows) + 1
        result = run_forward(
            eta,
            mesh=args.mesh,
            protocol=DEFAULT_PROTOCOL,
            label=f"bo_{call:02d}_{stage}_eta_{eta:.10f}",
            reuse_cache=args.reuse_cache,
            mpi_ranks=args.mpi_ranks,
        )
        attempted.append(eta)
        row = _history_row(
            call=call,
            stage=stage,
            result=result,
            target=target,
            acquisition=acquisition,
            gp_time=gp_time,
        )
        rows.append(row)
        running = [float(item["objective_J"]) for item in rows if item["objective_J"] != ""]
        best = min(running) if running else float("nan")
        for item in rows:
            item["best_objective_J"] = min(
                [float(previous["objective_J"]) for previous in rows[: int(item["call"])] if previous["objective_J"] != ""],
                default="",
            )
        _write_history(rows)
        print(f"BO call {call}: eta={eta:.9f}, status={row['status']}, best J={best:.6e}", flush=True)

    # All deterministic initial points are completed before tolerance-driven sequential BO.
    for eta in INITIAL_ETA[: args.initial_samples]:
        evaluate(float(eta), "initial")

    def tolerance_reached() -> bool:
        return any(
            row["status"] == "converged"
            and early_stop(
                float(row["relative_error_1pct"]), float(row["relative_error_2pct"]), args.target_tolerance
            )
            for row in rows
        )

    sequential = 0
    while sequential < args.max_bo_evals and not tolerance_reached():
        valid_eta, valid_objective = valid_training_data(rows)
        if len(valid_eta) < 2:
            raise RuntimeError("Too few converged FEM evaluations to fit the GP")
        eta, acquisition, gp_time = propose_eta(
            valid_eta, valid_objective, attempted, args.seed + sequential + 1
        )
        evaluate(eta, "sequential", acquisition, gp_time)
        sequential += 1

    converged_rows = [row for row in rows if row["status"] == "converged"]
    if not converged_rows:
        raise RuntimeError("No converged FEM result is available")
    best = min(converged_rows, key=lambda row: float(row["objective_J"]))
    best_eta = float(best["eta"])
    first_success = next(
        (
            int(row["call"])
            for row in rows
            if row["status"] == "converged"
            and early_stop(
                float(row["relative_error_1pct"]), float(row["relative_error_2pct"]), args.target_tolerance
            )
        ),
        None,
    )

    random_rows = run_random_search(
        target,
        budget=len(rows),
        mesh=args.mesh,
        tolerance=args.target_tolerance,
        seed=args.seed + 100003,
        reuse_cache=args.reuse_cache,
        mpi_ranks=args.mpi_ranks,
    )

    # Verification is intentionally a separate post-optimization stage.
    from verify_cp_inverse_design import verify_design

    verification = verify_design(
        best_eta,
        target,
        mesh=args.mesh,
        tolerance=args.target_tolerance,
        high_fidelity=args.validate_high_fidelity,
        mpi_ranks=args.mpi_ranks,
    )
    summary: dict[str, object] = {
        "workflow": "SPP2489 finite-strain FCC crystal-plasticity texture inverse design",
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "botorch": botorch.__version__,
            "gpytorch": gpytorch.__version__,
        },
        "seed": args.seed,
        "mesh": args.mesh,
        "mpi_ranks": args.mpi_ranks,
        "target_tolerance": args.target_tolerance,
        "target_properties": target,
        "texture_validation": texture_checks,
        "time_integration_study": time_study,
        "bo_fem_calls": len(rows),
        "bo_sequential_calls": sequential,
        "bo_first_tolerance_call": first_success,
        "bo_failed_calls": sum(row["status"] != "converged" for row in rows),
        "bo_fem_wall_time_s": sum(float(row["wall_time_s"]) for row in rows if row["wall_time_s"] != ""),
        "bo_gp_time_s": sum(float(row["gp_fit_and_acquisition_time_s"]) for row in rows),
        "best_observed_design": best,
        "random_search_calls": len(random_rows),
        "random_search_first_tolerance_call": next(
            (
                int(row["call"])
                for row in random_rows
                if row["status"] == "converged"
                and early_stop(
                    float(row["relative_error_1pct"]),
                    float(row["relative_error_2pct"]),
                    args.target_tolerance,
                )
            ),
            None,
        ),
        "random_search_fem_wall_time_s": sum(float(row["wall_time_s"]) for row in random_rows),
        "verification": verification,
        "workflow_wall_time_s": time.perf_counter() - started,
        "application_commit": git_commit(ROOT),
    }
    write_json(BEST_DESIGN, {"eta": best_eta, "observed": best, "verification": verification})
    write_json(SUMMARY, summary)

    from plot_cp_inverse_design import make_all_figures

    make_all_figures(summary)
    from animate_cp_inverse_design import make_animation

    try:
        summary["animation_path"] = make_animation(summary)
    except Exception as error:  # The requested GIF is lower priority than numerical validation.
        summary["animation_error"] = str(error)
    from write_cp_inverse_design_report import write_report

    write_report(summary)
    summary["workflow_wall_time_s"] = time.perf_counter() - started
    write_json(SUMMARY, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, default=32)
    parser.add_argument("--max-bo-evals", type=int, default=12)
    parser.add_argument("--initial-samples", type=int, default=6)
    parser.add_argument("--target-tolerance", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-high-fidelity", action="store_true")
    parser.add_argument("--mpi-ranks", type=int, default=8)
    args = parser.parse_args()
    if args.mesh <= 0 or args.max_bo_evals < 0 or args.target_tolerance <= 0.0:
        parser.error("mesh and tolerance must be positive; max-bo-evals must be nonnegative")
    summary = run_optimization(args)
    print("\nCP INVERSE DESIGN COMPLETE")
    print(f"best eta = {summary['best_observed_design']['eta']:.10f}")
    verified = summary["verification"]["independent_low_fidelity"]
    print(
        f"independent errors = {100*verified['relative_error_1pct']:.6f}% / "
        f"{100*verified['relative_error_2pct']:.6f}%"
    )
    print(f"summary = {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
