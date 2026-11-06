#!/usr/bin/env python3
"""Run deterministic DOE plus sequential 2D Bayesian optimization with real MOOSE FEM."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore", message="`torch.jit.script` is deprecated.*", category=FutureWarning
)

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
    DEFAULT_PROTOCOL,
    ETA_BOUNDS,
    INITIAL_DESIGNS,
    OPTIMIZATION_HISTORY,
    ROOT,
    SUMMARY,
    TARGET_PROPERTIES,
    THETA_BOUNDS_DEG,
    git_commit,
    write_json,
)
from cp_forward_solver import run_forward
from texture_parameterization import parameterization_validation


def objective_and_errors(
    proof: float, flow: float, target_proof: float, target_flow: float
) -> tuple[float, float, float]:
    error_proof = abs(proof - target_proof) / abs(target_proof)
    error_flow = abs(flow - target_flow) / abs(target_flow)
    objective = 0.5 * (error_proof * error_proof + error_flow * error_flow)
    return error_proof, error_flow, objective


def normalize_design(eta: float, theta_deg: float) -> tuple[float, float]:
    return (
        (eta - ETA_BOUNDS[0]) / (ETA_BOUNDS[1] - ETA_BOUNDS[0]),
        (theta_deg - THETA_BOUNDS_DEG[0]) / (THETA_BOUNDS_DEG[1] - THETA_BOUNDS_DEG[0]),
    )


def denormalize_design(point: np.ndarray | list[float]) -> tuple[float, float]:
    return (
        ETA_BOUNDS[0] + float(point[0]) * (ETA_BOUNDS[1] - ETA_BOUNDS[0]),
        THETA_BOUNDS_DEG[0]
        + float(point[1]) * (THETA_BOUNDS_DEG[1] - THETA_BOUNDS_DEG[0]),
    )


def fit_objective_gp(
    designs: list[tuple[float, float]], objectives: list[float]
) -> SingleTaskGP:
    if len(designs) != len(objectives) or len(designs) < 3:
        raise ValueError("2D GP training requires at least three matching observations")
    train_x = torch.tensor(
        [normalize_design(eta, theta) for eta, theta in designs], dtype=torch.double
    )
    train_y = -torch.tensor(objectives, dtype=torch.double).reshape(-1, 1)
    train_yvar = torch.full_like(train_y, 1.0e-12)
    with gpytorch.settings.min_fixed_noise(double_value=1.0e-12):
        model = SingleTaskGP(
            train_x,
            train_y,
            train_Yvar=train_yvar,
            outcome_transform=Standardize(m=1),
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def _normalized_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    aa = np.asarray(normalize_design(*a))
    bb = np.asarray(normalize_design(*b))
    return float(np.linalg.norm(aa - bb))


def propose_design(
    designs: list[tuple[float, float]],
    objectives: list[float],
    attempted: list[tuple[float, float]],
    seed: int,
) -> tuple[float, float, float, float, SingleTaskGP]:
    started = time.perf_counter()
    torch.manual_seed(seed)
    model = fit_objective_gp(designs, objectives)
    acquisition = LogExpectedImprovement(
        model=model, best_f=-min(objectives), maximize=True
    )
    candidate, score = optimize_acqf(
        acquisition,
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        q=1,
        num_restarts=24,
        raw_samples=512,
        options={"batch_limit": 8, "maxiter": 300},
    )
    eta, theta = denormalize_design(candidate.detach().cpu().numpy().reshape(-1))
    if min(_normalized_distance((eta, theta), old) for old in attempted) < 1.0e-5:
        eta_grid = np.linspace(ETA_BOUNDS[0], ETA_BOUNDS[1], 121)
        theta_grid = np.linspace(THETA_BOUNDS_DEG[0], THETA_BOUNDS_DEG[1], 181)
        grid = np.asarray(
            [
                (eta_value, theta_value)
                for eta_value in eta_grid
                for theta_value in theta_grid
            ]
        )
        allowed = np.asarray(
            [
                min(_normalized_distance(tuple(point), old) for old in attempted)
                >= 2.0e-3
                for point in grid
            ]
        )
        normalized = torch.tensor(
            [normalize_design(float(point[0]), float(point[1])) for point in grid],
            dtype=torch.double,
        ).reshape(-1, 1, 2)
        with torch.no_grad():
            values = acquisition(normalized).reshape(-1).cpu().numpy()
        values[~allowed] = -np.inf
        index = int(np.argmax(values))
        eta, theta = float(grid[index, 0]), float(grid[index, 1])
        score = torch.tensor(float(values[index]))
    return eta, theta, float(score.item()), time.perf_counter() - started, model


def _write_history(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    OPTIMIZATION_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with OPTIMIZATION_HISTORY.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_or_generate_target(reuse_cache: bool, mpi_ranks: int) -> dict[str, object]:
    valid = False
    if TARGET_PROPERTIES.exists():
        target = json.loads(TARGET_PROPERTIES.read_text(encoding="utf-8"))
        valid = (
            target.get("schema") == "spp2489_cp_inverse_design_2d_target_v1"
            and target.get("mpi_ranks") == mpi_ranks
            and target.get("converged") is True
            and target.get("application_commit") == git_commit(ROOT)
            and target.get("time_protocol", {}).get("name") == DEFAULT_PROTOCOL.name
        )
    if not valid:
        command = [
            sys.executable,
            str(Path(__file__).with_name("generate_hidden_target.py")),
            "--mpi-ranks",
            str(mpi_ranks),
        ]
        if not reuse_cache:
            command.append("--no-reuse")
        subprocess.run(command, cwd=ROOT, check=True)
        target = json.loads(TARGET_PROPERTIES.read_text(encoding="utf-8"))
    forbidden = {"eta", "theta", "orientation_path", "run_directory", "cache_key"}
    if forbidden.intersection(target):
        raise RuntimeError("Public target file leaks the hidden design")
    return target


def _history_row(
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
        "theta_deg": result.get("theta_deg", ""),
        "status": "converged" if result.get("converged") else "failed",
        "proof_stress_MPa": "",
        "flow_stress_2pct_MPa": "",
        "relative_error_proof": "",
        "relative_error_flow": "",
        "maximum_relative_error": "",
        "objective_J": "",
        "best_objective_J": "",
        "acquisition_log_ei": "" if acquisition is None else acquisition,
        "gp_fit_and_acquisition_time_s": gp_time,
        "initial_effective_stiffness_MPa": "",
        "P11_1pct_MPa": "",
        "secant_slope_1_to_2pct_MPa": "",
        "mean_accumulated_slip": "",
        "maximum_accumulated_slip": "",
        "mean_slip_resistance_MPa": "",
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
        e_proof, e_flow, objective = objective_and_errors(
            float(result["proof_stress_MPa"]),
            float(result["flow_stress_2pct_MPa"]),
            float(target["target_proof_stress_MPa"]),
            float(target["target_flow_stress_2pct_MPa"]),
        )
        row.update(
            {
                "proof_stress_MPa": result["proof_stress_MPa"],
                "flow_stress_2pct_MPa": result["flow_stress_2pct_MPa"],
                "relative_error_proof": e_proof,
                "relative_error_flow": e_flow,
                "maximum_relative_error": max(e_proof, e_flow),
                "objective_J": objective,
                "initial_effective_stiffness_MPa": result[
                    "initial_effective_stiffness_MPa"
                ],
                "P11_1pct_MPa": result["P11_1pct_MPa"],
                "secant_slope_1_to_2pct_MPa": result["secant_slope_1_to_2pct_MPa"],
                "mean_accumulated_slip": result["mean_accumulated_slip"],
                "maximum_accumulated_slip": result["maximum_accumulated_slip"],
                "mean_slip_resistance_MPa": result["mean_slip_resistance_MPa"],
            }
        )
    return row


def run_optimization(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.double)
    target = _load_or_generate_target(args.reuse_cache, args.mpi_ranks)
    texture_check = parameterization_validation()

    rows: list[dict[str, object]] = []
    attempted: list[tuple[float, float]] = []

    def evaluate(
        eta: float,
        theta: float,
        stage: str,
        acquisition: float | None = None,
        gp_time: float = 0.0,
    ) -> None:
        result = run_forward(
            eta,
            theta,
            mesh_name="coarse",
            protocol=DEFAULT_PROTOCOL,
            label=f"bo_{len(rows)+1:02d}_{stage}_eta_{eta:.8f}_theta_{theta:+.6f}",
            reuse_cache=args.reuse_cache,
            mpi_ranks=args.mpi_ranks,
        )
        attempted.append((eta, theta))
        rows.append(
            _history_row(len(rows) + 1, stage, result, target, acquisition, gp_time)
        )
        for index, row in enumerate(rows):
            observed = [
                float(previous["objective_J"])
                for previous in rows[: index + 1]
                if previous["objective_J"] != ""
            ]
            row["best_objective_J"] = min(observed) if observed else ""
        _write_history(rows)
        print(
            f"BO call {len(rows):02d}: eta={eta:.7f}, theta={theta:+.5f}, "
            f"status={rows[-1]['status']}, best_J={rows[-1]['best_objective_J']}",
            flush=True,
        )

    for eta, theta in INITIAL_DESIGNS:
        evaluate(eta, theta, "initial")

    def reached_tolerance() -> bool:
        return any(
            row["status"] == "converged"
            and float(row["relative_error_proof"]) < args.target_tolerance
            and float(row["relative_error_flow"]) < args.target_tolerance
            for row in rows
        )

    sequential = 0
    last_model: SingleTaskGP | None = None
    while sequential < args.max_bo_evals and not reached_tolerance():
        valid = [row for row in rows if row["status"] == "converged"]
        designs = [(float(row["eta"]), float(row["theta_deg"])) for row in valid]
        objectives = [float(row["objective_J"]) for row in valid]
        eta, theta, acquisition, gp_time, last_model = propose_design(
            designs, objectives, attempted, args.seed + sequential + 1
        )
        evaluate(eta, theta, "sequential", acquisition, gp_time)
        sequential += 1

    converged = [row for row in rows if row["status"] == "converged"]
    if not converged:
        raise RuntimeError("No converged MOOSE candidate is available")
    best = min(converged, key=lambda row: float(row["objective_J"]))
    first_tolerance_call = next(
        (
            int(row["call"])
            for row in rows
            if row["status"] == "converged"
            and float(row["relative_error_proof"]) < args.target_tolerance
            and float(row["relative_error_flow"]) < args.target_tolerance
        ),
        None,
    )
    summary = {
        "workflow": "SPP2489 2D texture inverse design on a Neper TRI6 periodic RVE",
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "botorch": botorch.__version__,
            "gpytorch": gpytorch.__version__,
        },
        "seed": args.seed,
        "mpi_ranks": args.mpi_ranks,
        "target_tolerance": args.target_tolerance,
        "target_properties": target,
        "texture_validation": texture_check,
        "bo_fem_calls": len(rows),
        "bo_initial_calls": len(INITIAL_DESIGNS),
        "bo_sequential_calls": sequential,
        "bo_first_tolerance_call": first_tolerance_call,
        "bo_failed_calls": sum(row["status"] != "converged" for row in rows),
        "bo_fem_wall_time_s": sum(
            float(row["wall_time_s"]) for row in rows if row["wall_time_s"] != ""
        ),
        "bo_gp_time_s": sum(
            float(row["gp_fit_and_acquisition_time_s"]) for row in rows
        ),
        "best_observed_design": best,
        "optimization_stopped_on_tolerance": reached_tolerance(),
        "workflow_wall_time_s": time.perf_counter() - started,
        "application_commit": git_commit(ROOT),
    }
    write_json(
        BEST_DESIGN,
        {
            "eta": float(best["eta"]),
            "theta_deg": float(best["theta_deg"]),
            "observed": best,
        },
    )
    write_json(SUMMARY, summary)
    if args.verify:
        from verify_high_fidelity import verify_design

        verification = verify_design(
            float(best["eta"]),
            float(best["theta_deg"]),
            tolerance=args.target_tolerance,
            mpi_ranks=args.mpi_ranks,
            run_fine=args.validate_fine,
            run_time_study=args.validate_time_integration,
        )
        summary["verification"] = verification
        summary["workflow_wall_time_s"] = time.perf_counter() - started
        write_json(SUMMARY, summary)
        if args.make_figures:
            from plot_optimization import make_all_figures

            make_all_figures()
        if args.write_report:
            from write_report import write_report

            write_report()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-bo-evals", type=int, default=20)
    parser.add_argument("--target-tolerance", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mpi-ranks", type=int, default=8)
    parser.add_argument(
        "--reuse-cache", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--validate-fine", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--validate-time-integration",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--make-figures", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--write-report", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    summary = run_optimization(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
