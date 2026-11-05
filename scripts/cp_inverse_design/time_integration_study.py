#!/usr/bin/env python3
"""Compare current adaptive, strict adaptive, and fixed CP time integration."""

from __future__ import annotations

import argparse
import csv
import json

from common import DEFAULT_PROTOCOL, PROTOCOLS, RESULTS, VERIFICATION, ensure_directories, write_json
from cp_forward_solver import run_forward


STUDY_PATH = RESULTS / "time_integration_study.json"
STUDY_CSV = RESULTS / "time_integration_study.csv"


def run_study(mesh: int = 32, reuse_cache: bool = True, mpi_ranks: int = 4) -> dict[str, object]:
    ensure_directories()
    records: list[dict[str, object]] = []
    results: dict[tuple[float, str], dict[str, object]] = {}
    for eta in (0.0, 0.5, 1.0):
        for name in ("adaptive_current", "adaptive_strict", "fixed_200"):
            result = run_forward(
                eta,
                mesh=mesh,
                protocol=PROTOCOLS[name],
                label=f"time_study_eta_{eta:.2f}_{name}",
                run_root=VERIFICATION / "time_integration",
                reuse_cache=reuse_cache,
                mpi_ranks=mpi_ranks,
            )
            if not result.get("converged"):
                raise RuntimeError(f"Time integration study failed for eta={eta}, protocol={name}")
            results[(eta, name)] = result
            records.append(
                {
                    "eta": eta,
                    "protocol": name,
                    "mpi_ranks": mpi_ranks,
                    "P11_1pct_MPa": result["P11_at_F11_1p01_MPa"],
                    "P11_2pct_MPa": result["P11_at_F11_1p02_MPa"],
                    "wall_time_s": result["wall_time_s"],
                    "accepted_increments": result["accepted_increments"],
                    "failed_attempts": result["failed_attempts"],
                    "total_newton_iterations": result["total_newton_iterations"],
                }
            )
    comparisons = []
    for eta in (0.0, 0.5, 1.0):
        reference = results[(eta, "fixed_200")]
        for name in ("adaptive_current", "adaptive_strict"):
            candidate = results[(eta, name)]
            e1 = abs(candidate["P11_at_F11_1p01_MPa"] - reference["P11_at_F11_1p01_MPa"]) / abs(reference["P11_at_F11_1p01_MPa"])
            e2 = abs(candidate["P11_at_F11_1p02_MPa"] - reference["P11_at_F11_1p02_MPa"]) / abs(reference["P11_at_F11_1p02_MPa"])
            comparisons.append({"eta": eta, "protocol": name, "relative_error_1pct_vs_fixed": e1, "relative_error_2pct_vs_fixed": e2})
    strict_max = max(
        max(row["relative_error_1pct_vs_fixed"], row["relative_error_2pct_vs_fixed"])
        for row in comparisons
        if row["protocol"] == "adaptive_strict"
    )
    summary = {
        "mesh": mesh,
        "mpi_ranks": mpi_ranks,
        "physical_loading_time_s": 1.0,
        "synchronized_times_s": [0.5, 1.0],
        "records": records,
        "comparisons": comparisons,
        "selected_protocol": DEFAULT_PROTOCOL.name,
        "selection_reason": (
            "Strict adaptive stepping reaches t=0.5 and t=1 exactly through Outputs/sync_times, "
            "keeps the benchmark convergence controls, and is compared against fixed_200."
        ),
        "selected_protocol_max_relative_difference_vs_fixed": strict_max,
    }
    write_json(STUDY_PATH, summary)
    with STUDY_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, default=32)
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--mpi-ranks", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run_study(args.mesh, not args.no_reuse, args.mpi_ranks), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
