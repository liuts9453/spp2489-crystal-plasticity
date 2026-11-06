#!/usr/bin/env python3
"""Verify that morphology partitions do not alter a homogeneous single crystal."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ID2_SCRIPT_DIR = SCRIPT_DIR.parent / "cp_inverse_design_2d"
for directory in (SCRIPT_DIR, ID2_SCRIPT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common import PROTOCOLS, write_json  # noqa: E402
from morphology_common import (  # noqa: E402
    BASE_GEOMETRY_SEED,
    DEFAULT_PROTOCOL,
    VERIFICATION,
    baseline_texture,
)
from texture_morphology_forward_solver import (  # noqa: E402
    run_texture_morphology_forward,
)


CONTROL_LEVELS = (0.70, 1.00, 1.50)


def run_control(
    *,
    protocol_name: str = DEFAULT_PROTOCOL.name,
    mpi_ranks: int = 8,
    reuse_cache: bool = True,
) -> dict[str, object]:
    _, theta = baseline_texture()
    results = []
    started = time.perf_counter()
    for a_value in CONTROL_LEVELS:
        result = run_texture_morphology_forward(
            1.0,
            theta,
            a_value,
            geometry_seed=BASE_GEOMETRY_SEED,
            mesh_name="coarse",
            protocol=PROTOCOLS[protocol_name],
            mpi_ranks=mpi_ranks,
            label_prefix="homogeneous_control",
            reuse_cache=reuse_cache,
        )
        if not result.get("converged"):
            raise RuntimeError(
                f"Homogeneous control failed for a={a_value}: "
                f"{result.get('failure_reason')}"
            )
        results.append(result)
    proof = [float(item["proof_stress_MPa"]) for item in results]
    flow = [float(item["flow_stress_2pct_MPa"]) for item in results]
    local_ranges = [
        float(item["exodus"]["element_statistics"]["P11"]["range"])
        for item in results
    ]
    summary = {
        "definition": (
            "All 20 grain blocks use the same crystal orientation and constitutive "
            "parameters; only the conforming morphology partition changes."
        ),
        "eta": 1.0,
        "theta_deg": theta,
        "levels": list(CONTROL_LEVELS),
        "protocol": protocol_name,
        "mpi_ranks": mpi_ranks,
        "proof_stress_spread_MPa": max(proof) - min(proof),
        "proof_stress_relative_spread": (max(proof) - min(proof)) / sum(proof)
        * len(proof),
        "flow_stress_spread_MPa": max(flow) - min(flow),
        "flow_stress_relative_spread": (max(flow) - min(flow)) / sum(flow)
        * len(flow),
        "maximum_local_P11_range_MPa": max(local_ranges),
        "all_checks_passed": all(item["all_checks_passed"] for item in results),
        "wall_time_s": time.perf_counter() - started,
        "results": [
            {
                "input_a": item["morphology_input_a"],
                "achieved_a": item[
                    "achieved_directional_extent_ratio_x_over_y"
                ],
                "proof_stress_MPa": item["proof_stress_MPa"],
                "flow_stress_2pct_MPa": item["flow_stress_2pct_MPa"],
                "local_P11_range_MPa": item["exodus"]["element_statistics"][
                    "P11"
                ]["range"],
                "failed_attempts": item["failed_attempts"],
                "accepted_increments": item["accepted_increments"],
                "run_directory": item["run_directory"],
            }
            for item in results
        ],
    }
    path = VERIFICATION / "homogeneous_control_summary.json"
    write_json(path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL.name)
    parser.add_argument("--mpi-ranks", type=int, default=8)
    parser.add_argument("--no-reuse", action="store_true")
    args = parser.parse_args()
    summary = run_control(
        protocol_name=args.protocol,
        mpi_ranks=args.mpi_ranks,
        reuse_cache=not args.no_reuse,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
