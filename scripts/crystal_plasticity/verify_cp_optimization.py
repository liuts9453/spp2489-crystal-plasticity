#!/usr/bin/env python3
"""Run focused legacy-versus-optimized FCC constitutive regression checks."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "spp2489-opt"
INPUT = ROOT / "test/tests/crystal_plasticity/single_crystal_full_state.i"
RESULTS = ROOT / "results/crystal_plasticity/optimization/verification"

CONFIGS = {
    "legacy": {
        "cp_local_solver": "legacy",
        "cp_reuse_lu": "false",
        "cp_optimize_tensors": "false",
        "cp_optimize_scalars": "false",
        "cp_predictor": "previous",
    },
    "optimized": {
        "cp_local_solver": "grouped",
        "cp_reuse_lu": "true",
        "cp_optimize_tensors": "true",
        "cp_optimize_scalars": "true",
        "cp_predictor": "previous",
    },
}

NONPHYSICAL_COLUMNS = {
    "linear_iterations",
    "nonlinear_iterations",
    "local_iterations",
    "consistent_tangent_relative_error",
}


def run_case(name: str, configuration: str, overrides: dict[str, object]) -> Path:
    stem = RESULTS / f"{name}_{configuration}"
    command = [
        str(APP),
        "-i",
        str(INPUT),
        "Outputs/active=csv",
        f"Outputs/csv/file_base={stem}",
        "--disable-perf-graph-live",
    ]
    values = dict(CONFIGS[configuration])
    values.update(overrides)
    command.extend(f"{key}={value}" for key, value in values.items())
    with stem.with_suffix(".log").open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if process.returncode:
        raise RuntimeError(f"{name}/{configuration} failed; see {stem.with_suffix('.log')}")
    return stem.with_suffix(".csv")


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def compare_histories(reference: Path, candidate: Path) -> dict[str, object]:
    reference_rows = load(reference)
    candidate_rows = load(candidate)
    if len(reference_rows) != len(candidate_rows):
        raise RuntimeError(f"History length differs: {reference} vs {candidate}")
    worst = {"scaled_error": 0.0, "column": "", "row": 0, "reference": 0.0, "candidate": 0.0}
    for row_index, (left, right) in enumerate(zip(reference_rows, candidate_rows)):
        if left.keys() != right.keys():
            raise RuntimeError(f"CSV columns differ: {reference} vs {candidate}")
        for column in left:
            if column in NONPHYSICAL_COLUMNS:
                continue
            reference_value = float(left[column])
            candidate_value = float(right[column])
            scale = max(1.0, abs(reference_value), abs(candidate_value))
            error = abs(reference_value - candidate_value) / scale
            if error > worst["scaled_error"]:
                worst = {
                    "scaled_error": error,
                    "column": column,
                    "row": row_index,
                    "reference": reference_value,
                    "candidate": candidate_value,
                }

    minimum_dissipation = min(float(row["plastic_dissipation"]) for row in candidate_rows[1:])
    maximum_group_spread = 0.0
    for row in candidate_rows[1:]:
        resistance = [float(row[f"resistance_full_{index:02d}"]) for index in range(12)]
        maximum_group_spread = max(
            maximum_group_spread,
            *(max(resistance[start : start + 3]) - min(resistance[start : start + 3])
              for start in range(0, 12, 3)),
        )
    return {
        "rows": len(reference_rows),
        "worst": worst,
        "minimum_plastic_dissipation": minimum_dissipation,
        "maximum_group_resistance_spread": maximum_group_spread,
        "final_P11_reference_MPa": float(reference_rows[-1]["P11"]),
        "final_P11_optimized_MPa": float(candidate_rows[-1]["P11"]),
        "passed": worst["scaled_error"] <= 1e-9
        and minimum_dissipation >= -1e-12
        and maximum_group_spread <= 1e-12,
    }


def main() -> int:
    if not APP.exists():
        raise SystemExit(f"Missing {APP}; build the application first")
    RESULTS.mkdir(parents=True, exist_ok=True)

    cases = {
        "orientation_000": {},
        "orientation_37_52_11": {"phi1": 37, "Phi": 52, "phi2": 11},
        "general_scalar_model": {
            "phi1": 37,
            "Phi": 52,
            "phi2": 11,
            "Materials/analytic_cp/rate_sensitivity": 0.13,
            "Materials/analytic_cp/hardening_exponent": 1.3,
        },
    }
    comparisons: dict[str, object] = {}
    for name, overrides in cases.items():
        reference = run_case(name, "legacy", overrides)
        candidate = run_case(name, "optimized", overrides)
        comparisons[name] = compare_histories(reference, candidate)

    substep_csv = run_case(
        "substepped_tangent",
        "optimized",
        {
            "steps": 4,
            "max_strain": 0.02,
            "Materials/analytic_cp/slip_increment_tolerance": 0.0005,
            "Materials/analytic_cp/verify_analytic_tangent": "true",
        },
    )
    substep_rows = load(substep_csv)[1:]
    tangent = {
        "maximum_relative_error": max(
            float(row["consistent_tangent_relative_error"]) for row in substep_rows
        ),
        "maximum_internal_substeps": max(float(row["local_substeps"]) for row in substep_rows),
    }
    tangent["passed"] = (
        tangent["maximum_relative_error"] <= 1e-6
        and tangent["maximum_internal_substeps"] > 1
    )

    passed = all(item["passed"] for item in comparisons.values()) and tangent["passed"]
    payload = {"passed": passed, "comparisons": comparisons, "substepped_tangent": tangent}
    output = RESULTS / "verification.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
