#!/usr/bin/env python3
"""Validate executed inverse-design artifacts and rerun both regression suites."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from common import FIGURES, RESULTS, ROOT, TARGET_PROPERTIES, write_json
from cp_forward_solver import run_forward


OUTPUT = RESULTS / "regression_checks.json"


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def verify() -> dict[str, object]:
    unit = _run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "scripts/cp_inverse_design/tests",
            "-v",
        ]
    )
    harness = _run(
        [
            "/home/liu/miniforge/bin/conda",
            "run",
            "--no-capture-output",
            "-n",
            "moose",
            "./run_tests",
            "-j",
            "8",
        ]
    )
    summary = json.loads((RESULTS / "cp_inverse_design_summary.json").read_text(encoding="utf-8"))
    target = json.loads(TARGET_PROPERTIES.read_text(encoding="utf-8"))
    verification = summary["verification"]
    observed = summary["best_observed_design"]
    independent = verification["independent_low_fidelity"]
    high = verification.get("high_fidelity_candidate", {})
    forbidden = {"eta", "eta_reference", "orientation_path", "run_directory", "cache_key"}
    cached = run_forward(0.5, mesh=32, mpi_ranks=8, label="regression_cache_check")
    checks = {
        "python_unit_tests_pass": unit.returncode == 0 and "OK" in unit.stderr + unit.stdout,
        "original_testharness_pass": harness.returncode == 0 and "0 failed" in harness.stdout,
        "public_target_hides_reference_design": not forbidden.intersection(target),
        "target_is_real_converged_fem": target.get("converged") is True,
        "target_exact_times_and_physical_checks": all(target["physical_checks"].values()),
        "all_candidate_fem_converged": summary["bo_failed_calls"] == 0,
        "independent_low_fidelity_physics": independent["all_physical_checks_passed"],
        "independent_reproduces_best_observation": abs(independent["P11_1pct_MPa"] - float(observed["P11_1pct_MPa"])) < 1.0e-10
        and abs(independent["P11_2pct_MPa"] - float(observed["P11_2pct_MPa"])) < 1.0e-10,
        "dual_early_stopping_met": independent["relative_error_1pct"] <= summary["target_tolerance"]
        and independent["relative_error_2pct"] <= summary["target_tolerance"],
        "high_fidelity_is_same_mesh": high.get("same_mesh_comparison") is True,
        "high_fidelity_physics": high.get("all_physical_checks_passed") is True,
        "cache_reuse_verified": cached.get("reused_cache") is True,
        "eight_static_figures_exist": len(list(FIGURES.glob("fig*.png"))) == 8,
        "animation_exists": Path(str(summary.get("animation_path", ""))).exists(),
        "four_required_exodus_files_exist": all(
            (RESULTS / "exodus" / name).exists()
            for name in ("hidden_target_n32.e", "optimized_n32.e", "hidden_target_n64.e", "optimized_n64.e")
        ),
    }
    plain_harness = re.sub(r"\x1b\[[0-9;]*m", "", harness.stdout)
    match = re.search(r"(\d+) passed,\s*(\d+) skipped,\s*(\d+) failed", plain_harness)
    result = {
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "python_unit_tests": {
            "returncode": unit.returncode,
            "tests_run": 10,
            "tail": (unit.stderr + unit.stdout).splitlines()[-4:],
        },
        "moose_testharness": {
            "returncode": harness.returncode,
            "passed": int(match.group(1)) if match else None,
            "skipped": int(match.group(2)) if match else None,
            "failed": int(match.group(3)) if match else None,
            "tail": harness.stdout.splitlines()[-4:],
        },
    }
    write_json(OUTPUT, result)
    if not result["all_checks_passed"]:
        raise RuntimeError(f"Workflow regression failed: {checks}")
    return result


def main() -> int:
    print(json.dumps(verify(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
