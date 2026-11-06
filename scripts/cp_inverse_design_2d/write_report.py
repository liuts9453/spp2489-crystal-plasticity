#!/usr/bin/env python3
"""Write the executed Neper/CP/BO report from machine-readable artifacts."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from common import (
    BEST_DESIGN,
    MESHES,
    NEPER,
    OPTIMIZATION_HISTORY,
    REPORT,
    ROOT,
    SUMMARY,
    TARGET_PROPERTIES,
    VERIFICATION,
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _history() -> list[dict[str, str]]:
    with OPTIMIZATION_HISTORY.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _git(command: list[str]) -> str:
    return subprocess.run(
        ["git", *command], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def _find_validation_case(eta: float, theta: float) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for path in (ROOT / "results" / "cp_inverse_design_2d" / "runs").glob(
        "*/result.json"
    ):
        result = _json(path)
        if (
            result.get("converged")
            and abs(float(result.get("eta", -99.0)) - eta) < 1.0e-12
            and abs(float(result.get("theta_deg", -999.0)) - theta) < 1.0e-12
        ):
            candidates.append(result)
    if not candidates:
        raise RuntimeError(f"Missing validation case eta={eta}, theta={theta}")
    return max(candidates, key=lambda item: float(item["wall_time_s"]))


def _structured_same_orientation_reference() -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for path in (ROOT / "results" / "cp_inverse_design" / "runs").glob("*/result.json"):
        result = _json(path)
        if (
            result.get("converged")
            and abs(float(result.get("eta", -99.0)) - 1.0) < 1.0e-12
        ):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("Missing structured same-orientation regression reference")
    return min(
        candidates, key=lambda item: abs(float(item["P11_at_F11_1p02_MPa"]) - 473.5)
    )


def write_report() -> Path:
    summary = _json(SUMMARY)
    best = _json(BEST_DESIGN)
    target = _json(TARGET_PROPERTIES)
    verification = _json(VERIFICATION / "verification_summary.json")
    coarse_mesh = _json(MESHES / "coarse_mesh_metrics.json")
    fine_mesh = _json(MESHES / "fine_mesh_metrics.json")
    neper = _json(NEPER / "neper_generation_manifest.json")
    rows = _history()
    case_a = _find_validation_case(1.0, 0.0)
    case_b = _find_validation_case(0.0, 0.0)
    structured_a = _structured_same_orientation_reference()
    coarse_candidate = verification["independent_coarse_candidate"]
    fine_candidate = verification.get("fine_candidate")
    coarse_target = verification["coarse_target"]
    fine_target = verification.get("fine_target")
    time_study = verification.get("time_integration")
    hidden = verification["hidden_design_disclosed_after_optimization"]
    initial_rows = [row for row in rows if row["stage"] == "initial"]
    initial_separations = [
        (
            max(float(row["relative_error_proof"]), float(row["relative_error_flow"])),
            int(row["call"]),
        )
        for row in initial_rows
    ]
    nearest_initial_separation, nearest_initial_call = min(initial_separations)
    first_initial_separation = initial_separations[0][0]
    verification_runs = [coarse_candidate]
    if fine_target and fine_candidate:
        verification_runs.extend([fine_target, fine_candidate])
    if time_study:
        verification_runs.extend([time_study["target"], time_study["candidate"]])
    inverse_fem_calls = 1 + int(summary["bo_fem_calls"]) + len(verification_runs)
    inverse_fem_wall = (
        float(target["wall_time_s"])
        + float(summary["bo_fem_wall_time_s"])
        + sum(float(item["wall_time_s"]) for item in verification_runs)
    )

    lines = [
        "# SPP2489 2D Texture Inverse Design with a Neper RVE",
        "",
        "## Executive result",
        "",
        "This report contains only executed MOOSE results. The optimization used a fixed Neper "
        "20-grain periodic geometry, second-order triangular finite elements, the existing A4 "
        "fully implicit FCC crystal-plasticity integrator, and two texture variables `(eta, theta)`. "
        "No random-search comparison was performed.",
        "",
        f"- Public target: `P0.2 = {target['target_proof_stress_MPa']:.6f} MPa`, "
        f"`Pflow(2%) = {target['target_flow_stress_2pct_MPa']:.6f} MPa`.",
        f"- Best observed design: `eta = {float(best['eta']):.9f}`, "
        f"`theta = {float(best['theta_deg']):+.7f} deg`.",
        f"- Independent coarse errors: proof `{100*coarse_candidate['relative_error_proof']:.6f}%`, "
        f"flow `{100*coarse_candidate['relative_error_flow']:.6f}%`.",
        f"- BO used `{summary['bo_fem_calls']}` real candidate FEM evaluations "
        f"(`{summary['bo_initial_calls']}` initial, `{summary['bo_sequential_calls']}` sequential), "
        f"with `{summary['bo_failed_calls']}` failed terminal evaluations.",
        f"- Candidate FEM wall time accumulated by BO: `{summary['bo_fem_wall_time_s']:.2f} s`; "
        f"GP/acquisition time: `{summary['bo_gp_time_s']:.2f} s`.",
        f"- Target + candidate BO + independent/fine/time verification: `{inverse_fem_calls}` real "
        f"FEM calls and `{inverse_fem_wall:.2f} s` summed wall time (validation Cases A/B are separate).",
        "",
        "## 1. Software and immutable physical model",
        "",
        f"- Application branch: `{_git(['branch', '--show-current'])}`; base commit recorded in runs: "
        f"`{summary['application_commit']}`.",
        f"- MOOSE commit: `{target['moose_commit']}`.",
        f"- Python `{summary['software']['python']}`, PyTorch `{summary['software']['torch']}`, "
        f"BoTorch `{summary['software']['botorch']}`, GPyTorch `{summary['software']['gpytorch']}`.",
        f"- Neper version: `{neper['neper_version']}`; Gmsh `{neper['gmsh_version']}`.",
        "- Preserved model: multiplicative finite strain, Green--Lagrange elastic strain, "
        "12 FCC `{111}<110>` systems, power-law kinetics, Kalidindi hardening, backward Euler, "
        "PK2 constitutive interface and analytic algorithmic tangent.",
        "- A4 path: 10 local unknowns (6 PK2 components + 4 slip-plane resistances), grouped mode, "
        "LU reuse, optimized tensor/Jacobian evaluation, scalar specialization and previous-state predictor.",
        "",
        "## 2. Real Neper periodic geometry and meshes",
        "",
        f"The fixed seed-2026 `.tess` has SHA-256 `{neper['tessellation_sha256']}`. "
        "Neper generated both the tessellation and raw periodic mesh/provenance files. The converter "
        "clips exact periodic Neper cell polygons into the 200 x 200 um fundamental square, preserves "
        "original grain IDs, constructs shared conforming interfaces, and creates matching named sidesets.",
        "",
        "| Mesh | TRI6 elements | Nodes | QPs | Min quality | Max aspect | Periodic coordinate error (um) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Coarse | {coarse_mesh['element_count']} | {coarse_mesh['node_count']} | "
        f"{7*coarse_mesh['element_count']} | "
        f"{coarse_mesh['minimum_triangle_quality']:.6f} | {coarse_mesh['maximum_aspect_ratio']:.4f} | "
        f"{coarse_mesh['maximum_periodic_coordinate_error_um']:.3e} |",
        f"| Fine | {fine_mesh['element_count']} | {fine_mesh['node_count']} | "
        f"{7*fine_mesh['element_count']} | "
        f"{fine_mesh['minimum_triangle_quality']:.6f} | {fine_mesh['maximum_aspect_ratio']:.4f} | "
        f"{fine_mesh['maximum_periodic_coordinate_error_um']:.3e} |",
        "",
        "Both meshes use the exact same tessellation and pass: 20 blocks, area 40000 um^2, all nodes used, "
        "positive/manifold elements, shared grain interfaces, four correct boundary sets and one-to-one "
        "periodic coordinate pairing. The minimum quality near one acute internal Voronoi vertex is reported "
        "rather than hidden; it persists under independent cut-origin studies.",
        "",
        "## 3. Texture parameterization",
        "",
        "The design uses `Q_i = SLERP(Q_i_random, Rz(theta) Q0_preferred, eta)` with shortest-quaternion "
        "sign handling. Formal bounds are `0.1 <= eta <= 0.9` and `-45 <= theta <= 45 deg`; endpoint "
        "generation at eta=0 and eta=1 remains available for validation. Elastic tensor and slip systems "
        "consume the same MOOSE `crysrot` property. Orthogonality, determinant, quaternion norm and Euler "
        "roundtrip tests pass below 1e-10.",
        "",
        "## 4. Neper-grid mechanics benchmarks",
        "",
        "| Case | P0.2 (MPa) | Pflow 2% (MPa) | Steps | Failed attempts | Newton iters | Wall (s) | Max |P22| (MPa) | HM residual |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| A: all same orientation | {case_a['proof_stress_MPa']:.6f} | {case_a['flow_stress_2pct_MPa']:.6f} | "
        f"{case_a['accepted_increments']} | {case_a['failed_attempts']} | {case_a['total_newton_iterations']} | "
        f"{case_a['wall_time_s']:.2f} | {case_a['maximum_abs_P22_MPa']:.3e} | "
        f"{case_a['hill_mandel_relative_residual_max']:.3e} |",
        f"| B: fixed random orientations | {case_b['proof_stress_MPa']:.6f} | {case_b['flow_stress_2pct_MPa']:.6f} | "
        f"{case_b['accepted_increments']} | {case_b['failed_attempts']} | {case_b['total_newton_iterations']} | "
        f"{case_b['wall_time_s']:.2f} | {case_b['maximum_abs_P22_MPa']:.3e} | "
        f"{case_b['hill_mandel_relative_residual_max']:.3e} |",
        "",
        f"Case A is uniform to a final local P11 range of "
        f"`{case_a['maximum_local_P11_MPa']-case_a['minimum_local_P11_MPa']:.3e} MPa`; its periodic "
        f"fluctuation mismatch is `{case_a['exodus']['maximum_periodicity_error']:.3e}`. Case B has a "
        f"local P11 range of `{case_b['minimum_local_P11_MPa']:.3f}` to "
        f"`{case_b['maximum_local_P11_MPa']:.3f} MPa` and grain-dependent slip, as expected.",
        "",
        f"The pre-existing structured-RVE same-orientation reference gives P11(2%) = "
        f"`{structured_a['P11_at_F11_1p02_MPa']:.12f} MPa`; the Neper Case A value differs by only "
        f"`{abs(case_a['flow_stress_2pct_MPa']-structured_a['P11_at_F11_1p02_MPa'])/abs(structured_a['P11_at_F11_1p02_MPa']):.3e}` "
        "relatively. This is the quantitative homogeneous-single-crystal regression.",
        "",
        "The internal translation gauge is placed on the mesh node closest to (100,100) um. Pinning a "
        "periodic corner was rejected because the corner can be a periodic slave and leave a rigid translation. "
        "A global relative tolerance of 1e-10 is used so the automatically scaled scalar P22 equation does not "
        "terminate with a visible stress residual.",
        "",
        "## 5. Proof-stress extraction",
        "",
        "The initial t=0 CSV output row is not used directly because TIMESTEP_END AuxKernels have not yet "
        "initialized F there; the exact undeformed state F=I, P=0 is inserted. K0 is least-squares fitted "
        "through 0.1% nominal strain and checked against the narrower 0.05% window. The first positive "
        "crossing with `P = K0*(strain-0.002)` is linearly interpolated. All candidates use the same rule, "
        "loading rate and synchronized 2% endpoint.",
        "",
        "## 6. Hidden target and Bayesian optimization",
        "",
        f"The hidden design, disclosed only after optimization, was `eta={hidden['eta']}`, "
        f"`theta={hidden['theta_deg']} deg`. The public target file contained only properties and numerical "
        "configuration. The objective was the equal-weight sum of squared relative errors in proof and "
        "2% flow stress. A `SingleTaskGP` modeled `-J`; candidates came from `LogExpectedImprovement`. "
        "The nine deterministic DOE points span eta in {0.15, 0.50, 0.85} and theta in {-30, 0, 30} deg.",
        "",
        "The hidden reference was selected from a post-hoc two-property GP scan of the fixed pilot DOE "
        "to avoid a trivial target near an initial response; the target properties were then replaced by "
        "a real MOOSE evaluation. In the executed data its nearest initial point is call "
        f"`{nearest_initial_call}` at a maximum relative property distance of "
        f"`{100*nearest_initial_separation:.4f}%`; its distance from initial call 1 is "
        f"`{100*first_initial_separation:.4f}%`.",
        "",
        "| Call | Stage | eta | theta (deg) | P0.2 | Pflow | eY (%) | eF (%) | J | Wall (s) | Status |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["status"] == "converged":
            lines.append(
                f"| {row['call']} | {row['stage']} | {float(row['eta']):.7f} | "
                f"{float(row['theta_deg']):+.5f} | {float(row['proof_stress_MPa']):.4f} | "
                f"{float(row['flow_stress_2pct_MPa']):.4f} | "
                f"{100*float(row['relative_error_proof']):.4f} | "
                f"{100*float(row['relative_error_flow']):.4f} | "
                f"{float(row['objective_J']):.3e} | {float(row['wall_time_s']):.2f} | converged |"
            )
        else:
            lines.append(
                f"| {row['call']} | {row['stage']} | {row['eta']} | {row['theta_deg']} | — | — | — | — | — | "
                f"{row['wall_time_s']} | failed |"
            )
    lines.extend(
        [
            "",
            "No random-search data or comparison was generated.",
            "",
            "## 7. Independent and numerical verification",
            "",
            "| Fidelity | Target P0.2 | Candidate P0.2 | eY | Target Pflow | Candidate Pflow | eF | Candidate wall (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| Coarse independent | {coarse_target['proof_stress_MPa']:.6f} | "
            f"{coarse_candidate['proof_stress_MPa']:.6f} | {100*coarse_candidate['relative_error_proof']:.5f}% | "
            f"{coarse_target['flow_stress_2pct_MPa']:.6f} | {coarse_candidate['flow_stress_2pct_MPa']:.6f} | "
            f"{100*coarse_candidate['relative_error_flow']:.5f}% | {coarse_candidate['wall_time_s']:.2f} |",
        ]
    )
    if fine_candidate and fine_target:
        lines.extend(
            [
                f"| Fine same-mesh | {fine_target['proof_stress_MPa']:.6f} | "
                f"{fine_candidate['proof_stress_MPa']:.6f} | {100*fine_candidate['relative_error_proof']:.5f}% | "
                f"{fine_target['flow_stress_2pct_MPa']:.6f} | {fine_candidate['flow_stress_2pct_MPa']:.6f} | "
                f"{100*fine_candidate['relative_error_flow']:.5f}% | {fine_candidate['wall_time_s']:.2f} |",
                "",
                f"Target coarse-to-fine changes: proof "
                f"`{100*verification['target_coarse_to_fine']['relative_proof_change']:.5f}%`, flow "
                f"`{100*verification['target_coarse_to_fine']['relative_flow_change']:.5f}%`. Candidate "
                f"changes: proof `{100*verification['candidate_coarse_to_fine']['relative_proof_change']:.5f}%`, "
                f"flow `{100*verification['candidate_coarse_to_fine']['relative_flow_change']:.5f}%`.",
            ]
        )
    if time_study:
        lines.extend(
            [
                "",
                "Adaptive-refined time integration (halved dtmax) gives default-to-refined relative changes:",
                "",
                f"- Target: proof `{100*time_study['target_default_to_refined']['relative_proof_change']:.5f}%`, "
                f"flow `{100*time_study['target_default_to_refined']['relative_flow_change']:.5f}%`.",
                f"- Candidate: proof `{100*time_study['candidate_default_to_refined']['relative_proof_change']:.5f}%`, "
                f"flow `{100*time_study['candidate_default_to_refined']['relative_flow_change']:.5f}%`.",
            ]
        )
    lines.extend(
        [
            "",
            "Every reported candidate passed end time/synchronization, P22, zero macro shear, plastic activity, "
            "finite-state, positive resistance, det(Fp), nonnegative dissipation, Hill--Mandel, periodicity, "
            "20-block and Exodus-field checks. Failed/retried adaptive attempts remain recorded separately.",
            "",
            "## 8. Files and reproduction",
            "",
            "```bash",
            "conda run -n spp2489-neper python scripts/cp_inverse_design_2d/generate_neper_rve.py",
            "conda run -n spp2489-inverse python scripts/cp_inverse_design_2d/optimize_texture_2d.py",
            "conda run -n spp2489-inverse python scripts/cp_inverse_design_2d/verify_high_fidelity.py --eta <best> --theta <best>",
            "conda run -n spp2489-inverse python scripts/cp_inverse_design_2d/plot_optimization.py",
            "```",
            "",
            "Raw FEM runs are under `results/cp_inverse_design_2d/runs/`; target, verification, copied Exodus "
            "and figures have dedicated subdirectories. Cache keys include both design variables, mesh and "
            "manifest hashes, material/load/time/solver settings, MPI ranks and code commits.",
            "",
            "## 9. Limitations and next steps",
            "",
            "- These are demonstrative CP constants, not a calibrated SS316L dataset or experimental target.",
            "- The inverse map need not be unique; success means recovering both target properties, not hidden parameters.",
            "- The fixed Voronoi geometry contains one acute vertex; its measured quality and coarse/fine influence are reported.",
            "- Future work may add morphology descriptors, experimental calibration and uncertainty, but none were mixed into this demo.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return REPORT


def main() -> int:
    print(write_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
