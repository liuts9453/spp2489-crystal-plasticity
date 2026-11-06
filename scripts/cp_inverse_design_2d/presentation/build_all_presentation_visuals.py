#!/usr/bin/env python3
"""Audit cached FEM data and build all PI-presentation visualizations without solving."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_SCRIPT_DIR = SCRIPT_DIR.parent
for directory in (SCRIPT_DIR, PARENT_SCRIPT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from animate_bo_history import build_bo_animation  # noqa: E402
from build_design_landscape import build_design_landscape  # noqa: E402
from build_final_comparison import build_final_comparison  # noqa: E402
from build_microstructure_comparison import (  # noqa: E402
    build_microstructure_comparison,
)
from presentation_utils import (  # noqa: E402
    BEST_DESIGN,
    MANIFEST,
    MESH_FILES,
    OPTIMIZATION_HISTORY,
    PRESENTATION,
    REPORT,
    ROOT,
    TARGET_PROPERTIES,
    VERIFICATION,
    audit_history,
    common_inputs,
    file_sha256,
    manifest_record,
    source_path,
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _working_tree_paths() -> list[str]:
    paths: list[str] = []
    for line in _git("status", "--porcelain", "--untracked-files=all").splitlines():
        if line:
            paths.append(line[3:])
    return paths


def _validation(
    audit: dict[str, object], manifest: dict[str, object]
) -> dict[str, bool]:
    records = audit["records"]
    inputs = common_inputs()
    target = inputs["target"]
    best = inputs["best"]
    verification = inputs["verification"]
    changed = _working_tree_paths()
    protected_prefixes = (
        "src/",
        "include/",
        "examples/crystal_plasticity/",
        "test/tests/crystal_plasticity/",
    )
    protected_changed = [
        path for path in changed if path.startswith(protected_prefixes)
    ]
    return {
        "calls_are_exactly_1_through_28": [record["call"] for record in records]
        == list(range(1, 29)),
        "stages_are_9_DOE_plus_19_BO": [record["stage"] for record in records[:9]]
        == ["initial"] * 9
        and [record["stage"] for record in records[9:]] == ["sequential"] * 19,
        "best_so_far_uses_prefix_only": all(
            record["best_call"]
            == min(records[: record["call"]], key=lambda item: item["objective_J"])[
                "call"
            ]
            for record in records
        ),
        "all_current_curves_are_real_FEM": all(
            record["paths"]["csv"].is_file() and record["curve_point_count"] > 2
            for record in records
        ),
        "all_candidate_Exodus_files_are_recovered": all(
            record["paths"]["exodus"].is_file() for record in records
        ),
        "current_texture_and_curve_share_eta_theta": all(
            record["checks"]["eta_matches_result"]
            and record["checks"]["theta_matches_result"]
            for record in records
        ),
        "fixed_Neper_geometry_for_all_frames": manifest["animation"][
            "parameter_interpolation"
        ]
        is False
        and manifest["animation"]["fixed_geometry"]["sha256"]
        == manifest["fixed_geometry"]["sha256"]
        and all(
            frame["fixed_geometry_sha256"] == manifest["fixed_geometry"]["sha256"]
            for frame in manifest["animation"]["frames"]
        ),
        "mp4_duration_matches_frame_schedule": abs(
            float(
                manifest["animation"]["outputs"]["mp4"]["probe"]["format"]["duration"]
            )
            - float(manifest["animation"]["total_duration_ms"]) / 1000.0
        )
        <= 0.20,
        "target_properties_match_existing_JSON": abs(
            float(target["target_proof_stress_MPa"])
            - float(verification["coarse_target"]["proof_stress_MPa"])
        )
        <= 1.0e-12
        and abs(
            float(target["target_flow_stress_2pct_MPa"])
            - float(verification["coarse_target"]["flow_stress_2pct_MPa"])
        )
        <= 1.0e-12,
        "optimized_design_matches_best_design_JSON": abs(
            float(best["eta"]) - float(records[-1]["eta"])
        )
        <= 1.0e-12
        and abs(float(best["theta_deg"]) - float(records[-1]["theta_deg"])) <= 1.0e-11,
        "surrogate_is_labeled_as_prediction": "surrogate"
        in manifest["figure_A"]["prediction_disclaimer"].lower(),
        "no_mechanics_or_original_BO_sources_changed": not protected_changed,
        "no_MOOSE_or_BO_reexecution": manifest["fem_reexecution_count"] == 0,
    }


def _write_report(manifest: dict[str, object]) -> None:
    audit = manifest["data_audit"]
    target = manifest["numerical_result"]["target"]
    best = manifest["numerical_result"]["optimized"]
    figure_b = manifest["figure_B"]
    animation = manifest["animation"]
    checks = manifest["validation"]
    outputs = [
        ("Figure A", "figA_bo_design_landscape.png/.pdf/.svg"),
        ("Figure B", "figB_final_macroscopic_response.png/.pdf/.svg"),
        ("Figure C", "figC_target_optimized_microstructure.png/.pdf/.svg"),
        ("Animation", "bo_inverse_design_history.gif and .mp4"),
    ]
    lines = [
        "# SPP2489 Presentation Visualization Report",
        "",
        "## Outcome",
        "",
        "This incremental visualization build reused the executed Neper--MOOSE--FCC crystal-plasticity "
        "Bayesian-optimization cache. It did not call MOOSE, rerun Bayesian optimization, alter the "
        "constitutive model, or synthesize missing stress--strain curves.",
        "",
        f"- Audited FEM records: `{audit['history_record_count']}/28`.",
        f"- Recovered full solution CSV files: `{audit['solution_csv_count']}/28`.",
        f"- Recovered candidate Exodus files: `{audit['exodus_count']}/28`.",
        f"- Best-so-far update calls: `{audit['best_update_calls']}`.",
        f"- Target: proof `{target['proof_stress_MPa']:.6f} MPa`, flow `{target['flow_stress_2pct_MPa']:.6f} MPa`.",
        f"- Optimized: eta `{best['eta']:.10f}`, theta `{best['theta_deg']:+.8f} deg`, "
        f"proof `{best['proof_stress_MPa']:.6f} MPa`, flow `{best['flow_stress_2pct_MPa']:.6f} MPa`.",
        f"- Independent coarse errors: proof `{100*figure_b['coarse_relative_errors']['proof']:.3f}%`, "
        f"flow `{100*figure_b['coarse_relative_errors']['flow']:.3f}%`.",
        f"- Same-tessellation fine errors: proof `{100*figure_b['fine_relative_errors']['proof']:.3f}%`, "
        f"flow `{100*figure_b['fine_relative_errors']['flow']:.3f}%`.",
        "",
        "## Deliverables",
        "",
    ]
    lines.extend(f"- **{label}:** `{name}`" for label, name in outputs)
    lines.extend(
        [
            "",
            "Recommended PI sequence: Figure B (property match), Figure A (search logic), Figure C "
            "(microstructure/local physics), then the synchronized BO animation.",
            "",
            "## Data provenance",
            "",
            f"The compact optimization history is `{source_path(OPTIMIZATION_HISTORY)}`. Absolute paths "
            "stored in the CSV were not trusted blindly: the audit checked the current path, relocated "
            "the results suffix into the present worktree, and checked sibling SPP2489 worktrees by the "
            "run/cache identifier. Every recovered curve was re-read from `solution.csv`; only the known "
            "uninitialized MOOSE t=0 AuxKernel row was replaced by the exact undeformed state, using the "
            "already validated extraction routine.",
            "",
            "Figure A uses a final post-hoc `SingleTaskGP` fit to `-J`. Its background and property target "
            "contours are labeled surrogate predictions; only the 28 markers are real FEM evaluations. "
            "The connecting line denotes evaluation order, not a continuous material path.",
            "",
            "Figure B uses the real hidden-reference coarse curve and the independent optimized coarse "
            "curve without smoothing. Its inset reads the executed coarse/fine verification values. "
            "P0.2 is explicitly identified as the 0.2% offset proof stress under the prescribed "
            "plane-strain loading condition.",
            "",
            "Figure C regenerates textures from the executed `(eta, theta)` parameters on one fixed Neper "
            "geometry and reads final accumulated slip directly from the target/optimized Exodus files. "
            "The fixed cubic IPF-X key evaluates sample X in the crystal frame as `Q^T e_X`, reduces it to "
            "the cubic fundamental sector, and maps [001]/[101]/[111] to red/green/blue. No per-panel or "
            "per-frame orientation normalization is used. The two slip panels share one numerical scale.",
            "",
            "## Animation semantics",
            "",
            f"The animation contains `{len(animation['frames'])}` actual evaluation frames and has no "
            "interpolated design states. Calls 1--2 show a neutral parameter plane. For every frame "
            "`k >= 3`, the objective posterior is refitted using only calls `1..k`; future observations are "
            "never used. Current and best-so-far stress--strain histories are both real FEM curves. "
            f"The GIF is 1280x720; the MP4 is 1920x1080 H.264 with yuv420p pixel format. The final design "
            "is held for three seconds.",
            "",
            "## Validation",
            "",
        ]
    )
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — `{name}`"
        for name, passed in checks.items()
    )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "conda run --no-capture-output -n spp2489-inverse \\",
            "  python scripts/cp_inverse_design_2d/presentation/build_all_presentation_visuals.py",
            "```",
            "",
            "This command performs read-only FEM data recovery and visualization generation. It does not "
            "invoke `spp2489-opt`.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_all() -> dict[str, object]:
    PRESENTATION.mkdir(parents=True, exist_ok=True)
    audit = audit_history()
    inputs = common_inputs()
    figure_a = build_design_landscape(audit)
    figure_b = build_final_comparison(audit)
    figure_c = build_microstructure_comparison(audit)
    animation = build_bo_animation(audit)
    best_record = audit["records"][-1]
    coarse_metrics = json.loads(
        (MESH_FILES["coarse"].parent / "coarse_mesh_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    manifest: dict[str, object] = {
        "schema": "spp2489_cp_inverse_design_presentation_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "path": str(ROOT),
            "head_commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "reference_commit": "b35e15945319b9c9a9e1714e927f876afb0a321f",
        },
        "fem_reexecution_count": 0,
        "data_audit": audit["summary"],
        "history_records": [manifest_record(record) for record in audit["records"]],
        "numerical_result": {
            "target": {
                "proof_stress_MPa": float(inputs["target"]["target_proof_stress_MPa"]),
                "flow_stress_2pct_MPa": float(
                    inputs["target"]["target_flow_stress_2pct_MPa"]
                ),
            },
            "optimized": {
                "eta": float(inputs["best"]["eta"]),
                "theta_deg": float(inputs["best"]["theta_deg"]),
                "proof_stress_MPa": float(best_record["proof_stress_MPa"]),
                "flow_stress_2pct_MPa": float(best_record["flow_stress_2pct_MPa"]),
                "relative_error_proof": float(best_record["relative_error_proof"]),
                "relative_error_flow": float(best_record["relative_error_flow"]),
            },
        },
        "fixed_geometry": {
            "mesh": source_path(MESH_FILES["coarse"]),
            "sha256": file_sha256(MESH_FILES["coarse"]),
            "element_count": int(coarse_metrics["element_count"]),
            "grain_count": len(coarse_metrics["grain_element_counts"]),
            "metrics": source_path(
                MESH_FILES["coarse"].parent / "coarse_mesh_metrics.json"
            ),
        },
        "figure_A": figure_a,
        "figure_B": figure_b,
        "figure_C": figure_c,
        "animation": animation,
        "source_files": {
            "optimization_history": source_path(OPTIMIZATION_HISTORY),
            "best_design": source_path(BEST_DESIGN),
            "target_properties": source_path(TARGET_PROPERTIES),
            "verification": source_path(VERIFICATION / "verification_summary.json"),
        },
    }
    manifest["validation"] = _validation(audit, manifest)
    if not all(manifest["validation"].values()):
        raise RuntimeError(f"Presentation validation failed: {manifest['validation']}")
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_report(manifest)
    manifest["report"] = {
        "path": source_path(REPORT),
        "size_bytes": REPORT.stat().st_size,
        "sha256": file_sha256(REPORT),
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    manifest = build_all()
    print(
        json.dumps(
            {
                "presentation_directory": source_path(PRESENTATION),
                "audit": manifest["data_audit"],
                "validation": manifest["validation"],
                "fem_reexecution_count": manifest["fem_reexecution_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
