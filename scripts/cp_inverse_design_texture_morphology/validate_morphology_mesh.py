#!/usr/bin/env python3
"""Validate morphology meshes and record geometry-driven quality risk explicitly."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ID2_SCRIPT_DIR = SCRIPT_DIR.parent / "cp_inverse_design_2d"
for directory in (SCRIPT_DIR, ID2_SCRIPT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from morphology_common import (  # noqa: E402
    MINIMUM_SCREENING_QUALITY,
    OLD_MESH_METRICS,
)
from common import write_json  # noqa: E402
from validate_periodic_mesh import validate_mesh  # noqa: E402


def validate_morphology_mesh(path: Path, mesh_name: str) -> dict[str, object]:
    metrics = validate_mesh(
        path, minimum_quality_threshold=MINIMUM_SCREENING_QUALITY
    )
    reference = json.loads(OLD_MESH_METRICS[mesh_name].read_text(encoding="utf-8"))
    quality = float(metrics["minimum_triangle_quality"])
    aspect = float(metrics["maximum_aspect_ratio"])
    reference_quality = float(reference["minimum_triangle_quality"])
    reference_aspect = float(reference["maximum_aspect_ratio"])
    metrics["quality_reference"] = {
        "mesh": str(OLD_MESH_METRICS[mesh_name]),
        "minimum_triangle_quality": reference_quality,
        "maximum_aspect_ratio": reference_aspect,
    }
    metrics["morphology_quality_assessment"] = {
        "meets_or_exceeds_reference_minimum_quality": quality
        >= reference_quality - 1.0e-12,
        "does_not_exceed_reference_maximum_aspect": aspect
        <= reference_aspect + 1.0e-12,
        "geometry_driven_acute_corner": quality < reference_quality,
        "mandatory_same_tessellation_fine_verification": quality < reference_quality
        or aspect > reference_aspect,
        "accepted_for_screening": quality > MINIMUM_SCREENING_QUALITY,
        "interpretation": (
            "A low value at elongated morphologies is retained as a measured geometric risk, "
            "not silently relabeled as reference-quality; those cases require fine verification."
        ),
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--mesh-name", choices=("coarse", "fine"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = validate_morphology_mesh(args.mesh, args.mesh_name)
    if args.output:
        write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
