#!/usr/bin/env python3
"""Collect traceable sample-coordinate evidence without guessing section mappings."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TEXT_SUFFIXES = {".m", ".py", ".md", ".txt", ".yaml", ".yml", ".json"}
PATTERN = re.compile(
    r"(Al1_oBD|Al1_pBD|GrainExtractionZ1_15_00|GrainExtractionZ1_15_90|"
    r"\boBD\b|\bpBD\b|build.?direction|sample.?direction|reference.?frame|"
    r"euler|bunge|scan.?plane)",
    flags=re.IGNORECASE,
)
CLEANING_PATTERN = re.compile(
    r"(grains_all\s*\(\s*['\"]fcc['\"]\s*\)|min_grain_size|"
    r"minimum grain size|MIN_AREA_THRESHOLD|MIN_B_THRESHOLD|"
    r"area threshold|b-axis threshold|equivalentRadius)",
    flags=re.IGNORECASE,
)
ALPHA_PATTERN = re.compile(
    r"(alpha_val\s*=|df\[['\"]alpha['\"]\]\s*=|process_alpha_column|"
    r"set all alpha|_pBD_|_oBD_|_00_|_90_)",
    flags=re.IGNORECASE,
)


def audit_coordinate_metadata(
    root: Path, maximum_matches: int = 500
) -> dict[str, object]:
    evidence = []
    cleaning_evidence = []
    alpha_evidence = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        try:
            lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if PATTERN.search(line):
                evidence.append(
                    {
                        "relative_path": str(path.relative_to(root)),
                        "line_number": line_number,
                        "text": line.strip()[:500],
                    }
                )
                if len(evidence) >= maximum_matches:
                    break
            if CLEANING_PATTERN.search(line):
                cleaning_evidence.append(
                    {
                        "relative_path": str(path.relative_to(root)),
                        "line_number": line_number,
                        "text": line.strip()[:500],
                    }
                )
            if ALPHA_PATTERN.search(line):
                alpha_evidence.append(
                    {
                        "relative_path": str(path.relative_to(root)),
                        "line_number": line_number,
                        "text": line.strip()[:500],
                    }
                )
        if len(evidence) >= maximum_matches:
            break
    return {
        "evidence_count": len(evidence),
        "evidence": evidence,
        "cleaning_evidence_count": len(cleaning_evidence),
        "cleaning_evidence": cleaning_evidence,
        "alpha_semantics_evidence_count": len(alpha_evidence),
        "alpha_semantics_evidence": alpha_evidence,
        "interpretation_status": "requires_manual_scientific_review",
        "phase_mapping_status": (
            "The MTEX analysis explicitly selects fcc grains, while the exported CSVs "
            "contain only the numeric phase label 0. The numeric label is retained "
            "unless an explicit phase-index mapping is found."
        ),
        "cleaning_interpretation": (
            "The experimental MTEX statistics use a 2.8 um minimum equivalent "
            "diameter. Separate DRAGen preprocessing uses a 0.5 um minimum ellipse "
            "minor axis for downstream geometry/meshing. These filters serve "
            "different purposes and are reported separately."
        ),
        "alpha_interpretation": (
            "For the four supplied CSVs alpha is constant within each section. "
            "The Al1 conversion script explicitly assigns 0/90 from the section "
            "filename. It is therefore retained as section metadata and excluded "
            "from per-grain morphology-direction calibration."
        ),
        "guardrail": (
            "Finding the same label in a script does not prove that Al1 oBD/pBD and "
            "Al0 00/90 share a coordinate convention. The workflow keeps all four "
            "sections separate until an explicit mapping is approved."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_coordinate_metadata(args.data_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
