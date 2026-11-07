#!/usr/bin/env python3
"""Validate Gmsh-v2 Neper meshes before they are passed to MOOSE/libMesh."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import write_json


ELEMENT_NODE_COUNT = {1: 2, 2: 3, 8: 3, 9: 6, 15: 1}


def read_msh(path: Path) -> tuple[dict[int, np.ndarray], list[dict[str, object]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    node_start = lines.index("$Nodes")
    node_count = int(lines[node_start + 1])
    nodes: dict[int, np.ndarray] = {}
    for line in lines[node_start + 2 : node_start + 2 + node_count]:
        fields = line.split()
        nodes[int(fields[0])] = np.asarray([float(value) for value in fields[1:4]])
    element_start = lines.index("$Elements")
    element_count = int(lines[element_start + 1])
    elements: list[dict[str, object]] = []
    for line in lines[element_start + 2 : element_start + 2 + element_count]:
        fields = [int(value) for value in line.split()]
        element_type, tag_count = fields[1], fields[2]
        if element_type not in ELEMENT_NODE_COUNT:
            continue
        elements.append(
            {
                "id": fields[0],
                "type": element_type,
                "tags": fields[3 : 3 + tag_count],
                "nodes": fields[3 + tag_count :],
            }
        )
    return nodes, elements


def _triangle_quality(points: np.ndarray) -> tuple[float, float, float]:
    lengths = np.asarray(
        [
            np.linalg.norm(points[1] - points[0]),
            np.linalg.norm(points[2] - points[1]),
            np.linalg.norm(points[0] - points[2]),
        ]
    )
    first = points[1, :2] - points[0, :2]
    second = points[2, :2] - points[0, :2]
    area = abs(float(first[0] * second[1] - first[1] * second[0])) / 2.0
    quality = 4.0 * math.sqrt(3.0) * area / float(np.sum(lengths * lengths))
    altitude = 2.0 * area / float(np.max(lengths))
    aspect = float(np.max(lengths)) / altitude
    return area, quality, aspect


def validate_mesh(
    path: Path,
    *,
    minimum_quality_threshold: float = 0.10,
    expected_grain_count: int = 20,
    domain_size_um: float = 200.0,
) -> dict[str, object]:
    nodes, elements = read_msh(path)
    triangles = [element for element in elements if element["type"] == 9]
    boundaries = [element for element in elements if element["type"] == 8]
    block_counts = Counter(int(element["tags"][0]) for element in triangles)
    boundary_elements: dict[int, list[dict[str, object]]] = defaultdict(list)
    for element in boundaries:
        boundary_elements[int(element["tags"][0])].append(element)

    areas: list[float] = []
    qualities: list[float] = []
    aspects: list[float] = []
    block_areas: Counter[int] = Counter()
    edge_grains: dict[tuple[int, int], list[int]] = defaultdict(list)
    for element in triangles:
        connectivity = list(element["nodes"])
        coordinates = np.asarray([nodes[node] for node in connectivity[:3]])
        area, quality, aspect = _triangle_quality(coordinates)
        areas.append(area)
        qualities.append(quality)
        aspects.append(aspect)
        grain = int(element["tags"][0])
        block_areas[grain] += area
        for first, second in (
            (connectivity[0], connectivity[1]),
            (connectivity[1], connectivity[2]),
            (connectivity[2], connectivity[0]),
        ):
            edge_grains[tuple(sorted((first, second)))].append(grain)

    def boundary_node_ids(identifier: int) -> list[int]:
        return sorted(
            {
                node
                for element in boundary_elements[identifier]
                for node in element["nodes"]
            }
        )

    periodic_errors: dict[str, float] = {}
    periodic_counts: dict[str, list[int]] = {}
    for name, first_id, second_id, coordinate in (
        ("x", 101, 102, 1),
        ("y", 103, 104, 0),
    ):
        first = boundary_node_ids(first_id)
        second = boundary_node_ids(second_id)
        first.sort(key=lambda node: nodes[node][coordinate])
        second.sort(key=lambda node: nodes[node][coordinate])
        periodic_counts[name] = [len(first), len(second)]
        if len(first) != len(second):
            periodic_errors[name] = math.inf
        else:
            periodic_errors[name] = max(
                abs(float(nodes[a][coordinate] - nodes[b][coordinate]))
                for a, b in zip(first, second)
            )

    used_nodes = {node for element in triangles for node in element["nodes"]}
    boundary_edge_count = sum(len(values) == 1 for values in edge_grains.values())
    nonmanifold_edges = sum(len(values) > 2 for values in edge_grains.values())
    shared_grain_edges = sum(len(set(values)) == 2 for values in edge_grains.values())
    checks = {
        "consecutive_grain_blocks": sorted(block_counts)
        == list(range(1, expected_grain_count + 1)),
        "all_nodes_used": used_nodes == set(nodes),
        "positive_elements": min(areas, default=0.0) > 0.0,
        "domain_area": abs(sum(areas) - domain_size_um**2) < 1.0e-6,
        "four_boundary_sets": set(boundary_elements) == {101, 102, 103, 104},
        "periodic_node_counts": all(
            first == second for first, second in periodic_counts.values()
        ),
        "periodic_coordinates": max(periodic_errors.values(), default=math.inf)
        < 1.0e-9,
        "manifold": nonmanifold_edges == 0,
        "boundary_topology": boundary_edge_count == len(boundaries),
        "shared_grain_boundaries": shared_grain_edges > 0,
        # The fixed seed-2026 Voronoi geometry contains one acute internal grain vertex.  Both
        # cut-origin studies and independent remeshing retain a quality of about 0.101 there;
        # rejecting below 0.20 would silently change the prescribed grain geometry.
        "acceptable_minimum_quality": min(qualities, default=0.0)
        > minimum_quality_threshold,
    }
    metrics: dict[str, object] = {
        "mesh": str(path),
        "expected_grain_count": expected_grain_count,
        "domain_size_um": domain_size_um,
        "node_count": len(nodes),
        "element_count": len(triangles),
        # MOOSE's default Galerkin rule is order 2*n+1.  SECOND-order displacement
        # variables therefore use a fifth-order triangular Gauss/Dunavant rule
        # with seven points per TRI6 element.
        "quadrature_point_count_TRI6_default_fifth_order": 7 * len(triangles),
        "grain_element_counts": {
            str(key): block_counts[key] for key in range(1, expected_grain_count + 1)
        },
        "grain_area_um2": {
            str(key): block_areas[key] for key in range(1, expected_grain_count + 1)
        },
        "grain_area_fraction": {
            str(key): block_areas[key] / domain_size_um**2
            for key in range(1, expected_grain_count + 1)
        },
        "domain_area_um2": sum(areas),
        "minimum_triangle_quality": min(qualities),
        "mean_triangle_quality": float(np.mean(qualities)),
        "maximum_aspect_ratio": max(aspects),
        "minimum_quality_threshold": minimum_quality_threshold,
        "meets_original_reference_quality": min(qualities, default=0.0) > 0.10,
        "boundary_edge_count": boundary_edge_count,
        "shared_grain_boundary_edge_count": shared_grain_edges,
        "nonmanifold_edge_count": nonmanifold_edges,
        "boundary_element_counts": {
            name: len(boundary_elements[identifier])
            for name, identifier in (
                ("left", 101),
                ("right", 102),
                ("bottom", 103),
                ("top", 104),
            )
        },
        "periodic_node_counts": periodic_counts,
        "maximum_periodic_coordinate_error_um": max(periodic_errors.values()),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-grains", type=int, default=20)
    parser.add_argument("--domain-size", type=float, default=200.0)
    parser.add_argument("--minimum-quality", type=float, default=0.10)
    args = parser.parse_args()
    result = validate_mesh(
        args.mesh,
        minimum_quality_threshold=args.minimum_quality,
        expected_grain_count=args.expected_grains,
        domain_size_um=args.domain_size,
    )
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(
            f"Mesh validation failed for {args.mesh}: {result['checks']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
