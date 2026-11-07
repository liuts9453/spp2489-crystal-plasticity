#!/usr/bin/env python3
"""Convert an exact Neper periodic tessellation to a conforming periodic MOOSE mesh.

Neper represents periodic cells as uncut polygons on a torus.  This converter clips periodic
copies of those exact polygons to the fundamental square, preserves the original Neper cell ID,
and asks Gmsh to create a boundary-fitted TRI6 mesh.  Opposite boundary curves are constrained as
periodic copies during meshing, while the Gmsh `$Periodic` section is removed from the MOOSE file;
MOOSE applies the actual displacement periodicity from the named sidesets.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from common import GMSH_EXECUTABLE, write_json


DOMAIN_SIZE = 200.0
COORDINATE_DIGITS = 9


def _corner_quality(
    first: tuple[float, float],
    middle: tuple[float, float],
    last: tuple[float, float],
) -> float:
    lengths_squared = (
        (first[0] - middle[0]) ** 2 + (first[1] - middle[1]) ** 2,
        (last[0] - middle[0]) ** 2 + (last[1] - middle[1]) ** 2,
        (last[0] - first[0]) ** 2 + (last[1] - first[1]) ** 2,
    )
    twice_area = abs(
        (first[0] - middle[0]) * (last[1] - middle[1])
        - (first[1] - middle[1]) * (last[0] - middle[0])
    )
    denominator = sum(lengths_squared)
    return 2.0 * math.sqrt(3.0) * twice_area / denominator if denominator else 0.0


def _smooth_shared_vertices(
    coordinates: list[tuple[float, float]],
    polygon_points: list[list[int]],
    *,
    domain_size: float,
    passes: int,
    maximum_move_fraction: float,
    activation_quality_threshold: float,
) -> dict[str, object]:
    """Improve acute triple junctions without changing the planar graph topology.

    A trial move is accepted only when all incident polygons remain convex with
    positive area and the worst local corner quality increases.  Domain-boundary
    vertices remain fixed, so the exact opposite-boundary segmentation and node
    pairing are preserved.
    """
    if passes < 0:
        raise ValueError("geometric smoothing passes must be non-negative")
    if not 0.0 <= maximum_move_fraction <= 0.5:
        raise ValueError("maximum vertex move fraction must lie in [0, 0.5]")
    if not 0.0 < activation_quality_threshold <= 1.0:
        raise ValueError("geometric smoothing activation threshold must lie in (0, 1]")
    original = list(coordinates)
    incidence: dict[int, list[int]] = defaultdict(list)
    neighbors: dict[int, set[int]] = defaultdict(set)
    for polygon_index, polygon in enumerate(polygon_points):
        for index, point in enumerate(polygon):
            incidence[point].append(polygon_index)
            neighbors[point].update(
                (polygon[index - 1], polygon[(index + 1) % len(polygon)])
            )

    def polygon_coordinates(index: int) -> list[tuple[float, float]]:
        return [coordinates[point - 1] for point in polygon_points[index]]

    def valid_convex_polygon(index: int) -> bool:
        polygon = polygon_coordinates(index)
        if _signed_area(polygon) <= 1.0e-10:
            return False
        scale = max(domain_size, 1.0)
        tolerance = -1.0e-11 * scale * scale
        for first, middle, last in zip(
            polygon[-1:] + polygon[:-1],
            polygon,
            polygon[1:] + polygon[:1],
        ):
            cross = (middle[0] - first[0]) * (last[1] - middle[1]) - (
                middle[1] - first[1]
            ) * (last[0] - middle[0])
            if cross < tolerance:
                return False
        return True

    def point_score(point: int) -> float:
        values = []
        for polygon_index in set(incidence[point]):
            identifiers = polygon_points[polygon_index]
            index = identifiers.index(point)
            polygon = polygon_coordinates(polygon_index)
            for corner in (index - 1, index, (index + 1) % len(polygon)):
                values.append(
                    _corner_quality(
                        polygon[corner - 1],
                        polygon[corner],
                        polygon[(corner + 1) % len(polygon)],
                    )
                )
        return min(values, default=0.0)

    moved = 0
    for _ in range(passes):
        improved_this_pass = 0
        candidates = []
        for point, polygon_indices in incidence.items():
            x_value, y_value = coordinates[point - 1]
            if (
                x_value in (0.0, domain_size)
                or y_value in (0.0, domain_size)
                or len(set(polygon_indices)) < 3
            ):
                continue
            affected = set(polygon_indices)
            score = point_score(point)
            if score < activation_quality_threshold:
                candidates.append((score, point, affected))
        for old_score, point, affected in sorted(candidates):
            current = coordinates[point - 1]
            old_score = point_score(point)
            adjacent = [coordinates[item - 1] for item in neighbors[point]]
            minimum_edge = min(math.dist(current, item) for item in adjacent)
            radius = maximum_move_fraction * minimum_edge
            if radius <= 1.0e-12:
                continue
            centroid = (
                sum(item[0] for item in adjacent) / len(adjacent),
                sum(item[1] for item in adjacent) / len(adjacent),
            )
            trial_points = []
            direction = (centroid[0] - current[0], centroid[1] - current[1])
            norm = math.hypot(*direction)
            if norm > 0.0:
                trial_points.extend(
                    (
                        current[0] + fraction * radius * direction[0] / norm,
                        current[1] + fraction * radius * direction[1] / norm,
                    )
                    for fraction in (0.25, 0.5, 1.0)
                )
            for direction_index in range(24):
                angle = 2.0 * math.pi * direction_index / 24.0
                trial_points.append(
                    (
                        current[0] + radius * math.cos(angle),
                        current[1] + radius * math.sin(angle),
                    )
                )
            best = current
            best_score = old_score
            for trial in trial_points:
                coordinates[point - 1] = trial
                if not all(valid_convex_polygon(index) for index in affected):
                    continue
                score = point_score(point)
                if score > best_score + 1.0e-9:
                    best = trial
                    best_score = score
            coordinates[point - 1] = best
            if best != current:
                improved_this_pass += 1
                moved += 1
        if improved_this_pass == 0:
            break
    displacements = [
        math.dist(first, second) for first, second in zip(original, coordinates)
    ]
    return {
        "enabled": passes > 0,
        "passes_requested": passes,
        "accepted_vertex_moves": moved,
        "maximum_vertex_displacement_um": max(displacements, default=0.0),
        "mean_vertex_displacement_um": sum(displacements) / max(len(displacements), 1),
        "maximum_move_fraction_per_pass": maximum_move_fraction,
        "corner_quality_activation_threshold": activation_quality_threshold,
    }


def _signed_area(polygon: list[tuple[float, float]]) -> float:
    return 0.5 * sum(
        a[0] * b[1] - a[1] * b[0] for a, b in zip(polygon, polygon[1:] + polygon[:1])
    )


def _clip_half_plane(
    polygon: list[tuple[float, float]], axis: int, bound: float, keep_greater: bool
) -> list[tuple[float, float]]:
    tolerance = 1.0e-10
    output: list[tuple[float, float]] = []
    for first, second in zip(polygon, polygon[1:] + polygon[:1]):
        value_first = first[axis] - bound
        value_second = second[axis] - bound
        inside_first = (
            value_first >= -tolerance if keep_greater else value_first <= tolerance
        )
        inside_second = (
            value_second >= -tolerance if keep_greater else value_second <= tolerance
        )
        if inside_first:
            output.append(first)
        if inside_first != inside_second:
            fraction = (bound - first[axis]) / (second[axis] - first[axis])
            point = [
                first[0] + fraction * (second[0] - first[0]),
                first[1] + fraction * (second[1] - first[1]),
            ]
            point[axis] = bound
            output.append((point[0], point[1]))
    return output


def _clean_polygon(
    polygon: Iterable[tuple[float, float]], domain_size: float = DOMAIN_SIZE
) -> list[tuple[float, float]]:
    clean: list[tuple[float, float]] = []
    for x_value, y_value in polygon:
        if abs(x_value) < 1.0e-9:
            x_value = 0.0
        elif abs(x_value - domain_size) < 1.0e-9:
            x_value = domain_size
        if abs(y_value) < 1.0e-9:
            y_value = 0.0
        elif abs(y_value - domain_size) < 1.0e-9:
            y_value = domain_size
        point = (round(x_value, COORDINATE_DIGITS), round(y_value, COORDINATE_DIGITS))
        if not clean or point != clean[-1]:
            clean.append(point)
    if len(clean) > 1 and clean[0] == clean[-1]:
        clean.pop()
    if len(clean) >= 3 and _signed_area(clean) < 0.0:
        clean.reverse()
    return clean


def read_neper_faces(
    path: Path, expected_grain_count: int | None = None
) -> list[tuple[int, list[tuple[float, float]]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        vertex_start = lines.index(" **vertex")
        face_start = lines.index(" **face")
    except ValueError as error:
        raise RuntimeError(f"Unsupported Neper tessellation format: {path}") from error
    vertex_count = int(lines[vertex_start + 1])
    vertices: dict[int, tuple[float, float]] = {}
    for line in lines[vertex_start + 2 : vertex_start + 2 + vertex_count]:
        fields = line.split()
        vertices[int(fields[0])] = (float(fields[1]), float(fields[2]))
    face_count = int(lines[face_start + 1])
    faces: list[tuple[int, list[tuple[float, float]]]] = []
    cursor = face_start + 2
    for _ in range(face_count):
        fields = lines[cursor].split()
        grain_id, count = int(fields[0]), int(fields[1])
        faces.append(
            (grain_id, [vertices[int(value)] for value in fields[2 : 2 + count]])
        )
        cursor += 4
    grain_ids = sorted(grain for grain, _ in faces)
    expected = list(range(1, len(faces) + 1))
    if grain_ids != expected:
        raise RuntimeError("Neper grain faces must have consecutive IDs beginning at 1")
    if expected_grain_count is not None and len(faces) != expected_grain_count:
        raise RuntimeError(
            f"Expected {expected_grain_count} Neper grains, found {len(faces)}"
        )
    return faces


def fundamental_pieces(
    faces: list[tuple[int, list[tuple[float, float]]]],
    domain_size: float = DOMAIN_SIZE,
) -> list[tuple[int, list[tuple[float, float]]]]:
    pieces: list[tuple[int, list[tuple[float, float]]]] = []
    length = domain_size
    for grain_id, polygon in faces:
        x_values = [point[0] for point in polygon]
        y_values = [point[1] for point in polygon]
        x_min = math.ceil((min(x_values) - length) / length)
        x_max = math.floor(max(x_values) / length)
        y_min = math.ceil((min(y_values) - length) / length)
        y_max = math.floor(max(y_values) / length)
        for x_shift in range(x_min, x_max + 1):
            for y_shift in range(y_min, y_max + 1):
                clipped = [
                    (x_value - x_shift * length, y_value - y_shift * length)
                    for x_value, y_value in polygon
                ]
                for axis, bound, keep_greater in (
                    (0, 0.0, True),
                    (0, length, False),
                    (1, 0.0, True),
                    (1, length, False),
                ):
                    clipped = _clip_half_plane(clipped, axis, bound, keep_greater)
                clipped = _clean_polygon(clipped, length)
                if len(clipped) >= 3 and abs(_signed_area(clipped)) > 1.0e-8:
                    pieces.append((grain_id, clipped))
    total_area = sum(_signed_area(polygon) for _, polygon in pieces)
    if abs(total_area - length * length) > 1.0e-6:
        raise RuntimeError(
            f"Clipped Neper geometry does not conserve area: {total_area}"
        )
    return pieces


def write_gmsh_geometry(
    tessellation: Path,
    output: Path,
    characteristic_length: float,
    domain_size: float = DOMAIN_SIZE,
    expected_grain_count: int | None = None,
    geometric_smoothing_passes: int = 0,
    maximum_vertex_move_fraction: float = 0.15,
    smoothing_activation_threshold: float = 0.01,
) -> dict[str, object]:
    faces = read_neper_faces(tessellation, expected_grain_count)
    grain_ids = sorted(grain_id for grain_id, _ in faces)
    pieces = fundamental_pieces(faces, domain_size)
    point_ids: dict[tuple[float, float], int] = {}
    point_coordinates: list[tuple[float, float]] = []
    line_ids: dict[tuple[int, int], int] = {}
    lines: list[tuple[int, int]] = []
    loops: list[list[int]] = []
    polygon_points: list[list[int]] = []
    grain_surfaces: dict[int, list[int]] = defaultdict(list)

    def point_id(point: tuple[float, float]) -> int:
        if point not in point_ids:
            point_ids[point] = len(point_coordinates) + 1
            point_coordinates.append(point)
        return point_ids[point]

    def oriented_line(first: int, second: int) -> int:
        canonical = (first, second) if first < second else (second, first)
        if canonical not in line_ids:
            line_ids[canonical] = len(lines) + 1
            lines.append(canonical)
        identifier = line_ids[canonical]
        return identifier if canonical == (first, second) else -identifier

    for grain_id, polygon in pieces:
        points = [point_id(point) for point in polygon]
        polygon_points.append(points)
        loop = [
            oriented_line(first, second)
            for first, second in zip(points, points[1:] + points[:1])
        ]
        loops.append(loop)
        grain_surfaces[grain_id].append(len(loops))

    smoothing = _smooth_shared_vertices(
        point_coordinates,
        polygon_points,
        domain_size=domain_size,
        passes=geometric_smoothing_passes,
        maximum_move_fraction=maximum_vertex_move_fraction,
        activation_quality_threshold=smoothing_activation_threshold,
    )
    smoothed_area = sum(
        _signed_area([point_coordinates[point - 1] for point in polygon])
        for polygon in polygon_points
    )
    smoothing["domain_area_after_smoothing_um2"] = smoothed_area
    smoothing["domain_area_error_um2"] = smoothed_area - domain_size**2
    if abs(smoothing["domain_area_error_um2"]) > 1.0e-6:
        raise RuntimeError("Geometric smoothing failed to conserve domain area")

    sides: dict[str, list[int]] = {
        name: [] for name in ("left", "right", "bottom", "top")
    }
    for identifier, (first, second) in enumerate(lines, start=1):
        a = point_coordinates[first - 1]
        b = point_coordinates[second - 1]
        if a[0] == b[0] == 0.0:
            sides["left"].append(identifier)
        elif a[0] == b[0] == domain_size:
            sides["right"].append(identifier)
        elif a[1] == b[1] == 0.0:
            sides["bottom"].append(identifier)
        elif a[1] == b[1] == domain_size:
            sides["top"].append(identifier)

    def interval(identifier: int, coordinate: int) -> tuple[float, float]:
        first, second = lines[identifier - 1]
        values = (
            point_coordinates[first - 1][coordinate],
            point_coordinates[second - 1][coordinate],
        )
        return tuple(sorted(values))

    for name in sides:
        coordinate = 1 if name in ("left", "right") else 0
        sides[name].sort(key=lambda identifier: interval(identifier, coordinate))
    x_left = [interval(identifier, 1) for identifier in sides["left"]]
    x_right = [interval(identifier, 1) for identifier in sides["right"]]
    y_bottom = [interval(identifier, 0) for identifier in sides["bottom"]]
    y_top = [interval(identifier, 0) for identifier in sides["top"]]
    if x_left != x_right or y_bottom != y_top:
        raise RuntimeError("Opposite Neper boundary segmentations are not periodic")

    text: list[str] = []
    for identifier, (x_value, y_value) in enumerate(point_coordinates, start=1):
        text.append(
            f"Point({identifier}) = {{{x_value:.12g}, {y_value:.12g}, 0, {characteristic_length:.12g}}};"
        )
    for identifier, (first, second) in enumerate(lines, start=1):
        text.append(f"Line({identifier}) = {{{first}, {second}}};")
    for identifier, loop in enumerate(loops, start=1):
        text.append(f"Curve Loop({identifier}) = {{{', '.join(map(str, loop))}}};")
        text.append(f"Plane Surface({identifier}) = {{{identifier}}};")
    for grain_id in grain_ids:
        surfaces = ", ".join(map(str, grain_surfaces[grain_id]))
        text.append(
            f'Physical Surface("grain_{grain_id}", {grain_id}) = {{{surfaces}}};'
        )
    for identifier, name in zip(
        (101, 102, 103, 104), ("left", "right", "bottom", "top")
    ):
        text.append(
            f'Physical Curve("{name}", {identifier}) = '
            f'{{{", ".join(map(str, sides[name]))}}};'
        )
    for master, secondary in zip(sides["left"], sides["right"]):
        text.append(
            f"Periodic Curve {{{secondary}}} = {{{master}}} Translate {{{domain_size}, 0, 0}};"
        )
    for master, secondary in zip(sides["bottom"], sides["top"]):
        text.append(
            f"Periodic Curve {{{secondary}}} = {{{master}}} Translate {{0, {domain_size}, 0}};"
        )
    text.extend(
        [
            "Mesh.MshFileVersion = 2.2;",
            "Mesh.ElementOrder = 2;",
            "Mesh.SecondOrderLinear = 1;",
            "Mesh.HighOrderOptimize = 0;",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(text) + "\n", encoding="utf-8")
    return {
        "neper_grains": len(grain_ids),
        "grain_ids": grain_ids,
        "domain_size_um": float(domain_size),
        "fundamental_domain_pieces": len(pieces),
        "fundamental_domain_area_um2": sum(
            _signed_area(polygon) for _, polygon in pieces
        ),
        "geometric_points": len(point_coordinates),
        "geometric_lines": len(lines),
        "boundary_segments": {
            name: len(identifiers) for name, identifiers in sides.items()
        },
        "opposite_boundary_geometry_matches": True,
        "geometric_smoothing": smoothing,
    }


def _remove_gmsh_periodic_section(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if "$Periodic" not in lines:
        return
    first = lines.index("$Periodic")
    last = lines.index("$EndPeriodic")
    path.write_text(
        "\n".join(lines[:first] + lines[last + 1 :]) + "\n", encoding="utf-8"
    )


def convert(
    tessellation: Path,
    output_mesh: Path,
    characteristic_length: float,
    gmsh: Path = GMSH_EXECUTABLE,
    domain_size: float = DOMAIN_SIZE,
    expected_grain_count: int | None = None,
    geometric_smoothing_passes: int = 0,
    maximum_vertex_move_fraction: float = 0.15,
    smoothing_activation_threshold: float = 0.01,
) -> dict[str, object]:
    geometry = output_mesh.with_suffix(".geo")
    geometry_metrics = write_gmsh_geometry(
        tessellation,
        geometry,
        characteristic_length,
        domain_size,
        expected_grain_count,
        geometric_smoothing_passes,
        maximum_vertex_move_fraction,
        smoothing_activation_threshold,
    )
    command = [
        str(gmsh),
        str(geometry),
        "-2",
        "-format",
        "msh2",
        "-order",
        "2",
        "-o",
        str(output_mesh),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path = output_mesh.with_suffix(".gmsh.log")
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not output_mesh.exists():
        raise RuntimeError(f"Gmsh conversion failed; see {log_path}")
    _remove_gmsh_periodic_section(output_mesh)
    result = {
        **geometry_metrics,
        "tessellation": str(tessellation),
        "geometry": str(geometry),
        "mesh": str(output_mesh),
        "characteristic_length_um": characteristic_length,
        "gmsh_command": command,
        "gmsh_log": str(log_path),
    }
    write_json(output_mesh.with_suffix(".conversion.json"), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tessellation", type=Path)
    parser.add_argument("output_mesh", type=Path)
    parser.add_argument("--characteristic-length", type=float, required=True)
    parser.add_argument("--gmsh", type=Path, default=GMSH_EXECUTABLE)
    parser.add_argument("--domain-size", type=float, default=DOMAIN_SIZE)
    parser.add_argument("--expected-grains", type=int)
    parser.add_argument("--geometric-smoothing-passes", type=int, default=0)
    parser.add_argument("--maximum-vertex-move-fraction", type=float, default=0.15)
    parser.add_argument("--smoothing-activation-threshold", type=float, default=0.01)
    args = parser.parse_args()
    print(
        json.dumps(
            convert(
                args.tessellation,
                args.output_mesh,
                args.characteristic_length,
                args.gmsh,
                args.domain_size,
                args.expected_grains,
                args.geometric_smoothing_passes,
                args.maximum_vertex_move_fraction,
                args.smoothing_activation_threshold,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
