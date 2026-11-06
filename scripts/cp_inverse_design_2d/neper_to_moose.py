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


def _clean_polygon(polygon: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    clean: list[tuple[float, float]] = []
    for x_value, y_value in polygon:
        if abs(x_value) < 1.0e-9:
            x_value = 0.0
        elif abs(x_value - DOMAIN_SIZE) < 1.0e-9:
            x_value = DOMAIN_SIZE
        if abs(y_value) < 1.0e-9:
            y_value = 0.0
        elif abs(y_value - DOMAIN_SIZE) < 1.0e-9:
            y_value = DOMAIN_SIZE
        point = (round(x_value, COORDINATE_DIGITS), round(y_value, COORDINATE_DIGITS))
        if not clean or point != clean[-1]:
            clean.append(point)
    if len(clean) > 1 and clean[0] == clean[-1]:
        clean.pop()
    if len(clean) >= 3 and _signed_area(clean) < 0.0:
        clean.reverse()
    return clean


def read_neper_faces(path: Path) -> list[tuple[int, list[tuple[float, float]]]]:
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
    if sorted(grain for grain, _ in faces) != list(range(1, 21)):
        raise RuntimeError("Expected exactly 20 Neper grain faces with IDs 1--20")
    return faces


def fundamental_pieces(
    faces: list[tuple[int, list[tuple[float, float]]]],
) -> list[tuple[int, list[tuple[float, float]]]]:
    pieces: list[tuple[int, list[tuple[float, float]]]] = []
    length = DOMAIN_SIZE
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
                clipped = _clean_polygon(clipped)
                if len(clipped) >= 3 and abs(_signed_area(clipped)) > 1.0e-8:
                    pieces.append((grain_id, clipped))
    total_area = sum(_signed_area(polygon) for _, polygon in pieces)
    if abs(total_area - length * length) > 1.0e-6:
        raise RuntimeError(
            f"Clipped Neper geometry does not conserve area: {total_area}"
        )
    return pieces


def write_gmsh_geometry(
    tessellation: Path, output: Path, characteristic_length: float
) -> dict[str, object]:
    pieces = fundamental_pieces(read_neper_faces(tessellation))
    point_ids: dict[tuple[float, float], int] = {}
    point_coordinates: list[tuple[float, float]] = []
    line_ids: dict[tuple[int, int], int] = {}
    lines: list[tuple[int, int]] = []
    loops: list[list[int]] = []
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
        loop = [
            oriented_line(first, second)
            for first, second in zip(points, points[1:] + points[:1])
        ]
        loops.append(loop)
        grain_surfaces[grain_id].append(len(loops))

    sides: dict[str, list[int]] = {
        name: [] for name in ("left", "right", "bottom", "top")
    }
    for identifier, (first, second) in enumerate(lines, start=1):
        a = point_coordinates[first - 1]
        b = point_coordinates[second - 1]
        if a[0] == b[0] == 0.0:
            sides["left"].append(identifier)
        elif a[0] == b[0] == DOMAIN_SIZE:
            sides["right"].append(identifier)
        elif a[1] == b[1] == 0.0:
            sides["bottom"].append(identifier)
        elif a[1] == b[1] == DOMAIN_SIZE:
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
    for grain_id in range(1, 21):
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
            f"Periodic Curve {{{secondary}}} = {{{master}}} Translate {{{DOMAIN_SIZE}, 0, 0}};"
        )
    for master, secondary in zip(sides["bottom"], sides["top"]):
        text.append(
            f"Periodic Curve {{{secondary}}} = {{{master}}} Translate {{0, {DOMAIN_SIZE}, 0}};"
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
        "neper_grains": 20,
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
) -> dict[str, object]:
    geometry = output_mesh.with_suffix(".geo")
    geometry_metrics = write_gmsh_geometry(
        tessellation, geometry, characteristic_length
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
    args = parser.parse_args()
    print(
        json.dumps(
            convert(
                args.tessellation,
                args.output_mesh,
                args.characteristic_length,
                args.gmsh,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
