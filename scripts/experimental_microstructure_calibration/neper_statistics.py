#!/usr/bin/env python3
"""Generic periodic Neper cell parsing and polygon statistics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


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
    identifiers = sorted(grain for grain, _ in faces)
    if identifiers != list(range(1, face_count + 1)):
        raise RuntimeError("Neper cell IDs are not contiguous from one")
    return faces


def periodicity_metadata(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        periodicity_start = lines.index(" **periodicity")
        general_offset = lines[periodicity_start:].index("  *general")
    except ValueError as error:
        raise RuntimeError(f"Tessellation is not marked periodic: {path}") from error
    cursor = periodicity_start + general_offset
    flags = [int(value) for value in lines[cursor + 1].split()]
    lengths = [float(value) for value in lines[cursor + 2].split()]
    if flags[:2] != [1, 1]:
        raise RuntimeError(f"Expected x/y periodicity, found {flags} in {path}")
    return {
        "periodic_flags_xyz": flags,
        "period_lengths_xyz": lengths,
        "xy_periodic": True,
    }


def polygon_statistics(polygon: list[tuple[float, float]]) -> dict[str, float]:
    points = np.asarray(polygon, dtype=float)
    following = np.roll(points, -1, axis=0)
    cross = points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]
    signed_area = 0.5 * float(np.sum(cross))
    if abs(signed_area) <= 1.0e-12:
        raise RuntimeError("Degenerate Neper grain polygon")
    centroid_x = float(np.sum((points[:, 0] + following[:, 0]) * cross)) / (
        6.0 * signed_area
    )
    centroid_y = float(np.sum((points[:, 1] + following[:, 1]) * cross)) / (
        6.0 * signed_area
    )
    second_xx = float(
        np.sum(
            (points[:, 1] ** 2 + points[:, 1] * following[:, 1] + following[:, 1] ** 2)
            * cross
        )
        / 12.0
    )
    second_yy = float(
        np.sum(
            (points[:, 0] ** 2 + points[:, 0] * following[:, 0] + following[:, 0] ** 2)
            * cross
        )
        / 12.0
    )
    second_xy = float(
        np.sum(
            (
                2.0 * points[:, 0] * points[:, 1]
                + points[:, 0] * following[:, 1]
                + following[:, 0] * points[:, 1]
                + 2.0 * following[:, 0] * following[:, 1]
            )
            * cross
        )
        / 24.0
    )
    covariance = np.array(
        [
            [
                second_yy / signed_area - centroid_x**2,
                second_xy / signed_area - centroid_x * centroid_y,
            ],
            [
                second_xy / signed_area - centroid_x * centroid_y,
                second_xx / signed_area - centroid_y**2,
            ],
        ]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(eigenvalues[0]) <= 0.0:
        raise RuntimeError("Non-positive polygon central moment")
    principal = eigenvectors[:, 1]
    principal_angle = math.degrees(math.atan2(principal[1], principal[0])) % 180.0
    major_length = 4.0 * math.sqrt(float(eigenvalues[1]))
    minor_length = 4.0 * math.sqrt(float(eigenvalues[0]))
    perimeter = float(np.sum(np.linalg.norm(following - points, axis=1)))
    area = abs(signed_area)
    return {
        "area_um2": area,
        "perimeter_um": perimeter,
        "centroid_x_unwrapped_um": centroid_x,
        "centroid_y_unwrapped_um": centroid_y,
        "directional_extent_x_um": 4.0 * math.sqrt(float(covariance[0, 0])),
        "directional_extent_y_um": 4.0 * math.sqrt(float(covariance[1, 1])),
        "principal_major_length_um": major_length,
        "principal_minor_length_um": minor_length,
        "principal_aspect_ratio": major_length / minor_length,
        "principal_axis_angle_deg": principal_angle,
        "circularity": 4.0 * math.pi * area / perimeter**2,
    }
