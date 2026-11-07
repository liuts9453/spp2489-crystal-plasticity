#!/usr/bin/env python3
"""Measure periodic grain morphology from unwrapped Neper scalar cell polygons."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ID2_SCRIPT_DIR = SCRIPT_DIR.parent / "cp_inverse_design_2d"
for directory in (SCRIPT_DIR, ID2_SCRIPT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from morphology_common import baseline_texture  # noqa: E402
from neper_to_moose import read_neper_faces  # noqa: E402
from texture_parameterization import (  # noqa: E402
    moose_bunge_rotation,
    orientations_for_design,
)


def _polygon_statistics(polygon: list[tuple[float, float]]) -> dict[str, float]:
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
    edges = following - points
    perimeter = float(np.sum(np.linalg.norm(edges, axis=1)))
    area = abs(signed_area)
    return {
        "area_um2": area,
        "perimeter_um": perimeter,
        "centroid_x_unwrapped_um": centroid_x,
        "centroid_y_unwrapped_um": centroid_y,
        "variance_x_um2": float(covariance[0, 0]),
        "variance_y_um2": float(covariance[1, 1]),
        "covariance_xy_um2": float(covariance[0, 1]),
        "directional_extent_x_um": 4.0 * math.sqrt(float(covariance[0, 0])),
        "directional_extent_y_um": 4.0 * math.sqrt(float(covariance[1, 1])),
        "principal_major_length_um": major_length,
        "principal_minor_length_um": minor_length,
        "principal_aspect_ratio": major_length / minor_length,
        "principal_axis_angle_deg": principal_angle,
        "circularity": 4.0 * math.pi * area / perimeter**2,
    }


def _orientation_descriptor(
    grain_rows: list[dict[str, float]], eta: float, theta_deg: float
) -> dict[str, object]:
    eulers = orientations_for_design(eta, theta_deg)[0]
    weights = np.asarray([row["area_fraction"] for row in grain_rows])
    dyads = []
    rotations = []
    for euler in eulers:
        rotation = moose_bunge_rotation(*euler)
        direction = rotation.T @ np.array([1.0, 0.0, 0.0])
        dyads.append(np.outer(direction, direction))
        rotations.append(rotation)
    dyads_array = np.asarray(dyads)
    rotations_array = np.asarray(rotations)
    return {
        "definition": "sample-X direction in each crystal frame; dyadic removes sign ambiguity",
        "area_weighted_sample_x_crystal_dyad": np.tensordot(
            weights, dyads_array, axes=(0, 0)
        ).tolist(),
        "unweighted_sample_x_crystal_dyad": np.mean(dyads_array, axis=0).tolist(),
        "area_weighted_rotation_matrix_mean": np.tensordot(
            weights, rotations_array, axes=(0, 0)
        ).tolist(),
        "unweighted_rotation_matrix_mean": np.mean(rotations_array, axis=0).tolist(),
    }


def measure(
    tessellation: Path,
    *,
    input_a: float,
    seed: int,
    eta: float | None = None,
    theta_deg: float | None = None,
    csv_path: Path | None = None,
) -> dict[str, object]:
    if eta is None or theta_deg is None:
        eta, theta_deg = baseline_texture()
    faces = read_neper_faces(tessellation)
    rows: list[dict[str, float]] = []
    for grain_id, polygon in faces:
        row = {"grain_id": grain_id, **_polygon_statistics(polygon)}
        rows.append(row)
    total_area = sum(row["area_um2"] for row in rows)
    if abs(total_area - 40000.0) > 1.0e-6:
        raise RuntimeError(f"Periodic unwrapped grain area is {total_area}, not 40000")
    for row in rows:
        row["area_fraction"] = row["area_um2"] / total_area
    areas = np.asarray([row["area_um2"] for row in rows])
    weighted_variance_x = sum(
        row["area_um2"] * row["variance_x_um2"] for row in rows
    ) / total_area
    weighted_variance_y = sum(
        row["area_um2"] * row["variance_y_um2"] for row in rows
    ) / total_area
    angles = np.radians([2.0 * row["principal_axis_angle_deg"] for row in rows])
    weights = np.asarray([row["area_fraction"] for row in rows])
    axial_mean = 0.5 * math.degrees(
        math.atan2(float(weights @ np.sin(angles)), float(weights @ np.cos(angles)))
    )
    summary: dict[str, object] = {
        "tessellation": str(tessellation.resolve()),
        "input_a": float(input_a),
        "geometry_seed": int(seed),
        "grain_count": len(rows),
        "domain_area_um2": total_area,
        "mean_grain_area_um2": float(np.mean(areas)),
        "area_standard_deviation_um2": float(np.std(areas)),
        "area_coefficient_of_variation": float(np.std(areas) / np.mean(areas)),
        "minimum_grain_area_um2": float(np.min(areas)),
        "maximum_grain_area_um2": float(np.max(areas)),
        "area_weighted_directional_extent_ratio_x_over_y": math.sqrt(
            weighted_variance_x / weighted_variance_y
        ),
        "area_weighted_principal_aspect_ratio": sum(
            row["area_fraction"] * row["principal_aspect_ratio"] for row in rows
        ),
        "area_weighted_principal_axis_angle_deg": axial_mean % 180.0,
        "area_weighted_mean_circularity": sum(
            row["area_fraction"] * row["circularity"] for row in rows
        ),
        "texture_eta": float(eta),
        "texture_theta_deg": float(theta_deg),
        "orientation_statistics": _orientation_descriptor(rows, eta, theta_deg),
        "periodic_measurement_method": (
            "Neper scalar cell polygons are measured in their unwrapped torus representation; "
            "boundary-crossing grains are not split in the statistics"
        ),
        "grains": rows,
    }
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tessellation", type=Path)
    parser.add_argument("--input-a", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--theta", type=float)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            measure(
                args.tessellation,
                input_a=args.input_a,
                seed=args.seed,
                eta=args.eta,
                theta_deg=args.theta,
                csv_path=args.csv,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
