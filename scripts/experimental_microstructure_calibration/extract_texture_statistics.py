#!/usr/bin/env python3
"""Extract cubic-symmetry-aware EBSD texture descriptors and empirical samples."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def moose_bunge_rotation(
    phi1_deg: float, Phi_deg: float, phi2_deg: float
) -> np.ndarray:
    """Exact crystal-to-sample convention used by the existing SPP2489 workflow."""
    phi1, Phi, phi2 = np.radians([phi1_deg, Phi_deg, phi2_deg])
    c1, c2, c3 = np.cos([phi1, Phi, phi2])
    s1, s2, s3 = np.sin([phi1, Phi, phi2])
    return np.array(
        [
            [c1 * c3 - c2 * s1 * s3, c3 * s1 + c1 * c2 * s3, s2 * s3],
            [-c1 * s3 - c2 * c3 * s1, c1 * c2 * c3 - s1 * s3, c3 * s2],
            [s1 * s2, -c1 * s2, c2],
        ]
    )


def proper_cubic_symmetries() -> np.ndarray:
    matrices: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3))
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if np.linalg.det(matrix) > 0.5:
                matrices.append(matrix)
    unique: list[np.ndarray] = []
    for matrix in matrices:
        if not any(np.array_equal(matrix, previous) for previous in unique):
            unique.append(matrix)
    result = np.asarray(unique)
    if result.shape != (24, 3, 3):
        raise RuntimeError(f"Expected 24 proper cubic symmetries, found {result.shape}")
    return result


CUBIC_SYMMETRIES = proper_cubic_symmetries()

POLE_FAMILIES = {
    "100": np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    "110": np.asarray(
        [
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, -1.0],
        ]
    )
    / np.sqrt(2.0),
    "111": np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    / np.sqrt(3.0),
}


def reduce_cubic_direction(direction: np.ndarray) -> np.ndarray:
    """Reduce an unoriented cubic direction to 0 <= u <= v <= w."""
    vector = np.sort(np.abs(np.asarray(direction, dtype=float)))
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("Cannot reduce a zero direction")
    return vector / norm


def ipf_coordinates(rotations: np.ndarray, sample_axis: np.ndarray) -> np.ndarray:
    sample = np.asarray(sample_axis, dtype=float)
    sample /= np.linalg.norm(sample)
    coordinates = []
    for rotation in rotations:
        crystal_direction = rotation.T @ sample
        reduced = reduce_cubic_direction(crystal_direction)
        coordinates.append((reduced[0] / reduced[2], reduced[1] / reduced[2]))
    return np.asarray(coordinates)


def pole_figure_coordinates(rotations: np.ndarray, family: str) -> np.ndarray:
    """Upper-hemisphere stereographic coordinates for a cubic pole family."""
    crystal_directions = POLE_FAMILIES[family]
    sample_directions = np.einsum("nij,kj->nki", rotations, crystal_directions)
    sign = np.where(sample_directions[:, :, 2:3] < 0.0, -1.0, 1.0)
    upper = sample_directions * sign
    return upper[:, :, :2] / (1.0 + upper[:, :, 2:3])


def _pole_histogram_descriptor(
    rotations: np.ndarray, grain_weights: np.ndarray, family: str, bins: int
) -> dict[str, object]:
    coordinates = pole_figure_coordinates(rotations, family)
    weights = np.repeat(grain_weights, coordinates.shape[1])
    flattened = coordinates.reshape(-1, 2)
    histogram, x_edges, y_edges = np.histogram2d(
        flattened[:, 0],
        flattened[:, 1],
        bins=(bins, bins),
        range=((-1.0, 1.0), (-1.0, 1.0)),
        weights=weights,
    )
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    outside = x_centers[:, None] ** 2 + y_centers[None, :] ** 2 > 1.0
    histogram[outside] = 0.0
    histogram /= np.sum(histogram)
    return {
        "family": family,
        "projection": "upper-hemisphere stereographic",
        "bins": bins,
        "x_edges": x_edges.tolist(),
        "y_edges": y_edges.tolist(),
        "probability": histogram.tolist(),
    }


def _histogram_descriptor(
    coordinates: np.ndarray, weights: np.ndarray, bins: int
) -> dict[str, object]:
    histogram, x_edges, y_edges = np.histogram2d(
        coordinates[:, 0],
        coordinates[:, 1],
        bins=(bins, bins),
        range=((0.0, 1.0), (0.0, 1.0)),
        weights=weights,
    )
    histogram /= np.sum(histogram)
    return {
        "bins": bins,
        "x_edges": x_edges.tolist(),
        "y_edges": y_edges.tolist(),
        "probability": histogram.tolist(),
    }


def rotations_from_frame(frame: pd.DataFrame) -> np.ndarray:
    rotations = np.asarray(
        [
            moose_bunge_rotation(row.phi1_deg, row.Phi_deg, row.phi2_deg)
            for row in frame.itertuples()
        ]
    )
    orthogonality = np.max(
        np.abs(np.einsum("nji,njk->nik", rotations, rotations) - np.eye(3))
    )
    determinant = np.max(np.abs(np.linalg.det(rotations) - 1.0))
    if max(float(orthogonality), float(determinant)) > 1.0e-10:
        raise RuntimeError("Bunge Euler conversion did not produce proper rotations")
    return rotations


def extract_texture_statistics(
    frame: pd.DataFrame, bins: int = 24
) -> dict[str, object]:
    phase_results: dict[str, object] = {}
    for phase, group in frame.groupby(frame["phase"].astype(str), sort=True):
        rotations = rotations_from_frame(group)
        area = group["grain_area_um2"].to_numpy(dtype=float)
        number_weights = np.ones(len(group), dtype=float) / len(group)
        area_weights = area / np.sum(area)
        axes = {
            "sample_x": np.array([1.0, 0.0, 0.0]),
            "sample_y": np.array([0.0, 1.0, 0.0]),
            "sample_z": np.array([0.0, 0.0, 1.0]),
        }
        ipf = {}
        for name, axis in axes.items():
            coordinates = ipf_coordinates(rotations, axis)
            ipf[name] = {
                "number_weighted": _histogram_descriptor(
                    coordinates, number_weights, bins
                ),
                "area_weighted": _histogram_descriptor(coordinates, area_weights, bins),
            }
        pole_figures = {
            family: {
                "number_weighted": _pole_histogram_descriptor(
                    rotations, number_weights, family, bins
                ),
                "area_weighted": _pole_histogram_descriptor(
                    rotations, area_weights, family, bins
                ),
            }
            for family in POLE_FAMILIES
        }
        phase_results[str(phase)] = {
            "grain_count": int(len(group)),
            "area_fraction_within_export": float(
                np.sum(area) / frame["grain_area_um2"].sum()
            ),
            "crystal_symmetry": "cubic, 24 proper rotations",
            "rotation_convention": (
                "MOOSE crystal-to-sample crysrot from passive Bunge Euler angles; "
                "sample directions are mapped by Q^T"
            ),
            "ipf_fundamental_sector_histograms": ipf,
            "pole_figure_histograms": pole_figures,
            "orientation_sampler": (
                "Empirical Bunge rotations are sampled as complete SO(3) objects; "
                "Euler components are never sampled independently."
            ),
        }
    return {
        "phase_separated": True,
        "phase_statistics": phase_results,
        "single_phase_cp_warning": (
            "Only the experimentally identified FCC phase may be passed to the current "
            "single-phase FCC MOOSE model; BCC orientations must remain separate."
        ),
    }


def stratified_empirical_orientation_sample(
    frame: pd.DataFrame,
    count: int,
    seed: int,
    *,
    area_weighted: bool = False,
) -> pd.DataFrame:
    """Systematic probability sampling of complete experimental orientations."""
    if count <= 0 or frame.empty:
        raise ValueError("Orientation sample requires positive count and nonempty data")
    weights = (
        frame["grain_area_um2"].to_numpy(dtype=float)
        if area_weighted
        else np.ones(len(frame), dtype=float)
    )
    weights /= np.sum(weights)
    cumulative = np.cumsum(weights)
    rng = np.random.default_rng(seed)
    offset = rng.uniform(0.0, 1.0 / count)
    positions = offset + np.arange(count, dtype=float) / count
    indices = np.searchsorted(cumulative, positions, side="right")
    sampled = (
        frame.iloc[np.minimum(indices, len(frame) - 1)].copy().reset_index(drop=True)
    )
    sampled.insert(0, "synthetic_grain_id", np.arange(1, count + 1))
    sampled["orientation_sampling_seed"] = seed
    sampled["orientation_sampling_weight"] = "area" if area_weighted else "number"
    return sampled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cleaned_csv", type=Path)
    parser.add_argument("--bins", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = extract_texture_statistics(pd.read_csv(args.cleaned_csv), args.bins)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
