#!/usr/bin/env python3
"""Generate periodic Neper RVEs from empirical EBSD morphology and texture targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common import NEPER, sha256, write_json
from neper_statistics import periodicity_metadata, polygon_statistics, read_neper_faces


@dataclass(frozen=True)
class NeperMorphologyParameters:
    diameter_log_spread_scale: float = 1.0
    directional_ratio_x_over_y: float = 1.0
    aspect_ratio_proxy_scale: float | None = None
    input_size_shape_rank_correlation: float | None = None
    one_minus_circularity_mean: float | None = None
    one_minus_circularity_standard_deviation: float | None = None

    def validate(self) -> None:
        if not 0.25 <= self.diameter_log_spread_scale <= 2.5:
            raise ValueError("diameter_log_spread_scale must lie in [0.25, 2.5]")
        if not 0.25 <= self.directional_ratio_x_over_y <= 4.0:
            raise ValueError("directional_ratio_x_over_y must lie in [0.25, 4]")
        if self.aspect_ratio_proxy_scale is not None and not (
            0.05 <= self.aspect_ratio_proxy_scale <= 4.0
        ):
            raise ValueError("aspect_ratio_proxy_scale must lie in [0.05, 4]")
        if self.aspect_ratio_proxy_scale is not None and (
            self.one_minus_circularity_mean is not None
            or self.one_minus_circularity_standard_deviation is not None
        ):
            raise ValueError(
                "Cell-wise aspect-ratio proxy and distribution-level circularity "
                "targets are mutually exclusive"
            )
        if self.input_size_shape_rank_correlation is not None and not (
            -1.0 <= self.input_size_shape_rank_correlation <= 1.0
        ):
            raise ValueError("input_size_shape_rank_correlation must lie in [-1, 1]")
        if (
            self.input_size_shape_rank_correlation is not None
            and self.aspect_ratio_proxy_scale is None
        ):
            raise ValueError(
                "input_size_shape_rank_correlation requires cell-wise shape targets"
            )
        if (self.one_minus_circularity_mean is None) != (
            self.one_minus_circularity_standard_deviation is None
        ):
            raise ValueError("Both 1-circularity parameters must be supplied together")
        if self.one_minus_circularity_mean is not None:
            if self.one_minus_circularity_mean <= 0.0:
                raise ValueError("Mean 1-circularity must be positive")
            if self.one_minus_circularity_standard_deviation <= 0.0:
                raise ValueError("1-circularity standard deviation must be positive")


def choose_grain_count(frame: pd.DataFrame, side_length_um: float) -> int:
    mean_area = float(np.mean(frame["grain_area_um2"]))
    if mean_area <= 0.0:
        raise ValueError("Experimental mean grain area must be positive")
    return max(4, int(round(side_length_um**2 / mean_area)))


def _systematic_empirical_values(
    values: np.ndarray, count: int, seed: int
) -> np.ndarray:
    ordered = np.sort(np.asarray(values, dtype=float))
    rng = np.random.default_rng(seed)
    positions = (np.arange(count) + rng.uniform(0.0, 1.0)) / count
    source_positions = (np.arange(len(ordered)) + 0.5) / len(ordered)
    return np.interp(positions, source_positions, ordered)


def diameter_targets(
    frame: pd.DataFrame,
    count: int,
    seed: int,
    log_spread_scale: float,
) -> np.ndarray:
    sample = _systematic_empirical_values(
        frame["equivalent_diameter_um"].to_numpy(dtype=float), count, seed
    )
    log_values = np.log(sample)
    warped = np.exp(
        np.mean(log_values) + log_spread_scale * (log_values - np.mean(log_values))
    )
    # Neper rescales absolute sizes to satisfy domain area and cell count. Unit
    # geometric mean makes the intended input shape explicit and portable.
    return warped / np.exp(np.mean(np.log(warped)))


def _ellipse_one_minus_circularity(aspect_ratio: np.ndarray) -> np.ndarray:
    """Map ellipse aspect ratios to a bounded circularity proxy.

    The EBSD CSVs provide fitted-ellipse axes but no grain perimeter.  The
    Ramanujan ellipse-perimeter approximation therefore supplies a transparent
    proxy for Neper's 2-D circularity objective.  It is used only to transmit
    the observed cell-wise size--shape pairing; achieved polygon AR is always
    measured independently after generation.
    """
    ratio = np.maximum(np.asarray(aspect_ratio, dtype=float), 1.0)
    h = np.square((ratio - 1.0) / (ratio + 1.0))
    perimeter = (
        np.pi * (ratio + 1.0) * (1.0 + 3.0 * h / (10.0 + np.sqrt(4.0 - 3.0 * h)))
    )
    circularity = 4.0 * np.pi**2 * ratio / np.square(perimeter)
    return np.clip(1.0 - circularity, 1.0e-5, 0.999)


def joint_morphology_targets(
    frame: pd.DataFrame,
    count: int,
    seed: int,
    log_spread_scale: float,
    aspect_ratio_proxy_scale: float,
    input_size_shape_rank_correlation: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired ECD and shape targets from stratified experimental rows."""
    ordered = frame.sort_values("equivalent_diameter_um").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    positions = (np.arange(count) + rng.uniform(0.0, 1.0)) / count
    indices = np.minimum(
        np.floor(positions * len(ordered)).astype(int), len(ordered) - 1
    )
    selected = ordered.iloc[indices]
    diameters = selected["equivalent_diameter_um"].to_numpy(dtype=float)
    log_values = np.log(diameters)
    diameter_target = np.exp(
        np.mean(log_values) + log_spread_scale * (log_values - np.mean(log_values))
    )
    diameter_target /= np.exp(np.mean(np.log(diameter_target)))
    shape_target = _ellipse_one_minus_circularity(
        selected["aspect_ratio_recomputed"].to_numpy(dtype=float)
    )
    shape_target = np.clip(aspect_ratio_proxy_scale * shape_target, 1.0e-5, 0.999)
    if input_size_shape_rank_correlation is not None:
        rho = float(input_size_shape_rank_correlation)
        diameter_ranks = np.argsort(np.argsort(diameter_target)).astype(float)
        diameter_scores = (diameter_ranks - np.mean(diameter_ranks)) / max(
            np.std(diameter_ranks), 1.0
        )
        noise = rng.normal(size=count)
        noise -= np.mean(noise)
        noise /= max(np.std(noise), 1.0e-12)
        copula_score = rho * diameter_scores + np.sqrt(max(0.0, 1.0 - rho**2)) * noise
        reordered = np.empty_like(shape_target)
        reordered[np.argsort(copula_score)] = np.sort(shape_target)
        shape_target = reordered
    return diameter_target, shape_target


def _run(
    command: list[str],
    cwd: Path,
    log: Path,
    *,
    timeout_s: float | None = None,
) -> dict[str, object]:
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "4")
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        elapsed = time.perf_counter() - start
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        log.write_text(
            stdout
            + stderr
            + f"\nSPP2489: Neper exceeded the {timeout_s:g} s candidate limit.\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Neper exceeded the {timeout_s:g} s candidate limit after "
            f"{elapsed:.3f} s; see {log}"
        ) from error
    elapsed = time.perf_counter() - start
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    record = {
        "command": command,
        "returncode": completed.returncode,
        "wall_time_s": elapsed,
        "log_filename": log.name,
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"Neper failed with status {completed.returncode}; see {log}"
        )
    return record


def measure_tessellation(tessellation: Path) -> pd.DataFrame:
    rows = []
    for grain_id, polygon in read_neper_faces(tessellation):
        statistics = polygon_statistics(polygon)
        rows.append(
            {
                "synthetic_grain_id": int(grain_id),
                "grain_area_um2": statistics["area_um2"],
                "equivalent_diameter_um": 2.0 * np.sqrt(statistics["area_um2"] / np.pi),
                "semi_axis_a_um": statistics["principal_major_length_um"] / 2.0,
                "semi_axis_b_um": statistics["principal_minor_length_um"] / 2.0,
                "aspect_ratio_recomputed": statistics["principal_aspect_ratio"],
                "alpha_deg": statistics["principal_axis_angle_deg"],
                "perimeter_um": statistics["perimeter_um"],
                "circularity": statistics["circularity"],
                "directional_extent_x_um": statistics["directional_extent_x_um"],
                "directional_extent_y_um": statistics["directional_extent_y_um"],
            }
        )
    result = pd.DataFrame(rows).sort_values("synthetic_grain_id").reset_index(drop=True)
    if result.empty or np.any(result["grain_area_um2"] <= 0.0):
        raise RuntimeError("Neper output has missing or nonpositive cells")
    return result


def size_conditioned_orientation_sample(
    experimental: pd.DataFrame,
    synthetic: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Match synthetic size ranks to experimental size--orientation records."""
    ordered_experimental = experimental.sort_values(
        "equivalent_diameter_um"
    ).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    count = len(synthetic)
    probability = (np.arange(count) + rng.uniform(0.0, 1.0)) / count
    indices = np.minimum(
        (probability * len(ordered_experimental)).astype(int),
        len(ordered_experimental) - 1,
    )
    selected = ordered_experimental.iloc[indices].reset_index(drop=True)
    synthetic_order = np.argsort(synthetic["equivalent_diameter_um"].to_numpy())
    result = pd.DataFrame(
        index=np.arange(count),
        columns=["phi1_deg", "Phi_deg", "phi2_deg", "source_row", "phase"],
    )
    result.iloc[synthetic_order] = selected[
        ["phi1_deg", "Phi_deg", "phi2_deg", "source_row", "phase"]
    ].to_numpy()
    result.insert(0, "synthetic_grain_id", synthetic["synthetic_grain_id"].to_numpy())
    for column in ("phi1_deg", "Phi_deg", "phi2_deg", "source_row"):
        result[column] = pd.to_numeric(result[column])
    return result


def write_orientation_file(orientations: pd.DataFrame, path: Path) -> None:
    orientations[["phi1_deg", "Phi_deg", "phi2_deg"]].to_csv(
        path, sep=" ", index=False, header=False, float_format="%.15g"
    )


def _experimental_table_hash(frame: pd.DataFrame) -> str:
    columns = [
        "source_row",
        "phase",
        "grain_area_um2",
        "equivalent_diameter_um",
        "phi1_deg",
        "Phi_deg",
        "phi2_deg",
    ]
    encoded = frame[columns].to_csv(index=False, float_format="%.15g").encode()
    return hashlib.sha256(encoded).hexdigest()


def generate_rve(
    experimental: pd.DataFrame,
    output_directory: Path,
    *,
    dataset_key: str,
    phase: str,
    side_length_um: float,
    seed: int,
    parameters: NeperMorphologyParameters,
    grain_count: int | None = None,
    force: bool = False,
    generation_timeout_s: float | None = None,
    morphology_time_limit_s: float | None = None,
) -> dict[str, object]:
    parameters.validate()
    output_directory.mkdir(parents=True, exist_ok=True)
    phase_frame = (
        experimental.copy()
        if str(phase).lower() == "all"
        else experimental.loc[experimental["phase"].astype(str) == str(phase)].copy()
    )
    if phase_frame.empty:
        raise ValueError(f"No grains for requested phase {phase!r}")
    count = grain_count or choose_grain_count(phase_frame, side_length_um)
    if morphology_time_limit_s is not None and morphology_time_limit_s <= 0.0:
        raise ValueError("morphology_time_limit_s must be positive when supplied")
    cache_identity = {
        "dataset": dataset_key,
        "phase": str(phase),
        "side_length_um": float(side_length_um),
        "grain_count": int(count),
        "geometry_seed": int(seed),
        "parameters": asdict(parameters),
        "experimental_table_sha256": _experimental_table_hash(phase_frame),
        "morphology_time_limit_s": morphology_time_limit_s,
    }
    manifest_path = output_directory / "generation_manifest.json"
    previous_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )
    reuse_exact_cache = (
        not force
        and previous_manifest is not None
        and previous_manifest.get("cache_identity") == cache_identity
    )
    target_shape = None
    if parameters.aspect_ratio_proxy_scale is None:
        target_diameters = diameter_targets(
            phase_frame, count, seed, parameters.diameter_log_spread_scale
        )
    else:
        target_diameters, target_shape = joint_morphology_targets(
            phase_frame,
            count,
            seed,
            parameters.diameter_log_spread_scale,
            parameters.aspect_ratio_proxy_scale,
            parameters.input_size_shape_rank_correlation,
        )
    diameter_file = output_directory / "target_diameq_normalized.txt"
    np.savetxt(diameter_file, target_diameters, fmt="%.15g")
    base = "rve_periodic_morphology"
    tessellation = output_directory / f"{base}.tess"
    ratio = parameters.directional_ratio_x_over_y
    morphology_parts = [
        f"diameq:file({diameter_file.name})",
        f"aspratio({ratio:.15g},1)",
    ]
    shape_file = None
    if target_shape is not None:
        # Neper 5.0 canonicalizes the 2-D alias ``1-circularity`` to the
        # internal property name ``1-sphericity`` and, for a file-valued
        # target, opens this canonical filename regardless of the spelling in
        # the command expression.  Use that actual 5.0 interface name while
        # retaining the physically correct 2-D circularity expression.
        shape_file = output_directory / "target_one_minus_sphericity.txt"
        np.savetxt(shape_file, target_shape, fmt="%.15g")
        morphology_parts.insert(1, f"1-circularity:file({shape_file.name})")
    if parameters.one_minus_circularity_mean is not None:
        morphology_parts.insert(
            1,
            "1-circularity:lognormal("
            f"{parameters.one_minus_circularity_mean:.15g},"
            f"{parameters.one_minus_circularity_standard_deviation:.15g})",
        )
    morphology = ",".join(morphology_parts)
    command = [
        str(NEPER),
        "-T",
        "-dim",
        "2",
        "-n",
        str(count),
        "-id",
        str(seed),
        "-domain",
        f"square({side_length_um:.15g},{side_length_um:.15g})",
        "-periodicity",
        "all",
        "-morpho",
        morphology,
    ]
    if morphology_time_limit_s is not None:
        command.extend(
            [
                "-morphooptistop",
                f"time>{morphology_time_limit_s:.15g}||eps<1e-6||val<1e-12",
            ]
        )
    command.extend(["-crysym", "cubic", "-format", "tess", "-o", base])
    if not reuse_exact_cache or not tessellation.exists():
        generation = _run(
            command,
            output_directory,
            output_directory / "neper_generation.log",
            timeout_s=generation_timeout_s,
        )
    else:
        generation = {"command": command, "returncode": 0, "reused": True}
    synthetic = measure_tessellation(tessellation)
    expected_area = side_length_um**2
    actual_area = float(synthetic["grain_area_um2"].sum())
    if abs(actual_area - expected_area) > max(1.0e-6, expected_area * 1.0e-10):
        raise RuntimeError(
            f"Periodic cell area is {actual_area}, expected {expected_area}"
        )
    orientations = size_conditioned_orientation_sample(
        phase_frame, synthetic, seed + 100000
    )
    orientation_file = output_directory / "orientations_euler_bunge_deg.txt"
    write_orientation_file(orientations, orientation_file)
    final_base = "rve_periodic_ebsd_matched"
    final_tessellation = output_directory / f"{final_base}.tess"
    orientation_command = [
        str(NEPER),
        "-T",
        "-loadtess",
        tessellation.name,
        "-ori",
        f"file({orientation_file.name},des=euler-bunge)",
        "-format",
        "tess",
        "-o",
        final_base,
    ]
    if not reuse_exact_cache or not final_tessellation.exists():
        orientation_run = _run(
            orientation_command,
            output_directory,
            output_directory / "neper_orientation_assignment.log",
        )
    else:
        orientation_run = {
            "command": orientation_command,
            "returncode": 0,
            "reused": True,
        }
    synthetic = synthetic.merge(
        orientations, on="synthetic_grain_id", validate="one_to_one"
    )
    synthetic["phase"] = orientations["phase"].astype(str).to_numpy()
    synthetic.to_csv(output_directory / "synthetic_grains.csv", index=False)
    orientations.to_csv(output_directory / "orientation_mapping.csv", index=False)
    periodicity = periodicity_metadata(final_tessellation)
    manifest = {
        "dataset": dataset_key,
        "phase": str(phase),
        "side_length_um": float(side_length_um),
        "grain_count": int(count),
        "geometry_seed": int(seed),
        "parameters": asdict(parameters),
        "cache_identity": cache_identity,
        "cache_reused": reuse_exact_cache,
        "morphology_expression": morphology,
        "generation": generation,
        "orientation_assignment": orientation_run,
        "tessellation": final_tessellation.name,
        "tessellation_sha256": sha256(final_tessellation),
        "grain_table_sha256": sha256(output_directory / "synthetic_grains.csv"),
        "domain_area_um2": actual_area,
        "periodicity": periodicity,
        "orientation_sampling": (
            "complete experimental Bunge triplets sampled conditionally by ECD rank; "
            "Euler components are not sampled independently"
        ),
        "joint_size_shape_target": (
            "paired experimental ECD and fitted-ellipse AR rows mapped to cell-wise "
            "Neper diameq and ellipse-derived 1-circularity targets"
            if target_shape is not None
            else "disabled"
        ),
        "input_size_shape_rank_correlation": (
            parameters.input_size_shape_rank_correlation
        ),
        "morphology_time_limit_s": morphology_time_limit_s,
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cleaned_csv", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", type=float, default=200.0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--diameter-spread", type=float, default=1.0)
    parser.add_argument("--directional-ratio", type=float, default=1.0)
    parser.add_argument("--aspect-ratio-proxy-scale", type=float)
    parser.add_argument("--input-size-shape-rank-correlation", type=float)
    parser.add_argument("--one-minus-circularity-mean", type=float)
    parser.add_argument("--one-minus-circularity-std", type=float)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--morphology-time-limit", type=float)
    args = parser.parse_args()
    manifest = generate_rve(
        pd.read_csv(args.cleaned_csv),
        args.output,
        dataset_key=args.dataset,
        phase=args.phase,
        side_length_um=args.side,
        seed=args.seed,
        grain_count=args.count,
        parameters=NeperMorphologyParameters(
            diameter_log_spread_scale=args.diameter_spread,
            directional_ratio_x_over_y=args.directional_ratio,
            aspect_ratio_proxy_scale=args.aspect_ratio_proxy_scale,
            input_size_shape_rank_correlation=(args.input_size_shape_rank_correlation),
            one_minus_circularity_mean=args.one_minus_circularity_mean,
            one_minus_circularity_standard_deviation=args.one_minus_circularity_std,
        ),
        force=args.force,
        morphology_time_limit_s=args.morphology_time_limit,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
