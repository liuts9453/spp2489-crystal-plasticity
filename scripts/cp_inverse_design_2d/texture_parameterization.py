#!/usr/bin/env python3
"""Deterministic two-parameter SO(3) texture generator for the Neper RVE."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from common import (
    MATERIAL_PARAMETERS,
    PREFERRED_EULER_DEG,
    RANDOM_MATERIALS,
    ROOT,
    SLIP_SYSTEM_FILE,
    write_json,
)


def moose_bunge_rotation(
    phi1_deg: float, Phi_deg: float, phi2_deg: float
) -> np.ndarray:
    """Return the passive-Bunge-derived crystal-to-sample tensor stored as `crysrot`."""
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


def sample_z_rotation(theta_deg: float) -> np.ndarray:
    """Active sample-frame rotation used in Qpreferred(theta)=Rz(theta) Q0."""
    angle = np.radians(theta_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=float)
    trace = float(np.trace(r))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * scale,
                (r[2, 1] - r[1, 2]) / scale,
                (r[0, 2] - r[2, 0]) / scale,
                (r[1, 0] - r[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(r)))
        if index == 0:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2]))
            quaternion = np.array(
                [
                    (r[2, 1] - r[1, 2]) / scale,
                    0.25 * scale,
                    (r[0, 1] + r[1, 0]) / scale,
                    (r[0, 2] + r[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + r[1, 1] - r[0, 0] - r[2, 2]))
            quaternion = np.array(
                [
                    (r[0, 2] - r[2, 0]) / scale,
                    (r[0, 1] + r[1, 0]) / scale,
                    0.25 * scale,
                    (r[1, 2] + r[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + r[2, 2] - r[0, 0] - r[1, 1]))
            quaternion = np.array(
                [
                    (r[1, 0] - r[0, 1]) / scale,
                    (r[0, 2] + r[2, 0]) / scale,
                    (r[1, 2] + r[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return quaternion / np.linalg.norm(quaternion)


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    w_value, x_value, y_value, z_value = np.asarray(
        quaternion, dtype=float
    ) / np.linalg.norm(quaternion)
    return np.array(
        [
            [
                1 - 2 * (y_value * y_value + z_value * z_value),
                2 * (x_value * y_value - z_value * w_value),
                2 * (x_value * z_value + y_value * w_value),
            ],
            [
                2 * (x_value * y_value + z_value * w_value),
                1 - 2 * (x_value * x_value + z_value * z_value),
                2 * (y_value * z_value - x_value * w_value),
            ],
            [
                2 * (x_value * z_value - y_value * w_value),
                2 * (y_value * z_value + x_value * w_value),
                1 - 2 * (x_value * x_value + y_value * y_value),
            ],
        ]
    )


def shortest_slerp(first: np.ndarray, second: np.ndarray, eta: float) -> np.ndarray:
    a_value = np.asarray(first, dtype=float) / np.linalg.norm(first)
    b_value = np.asarray(second, dtype=float) / np.linalg.norm(second)
    dot = float(np.dot(a_value, b_value))
    if dot < 0.0:
        b_value, dot = -b_value, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 1.0 - 1.0e-12:
        result = (1.0 - eta) * a_value + eta * b_value
        return result / np.linalg.norm(result)
    angle = np.arccos(dot)
    return (
        np.sin((1.0 - eta) * angle) * a_value + np.sin(eta * angle) * b_value
    ) / np.sin(angle)


def rotation_to_moose_bunge(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=float)
    Phi = float(np.arccos(np.clip(r[2, 2], -1.0, 1.0)))
    if abs(np.sin(Phi)) > 1.0e-10:
        phi1 = float(np.arctan2(r[2, 0], -r[2, 1]))
        phi2 = float(np.arctan2(r[0, 2], r[1, 2]))
    else:
        phi1, phi2 = float(np.arctan2(r[0, 1], r[0, 0])), 0.0
    return np.degrees([phi1 % (2.0 * np.pi), Phi, phi2 % (2.0 * np.pi)])


def random_endpoint_eulers() -> np.ndarray:
    text = RANDOM_MATERIALS.read_text(encoding="utf-8")
    arrays: list[np.ndarray] = []
    for name in ("orientation_phi1", "orientation_Phi", "orientation_phi2"):
        match = re.search(rf"^{name}\s*=\s*'([^']+)'", text, flags=re.MULTILINE)
        if not match:
            raise RuntimeError(f"Missing {name} in {RANDOM_MATERIALS}")
        arrays.append(np.fromstring(match.group(1), sep=" "))
    result = np.column_stack(arrays)
    if result.shape != (20, 3):
        raise RuntimeError(f"Expected 20 random orientations, found {result.shape}")
    return result


def preferred_rotation(theta_deg: float) -> np.ndarray:
    base = moose_bunge_rotation(*PREFERRED_EULER_DEG)
    return sample_z_rotation(theta_deg) @ base


def orientations_for_design(
    eta: float, theta_deg: float
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must lie in [0,1], got {eta}")
    source = random_endpoint_eulers()
    target_rotation = preferred_rotation(theta_deg)
    target_quaternion = rotation_to_quaternion(target_rotation)
    eulers: list[np.ndarray] = []
    quaternions: list[np.ndarray] = []
    maximum_orthogonality = 0.0
    maximum_determinant = 0.0
    maximum_roundtrip = 0.0
    maximum_norm = 0.0
    for original in source:
        source_quaternion = rotation_to_quaternion(moose_bunge_rotation(*original))
        quaternion = shortest_slerp(source_quaternion, target_quaternion, eta)
        rotation = quaternion_to_rotation(quaternion)
        euler = original.copy() if eta == 0.0 else rotation_to_moose_bunge(rotation)
        roundtrip = moose_bunge_rotation(*euler)
        maximum_orthogonality = max(
            maximum_orthogonality,
            float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
        )
        maximum_determinant = max(
            maximum_determinant, abs(float(np.linalg.det(rotation)) - 1.0)
        )
        maximum_roundtrip = max(
            maximum_roundtrip, float(np.max(np.abs(roundtrip - rotation)))
        )
        maximum_norm = max(maximum_norm, abs(float(np.linalg.norm(quaternion)) - 1.0))
        eulers.append(euler)
        quaternions.append(quaternion)
    metrics = {
        "eta": float(eta),
        "theta_deg": float(theta_deg),
        "maximum_orthogonality_error": maximum_orthogonality,
        "maximum_determinant_error": maximum_determinant,
        "maximum_euler_roundtrip_error": maximum_roundtrip,
        "maximum_quaternion_norm_error": maximum_norm,
    }
    if (
        max(value for key, value in metrics.items() if key not in ("eta", "theta_deg"))
        > 1.0e-10
    ):
        raise RuntimeError(f"Invalid SO(3) texture: {metrics}")
    return np.asarray(eulers), np.asarray(quaternions), metrics


def rotation_convention_validation() -> dict[str, bool]:
    source = (
        ROOT / "src" / "materials" / "ComputeLagrangianFCCCrystalPlasticityStress.C"
    ).read_text(encoding="utf-8")
    elasticity_source = Path(
        "/home/liu/projects/moose/modules/solid_mechanics/src/materials/crystal_plasticity/ComputeElasticityTensorCP.C"
    ).read_text(encoding="utf-8")
    checks = {
        "elasticity_material_produces_crysrot": "_crysrot(declareProperty<RankTwoTensor>"
        in elasticity_source,
        "elasticity_uses_transposed_bunge_rotation": "_crysrot[_qp] = _R.transpose();"
        in elasticity_source,
        "elasticity_rotated_by_shared_property": "_elasticity_tensor[_qp].rotate(_crysrot[_qp]);"
        in elasticity_source,
        "cp_rotation_property": '"crystal_rotation", "crysrot"' in source,
        "slip_direction_rotated": "_crystal_rotation[_qp] * _crystal_slip_direction[a]"
        in source,
        "plane_normal_rotated": "_crystal_rotation[_qp] * _crystal_plane_normal[a]"
        in source,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Crystal rotation convention check failed: {checks}")
    return checks


def parameterization_validation() -> dict[str, object]:
    random_zero, _, zero_metrics = orientations_for_design(0.0, -45.0)
    random_other, _, _ = orientations_for_design(0.0, 45.0)
    preferred_minus, _, minus_metrics = orientations_for_design(1.0, -45.0)
    preferred_plus, _, plus_metrics = orientations_for_design(1.0, 45.0)
    expected_minus = preferred_rotation(-45.0)
    expected_plus = preferred_rotation(45.0)
    checks = {
        "eta_zero_is_theta_independent": bool(
            np.array_equal(random_zero, random_other)
        ),
        "theta_changes_preferred_rotation": bool(
            np.linalg.norm(expected_plus - expected_minus) > 1.0e-3
        ),
        "eta_one_is_same_orientation": bool(
            max(
                np.linalg.norm(
                    moose_bunge_rotation(*euler)
                    - moose_bunge_rotation(*preferred_minus[0])
                )
                for euler in preferred_minus
            )
            < 1.0e-10
        ),
        "minus_endpoint_matrix": bool(
            np.max(np.abs(moose_bunge_rotation(*preferred_minus[0]) - expected_minus))
            < 1.0e-10
        ),
        "plus_endpoint_matrix": bool(
            np.max(np.abs(moose_bunge_rotation(*preferred_plus[0]) - expected_plus))
            < 1.0e-10
        ),
        "so3_errors_below_tolerance": max(
            zero_metrics["maximum_orthogonality_error"],
            minus_metrics["maximum_determinant_error"],
            plus_metrics["maximum_euler_roundtrip_error"],
        )
        < 1.0e-10,
    }
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "rotation_convention": rotation_convention_validation(),
    }


def _array(values: np.ndarray) -> str:
    return " ".join(f"{value:.15g}" for value in values)


def write_cp_material_include_from_eulers(
    path: Path,
    eulers: np.ndarray,
    *,
    description: str,
    include_grain_postprocessors: bool = False,
) -> dict[str, object]:
    """Write the verified FCC model for an arbitrary consecutive grain map."""
    eulers = np.asarray(eulers, dtype=float)
    if eulers.ndim != 2 or eulers.shape[1] != 3 or len(eulers) == 0:
        raise ValueError("eulers must have shape (grain_count, 3)")
    quaternions = []
    maximum_orthogonality = 0.0
    maximum_determinant = 0.0
    for euler in eulers:
        rotation = moose_bunge_rotation(*euler)
        maximum_orthogonality = max(
            maximum_orthogonality,
            float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
        )
        maximum_determinant = max(
            maximum_determinant, abs(float(np.linalg.det(rotation)) - 1.0)
        )
        quaternions.append(rotation_to_quaternion(rotation))
    quaternions = np.asarray(quaternions)
    metrics: dict[str, object] = {
        "grain_count": int(len(eulers)),
        "maximum_orthogonality_error": maximum_orthogonality,
        "maximum_determinant_error": maximum_determinant,
        "complete_bunge_triplets_used": True,
    }
    if max(maximum_orthogonality, maximum_determinant) > 1.0e-10:
        raise RuntimeError(f"Invalid SO(3) orientation table: {metrics}")
    elastic_constants = " ".join(
        f"{value:g}" for value in MATERIAL_PARAMETERS["C_ijkl_MPa"]
    )
    lines = [
        f"# {description}",
        f"orientation_phi1 = '{_array(eulers[:, 0])}'",
        f"orientation_Phi = '{_array(eulers[:, 1])}'",
        f"orientation_phi2 = '{_array(eulers[:, 2])}'",
        "",
        "[Materials]",
    ]
    for grain_id, euler in enumerate(eulers, start=1):
        tag = f"g{grain_id:02d}"
        lines.extend(
            [
                f"  [elasticity_{tag}]",
                "    type = ComputeElasticityTensorCP",
                f"    block = {grain_id}",
                "    fill_method = symmetric9",
                f"    C_ijkl = '{elastic_constants}'",
                f"    euler_angle_1 = {euler[0]:.15g}",
                f"    euler_angle_2 = {euler[1]:.15g}",
                f"    euler_angle_3 = {euler[2]:.15g}",
                "  []",
                f"  [analytic_cp_{tag}]",
                "    type = ComputeLagrangianFCCCrystalPlasticityStress",
                f"    block = {grain_id}",
                f"    slip_system_file = {SLIP_SYSTEM_FILE}",
                "    reference_slip_rate = 0.001",
                "    rate_sensitivity = ${cp_xm}",
                "    initial_slip_resistance = 80",
                "    hardening_modulus = 400",
                "    saturation_resistance = 160",
                "    hardening_exponent = 1.0",
                "    latent_hardening_ratio = 1.4",
                "    slip_increment_tolerance = 0.02",
                "    local_absolute_tolerance = 1e-10",
                "    local_relative_tolerance = 1e-9",
                "    local_max_iterations = 60",
                "    local_solver_mode = ${cp_local_solver}",
                "    reuse_tangent_factorization = ${cp_reuse_lu}",
                "    optimized_tensor_evaluation = ${cp_optimize_tensors}",
                "    specialize_scalar_operations = ${cp_optimize_scalars}",
                "    local_predictor = ${cp_predictor}",
                "  []",
                f"  [diagnostics_{tag}]",
                "    type = CrystalPlasticityDiagnostics",
                f"    block = {grain_id}",
                "  []",
            ]
        )
    lines.extend(["[]", ""])
    if include_grain_postprocessors:
        lines.append("[Postprocessors]")
        for grain_id in range(1, len(eulers) + 1):
            lines.extend(
                [
                    f"  [grain_{grain_id:02d}_slip]",
                    "    type = ElementAverageMaterialProperty",
                    "    mat_prop = accumulated_slip",
                    f"    block = {grain_id}",
                    "    execute_on = TIMESTEP_END",
                    "  []",
                ]
            )
        lines.extend(["[]", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    with path.with_name("orientations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ["grain_id", "phi1_deg", "Phi_deg", "phi2_deg", "q_w", "q_x", "q_y", "q_z"]
        )
        for grain_id, (euler, quaternion) in enumerate(
            zip(eulers, quaternions), start=1
        ):
            writer.writerow([grain_id, *euler, *quaternion])
    write_json(path.with_name("orientation_validation.json"), metrics)
    return metrics


def write_cp_material_include(
    path: Path, eta: float, theta_deg: float
) -> dict[str, object]:
    eulers, _, metrics = orientations_for_design(eta, theta_deg)
    generic = write_cp_material_include_from_eulers(
        path,
        eulers,
        description=(
            f"Generated 2D FCC CP texture: eta={eta:.17g}, "
            f"theta={theta_deg:.17g} deg."
        ),
        include_grain_postprocessors=True,
    )
    validation = {
        **generic,
        **metrics,
        "parameterization": parameterization_validation(),
    }
    write_json(path.with_name("orientation_validation.json"), validation)
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--theta", type=float)
    parser.add_argument(
        "--euler-csv",
        type=Path,
        help="CSV with phi1_deg,Phi_deg,phi2_deg for an arbitrary grain map",
    )
    parser.add_argument(
        "--description", default="Experimental EBSD sampled FCC texture"
    )
    parser.add_argument("--include-grain-postprocessors", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.euler_csv is not None:
        with args.euler_csv.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        eulers = np.asarray(
            [[row["phi1_deg"], row["Phi_deg"], row["phi2_deg"]] for row in rows],
            dtype=float,
        )
        result = write_cp_material_include_from_eulers(
            args.output_dir / "cp_materials.i",
            eulers,
            description=args.description,
            include_grain_postprocessors=args.include_grain_postprocessors,
        )
    else:
        if args.eta is None or args.theta is None:
            parser.error("supply either --euler-csv or both --eta and --theta")
        result = write_cp_material_include(
            args.output_dir / "cp_materials.i", args.eta, args.theta
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
