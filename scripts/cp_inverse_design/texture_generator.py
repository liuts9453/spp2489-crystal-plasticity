#!/usr/bin/env python3
"""Generate shortest-path SO(3) textures and matching FCC CP material blocks."""

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


def moose_bunge_rotation(phi1_deg: float, Phi_deg: float, phi2_deg: float) -> np.ndarray:
    """Return the passive ZXZ/Bunge matrix used by MOOSE RotationTensor."""
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


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=float)
    trace = float(np.trace(r))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        q = np.array([0.25 * scale, (r[2, 1] - r[1, 2]) / scale,
                      (r[0, 2] - r[2, 0]) / scale, (r[1, 0] - r[0, 1]) / scale])
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2]))
            q = np.array([(r[2, 1] - r[1, 2]) / scale, 0.25 * scale,
                          (r[0, 1] + r[1, 0]) / scale, (r[0, 2] + r[2, 0]) / scale])
        elif i == 1:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + r[1, 1] - r[0, 0] - r[2, 2]))
            q = np.array([(r[0, 2] - r[2, 0]) / scale, (r[0, 1] + r[1, 0]) / scale,
                          0.25 * scale, (r[1, 2] + r[2, 1]) / scale])
        else:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + r[2, 2] - r[0, 0] - r[1, 1]))
            q = np.array([(r[1, 0] - r[0, 1]) / scale, (r[0, 2] + r[2, 0]) / scale,
                          (r[1, 2] + r[2, 1]) / scale, 0.25 * scale])
    return q / np.linalg.norm(q)


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def shortest_slerp(q0: np.ndarray, q1: np.ndarray, eta: float) -> np.ndarray:
    a = np.asarray(q0, dtype=float) / np.linalg.norm(q0)
    b = np.asarray(q1, dtype=float) / np.linalg.norm(q1)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b, dot = -b, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 1.0 - 1.0e-12:
        q = (1.0 - eta) * a + eta * b
        return q / np.linalg.norm(q)
    angle = np.arccos(dot)
    return (np.sin((1.0 - eta) * angle) * a + np.sin(eta * angle) * b) / np.sin(angle)


def rotation_to_moose_bunge(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=float)
    Phi = float(np.arccos(np.clip(r[2, 2], -1.0, 1.0)))
    if abs(np.sin(Phi)) > 1.0e-10:
        phi1 = float(np.arctan2(r[2, 0], -r[2, 1]))
        phi2 = float(np.arctan2(r[0, 2], r[1, 2]))
    elif r[2, 2] > 0.0:
        phi1, phi2 = float(np.arctan2(r[0, 1], r[0, 0])), 0.0
    else:
        phi1, phi2 = float(np.arctan2(r[0, 1], r[0, 0])), 0.0
    return np.degrees([phi1 % (2.0 * np.pi), Phi, phi2 % (2.0 * np.pi)])


def random_endpoint_eulers() -> np.ndarray:
    text = RANDOM_MATERIALS.read_text(encoding="utf-8")
    arrays = []
    for name in ("orientation_phi1", "orientation_Phi", "orientation_phi2"):
        match = re.search(rf"^{name}\s*=\s*'([^']+)'", text, flags=re.MULTILINE)
        if not match:
            raise RuntimeError(f"Missing {name} in {RANDOM_MATERIALS}")
        arrays.append(np.fromstring(match.group(1), sep=" "))
    result = np.column_stack(arrays)
    if result.shape != (20, 3):
        raise RuntimeError(f"Expected 20 endpoint orientations, found {result.shape}")
    return result


def orientations_for_eta(eta: float) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must lie in [0,1], got {eta}")
    source = random_endpoint_eulers()
    target = np.asarray(PREFERRED_EULER_DEG)
    target_q = rotation_to_quaternion(moose_bunge_rotation(*target))
    eulers, quaternions = [], []
    max_orthogonality = max_roundtrip = max_quaternion_norm_error = 0.0
    max_det_error = 0.0
    for original in source:
        q = shortest_slerp(rotation_to_quaternion(moose_bunge_rotation(*original)), target_q, eta)
        rotation = quaternion_to_rotation(q)
        if eta == 0.0:
            euler = original.copy()
        elif eta == 1.0:
            euler = target.copy()
        else:
            euler = rotation_to_moose_bunge(rotation)
        roundtrip = moose_bunge_rotation(*euler)
        max_orthogonality = max(max_orthogonality, float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))))
        max_det_error = max(max_det_error, abs(float(np.linalg.det(rotation)) - 1.0))
        max_roundtrip = max(max_roundtrip, float(np.max(np.abs(roundtrip - rotation))))
        max_quaternion_norm_error = max(max_quaternion_norm_error, abs(float(np.linalg.norm(q)) - 1.0))
        eulers.append(euler)
        quaternions.append(q)
    metrics = {
        "eta": float(eta),
        "maximum_orthogonality_error": max_orthogonality,
        "maximum_determinant_error": max_det_error,
        "maximum_euler_roundtrip_error": max_roundtrip,
        "maximum_quaternion_norm_error": max_quaternion_norm_error,
    }
    if max(metrics[key] for key in metrics if key != "eta") > 1.0e-10:
        raise RuntimeError(f"Invalid SO(3) interpolation: {metrics}")
    return np.asarray(eulers), np.asarray(quaternions), metrics


def rotation_convention_validation() -> dict[str, object]:
    source = (ROOT / "src" / "materials" / "ComputeLagrangianFCCCrystalPlasticityStress.C").read_text()
    elasticity_source = Path(
        "/home/liu/projects/moose/modules/solid_mechanics/src/materials/crystal_plasticity/ComputeElasticityTensorCP.C"
    ).read_text()
    checks = {
        "elasticity_material_produces_crysrot": "_crysrot(declareProperty<RankTwoTensor>" in elasticity_source,
        "elasticity_uses_transposed_bunge_rotation": "_crysrot[_qp] = _R.transpose();" in elasticity_source,
        "elasticity_rotated_by_shared_property": "_elasticity_tensor[_qp].rotate(_crysrot[_qp]);" in elasticity_source,
        "cp_rotation_property": '"crystal_rotation", "crysrot"' in source,
        "slip_direction_rotated": "_crystal_rotation[_qp] * _crystal_slip_direction[a]" in source,
        "plane_normal_rotated": "_crystal_rotation[_qp] * _crystal_plane_normal[a]" in source,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Crystal rotation convention check failed: {checks}")
    return checks


def _array(values: np.ndarray) -> str:
    return " ".join(f"{value:.15g}" for value in values)


def write_cp_material_include(path: Path, eta: float) -> dict[str, object]:
    eulers, quaternions, metrics = orientations_for_eta(eta)
    c = " ".join(f"{value:g}" for value in MATERIAL_PARAMETERS["C_ijkl_MPa"])
    lines = [
        f"# Generated FCC CP texture for eta={eta:.17g}.",
        f"orientation_phi1 = '{_array(eulers[:, 0])}'",
        f"orientation_Phi = '{_array(eulers[:, 1])}'",
        f"orientation_phi2 = '{_array(eulers[:, 2])}'",
        "",
        "[Materials]",
    ]
    for index, euler in enumerate(eulers, start=1):
        tag = f"g{index:02d}"
        lines.extend(
            [
                f"  [elasticity_{tag}]",
                "    type = ComputeElasticityTensorCP",
                f"    block = {index}",
                "    fill_method = symmetric9",
                f"    C_ijkl = '{c}'",
                f"    euler_angle_1 = {euler[0]:.15g}",
                f"    euler_angle_2 = {euler[1]:.15g}",
                f"    euler_angle_3 = {euler[2]:.15g}",
                "  []",
                f"  [analytic_cp_{tag}]",
                "    type = ComputeLagrangianFCCCrystalPlasticityStress",
                f"    block = {index}",
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
                f"    block = {index}",
                "  []",
            ]
        )
    lines.extend(["[]", "", "[Postprocessors]"])
    for grain in range(1, 21):
        lines.extend(
            [
                f"  [grain_{grain:02d}_slip]",
                "    type = ElementAverageMaterialProperty",
                "    mat_prop = accumulated_slip",
                f"    block = {grain}",
                "    execute_on = TIMESTEP_END",
                "  []",
            ]
        )
    lines.extend(["[]", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    with path.with_name("orientations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["grain_id", "phi1_deg", "Phi_deg", "phi2_deg", "q_w", "q_x", "q_y", "q_z"])
        for grain, (euler, q) in enumerate(zip(eulers, quaternions), start=1):
            writer.writerow([grain, *euler, *q])
    validation = {**metrics, "rotation_convention": rotation_convention_validation()}
    write_json(path.with_name("orientation_validation.json"), validation)
    return validation


def endpoint_validation() -> dict[str, float]:
    source = random_endpoint_eulers()
    zero, _, zero_metrics = orientations_for_eta(0.0)
    one, _, one_metrics = orientations_for_eta(1.0)
    preferred = np.tile(PREFERRED_EULER_DEG, (20, 1))
    result = {
        "eta_0_random_euler_max_error_deg": float(np.max(np.abs(zero - source))),
        "eta_1_preferred_euler_max_error_deg": float(np.max(np.abs(one - preferred))),
        "maximum_rotation_orthogonality_error": max(zero_metrics["maximum_orthogonality_error"], one_metrics["maximum_orthogonality_error"]),
        "maximum_rotation_determinant_error": max(zero_metrics["maximum_determinant_error"], one_metrics["maximum_determinant_error"]),
    }
    if max(result.values()) > 1.0e-12:
        raise RuntimeError(f"Endpoint validation failed: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "endpoints": endpoint_validation(),
        "texture": write_cp_material_include(args.output_dir / "cp_materials.i", args.eta),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
