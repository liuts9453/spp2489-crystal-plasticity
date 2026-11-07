#!/usr/bin/env python3
"""Focused unit and artifact regression tests for the 2D workflow."""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import optimize_texture_2d
from common import (
    MESH_FILES,
    OPTIMIZATION_HISTORY,
    PROTOCOLS,
    TARGET_PROPERTIES,
    canonical_hash,
    simulation_configuration,
)
from extract_proof_stress import physical_history
from generate_hidden_target import REFERENCE_ETA, REFERENCE_THETA_DEG
from neper_to_moose import _signed_area, _smooth_shared_vertices
from optimize_texture_2d import (
    denormalize_design,
    fit_objective_gp,
    normalize_design,
    objective_and_errors,
)
from texture_parameterization import (
    orientations_for_design,
    parameterization_validation,
    preferred_rotation,
    write_cp_material_include_from_eulers,
)
from validate_periodic_mesh import validate_mesh


class InverseDesign2DTests(unittest.TestCase):
    def test_selective_geometry_smoothing_preserves_shared_topology_and_area(self):
        coordinates = [(0.1, 0.02), (0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        polygons = [[1, 2, 3], [1, 3, 4], [1, 4, 2]]
        original_area = sum(
            _signed_area([coordinates[point - 1] for point in polygon])
            for polygon in polygons
        )
        metrics = _smooth_shared_vertices(
            coordinates,
            polygons,
            domain_size=1.0,
            passes=20,
            maximum_move_fraction=0.5,
            activation_quality_threshold=0.2,
        )
        final_area = sum(
            _signed_area([coordinates[point - 1] for point in polygon])
            for polygon in polygons
        )
        self.assertGreater(metrics["accepted_vertex_moves"], 0)
        self.assertGreater(coordinates[0][1], 0.02)
        self.assertEqual(coordinates[1:], [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)])
        self.assertAlmostEqual(original_area, final_area, places=12)
        self.assertTrue(
            all(
                _signed_area([coordinates[point - 1] for point in polygon]) > 0.0
                for polygon in polygons
            )
        )

    def test_arbitrary_grain_orientation_include_uses_all_blocks(self) -> None:
        eulers = np.asarray([[0.0, 0.0, 0.0], [37.0, 52.0, 11.0], [12.0, 84.0, 173.0]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cp_materials.i"
            metrics = write_cp_material_include_from_eulers(
                path, eulers, description="synthetic unit test"
            )
            text = path.read_text(encoding="utf-8")
            self.assertEqual(metrics["grain_count"], 3)
            self.assertIn("block = 3", text)
            self.assertNotIn("grain_03_slip", text)

    def test_parameterization_endpoints_and_rotation_convention(self) -> None:
        result = parameterization_validation()
        self.assertTrue(result["all_checks_passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertTrue(all(result["rotation_convention"].values()))

    def test_texture_is_reproducible_and_on_so3(self) -> None:
        first = orientations_for_design(0.314159, -12.5)
        second = orientations_for_design(0.314159, -12.5)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertLess(first[2]["maximum_orthogonality_error"], 1.0e-12)
        self.assertLess(first[2]["maximum_determinant_error"], 1.0e-12)

    def test_eta_zero_is_theta_independent(self) -> None:
        minus = orientations_for_design(0.0, -45.0)[0]
        plus = orientations_for_design(0.0, 45.0)[0]
        np.testing.assert_array_equal(minus, plus)

    def test_theta_is_sample_frame_left_rotation(self) -> None:
        minus = preferred_rotation(-45.0)
        plus = preferred_rotation(45.0)
        self.assertGreater(float(np.linalg.norm(plus - minus)), 1.0e-3)
        self.assertAlmostEqual(float(np.linalg.det(minus)), 1.0, places=12)
        self.assertAlmostEqual(float(np.linalg.det(plus)), 1.0, places=12)

    def test_cache_key_separates_both_design_variables_mesh_and_time(self) -> None:
        configurations = [
            simulation_configuration(
                0.4, 0.0, "coarse", PROTOCOLS["adaptive_strict"], 8
            ),
            simulation_configuration(
                0.5, 0.0, "coarse", PROTOCOLS["adaptive_strict"], 8
            ),
            simulation_configuration(
                0.4, 1.0, "coarse", PROTOCOLS["adaptive_strict"], 8
            ),
            simulation_configuration(0.4, 0.0, "fine", PROTOCOLS["adaptive_strict"], 8),
            simulation_configuration(
                0.4, 0.0, "coarse", PROTOCOLS["adaptive_refined"], 8
            ),
            simulation_configuration(
                0.4, 0.0, "coarse", PROTOCOLS["adaptive_strict"], 4
            ),
        ]
        self.assertEqual(
            len({canonical_hash(item) for item in configurations}), len(configurations)
        )
        self.assertEqual(configurations[0], json.loads(json.dumps(configurations[0])))

    def test_design_normalization_roundtrip(self) -> None:
        for design in ((0.1, -45.0), (0.47, 12.25), (0.9, 45.0)):
            np.testing.assert_allclose(
                denormalize_design(normalize_design(*design)), design
            )

    def test_dual_property_objective(self) -> None:
        self.assertEqual(objective_and_errors(10.0, 20.0, 10.0, 20.0), (0.0, 0.0, 0.0))
        e1, e2, objective = objective_and_errors(11.0, 18.0, 10.0, 20.0)
        self.assertAlmostEqual(e1, 0.1)
        self.assertAlmostEqual(e2, 0.1)
        self.assertAlmostEqual(objective, 0.01)

    def test_gp_uses_two_normalized_inputs(self) -> None:
        model = fit_objective_gp(
            [(0.15, -30.0), (0.50, 0.0), (0.85, 30.0)], [0.1, 0.01, 0.2]
        )
        self.assertEqual(tuple(model.train_inputs[0].shape), (3, 2))
        self.assertEqual(tuple(model.train_targets.shape), (3,))

    def test_optimizer_does_not_contain_hidden_design(self) -> None:
        source = inspect.getsource(optimize_texture_2d)
        self.assertNotIn("REFERENCE_ETA", source)
        self.assertNotIn("REFERENCE_THETA", source)
        self.assertNotIn(str(REFERENCE_ETA), source)
        self.assertNotIn(str(REFERENCE_THETA_DEG), source)

    def test_executed_target_is_separated_from_every_initial_response(self) -> None:
        if not OPTIMIZATION_HISTORY.exists():
            self.skipTest("Optimization history has not been generated")
        import csv

        with OPTIMIZATION_HISTORY.open(newline="", encoding="utf-8") as stream:
            initial = [
                row for row in csv.DictReader(stream) if row["stage"] == "initial"
            ]
        self.assertEqual(len(initial), 9)
        minimum = min(
            max(float(row["relative_error_proof"]), float(row["relative_error_flow"]))
            for row in initial
        )
        self.assertGreater(minimum, 0.03)

    def test_initial_csv_row_is_replaced_by_identity_state(self) -> None:
        keys = {
            "time": 0.0,
            "F11_bar": 0.0,
            "F22_bar": 0.0,
            "det_fp_min": 0.0,
            "det_fp_max": 0.0,
            "slip_resistance_mean_bar": 0.0,
        }
        next_row = {**keys, "time": 0.1, "F11_bar": 1.002, "F22_bar": 0.999}
        history = physical_history([keys, next_row])
        self.assertEqual(history[0]["F11_bar"], 1.0)
        self.assertEqual(history[0]["F22_bar"], 1.0)
        self.assertEqual(history[0]["det_fp_min"], 1.0)
        self.assertEqual(history[1]["F11_bar"], 1.002)

    def test_generated_meshes_pass_static_validation(self) -> None:
        for name, mesh in MESH_FILES.items():
            if not mesh.exists():
                self.skipTest(f"Generated {name} mesh is absent")
            metrics = validate_mesh(mesh)
            self.assertTrue(metrics["all_checks_passed"])

    def test_public_target_does_not_leak_hidden_design(self) -> None:
        if not TARGET_PROPERTIES.exists():
            self.skipTest("Public target has not been generated")
        target = json.loads(TARGET_PROPERTIES.read_text(encoding="utf-8"))
        forbidden = {"eta", "theta", "orientation_path", "run_directory", "cache_key"}
        self.assertFalse(forbidden.intersection(target))


if __name__ == "__main__":
    unittest.main(verbosity=2)
