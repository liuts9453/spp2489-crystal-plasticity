#!/usr/bin/env python3
"""Focused unit/regression tests for the CP inverse-design orchestration."""

from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from common import PROTOCOLS, canonical_hash, simulation_configuration
from optimize_cp_texture import (
    early_stop,
    fit_objective_gp,
    objective_from_properties,
    valid_training_data,
)
import optimize_cp_texture
from texture_generator import (
    endpoint_validation,
    orientations_for_eta,
    rotation_convention_validation,
)


class CPInverseDesignTests(unittest.TestCase):
    def test_texture_endpoints(self) -> None:
        metrics = endpoint_validation()
        self.assertEqual(metrics["eta_0_random_euler_max_error_deg"], 0.0)
        self.assertEqual(metrics["eta_1_preferred_euler_max_error_deg"], 0.0)

    def test_rotation_shared_by_elasticity_and_slip_systems(self) -> None:
        self.assertTrue(all(rotation_convention_validation().values()))

    def test_texture_is_reproducible_and_on_so3(self) -> None:
        first = orientations_for_eta(0.314159)
        second = orientations_for_eta(0.314159)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertLess(first[2]["maximum_orthogonality_error"], 1.0e-12)
        self.assertLess(first[2]["maximum_determinant_error"], 1.0e-12)

    def test_cache_key_separates_mesh_and_time_protocol(self) -> None:
        base = canonical_hash(simulation_configuration(0.4, 32, PROTOCOLS["adaptive_strict"], 4))
        mesh = canonical_hash(simulation_configuration(0.4, 64, PROTOCOLS["adaptive_strict"], 4))
        time = canonical_hash(simulation_configuration(0.4, 32, PROTOCOLS["fixed_200"], 4))
        mpi = canonical_hash(simulation_configuration(0.4, 32, PROTOCOLS["adaptive_strict"], 8))
        self.assertEqual(len({base, mesh, time, mpi}), 4)

    def test_cache_configuration_survives_json_roundtrip(self) -> None:
        configuration = simulation_configuration(0.5, 32, PROTOCOLS["adaptive_strict"], 8)
        self.assertEqual(configuration, json.loads(json.dumps(configuration)))

    def test_dual_property_objective(self) -> None:
        self.assertEqual(objective_from_properties(10.0, 20.0, 10.0, 20.0), 0.0)
        self.assertAlmostEqual(objective_from_properties(11.0, 18.0, 10.0, 20.0), 0.01)

    def test_early_stopping_requires_both_properties(self) -> None:
        self.assertTrue(early_stop(0.01, 0.009, 0.01))
        self.assertFalse(early_stop(0.009, 0.01001, 0.01))

    def test_failed_fem_is_excluded_from_gp_training(self) -> None:
        rows = [
            {"status": "converged", "eta": 0.2, "objective_J": 0.1},
            {"status": "failed", "eta": 0.4, "objective_J": ""},
            {"status": "converged", "eta": 0.8, "objective_J": 0.02},
        ]
        eta, objective = valid_training_data(rows)
        self.assertEqual(eta, [0.2, 0.8])
        self.assertEqual(objective, [0.1, 0.02])

    def test_gp_shapes(self) -> None:
        model = fit_objective_gp([0.0, 0.5, 1.0], [0.1, 0.01, 0.2])
        self.assertEqual(tuple(model.train_inputs[0].shape), (3, 1))
        self.assertEqual(tuple(model.train_targets.shape), (3,))

    def test_optimizer_source_has_no_hidden_reference_value(self) -> None:
        source = inspect.getsource(optimize_cp_texture)
        self.assertNotIn("0.65", source)
        self.assertNotIn("0.90", source)
        self.assertNotIn("_HIDDEN_REFERENCE_ETA", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
