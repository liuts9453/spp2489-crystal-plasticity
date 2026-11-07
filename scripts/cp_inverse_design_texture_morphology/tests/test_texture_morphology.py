#!/usr/bin/env python3
"""Artifact and scientific-gate regressions for the executed morphology study."""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "cp_inverse_design_texture_morphology"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


class TextureMorphologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (RESULTS / "morphology_sensitivity.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            cls.rows = list(csv.DictReader(stream))
        with (RESULTS / "morphology_statistics.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            cls.statistics = list(csv.DictReader(stream))
        cls.sensitivity = load_json(RESULTS / "morphology_sensitivity_summary.json")
        cls.verification = load_json(
            RESULTS / "verification" / "verification_summary.json"
        )
        cls.homogeneous = load_json(
            RESULTS / "verification" / "homogeneous_control_summary.json"
        )

    def test_five_requested_levels_are_present(self) -> None:
        self.assertEqual(
            [float(row["input_a"]) for row in self.rows],
            [0.70, 0.85, 1.00, 1.25, 1.50],
        )

    def test_achieved_directional_morphology_is_monotonic(self) -> None:
        achieved = [float(row["achieved_a"]) for row in self.rows]
        self.assertEqual(achieved, sorted(achieved))
        self.assertLess(achieved[0], 1.0)
        self.assertGreater(achieved[-1], 1.0)

    def test_periodic_statistics_have_twenty_grains_and_exact_area(self) -> None:
        for row in self.statistics:
            self.assertEqual(int(row["grain_count"]), 20)
            self.assertAlmostEqual(float(row["domain_area_um2"]), 40000.0, places=7)

    def test_all_five_fem_runs_converged_and_passed_physics(self) -> None:
        self.assertTrue(self.sensitivity["all_converged"])
        self.assertTrue(self.sensitivity["all_physical_checks_passed"])
        for row in self.rows:
            self.assertEqual(row["converged"], "True")
            self.assertEqual(row["all_checks_passed"], "True")
            self.assertLess(float(row["maximum_abs_P22_MPa"]), 1.0e-3)
            self.assertLess(float(row["maximum_abs_macro_shear"]), 1.0e-10)
            self.assertLess(float(row["hill_mandel_relative_residual_max"]), 5.0e-4)
            self.assertLess(float(row["maximum_periodicity_error"]), 1.0e-8)

    def test_homogeneous_control_has_no_artificial_morphology_effect(self) -> None:
        self.assertTrue(self.homogeneous["all_checks_passed"])
        self.assertLess(self.homogeneous["proof_stress_relative_spread"], 1.0e-10)
        self.assertLess(self.homogeneous["flow_stress_relative_spread"], 1.0e-10)
        self.assertLess(self.homogeneous["maximum_local_P11_range_MPa"], 1.0e-6)

    def test_mesh_and_time_verification_passed(self) -> None:
        self.assertTrue(self.verification["all_checks_passed"])
        self.assertEqual(len(self.verification["fine_results"]), 3)
        self.assertEqual(len(self.verification["time_refined_results"]), 2)

    def test_realization_pairs_include_all_three_geometry_seeds(self) -> None:
        seeds = [
            int(row["geometry_seed"])
            for row in self.verification["realization_comparisons"]
        ]
        self.assertEqual(seeds, [2026, 2027, 2028])

    def test_paired_morphology_effect_reverses_sign(self) -> None:
        for key in ("proof_stress_MPa", "flow_stress_2pct_MPa"):
            values = self.verification["uncertainty"][key][
                "paired_a1p5_minus_a1_contrasts_MPa"
            ]
            self.assertLess(min(values), 0.0)
            self.assertGreater(max(values), 0.0)
            self.assertFalse(
                self.verification["uncertainty"][key]["paired_contrasts_same_sign"]
            )

    def test_scientific_gate_rejects_three_variable_bo(self) -> None:
        gate = self.verification["scientific_gate"]
        self.assertFalse(gate["enter_three_variable_bo"])
        self.assertFalse(gate["by_property"]["proof_stress_MPa"]["passes"])
        self.assertFalse(gate["by_property"]["flow_stress_2pct_MPa"]["passes"])
        self.assertFalse((RESULTS / "optimization_history.csv").exists())
        self.assertFalse((RESULTS / "best_design.json").exists())

    def test_existing_two_parameter_result_is_unchanged(self) -> None:
        best = load_json(ROOT / "results" / "cp_inverse_design_2d" / "best_design.json")
        self.assertAlmostEqual(best["eta"], 0.7729026253348713, places=14)
        self.assertAlmostEqual(best["theta_deg"], 4.242186925858952, places=13)

    def test_all_real_data_figures_exist(self) -> None:
        expected = {
            "fig01_grain_morphology_design_space.png",
            "fig02_morphology_sensitivity.png",
            "fig03_plastic_slip_fields.png",
            "fig05_forward_response_comparison.png",
            "fig07_coarse_fine_time_verification.png",
            "fig08_morphology_texture_property_summary.png",
        }
        self.assertEqual(
            {path.name for path in (RESULTS / "figures").glob("*.png")}, expected
        )


if __name__ == "__main__":
    unittest.main()
