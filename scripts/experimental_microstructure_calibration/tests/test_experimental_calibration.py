#!/usr/bin/env python3
"""Unit and local Neper capability tests using non-experimental synthetic data."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from clean_ebsd_data import (
    CleaningPolicy,
    add_derived_geometry,
    clean_grains,
)  # noqa: E402
from common import DatasetSpec, NEPER  # noqa: E402
from compare_experimental_neper import comparison_metrics  # noqa: E402
from extract_experimental_statistics import summarize_grains  # noqa: E402
from export_public_results import _sanitize  # noqa: E402
from extract_texture_statistics import (  # noqa: E402
    CUBIC_SYMMETRIES,
    extract_texture_statistics,
    ipf_coordinates,
    moose_bunge_rotation,
    pole_figure_coordinates,
    rotations_from_frame,
)
from fit_grain_distributions import fit_positive_distributions  # noqa: E402
from generate_ebsd_matched_neper import (  # noqa: E402
    NeperMorphologyParameters,
    generate_rve,
    joint_morphology_targets,
)
from load_ebsd_grains import load_ebsd_csv  # noqa: E402


def synthetic_frame(count: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(2489)
    diameters = np.exp(rng.normal(np.log(8.0), 0.28, count))
    area = np.pi * (diameters / 2.0) ** 2
    aspect = 1.0 + np.exp(rng.normal(-0.15, 0.25, count))
    semi_minor = np.sqrt(area / (np.pi * aspect))
    semi_major = aspect * semi_minor
    return pd.DataFrame(
        {
            "dataset": "SYNTHETIC_TEST_ONLY",
            "source_row": np.arange(2, count + 2),
            "grain_id": np.arange(1, count + 1),
            "phase": ["FCC"] * count,
            "semi_axis_a_um": semi_major,
            "semi_axis_b_um": semi_minor,
            "grain_radius_um": diameters / 2.0,
            "grain_area_um2": area,
            "aspect_ratio_reported": aspect,
            "phi1_deg": rng.uniform(0.0, 360.0, count),
            "Phi_deg": np.degrees(np.arccos(rng.uniform(-1.0, 1.0, count))),
            "phi2_deg": rng.uniform(0.0, 360.0, count),
            "alpha_deg": rng.uniform(0.0, 180.0, count),
        }
    )


class ExperimentalCalibrationTests(unittest.TestCase):
    def test_loader_maps_columns_and_degrees(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            rows = []
            for index in range(12):
                rows.append(
                    {
                        "Grain_ID": index + 1,
                        "Phase": "FCC",
                        "phi1": 10.0 + index,
                        "PHI": 20.0 + index,
                        "phi2": 30.0 + index,
                        "a": 2.0,
                        "b": 1.0,
                        "Grain Radius": 2.0,
                        "Grain Area (μm²)": 12.0 + index,
                        "Aspect Ratio": 2.0,
                        "alpha": 40.0,
                    }
                )
            pd.DataFrame(rows).to_csv(path, index=False)
            spec = DatasetSpec("test", path.name, "test", "test", 12, "")
            frame, metadata = load_ebsd_csv(path, spec, angle_unit="degree")
            self.assertEqual(len(frame), 12)
            self.assertTrue(metadata["raw_count_matches_reference"])
            self.assertAlmostEqual(frame.loc[0, "Phi_deg"], 20.0)
            self.assertFalse(metadata["alpha_is_grain_axis_direction"])

    def test_cleaning_is_audited_and_does_not_trim_upper_tail(self) -> None:
        frame = synthetic_frame()
        frame.loc[0, "grain_area_um2"] = np.pi * 0.5**2
        frame.loc[1, "aspect_ratio_reported"] = 100.0
        frame.loc[1, "semi_axis_a_um"] = 100.0 * frame.loc[1, "semi_axis_b_um"]
        cleaned, audited, audit = clean_grains(
            frame, CleaningPolicy(minimum_ecd_um=2.8)
        )
        self.assertEqual(audit["removed_count"], 1)
        self.assertEqual(audited.loc[0, "cleaning_status"], "below_minimum_ecd")
        self.assertGreater(cleaned["aspect_ratio_recomputed"].max(), 90.0)

    def test_ecd_and_aspect_ratio_definitions(self) -> None:
        frame = add_derived_geometry(synthetic_frame())
        expected = 2.0 * np.sqrt(frame["grain_area_um2"] / np.pi)
        np.testing.assert_allclose(frame["equivalent_diameter_um"], expected)
        self.assertTrue(np.all(frame["aspect_ratio_recomputed"] >= 1.0))
        summary = summarize_grains(frame)
        self.assertEqual(summary["grain_count"], len(frame))

    def test_distribution_models_include_mixture_and_tail_checks(self) -> None:
        rng = np.random.default_rng(3)
        values = np.concatenate(
            [rng.lognormal(1.0, 0.12, 150), rng.lognormal(2.1, 0.15, 100)]
        )
        fit = fit_positive_distributions(values)
        names = {item["model"] for item in fit["candidates"]}
        self.assertEqual(
            names, {"lognormal", "weibull", "gamma", "two_component_lognormal_mixture"}
        )
        self.assertEqual(
            fit["selected_by_bic"]["model"], "two_component_lognormal_mixture"
        )

    def test_cubic_ipf_descriptor_is_symmetry_invariant(self) -> None:
        rotation = moose_bunge_rotation(37.0, 52.0, 11.0)
        reference = ipf_coordinates(rotation[None, :, :], np.array([1.0, 0.0, 0.0]))[0]
        self.assertEqual(CUBIC_SYMMETRIES.shape, (24, 3, 3))
        for symmetry in CUBIC_SYMMETRIES:
            candidate = ipf_coordinates(
                (rotation @ symmetry)[None, :, :], np.array([1.0, 0.0, 0.0])
            )[0]
            np.testing.assert_allclose(candidate, reference, atol=1.0e-12)

    def test_rotation_matrices_are_proper(self) -> None:
        rotations = rotations_from_frame(synthetic_frame())
        np.testing.assert_allclose(
            np.einsum("nji,njk->nik", rotations, rotations),
            np.broadcast_to(np.eye(3), rotations.shape),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=1.0e-12)

    def test_phase_separated_pole_and_ipf_statistics(self) -> None:
        frame = add_derived_geometry(synthetic_frame())
        rotations = rotations_from_frame(frame)
        poles = pole_figure_coordinates(rotations, "111")
        self.assertLessEqual(
            float(np.max(np.linalg.norm(poles, axis=2))), 1.0 + 1.0e-12
        )
        texture = extract_texture_statistics(frame, bins=12)
        phase = texture["phase_statistics"]["FCC"]
        self.assertEqual(set(phase["pole_figure_histograms"]), {"100", "110", "111"})
        self.assertIn("sample_x", phase["ipf_fundamental_sector_histograms"])

    def test_numeric_phase_labels_are_type_invariant(self) -> None:
        experimental = add_derived_geometry(synthetic_frame())
        experimental["phase"] = 0
        synthetic = experimental.copy()
        synthetic["phase"] = 0.0
        metrics = comparison_metrics(experimental, synthetic)
        self.assertAlmostEqual(metrics["normalized_components"]["odf"], 0.0)
        self.assertAlmostEqual(
            metrics["diagnostics"]["phase_fraction_total_variation"], 0.0
        )

    def test_section_label_alpha_can_be_excluded_from_objective(self) -> None:
        experimental = add_derived_geometry(synthetic_frame())
        synthetic = experimental.copy()
        experimental["alpha_deg"] = 0.0
        synthetic["alpha_deg"] = 90.0
        included = comparison_metrics(experimental, synthetic, include_direction=True)
        excluded = comparison_metrics(experimental, synthetic, include_direction=False)
        self.assertGreater(included["normalized_components"]["direction"], 0.99)
        self.assertEqual(excluded["weights"]["direction"], 0.0)
        self.assertLess(excluded["objective"], included["objective"])
        self.assertFalse(excluded["direction_target_used"])

    def test_joint_targets_preserve_size_shape_rank_pairing(self) -> None:
        frame = add_derived_geometry(synthetic_frame(80)).sort_values(
            "equivalent_diameter_um"
        )
        frame["aspect_ratio_recomputed"] = np.linspace(1.05, 5.0, len(frame))
        diameters, one_minus_circularity = joint_morphology_targets(
            frame, len(frame), 2489, 1.0, 1.0
        )
        self.assertGreater(
            float(
                pd.Series(diameters).corr(
                    pd.Series(one_minus_circularity), method="spearman"
                )
            ),
            0.99,
        )
        self.assertTrue(
            np.all((one_minus_circularity > 0.0) & (one_minus_circularity < 1.0))
        )

    def test_joint_target_copula_preserves_marginal_and_controls_rank(self) -> None:
        frame = add_derived_geometry(synthetic_frame(80))
        diameter, original = joint_morphology_targets(frame, 80, 41, 1.0, 1.0)
        controlled_diameter, controlled = joint_morphology_targets(
            frame, 80, 41, 1.0, 1.0, 1.0
        )
        np.testing.assert_allclose(controlled_diameter, diameter)
        np.testing.assert_allclose(np.sort(controlled), np.sort(original))
        correlation = pd.Series(controlled_diameter).corr(
            pd.Series(controlled), method="spearman"
        )
        self.assertGreater(float(correlation), 0.999)

    def test_public_export_removes_private_provenance(self) -> None:
        payload = {
            "dataset": "Al1_oBD",
            "source_sha256": "private-hash",
            "cleaned_csv": "/private/cleaned.csv",
            "private_source_root_is_not_recorded": True,
            "nested": {"objective": 0.25, "statistics_json": "/private/stats.json"},
        }
        sanitized = _sanitize(payload)
        self.assertEqual(sanitized["dataset"], "Al1_oBD")
        self.assertEqual(sanitized["nested"], {"objective": 0.25})
        self.assertNotIn("source_sha256", sanitized)
        self.assertNotIn("cleaned_csv", sanitized)
        self.assertNotIn("private_source_root_is_not_recorded", sanitized)

    @unittest.skipUnless(NEPER.is_file(), "Neper executable is unavailable")
    def test_real_neper_periodic_generation_and_orientation_assignment(self) -> None:
        experimental = add_derived_geometry(synthetic_frame(80))
        with tempfile.TemporaryDirectory() as directory:
            manifest = generate_rve(
                experimental,
                Path(directory),
                dataset_key="SYNTHETIC_CAPABILITY_TEST_ONLY",
                phase="all",
                side_length_um=30.0,
                seed=92489,
                grain_count=10,
                parameters=NeperMorphologyParameters(
                    diameter_log_spread_scale=1.0,
                    directional_ratio_x_over_y=1.25,
                ),
                morphology_time_limit_s=2.0,
                force=True,
            )
            synthetic = pd.read_csv(Path(directory) / "synthetic_grains.csv")
            self.assertEqual(len(synthetic), 10)
            self.assertAlmostEqual(synthetic["grain_area_um2"].sum(), 900.0, places=8)
            self.assertTrue(
                (Path(directory) / "rve_periodic_ebsd_matched.tess").is_file()
            )
            self.assertEqual(manifest["generation"]["returncode"], 0)
            self.assertTrue(manifest["periodicity"]["xy_periodic"])
            metrics = comparison_metrics(experimental, synthetic)
            self.assertTrue(np.isfinite(metrics["objective"]))
            reused = generate_rve(
                experimental,
                Path(directory),
                dataset_key="SYNTHETIC_CAPABILITY_TEST_ONLY",
                phase="all",
                side_length_um=30.0,
                seed=92489,
                grain_count=10,
                parameters=NeperMorphologyParameters(
                    diameter_log_spread_scale=1.0,
                    directional_ratio_x_over_y=1.25,
                ),
                morphology_time_limit_s=2.0,
            )
            self.assertTrue(reused["cache_reused"])
            changed = generate_rve(
                experimental,
                Path(directory),
                dataset_key="SYNTHETIC_CAPABILITY_TEST_ONLY",
                phase="all",
                side_length_um=30.0,
                seed=92489,
                grain_count=10,
                parameters=NeperMorphologyParameters(
                    diameter_log_spread_scale=1.0,
                    directional_ratio_x_over_y=1.35,
                ),
                morphology_time_limit_s=2.0,
            )
            self.assertFalse(changed["cache_reused"])


if __name__ == "__main__":
    unittest.main()
