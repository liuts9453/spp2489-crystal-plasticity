#!/usr/bin/env python3
"""Data-provenance and media regressions for the PI presentation package."""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PRESENTATION_DIR = Path(__file__).resolve().parents[1]
if str(PRESENTATION_DIR) not in sys.path:
    sys.path.insert(0, str(PRESENTATION_DIR))

from presentation_utils import (  # noqa: E402
    MANIFEST,
    MESH_FILES,
    PRESENTATION,
    audit_history,
    cubic_ipf_x_color,
    file_sha256,
)


def _rotation_with_first_row(direction: tuple[float, float, float]) -> np.ndarray:
    first = np.asarray(direction, dtype=float)
    first /= np.linalg.norm(first)
    helper = np.array([0.0, 1.0, 0.0])
    if abs(float(first @ helper)) > 0.9:
        helper = np.array([0.0, 0.0, 1.0])
    second = np.cross(helper, first)
    second /= np.linalg.norm(second)
    third = np.cross(first, second)
    return np.vstack((first, second, third))


def _proper_cubic_symmetries() -> list[np.ndarray]:
    symmetries: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3))
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if np.linalg.det(matrix) > 0.5:
                symmetries.append(matrix)
    return symmetries


class PresentationVisualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = audit_history()

    def test_all_28_real_fem_records_are_recovered(self) -> None:
        records = self.audit["records"]
        self.assertEqual([record["call"] for record in records], list(range(1, 29)))
        self.assertEqual([record["stage"] for record in records[:9]], ["initial"] * 9)
        self.assertEqual(
            [record["stage"] for record in records[9:]], ["sequential"] * 19
        )
        for record in records:
            self.assertTrue(record["paths"]["csv"].is_file())
            self.assertTrue(record["paths"]["exodus"].is_file())
            self.assertTrue(record["paths"]["result"].is_file())
            self.assertGreater(record["curve_point_count"], 2)
            self.assertTrue(all(record["checks"].values()))

    def test_best_so_far_is_prefix_only(self) -> None:
        records = self.audit["records"]
        self.assertEqual(
            [record["call"] for record in records if record["is_best_update"]],
            [1, 2, 8, 9, 14, 21, 28],
        )
        for record in records:
            expected = min(
                records[: record["call"]], key=lambda item: item["objective_J"]
            )
            self.assertEqual(record["best_call"], expected["call"])

    def test_fixed_cubic_ipf_x_key_vertices(self) -> None:
        vertices = {
            (0.0, 0.0, 1.0): np.array([1.0, 0.0, 0.0]),
            (1.0, 0.0, 1.0): np.array([0.0, 1.0, 0.0]),
            (1.0, 1.0, 1.0): np.array([0.0, 0.0, 1.0]),
        }
        for direction, expected in vertices.items():
            np.testing.assert_allclose(
                cubic_ipf_x_color(_rotation_with_first_row(direction)),
                expected,
                atol=1.0e-12,
            )

    def test_ipf_x_color_is_invariant_under_cubic_symmetry(self) -> None:
        rotation = _rotation_with_first_row((0.31, -0.55, 0.775))
        reference = cubic_ipf_x_color(rotation)
        symmetries = _proper_cubic_symmetries()
        self.assertEqual(len(symmetries), 24)
        for symmetry in symmetries:
            np.testing.assert_allclose(
                cubic_ipf_x_color(rotation @ symmetry), reference, atol=1.0e-12
            )

    def test_manifest_records_frame_provenance_and_prefix_posteriors(self) -> None:
        if not MANIFEST.exists():
            self.skipTest("Run the unified presentation builder first")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        frames = manifest["animation"]["frames"]
        self.assertEqual(len(frames), 28)
        mesh_hash = file_sha256(MESH_FILES["coarse"])
        for index, frame in enumerate(frames, start=1):
            self.assertEqual(frame["call"], index)
            self.assertEqual(frame["fixed_geometry_sha256"], mesh_hash)
            self.assertEqual(
                frame["data_classification"], "real MOOSE FCC CP FEM evaluation"
            )
            self.assertTrue(Path(frame["solution_csv"]).is_file())
            self.assertTrue(Path(frame["exodus"]).is_file())
            if index < 3:
                self.assertEqual(
                    frame["posterior_background"], "neutral parameter plane"
                )
            else:
                self.assertIn(f"calls 1..{index}", frame["posterior_background"])
        self.assertTrue(all(manifest["validation"].values()))

    def test_animation_media_contract(self) -> None:
        gif_path = PRESENTATION / "bo_inverse_design_history.gif"
        mp4_path = PRESENTATION / "bo_inverse_design_history.mp4"
        if not gif_path.exists() or not mp4_path.exists():
            self.skipTest("Run the unified presentation builder first")
        with Image.open(gif_path) as image:
            self.assertEqual(image.n_frames, 28)
            self.assertEqual(image.size, (1280, 720))
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,width,height",
                "-of",
                "json",
                str(mp4_path),
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        stream = json.loads(completed.stdout)["streams"][0]
        self.assertEqual(stream["codec_name"], "h264")
        self.assertEqual(stream["pix_fmt"], "yuv420p")
        self.assertEqual((stream["width"], stream["height"]), (1920, 1080))
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        actual_duration = float(
            manifest["animation"]["outputs"]["mp4"]["probe"]["format"]["duration"]
        )
        scheduled_duration = manifest["animation"]["total_duration_ms"] / 1000.0
        self.assertLessEqual(abs(actual_duration - scheduled_duration), 0.20)
        self.assertEqual(manifest["animation"]["final_hold_ms"], 3000)

    def test_presentation_code_cannot_launch_moose(self) -> None:
        for script in PRESENTATION_DIR.glob("*.py"):
            source = script.read_text(encoding="utf-8")
            self.assertNotIn("run_forward(", source)
            self.assertNotIn("from forward_solver import", source)
            self.assertNotIn("import forward_solver", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
