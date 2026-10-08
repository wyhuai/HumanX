import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from humanoidverse.utils.motion_lib.motion_utils.set_fps import (
    _default_output_pkl_path,
    _format_override_fps_suffix,
    _parse_override_fps_factor,
    write_motion_with_overridden_fps,
)


class SetFpsTests(unittest.TestCase):
    def test_parse_override_fps_factor_accepts_p_and_dot_notation(self):
        self.assertAlmostEqual(_parse_override_fps_factor("0p7"), 0.7, places=6)
        self.assertAlmostEqual(_parse_override_fps_factor("1.3"), 1.3, places=6)

    def test_format_override_fps_suffix_uses_p_separator(self):
        self.assertEqual(_format_override_fps_suffix(0.7), "x0p7")
        self.assertEqual(_format_override_fps_suffix(1.25), "x1p25")

    def test_default_output_path_appends_fps_suffix(self):
        self.assertEqual(
            _default_output_pkl_path(Path("/tmp/motion_000073_modified.pkl"), 0.7),
            Path("/tmp/motion_000073_modified_x0p7.pkl"),
        )
        self.assertEqual(
            _default_output_pkl_path(Path("/tmp/motion_000073_modified_w_ball_w_contact.pkl"), 1.3),
            Path("/tmp/motion_000073_modified_x1p3_w_ball_w_contact.pkl"),
        )

    def test_write_motion_with_overridden_fps_updates_only_fps_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_pkl = tmpdir / "input.pkl"
            original = {
                "motion_a": {
                    "fps": 50.0,
                    "root_rot": np.zeros((2, 4), dtype=np.float32),
                    "root_trans_offset": np.zeros((2, 3), dtype=np.float32),
                    "pose_aa": np.zeros((2, 30, 3), dtype=np.float32),
                    "obj_pos": np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
                }
            }
            joblib.dump(original, input_pkl)

            output_path = write_motion_with_overridden_fps(
                input_pkl=input_pkl,
                factor=0.7,
                motion_key="motion_a",
            )

            self.assertEqual(output_path, tmpdir / "input_x0p7.pkl")
            out = joblib.load(output_path)["motion_a"]
            self.assertAlmostEqual(float(out["fps"]), 35.0, places=6)
            np.testing.assert_allclose(out["obj_pos"], original["motion_a"]["obj_pos"], atol=1e-6)
            np.testing.assert_allclose(out["root_rot"], original["motion_a"]["root_rot"], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
