import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np

from humanoidverse.utils.motion_lib.vis_rr import (
    _ball_colors_from_contact,
    build_motion_data_for_vis,
    build_object_contact_for_vis,
    build_object_positions_for_vis,
    format_startup_summary,
    frame_time_seconds,
    prefetch_count,
    rebase_yaw,
    stream_sleep_seconds,
    build_arg_parser,
)


class VisRrMotionLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.sample_pkl = cls.repo_root / "data/motions/humanx_demo/BMaster_catch_yaw0_mean.pkl"
        cls.sample_data = joblib.load(cls.sample_pkl)
        cls.motion_key = next(iter(cls.sample_data))

    def test_pose_aa_only_motion_is_supported(self):
        motion = dict(self.sample_data[self.motion_key])
        motion.pop("dof", None)

        motion_data = build_motion_data_for_vis(
            motion,
            frame_mode="root-centered",
        )

        self.assertEqual(motion_data.shape[0], motion["pose_aa"].shape[0])
        self.assertEqual(motion_data.shape[1], 36)
        np.testing.assert_allclose(motion_data[:, :3], 0.0, atol=1e-6)

    def test_pose_aa_is_preferred_over_dof_when_both_exist(self):
        motion = dict(self.sample_data[self.motion_key])
        bad_dof = np.zeros_like(motion["dof"], dtype=np.float32)
        motion["dof"] = bad_dof

        motion_data = build_motion_data_for_vis(
            motion,
            frame_mode="root-centered",
        )

        dof_from_motion_data = motion_data[:, 7:]
        expected_dof = self.sample_data[self.motion_key]["dof"].reshape(motion_data.shape[0], -1)
        np.testing.assert_allclose(dof_from_motion_data, expected_dof, atol=1e-5)

    def test_frame_time_seconds_uses_fps(self):
        self.assertAlmostEqual(frame_time_seconds(30, 60), 0.5, places=6)

    def test_format_startup_summary_includes_total_frames_and_fps(self):
        summary = format_startup_summary(total_frames=121, fps=60)
        self.assertIn("total frames: 121", summary)
        self.assertIn("fps: 60", summary)

    def test_prefetch_count_is_capped_by_num_frames(self):
        self.assertEqual(prefetch_count(10, 50), 10)
        self.assertEqual(prefetch_count(100, 50), 50)

    def test_stream_sleep_seconds_matches_stream_mode(self):
        self.assertAlmostEqual(stream_sleep_seconds("realtime", 20), 0.05, places=6)
        self.assertEqual(stream_sleep_seconds("max", 20), 0.0)

    def test_parser_defaults_enable_streaming(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--filepath", "dummy.pkl"])
        self.assertTrue(args.stream)
        self.assertEqual(args.stream_rate, "realtime")
        self.assertEqual(args.prefetch_frames, 50)

    def test_world_frame_preserves_root_pose(self):
        motion = dict(self.sample_data[self.motion_key])

        motion_data = build_motion_data_for_vis(
            motion,
            frame_mode="world",
        )

        np.testing.assert_allclose(
            motion_data[:, :3],
            motion["root_trans_offset"],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            motion_data[:, 3:7],
            motion["root_rot"],
            atol=1e-6,
        )

    def test_world_frame_preserves_object_positions(self):
        motion = dict(self.sample_data[self.motion_key])

        object_positions = build_object_positions_for_vis(
            motion,
            frame_mode="world",
        )

        np.testing.assert_allclose(
            object_positions,
            motion["obj_pos"],
            atol=1e-6,
        )

    def test_world_frame_preserves_object_contact(self):
        motion = dict(self.sample_data[self.motion_key])
        motion["contact"] = np.asarray([[1.0], [0.0], [1.0]], dtype=np.float32)

        object_contact = build_object_contact_for_vis(motion)

        np.testing.assert_allclose(object_contact[:, 0], [1.0, 0.0, 1.0], atol=1e-6)

    def test_missing_contact_returns_none(self):
        motion = dict(self.sample_data[self.motion_key])
        motion.pop("contact", None)

        self.assertIsNone(build_object_contact_for_vis(motion))

    def test_root_centered_frame_zeroes_translation_and_rebases_yaw(self):
        motion = dict(self.sample_data[self.motion_key])

        motion_data = build_motion_data_for_vis(
            motion,
            frame_mode="root-centered",
        )

        np.testing.assert_allclose(motion_data[:, :3], 0.0, atol=1e-6)
        np.testing.assert_allclose(
            motion_data[:, 3:7],
            rebase_yaw(np.asarray(motion["root_rot"], dtype=np.float32)),
            atol=1e-6,
        )

    def test_root_centered_frame_rebases_object_positions(self):
        motion = dict(self.sample_data[self.motion_key])

        object_positions = build_object_positions_for_vis(
            motion,
            frame_mode="root-centered",
        )

        # Recover the same planar rotation used by the implementation via the original quaternion.
        # For the first frame, root-centered applies only the initial-yaw rebasing transform after
        # subtracting root translation.
        from humanoidverse.utils.motion_lib.vis_rr import quat_to_euler
        import math

        root_yaw0 = quat_to_euler(*np.asarray(motion["root_rot"], dtype=np.float32)[0])[2]
        c = math.cos(-root_yaw0)
        s = math.sin(-root_yaw0)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        expected_first = (motion["obj_pos"][0] - motion["root_trans_offset"][0]) @ rot.T
        np.testing.assert_allclose(object_positions[0], expected_first, atol=1e-5)

    def test_ball_colors_turn_red_only_when_contact_is_active(self):
        np.testing.assert_array_equal(
            _ball_colors_from_contact(True),
            np.asarray([255, 0, 0], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            _ball_colors_from_contact(False),
            np.asarray([255, 140, 0], dtype=np.uint8),
        )


if __name__ == "__main__":
    unittest.main()
