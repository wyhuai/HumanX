import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from humanoidverse.utils.motion_lib.motion_utils.set_obj import (
    _default_output_pkl_path,
    _default_output_pkl_path_with_fps_override,
    _ball_vertex_colors,
    _calc_heading_from_quat,
    _palm_center_line_radius_ui_points,
    _palm_center_wireframe,
    _palm_center_radius,
    _palm_center_vertex_colors,
    _resolve_fk_device,
    build_updated_frame_state,
    build_ballistic_trajectory,
    custom_frame_state,
    get_current_rule_context,
    scale_ballistic_speed,
    write_motion_with_updated_obj_pos,
)


class SetObjTests(unittest.TestCase):
    def test_default_output_path_appends_w_ball_suffix(self):
        self.assertEqual(
            _default_output_pkl_path(Path("/tmp/motion_000005_modified.pkl")),
            Path("/tmp/motion_000005_modified_w_ball_w_contact.pkl"),
        )
        self.assertEqual(
            _default_output_pkl_path(Path("/tmp/motion_000005_modified_w_ball.pkl")),
            Path("/tmp/motion_000005_modified_w_ball_w_contact.pkl"),
        )

    def test_default_output_path_with_fps_override_inserts_suffix_before_ball_suffix(self):
        self.assertEqual(
            _default_output_pkl_path_with_fps_override(
                Path("/tmp/motion_000005_modified.pkl"),
                override_fps_factor=0.7,
            ),
            Path("/tmp/motion_000005_modified_x0p7_w_ball_w_contact.pkl"),
        )
        self.assertEqual(
            _default_output_pkl_path_with_fps_override(
                Path("/tmp/motion_000005_modified_w_ball.pkl"),
                override_fps_factor=1.25,
            ),
            Path("/tmp/motion_000005_modified_w_ball_x1p25_w_contact.pkl"),
        )

    def test_default_fk_device_prefers_cpu(self):
        self.assertEqual(_resolve_fk_device(None), "cpu")

    def test_explicit_fk_device_is_preserved(self):
        self.assertEqual(_resolve_fk_device("cuda:0"), "cuda:0")

    def test_ball_vertex_colors_turn_red_on_contact(self):
        np.testing.assert_array_equal(_ball_vertex_colors(True), np.asarray([255, 0, 0], dtype=np.uint8))
        np.testing.assert_array_equal(_ball_vertex_colors(False), np.asarray([255, 140, 0], dtype=np.uint8))

    def test_palm_center_vertex_colors_use_higher_transparency(self):
        np.testing.assert_array_equal(
            _palm_center_vertex_colors(False),
            np.asarray([0, 220, 220, 40], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            _palm_center_vertex_colors(True),
            np.asarray([0, 220, 220, 40], dtype=np.uint8),
        )

    def test_palm_center_radius_keeps_original_size(self):
        self.assertAlmostEqual(_palm_center_radius(), 0.10, places=6)

    def test_palm_center_line_radius_uses_half_ui_point(self):
        self.assertAlmostEqual(_palm_center_line_radius_ui_points(), 0.5, places=6)

    def test_palm_center_wireframe_builds_three_circles_at_radius(self):
        center = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        wire = _palm_center_wireframe(center, radius=0.04, latitude_segments=6, longitude_segments=8)
        self.assertEqual(wire.shape, (13, 9, 3))
        first_xy = wire[0, 0]
        self.assertAlmostEqual(first_xy[0], 1.02, places=5)
        self.assertAlmostEqual(first_xy[1], 2.0, places=5)
        self.assertAlmostEqual(first_xy[2], 2.96535898, places=5)

    def test_build_updated_frame_state_uses_fps_for_motion_time(self):
        palm_centers = np.asarray(
            [
                [0.0, 0.0, 0.1],
                [1.0, 1.0, 0.2],
                [2.0, 2.0, 0.3],
            ],
            dtype=np.float32,
        )
        motion = {"fps": 20}

        def custom_rule(frame_idx, motion_time, palm_center, fps):
            self.assertEqual(fps, 20.0)
            if motion_time < 0.1:
                return {
                    "obj_pos": np.asarray([9.0, 9.0, 9.0], dtype=np.float32),
                    "contact": 1.0,
                }
            return {
                "obj_pos": palm_center,
                "contact": 0.0,
            }

        updated_frame_state = build_updated_frame_state(
            motion=motion,
            palm_centers=palm_centers,
            rule_fn=custom_rule,
        )
        updated_obj_pos = updated_frame_state["obj_pos"]
        updated_contact = updated_frame_state["contact"]

        np.testing.assert_allclose(updated_obj_pos[0], [9.0, 9.0, 9.0], atol=1e-6)
        np.testing.assert_allclose(updated_obj_pos[1], [9.0, 9.0, 9.0], atol=1e-6)
        np.testing.assert_allclose(updated_obj_pos[2], palm_centers[2], atol=1e-6)
        np.testing.assert_allclose(updated_contact[:, 0], [1.0, 1.0, 0.0], atol=1e-6)

    def test_build_updated_frame_state_allows_partial_dict_updates(self):
        palm_centers = np.asarray(
            [
                [10.0, 10.0, 0.1],
                [20.0, 20.0, 0.2],
            ],
            dtype=np.float32,
        )
        motion = {
            "fps": 20,
            "obj_pos": np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "contact": np.asarray([[0.0], [1.0]], dtype=np.float32),
        }

        def custom_rule(frame_idx, motion_time, palm_center, fps):
            del motion_time, palm_center, fps
            if frame_idx == 0:
                return {"contact": 1.0}
            return {"obj_pos": np.asarray([7.0, 8.0, 9.0], dtype=np.float32)}

        updated_frame_state = build_updated_frame_state(
            motion=motion,
            palm_centers=palm_centers,
            rule_fn=custom_rule,
        )
        updated_obj_pos = updated_frame_state["obj_pos"]
        updated_contact = updated_frame_state["contact"]

        np.testing.assert_allclose(updated_obj_pos[0], [1.0, 2.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(updated_obj_pos[1], [7.0, 8.0, 9.0], atol=1e-6)
        np.testing.assert_allclose(updated_contact[:, 0], [1.0, 1.0], atol=1e-6)

    def test_build_updated_frame_state_exposes_rule_context(self):
        palm_centers = np.asarray([[1.0, 2.0, 0.3], [4.0, 5.0, 0.6]], dtype=np.float32)
        motion = {
            "fps": 20,
            "root_rot": np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        }

        def custom_rule(frame_idx, motion_time, palm_center, fps):
            del motion_time, palm_center, fps
            context = get_current_rule_context()
            self.assertIs(context["motion"], motion)
            np.testing.assert_allclose(context["palm_centers"], palm_centers, atol=1e-6)
            np.testing.assert_allclose(context["motion"]["root_rot"], motion["root_rot"], atol=1e-6)
            self.assertAlmostEqual(context["fps_override_factor"], 1.3, places=6)
            self.assertAlmostEqual(scale_ballistic_speed(5.0), 6.5, places=6)
            return {"obj_pos": np.asarray([frame_idx, 0.0, 0.2], dtype=np.float32)}

        updated_frame_state = build_updated_frame_state(
            motion=motion,
            palm_centers=palm_centers,
            rule_fn=custom_rule,
            fps_override_factor=1.3,
        )

        np.testing.assert_allclose(
            updated_frame_state["obj_pos"],
            np.asarray([[0.0, 0.0, 0.2], [1.0, 0.0, 0.2]], dtype=np.float32),
            atol=1e-6,
        )

    def test_calc_heading_from_quat_rotates_body_x_axis_into_world_xy(self):
        quarter_turn_yaw = np.asarray([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=np.float32)

        heading = _calc_heading_from_quat(quarter_turn_yaw)

        self.assertAlmostEqual(float(heading), np.pi / 2.0, places=6)

    def test_build_ballistic_trajectory_generates_frames_before_and_after_anchor(self):
        root_rot = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (5, 1))

        trajectory = build_ballistic_trajectory(
            root_rot=root_rot,
            event_frame=2,
            event_pos=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            speed=1.0,
            relative_heading_deg=0.0,
            elevation_deg=0.0,
            fps=10.0,
        )

        self.assertEqual(trajectory.shape, (5, 3))
        np.testing.assert_allclose(trajectory[2], [0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(trajectory[1], [-0.1, 0.0, 0.95095], atol=1e-5)
        np.testing.assert_allclose(trajectory[3], [0.1, 0.0, 0.95095], atol=1e-5)
        np.testing.assert_allclose(trajectory[0], [-0.2, 0.0, 0.8038], atol=1e-4)
        np.testing.assert_allclose(trajectory[4], [0.2, 0.0, 0.8038], atol=1e-4)

    def test_build_ballistic_trajectory_keeps_horizontal_speed_through_bounce(self):
        root_rot = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (6, 1))

        trajectory = build_ballistic_trajectory(
            root_rot=root_rot,
            event_frame=2,
            event_pos=np.asarray([0.0, 0.0, 0.12], dtype=np.float32),
            speed=2.0,
            relative_heading_deg=0.0,
            elevation_deg=-80.0,
            fps=30.0,
        )

        self.assertGreaterEqual(float(np.min(trajectory[:, 2])), 0.099999)
        np.testing.assert_allclose(np.diff(trajectory[:, 0]), np.diff(trajectory[:, 0])[0], atol=1e-6)

    def test_build_ballistic_trajectory_negative_speed_reverses_anchor_velocity_direction(self):
        root_rot = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (3, 1))

        trajectory = build_ballistic_trajectory(
            root_rot=root_rot,
            event_frame=1,
            event_pos=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            speed=-1.0,
            relative_heading_deg=0.0,
            elevation_deg=0.0,
            fps=10.0,
        )

        np.testing.assert_allclose(trajectory[0], [0.1, 0.0, 0.95095], atol=1e-5)
        np.testing.assert_allclose(trajectory[2], [-0.1, 0.0, 0.95095], atol=1e-5)

    def test_custom_frame_state_clamps_palm_center_z_to_minimum(self):
        palm_center = np.asarray([1.0, 2.0, 0.05], dtype=np.float32)

        state = custom_frame_state(
            frame_idx=10,
            motion_time=4.0,
            palm_center=palm_center,
            fps=50.0,
        )

        np.testing.assert_allclose(state["obj_pos"], [1.0, 2.0, 0.10], atol=1e-6)
        self.assertEqual(state["contact"], 1.0)

    def test_custom_frame_state_uses_built_in_segments(self):
        before_state = custom_frame_state(
            frame_idx=10,
            motion_time=2.0,
            palm_center=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            fps=50.0,
        )
        after_state = custom_frame_state(
            frame_idx=100,
            motion_time=4.0,
            palm_center=np.asarray([1.0, 2.0, 0.05], dtype=np.float32),
            fps=50.0,
        )

        np.testing.assert_allclose(before_state["obj_pos"], [1.0, 2.0, 3.0], atol=1e-6)
        self.assertEqual(before_state["contact"], 1.0)
        np.testing.assert_allclose(after_state["obj_pos"], [0.4347, 1.4841, 0.6990], atol=1e-6)
        self.assertEqual(after_state["contact"], 0.0)

    def test_write_motion_with_updated_obj_pos_preserves_other_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_pkl = tmpdir / "input.pkl"
            output_pkl = tmpdir / "output.pkl"
            original = {
                "motion_a": {
                    "fps": 30,
                    "root_trans_offset": np.zeros((2, 3), dtype=np.float32),
                    "root_rot": np.zeros((2, 4), dtype=np.float32),
                    "pose_aa": np.zeros((2, 30, 3), dtype=np.float32),
                    "obj_pos": np.zeros((2, 3), dtype=np.float32),
                    "contact": np.zeros((2, 1), dtype=np.float32),
                    "obj_rot_quat": np.ones((2, 4), dtype=np.float32),
                    "custom_key": np.asarray([1, 2, 3], dtype=np.int32),
                }
            }
            joblib.dump(original, input_pkl)

            updated_obj_pos = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
            updated_contact = np.asarray([[1.0], [0.0]], dtype=np.float32)
            written_path = write_motion_with_updated_obj_pos(
                input_pkl=input_pkl,
                motion_key="motion_a",
                updated_obj_pos=updated_obj_pos,
                updated_contact=updated_contact,
                output_pkl=output_pkl,
            )

            self.assertEqual(written_path, output_pkl)
            out = joblib.load(output_pkl)["motion_a"]
            np.testing.assert_allclose(out["obj_pos"], updated_obj_pos, atol=1e-6)
            np.testing.assert_allclose(out["contact"], updated_contact, atol=1e-6)
            np.testing.assert_allclose(out["obj_rot_quat"], original["motion_a"]["obj_rot_quat"], atol=1e-6)
            np.testing.assert_allclose(out["root_trans_offset"], original["motion_a"]["root_trans_offset"], atol=1e-6)
            np.testing.assert_array_equal(out["custom_key"], original["motion_a"]["custom_key"])

    def test_write_motion_with_updated_obj_pos_can_override_saved_fps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_pkl = tmpdir / "input.pkl"
            output_pkl = tmpdir / "output.pkl"
            original = {
                "motion_a": {
                    "fps": 50.0,
                    "root_trans_offset": np.zeros((2, 3), dtype=np.float32),
                    "root_rot": np.zeros((2, 4), dtype=np.float32),
                    "pose_aa": np.zeros((2, 30, 3), dtype=np.float32),
                    "obj_pos": np.zeros((2, 3), dtype=np.float32),
                    "contact": np.zeros((2, 1), dtype=np.float32),
                }
            }
            joblib.dump(original, input_pkl)

            write_motion_with_updated_obj_pos(
                input_pkl=input_pkl,
                motion_key="motion_a",
                updated_obj_pos=np.zeros((2, 3), dtype=np.float32),
                updated_contact=np.zeros((2, 1), dtype=np.float32),
                output_pkl=output_pkl,
                motion_overrides={"fps": 35.0},
            )

            out = joblib.load(output_pkl)["motion_a"]
            self.assertAlmostEqual(float(out["fps"]), 35.0, places=6)


if __name__ == "__main__":
    unittest.main()
