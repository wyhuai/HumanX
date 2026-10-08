import unittest
import warnings
from pathlib import Path

import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from humanoidverse.utils.motion_lib.motion_lib_base import MotionLibBase
from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch


class PoseAANormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        cfg = OmegaConf.load(
            repo_root / "humanoidverse/config/robot/g1/g1_29dof_model16.yaml"
        ).robot.motion
        cls.humanoid_batch = Humanoid_Batch(cfg)

    def test_accepts_base_joint_shape(self):
        pose = torch.randn(4, 30, 3)

        normalized = MotionLibBase._normalize_pose_aa_shape(
            pose,
            base_joints=30,
            extend_count=7,
            motion_id="unit-test",
        )

        self.assertEqual(normalized.shape, (4, 30, 3))
        self.assertTrue(torch.equal(normalized, pose))

    def test_truncates_legacy_extended_shape(self):
        pose = torch.randn(4, 37, 3)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            normalized = MotionLibBase._normalize_pose_aa_shape(
                pose,
                base_joints=30,
                extend_count=7,
                motion_id="unit-test",
            )

        self.assertEqual(normalized.shape, (4, 30, 3))
        self.assertTrue(torch.equal(normalized, pose[:, :30, :]))
        self.assertTrue(any("deprecated" in str(w.message) for w in caught))

    def test_rejects_invalid_shape(self):
        pose = torch.randn(4, 31, 3)

        with self.assertRaisesRegex(ValueError, "pose_aa"):
            MotionLibBase._normalize_pose_aa_shape(
                pose,
                base_joints=30,
                extend_count=7,
                motion_id="unit-test",
            )

    def test_optional_contact_defaults_to_half(self):
        contact = MotionLibBase._optional_contact_tensor(
            curr_file={},
            key="lfoot_contact",
            length=5,
        )

        self.assertEqual(contact.shape, (5, 1))
        self.assertEqual(contact.dtype, torch.float32)
        self.assertTrue(torch.equal(contact, torch.full((5, 1), 0.5, dtype=torch.float32)))

    def test_optional_contact_preserves_existing_values(self):
        raw = torch.tensor([[0.0], [1.0], [0.25]], dtype=torch.float32)

        contact = MotionLibBase._optional_contact_tensor(
            curr_file={"rfoot_contact": raw},
            key="rfoot_contact",
            length=3,
        )

        self.assertTrue(torch.equal(contact, raw))

    def test_optional_sequence_tensor_defaults_to_zeros(self):
        seq = MotionLibBase._optional_sequence_tensor(
            curr_file={},
            key="human_joints",
            length=5,
            trailing_shape=(24, 3),
        )

        self.assertEqual(seq.shape, (5, 24, 3))
        self.assertEqual(seq.dtype, torch.float32)
        self.assertTrue(torch.equal(seq, torch.zeros((5, 24, 3), dtype=torch.float32)))

    def test_optional_sequence_tensor_expands_single_frame(self):
        raw = torch.arange(24 * 3, dtype=torch.float32).view(1, 24, 3)

        seq = MotionLibBase._optional_sequence_tensor(
            curr_file={"human_joints": raw},
            key="human_joints",
            length=4,
            trailing_shape=(24, 3),
        )

        self.assertEqual(seq.shape, (4, 24, 3))
        self.assertTrue(torch.equal(seq[0], raw[0]))
        self.assertTrue(torch.equal(seq[3], raw[0]))

    def test_optional_sequence_tensor_or_none_preserves_existing_values(self):
        raw = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)

        seq = MotionLibBase._optional_sequence_tensor_or_none(
            curr_file={"obj_vel": raw},
            key="obj_vel",
            length=2,
            trailing_shape=(3,),
        )

        self.assertTrue(torch.equal(seq, raw))

    def test_optional_sequence_tensor_or_none_returns_none_when_missing(self):
        seq = MotionLibBase._optional_sequence_tensor_or_none(
            curr_file={},
            key="obj_vel",
            length=2,
            trailing_shape=(3,),
        )

        self.assertIsNone(seq)

    def test_openxr_translation_is_converted_to_isaac_world_frame(self):
        transl_openxr = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

        transl_isaac = MotionLibBase._openxr_to_isaac_translation(transl_openxr)

        self.assertTrue(
            torch.equal(
                transl_isaac,
                torch.tensor([[-3.0, -1.0, 2.0]], dtype=torch.float32),
            )
        )

    def test_human_joints_are_rebased_before_adding_openxr_root_offset(self):
        human_joints = torch.tensor(
            [[[0.1, 0.2, 0.3], [0.4, 0.8, 1.2]]],
            dtype=torch.float32,
        )
        transl_openxr = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

        world_joints = MotionLibBase._human_joints_world_from_local(
            human_joints,
            transl_openxr,
            z_offset=0.5,
        )

        self.assertTrue(
            torch.allclose(
                world_joints,
                torch.tensor(
                    [[[-3.0, -1.0, 2.5], [-2.7, -0.4, 3.4]]],
                    dtype=torch.float32,
                ),
            )
        )

    def test_human_joints_heading_alignment_preserves_root_position(self):
        human_joints_world = torch.tensor(
            [[[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]]],
            dtype=torch.float32,
        )
        source_root_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        target_root_rot = torch.from_numpy(
            R.from_euler("z", [90.0], degrees=True).as_quat().astype("float32")
        )

        aligned = MotionLibBase._align_human_joints_heading_to_target(
            human_joints_world,
            source_root_rot,
            target_root_rot,
        )

        self.assertTrue(
            torch.allclose(
                aligned,
                torch.tensor(
                    [[[1.0, 2.0, 3.0], [1.0, 3.0, 3.0]]],
                    dtype=torch.float32,
                ),
                atol=1e-5,
            )
        )

    def test_human_joints_heading_alignment_accepts_singleton_quat_axis(self):
        human_joints_world = torch.tensor(
            [[[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]]],
            dtype=torch.float32,
        )
        source_root_rot = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32)
        target_root_rot = torch.from_numpy(
            R.from_euler("z", [90.0], degrees=True).as_quat().astype("float32")
        )

        aligned = MotionLibBase._align_human_joints_heading_to_target(
            human_joints_world,
            source_root_rot,
            target_root_rot,
        )

        self.assertTrue(
            torch.allclose(
                aligned,
                torch.tensor(
                    [[[1.0, 2.0, 3.0], [1.0, 3.0, 3.0]]],
                    dtype=torch.float32,
                ),
                atol=1e-5,
            )
        )

    def test_human_joints_heading_alignment_accepts_extra_frame_axis(self):
        human_joints_world = torch.tensor(
            [[[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]]],
            dtype=torch.float32,
        )
        source_root_rot = torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        )
        target_root_rot = torch.from_numpy(
            R.from_euler("z", [[90.0], [90.0]], degrees=True).as_quat().astype("float32")
        ).view(1, 2, 4)

        aligned = MotionLibBase._align_human_joints_heading_to_target(
            human_joints_world,
            source_root_rot,
            target_root_rot,
        )

        self.assertTrue(
            torch.allclose(
                aligned,
                torch.tensor(
                    [[[1.0, 2.0, 3.0], [1.0, 3.0, 3.0]]],
                    dtype=torch.float32,
                ),
                atol=1e-5,
            )
        )

    def test_smpl_anchor_orientation_local_matches_identity_case(self):
        base_quat_current = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        init_base_quat_robot = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        init_ref_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        global_orient_quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32)

        rot6d = MotionLibBase._smpl_anchor_orientation_local_rot6d(
            base_quat_current,
            init_base_quat_robot,
            init_ref_quat,
            global_orient_quat,
        )

        self.assertTrue(
            torch.allclose(
                rot6d,
                torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
                atol=1e-5,
            )
        )

    def test_smpl_anchor_orientation_local_applies_heading_delta(self):
        base_quat_current = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        init_base_quat_robot = torch.from_numpy(
            R.from_euler("z", [90.0], degrees=True).as_quat().astype("float32")
        )
        init_ref_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        global_orient_quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32)

        rot6d = MotionLibBase._smpl_anchor_orientation_local_rot6d(
            base_quat_current,
            init_base_quat_robot,
            init_ref_quat,
            global_orient_quat,
        )

        self.assertTrue(
            torch.allclose(
                rot6d,
                torch.tensor([[0.0, -1.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
                atol=1e-5,
            )
        )

    def test_smpl_anchor_orientation_local_accepts_extra_frame_axis_on_init_quats(self):
        base_quat_current = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        init_base_quat_robot = torch.from_numpy(
            R.from_euler("z", [[90.0], [90.0]], degrees=True).as_quat().astype("float32")
        ).view(1, 2, 4)
        init_ref_quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32)
        global_orient_quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32)

        rot6d = MotionLibBase._smpl_anchor_orientation_local_rot6d(
            base_quat_current,
            init_base_quat_robot,
            init_ref_quat,
            global_orient_quat,
        )

        self.assertTrue(
            torch.allclose(
                rot6d,
                torch.tensor([[0.0, -1.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
                atol=1e-5,
            )
        )

    def test_object_angular_velocity_from_quat(self):
        angles = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float32)
        quats = torch.from_numpy(
            R.from_euler("z", angles.numpy()).as_quat().astype("float32")
        )

        ang_vel = MotionLibBase._compute_angular_velocity_from_quat(quats, fps=10)

        self.assertEqual(ang_vel.shape, (3, 3))
        self.assertAlmostEqual(float(ang_vel[1, 2]), 1.0, places=5)
        self.assertAlmostEqual(float(ang_vel[2, 2]), 1.0, places=5)

    def test_obj_rot_euler_triggers_deprecation_warning(self):
        curr_file = {
            "obj_rot_quat": torch.zeros((3, 4), dtype=torch.float32),
            "obj_rot_euler": torch.zeros((3, 3), dtype=torch.float32),
        }
        curr_file["obj_rot_quat"][:, 3] = 1.0

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            quat = MotionLibBase._required_obj_rot_quat_tensor(curr_file, motion_id="unit-test")

        self.assertEqual(quat.shape, (3, 4))
        self.assertTrue(any("obj_rot_euler" in str(w.message) for w in caught))

    def test_legacy_extended_pose_matches_truncated_fk_outputs(self):
        torch.manual_seed(0)
        pose_37 = torch.randn(8, 37, 3)
        trans = torch.randn(8, 3)
        pose_30 = pose_37[:, :30, :].clone()

        res_37 = self.humanoid_batch.fk_batch(
            pose_37.unsqueeze(0),
            trans.unsqueeze(0),
            return_full=True,
            dt=1 / 60,
        )
        res_30 = self.humanoid_batch.fk_batch(
            pose_30.unsqueeze(0),
            trans.unsqueeze(0),
            return_full=True,
            dt=1 / 60,
        )

        self.assertEqual(
            (res_37.global_translation_extend - res_30.global_translation_extend).abs().max().item(),
            0.0,
        )
        self.assertEqual(
            (res_37.global_rotation_extend - res_30.global_rotation_extend).abs().max().item(),
            0.0,
        )
        self.assertEqual(
            (res_37.dof_pos - res_30.dof_pos).abs().max().item(),
            0.0,
        )
        self.assertEqual(
            (res_37.dof_vels - res_30.dof_vels).abs().max().item(),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
