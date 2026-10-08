import unittest

import numpy as np
import torch
from easydict import EasyDict

from humanoidverse.utils.motion_lib.motion_lib_base import MotionLibBase
from humanoidverse.utils.motion_lib.motion_lib_robot_HOI import MotionLibRobotHOI


class MotionLoadPlanningTests(unittest.TestCase):
    def test_deduplicates_sampled_motion_indices_and_preserves_order_mapping(self):
        sample_idxes = np.asarray([7, 2, 7, 5, 2, 7], dtype=np.int64)

        unique_idxes, inverse_idxes = MotionLibBase._deduplicate_motion_indices(sample_idxes)

        np.testing.assert_array_equal(unique_idxes, np.asarray([7, 2, 5], dtype=np.int64))
        np.testing.assert_array_equal(
            inverse_idxes,
            np.asarray([0, 1, 0, 2, 1, 0], dtype=np.int64),
        )

    def test_hoi_loader_runs_skeleton_load_once_per_unique_motion(self):
        motion_lib = object.__new__(MotionLibRobotHOI)
        motion_lib.num_envs = 6
        motion_lib._device = torch.device("cpu")
        motion_lib.skeleton_tree = EasyDict(node_names=["root", "joint"])
        motion_lib._num_unique_motions = 3
        motion_lib._sampling_prob = torch.ones(3, dtype=torch.float32) / 3.0
        motion_lib._motion_data_list = np.asarray(
            [
                {"name": "a"},
                {"name": "b"},
                {"name": "c"},
            ],
            dtype=object,
        )
        motion_lib._motion_data_keys = np.asarray(["a", "b", "c"], dtype=object)
        motion_lib.fix_height = False
        motion_lib.has_action = False
        motion_lib.m_cfg = EasyDict()

        load_batch_sizes = []

        def make_motion():
            frames = 2
            joints = 2
            return EasyDict(
                fps=50,
                global_rotation=torch.zeros(frames, joints, 4),
                normalized_pose_aa=torch.zeros(frames, joints, 3),
                global_translation=torch.zeros(frames, joints, 3),
                local_rotation=torch.zeros(frames, joints, 4),
                global_root_velocity=torch.zeros(frames, 3),
                global_root_angular_velocity=torch.zeros(frames, 3),
                global_angular_velocity=torch.zeros(frames, joints, 3),
                global_velocity=torch.zeros(frames, joints, 3),
                dof_vels=torch.zeros(frames, 1),
                ref_origin_pos=torch.zeros(frames, 3),
                ref_origin_rot=torch.zeros(frames, 4),
            )

        def make_obj_status():
            frames = 2
            return EasyDict(
                obj_pos=torch.zeros(frames, 3),
                obj_rot_quat=torch.zeros(frames, 4),
                contact=torch.zeros(frames, 1),
                lfoot_contact=torch.zeros(frames, 1),
                rfoot_contact=torch.zeros(frames, 1),
                obj_vel=torch.zeros(frames, 3),
                obj_ang_vel=torch.zeros(frames, 3),
                obj_freeze=torch.zeros(frames, 1, dtype=torch.long),
                basket_pos=torch.zeros(frames, 3),
                skill_label=torch.zeros(frames, 32),
                need_support=torch.zeros(1, dtype=torch.long),
                support_top_pos=torch.zeros(3),
                human_joints=torch.zeros(frames, 0, 3),
                human_joints_valid=torch.zeros(frames, 1, dtype=torch.bool),
                human_root_rot=torch.zeros(frames, 4),
                human_root_rot_valid=torch.zeros(frames, 1, dtype=torch.bool),
                human_smpl_joints_local=torch.zeros(frames, 24, 3),
                human_smpl_joints_local_centered=torch.zeros(frames, 24, 3),
                human_smpl_pose=None,
                human_global_orient_quat=torch.zeros(frames, 4),
            )

        def fake_load_motion_with_skeleton(motion_data_list, fix_height, target_heading, max_len):
            load_batch_sizes.append(len(motion_data_list))
            return {
                idx: (motion_data, make_motion(), make_obj_status())
                for idx, motion_data in enumerate(motion_data_list)
            }

        motion_lib.load_motion_with_skeleton = fake_load_motion_with_skeleton

        motion_lib.load_motions(random_sample=False)

        self.assertEqual(load_batch_sizes, [3])
        self.assertEqual(motion_lib._num_motions, 6)
        self.assertEqual(motion_lib.gts.shape[0], 12)


if __name__ == "__main__":
    unittest.main()
