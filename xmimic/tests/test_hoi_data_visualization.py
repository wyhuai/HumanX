import unittest

import torch

from humanoidverse.utils.motion_data_control import (
    apply_motion_frame_to_state_tensors,
)


class HOIDataVisualizationTests(unittest.TestCase):
    def test_apply_motion_frame_to_state_tensors_updates_selected_env(self):
        env_ids = torch.tensor([1], dtype=torch.long)
        robot_root_states = torch.zeros((2, 13), dtype=torch.float32)
        object_root_state = torch.zeros((2, 13), dtype=torch.float32)
        dof_pos = torch.zeros((2, 3), dtype=torch.float32)
        dof_vel = torch.zeros((2, 3), dtype=torch.float32)

        motion_res = {
            "root_pos": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
            "root_rot": torch.tensor(
                [[0.0, 0.0, 0.0, 1.0], [0.1, 0.2, 0.3, 0.9]], dtype=torch.float32
            ),
            "root_vel": torch.tensor([[0.3, 0.2, 0.1], [1.0, 1.1, 1.2]], dtype=torch.float32),
            "root_ang_vel": torch.tensor(
                [[0.4, 0.5, 0.6], [1.3, 1.4, 1.5]], dtype=torch.float32
            ),
            "curr_obj_pos": torch.tensor(
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=torch.float32
            ),
            "curr_obj_rot": torch.tensor(
                [[0.0, 0.0, 0.0, 1.0], [0.6, 0.1, 0.2, 0.7]], dtype=torch.float32
            ),
            "curr_obj_vel": torch.tensor(
                [[0.7, 0.8, 0.9], [1.6, 1.7, 1.8]], dtype=torch.float32
            ),
            "curr_obj_ang_vel": torch.tensor(
                [[0.9, 0.8, 0.7], [1.9, 2.0, 2.1]], dtype=torch.float32
            ),
            "dof_pos": torch.tensor(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32
            ),
            "dof_vel": torch.tensor(
                [[0.6, 0.5, 0.4], [0.3, 0.2, 0.1]], dtype=torch.float32
            ),
        }

        apply_motion_frame_to_state_tensors(
            env_ids=env_ids,
            simulator_name="isaacgym",
            robot_root_states=robot_root_states,
            object_root_state=object_root_state,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            motion_res=motion_res,
        )

        self.assertTrue(torch.equal(robot_root_states[0], torch.zeros(13)))
        self.assertTrue(torch.equal(object_root_state[0], torch.zeros(13)))
        self.assertTrue(torch.equal(dof_pos[0], torch.zeros(3)))
        self.assertTrue(torch.equal(dof_vel[0], torch.zeros(3)))

        self.assertTrue(torch.equal(robot_root_states[1, :3], motion_res["root_pos"][1]))
        self.assertTrue(torch.equal(robot_root_states[1, 3:7], motion_res["root_rot"][1]))
        self.assertTrue(torch.equal(robot_root_states[1, 7:10], motion_res["root_vel"][1]))
        self.assertTrue(torch.equal(robot_root_states[1, 10:13], motion_res["root_ang_vel"][1]))
        self.assertTrue(torch.equal(object_root_state[1, :3], motion_res["curr_obj_pos"][1]))
        self.assertTrue(torch.equal(object_root_state[1, 3:7], motion_res["curr_obj_rot"][1]))
        self.assertTrue(torch.equal(object_root_state[1, 7:10], motion_res["curr_obj_vel"][1]))
        self.assertTrue(torch.equal(object_root_state[1, 10:13], motion_res["curr_obj_ang_vel"][1]))
        self.assertTrue(torch.equal(dof_pos[1], motion_res["dof_pos"][1]))
        self.assertTrue(torch.equal(dof_vel[1], motion_res["dof_vel"][1]))


if __name__ == "__main__":
    unittest.main()
