import unittest

import torch

from humanoidverse.utils.hoi_runtime import (
    apply_xy_only_init_noise,
    build_hoi_body_contact_force,
    get_hoi_object_contact,
    get_hoi_object_rigid_body_indices,
    get_rigid_bodies_per_env,
    support_top_to_world_center,
)


class HOIRuntimeTests(unittest.TestCase):
    def test_get_rigid_bodies_per_env_uses_actual_tensor_shape(self):
        rigid_body_state = torch.zeros((128, 13), dtype=torch.float32)

        bodies_per_env = get_rigid_bodies_per_env(rigid_body_state, num_envs=4)

        self.assertEqual(bodies_per_env, 32)

    def test_get_rigid_bodies_per_env_rejects_non_divisible_shape(self):
        rigid_body_state = torch.zeros((129, 13), dtype=torch.float32)

        with self.assertRaises(ValueError):
            get_rigid_bodies_per_env(rigid_body_state, num_envs=4)

    def test_get_hoi_object_rigid_body_indices_follow_robot_body_block(self):
        ball_idx, support_idx = get_hoi_object_rigid_body_indices(num_robot_bodies=30)

        self.assertEqual(ball_idx, 30)
        self.assertEqual(support_idx, 31)

    def test_build_hoi_body_contact_force_uses_ball_index_not_last_body(self):
        contact_forces = torch.zeros((2, 33, 3), dtype=torch.float32)
        hoi_no_contact_indices = torch.tensor([1, 2], dtype=torch.long)
        ball_idx, _ = get_hoi_object_rigid_body_indices(num_robot_bodies=30)

        contact_forces[0, ball_idx] = torch.tensor([0.2, 0.3, 0.4])
        contact_forces[0, 32] = torch.tensor([9.0, 9.0, 9.0])  # projectile or other extra body
        contact_force = build_hoi_body_contact_force(
            contact_forces,
            ball_rigid_body_idx=ball_idx,
            hoi_no_contact_indices=hoi_no_contact_indices,
        )

        self.assertTrue(torch.equal(contact_force[0, 0], torch.tensor([0.2, 0.3, 0.4])))
        self.assertTrue(torch.equal(contact_force[0, 1], contact_forces[0, 1]))
        self.assertTrue(torch.equal(contact_force[0, 2], contact_forces[0, 2]))

    def test_get_hoi_object_contact_uses_ball_index_not_last_body(self):
        contact_forces = torch.zeros((1, 33, 3), dtype=torch.float32)
        ball_idx, _ = get_hoi_object_rigid_body_indices(num_robot_bodies=30)

        contact_forces[0, ball_idx] = torch.tensor([0.2, 0.0, 0.0])
        contact_forces[0, 32] = torch.tensor([9.0, 0.0, 0.0])

        object_contact = get_hoi_object_contact(contact_forces, ball_rigid_body_idx=ball_idx)

        self.assertTrue(torch.equal(object_contact, torch.tensor([1.0], dtype=torch.float32)))

    def test_support_top_to_world_center_applies_env_origin_offset(self):
        support_top_pos = torch.tensor([[0.4, 1.2, 0.6], [0.4, 1.2, 0.6]], dtype=torch.float32)
        env_origins = torch.tensor([[0.0, 0.0, 0.0], [3.0, -2.0, 0.5]], dtype=torch.float32)

        support_center = support_top_to_world_center(
            support_top_pos,
            env_origins,
            thickness=0.02,
        )

        self.assertTrue(
            torch.allclose(
                support_center,
                torch.tensor([[0.4, 1.2, 0.59], [3.4, -0.8, 1.09]], dtype=torch.float32),
                atol=1e-6,
            )
        )

    def test_apply_xy_only_init_noise_preserves_z(self):
        base_pos = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.2]], dtype=torch.float32)
        noisy_pos = apply_xy_only_init_noise(base_pos, noise_scale=0.1)

        self.assertEqual(noisy_pos.shape, base_pos.shape)
        self.assertTrue(torch.equal(noisy_pos[:, 2], base_pos[:, 2]))
        self.assertTrue(torch.all(torch.abs(noisy_pos[:, :2] - base_pos[:, :2]) <= 0.1 + 1e-6))


if __name__ == "__main__":
    unittest.main()
