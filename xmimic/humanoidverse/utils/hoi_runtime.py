from __future__ import annotations

import torch


def get_rigid_bodies_per_env(rigid_body_state: torch.Tensor, num_envs: int) -> int:
    total_bodies = int(rigid_body_state.shape[0])
    if num_envs <= 0 or total_bodies % int(num_envs) != 0:
        raise ValueError(
            f"rigid body state shape {tuple(rigid_body_state.shape)} is incompatible with num_envs={num_envs}"
        )
    return total_bodies // int(num_envs)


def get_hoi_object_rigid_body_indices(num_robot_bodies: int):
    if num_robot_bodies < 0:
        raise ValueError(f"num_robot_bodies must be non-negative, got {num_robot_bodies}")
    ball_idx = int(num_robot_bodies)
    support_idx = ball_idx + 1
    return ball_idx, support_idx


def get_hoi_object_contact(contact_forces: torch.Tensor, ball_rigid_body_idx: int) -> torch.Tensor:
    return torch.any(
        torch.abs(contact_forces[:, int(ball_rigid_body_idx), 0:2]) > 0.1,
        dim=-1,
        keepdim=False,
    ).to(torch.float32)


def build_hoi_body_contact_force(
    contact_forces: torch.Tensor,
    ball_rigid_body_idx: int,
    hoi_no_contact_indices: torch.Tensor,
) -> torch.Tensor:
    ball_contact_force = contact_forces[:, int(ball_rigid_body_idx):int(ball_rigid_body_idx) + 1, :]
    return torch.cat([ball_contact_force, contact_forces[:, hoi_no_contact_indices, :]], dim=1)


def support_top_to_world_center(
    support_top_pos: torch.Tensor,
    env_origins: torch.Tensor,
    thickness: float,
) -> torch.Tensor:
    support_center = support_top_pos + env_origins
    support_center = support_center.clone()
    support_center[..., 2] = support_center[..., 2] - float(thickness) * 0.5
    return support_center


def apply_xy_only_init_noise(base_pos: torch.Tensor, noise_scale: float) -> torch.Tensor:
    noisy_pos = base_pos + torch.empty_like(base_pos).uniform_(-float(noise_scale), float(noise_scale))
    noisy_pos[:, 2] = base_pos[:, 2]
    return noisy_pos
