"""Helpers for SMPL reference-joint observation tensors."""

import torch

def center_smpl_joints_local(joints_local: torch.Tensor, pelvis_index: int = 0) -> torch.Tensor:
    """Return per-frame SMPL joints centered at the pelvis joint."""
    if joints_local.ndim < 3:
        raise ValueError(
            "joints_local must have shape (..., num_joints, 3); "
            f"got {tuple(joints_local.shape)}"
        )
    if joints_local.shape[-1] != 3:
        raise ValueError(
            "joints_local must have trailing xyz dimension of 3; "
            f"got {tuple(joints_local.shape)}"
        )
    pelvis = joints_local[..., pelvis_index : pelvis_index + 1, :]
    return joints_local - pelvis


def resolve_smpl_joints_local_centered(
    raw_joints_local: torch.Tensor,
    explicit_centered_joints: torch.Tensor = None,
) -> torch.Tensor:
    """Use explicit centered joints when present, otherwise derive them from raw local joints."""
    if explicit_centered_joints is not None:
        return explicit_centered_joints
    return center_smpl_joints_local(raw_joints_local)
