import torch

from humanoidverse.utils.smpl_reference import (
    center_smpl_joints_local,
    resolve_smpl_joints_local_centered,
)


def test_center_smpl_joints_local_subtracts_pelvis_per_frame():
    joints = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
                [0.0, 1.0, 2.0],
            ],
            [
                [-1.0, 0.5, 2.0],
                [1.0, 0.5, 3.0],
                [-2.0, 2.5, 1.0],
            ],
        ]
    )

    centered = center_smpl_joints_local(joints)

    expected = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
                [-1.0, -1.0, -1.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 1.0],
                [-1.0, 2.0, -1.0],
            ],
        ]
    )
    assert torch.allclose(centered, expected)


def test_resolve_smpl_joints_local_centered_prefers_explicit_centered_data():
    raw = torch.tensor([[[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]])
    explicit_centered = torch.tensor([[[0.0, 0.0, 0.0], [0.25, 0.5, 0.75]]])

    centered = resolve_smpl_joints_local_centered(
        raw_joints_local=raw,
        explicit_centered_joints=explicit_centered,
    )

    assert centered is explicit_centered
