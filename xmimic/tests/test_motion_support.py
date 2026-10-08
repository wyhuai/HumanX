import unittest

import numpy as np

from humanoidverse.utils.motion_lib.motion_utils.support import (
    derive_support_top_pose_from_motion,
    parked_support_center,
    support_top_to_center,
)


class MotionSupportTests(unittest.TestCase):
    def test_derive_support_top_pose_uses_static_window_around_contact_change(self):
        obj_pos = np.asarray(
            [[0.4, 1.2, 0.7]] * 6
            + [[0.42, 1.22, 0.72], [0.45, 1.3, 0.9]]
            + [[0.4, 1.2, 0.7]] * 6,
            dtype=np.float32,
        )
        contact = np.asarray(
            [[0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1]],
            dtype=np.float32,
        )

        support_top = derive_support_top_pose_from_motion(
            obj_pos,
            contact,
            window_size=5,
            static_threshold=1e-4,
            z_offset=0.1,
        )

        np.testing.assert_allclose(support_top, [0.4, 1.2, 0.6], atol=1e-6)

    def test_derive_support_top_pose_requires_static_window(self):
        obj_pos = np.asarray(
            [[0.0, 0.0, 0.2], [0.1, 0.0, 0.2], [0.2, 0.0, 0.2], [0.3, 0.0, 0.2]],
            dtype=np.float32,
        )
        contact = np.asarray([[0], [0], [1], [1]], dtype=np.float32)

        with self.assertRaises(ValueError):
            derive_support_top_pose_from_motion(
                obj_pos,
                contact,
                window_size=2,
                static_threshold=1e-6,
                z_offset=0.1,
            )

    def test_support_top_to_center_subtracts_half_thickness(self):
        center = support_top_to_center(
            np.asarray([0.1, -0.2, 0.6], dtype=np.float32),
            thickness=0.02,
        )
        np.testing.assert_allclose(center, [0.1, -0.2, 0.59], atol=1e-6)

    def test_parked_support_center_moves_actor_far_below_env(self):
        env_origins = np.asarray([[1.0, 2.0, 0.5], [-1.0, 0.0, 1.0]], dtype=np.float32)
        parked = parked_support_center(env_origins, dz=100.0)
        np.testing.assert_allclose(
            parked,
            [[1.0, 2.0, -99.5], [-1.0, 0.0, -99.0]],
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
