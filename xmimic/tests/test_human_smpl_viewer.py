import os
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from tools.human_smpl_viewer.server import discover_motions, load_motion_payload


class HumanSmplViewerTests(unittest.TestCase):
    def test_discover_motions_recurses_and_returns_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion_dir = root / "group" / "clip"
            motion_dir.mkdir(parents=True)
            joblib.dump({"clip": {"human_smpl_joints_local": np.zeros((2, 24, 3), dtype=np.float32)}}, motion_dir / "a.pkl")
            (root / "ignore.txt").write_text("not a motion")

            motions = discover_motions(root)

        self.assertEqual(
            motions,
            [{"path": "group/clip/a.pkl", "name": "group/clip/a.pkl"}],
        )

    def test_load_motion_payload_extracts_nested_human_smpl_joints_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = np.arange(5 * 24 * 3, dtype=np.float32).reshape(5, 24, 3)
            obj_pos = np.ones((5, 3), dtype=np.float32)
            global_orient = np.tile(
                np.array([[0.5, 0.1, 0.2, 0.3]], dtype=np.float32),
                (5, 1),
            )
            motion_path = root / "walk.pkl"
            joblib.dump(
                {
                    "walk": {
                        "human_smpl_joints_local": data,
                        "human_global_orient_quat": global_orient,
                        "fps": 50,
                        "obj_pos": obj_pos,
                    }
                },
                motion_path,
            )

            payload = load_motion_payload(root, "walk.pkl")

        self.assertEqual(payload["path"], "walk.pkl")
        self.assertEqual(payload["motion_key"], "walk")
        self.assertEqual(payload["fps"], 50)
        self.assertEqual(payload["num_frames"], 5)
        self.assertEqual(payload["num_joints"], 24)
        self.assertEqual(payload["joints"][0][20], data[0, 20].tolist())
        self.assertEqual(payload["obj_pos"][0], obj_pos[0].tolist())
        self.assertEqual(payload["human_global_orient_quat"][0], global_orient[0].tolist())
        self.assertEqual(payload["human_global_orient_quat_order"], "wxyz")
        np.testing.assert_allclose(payload["human_global_orient_quat_xyzw"][0], [0.1, 0.2, 0.3, 0.5])


if __name__ == "__main__":
    unittest.main()
