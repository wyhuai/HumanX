import unittest
from pathlib import Path

from humanoidverse.utils.motion_lib.vis_rr import resolve_robot_urdf_paths


class VisRrPathTests(unittest.TestCase):
    def test_resolve_robot_urdf_paths_returns_existing_paths(self):
        urdf_path, mesh_root = resolve_robot_urdf_paths("g1")

        self.assertTrue(Path(urdf_path).exists())
        self.assertTrue(Path(mesh_root).exists())


if __name__ == "__main__":
    unittest.main()
