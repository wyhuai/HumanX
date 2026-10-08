import os
import subprocess
import sys
import textwrap
import unittest


class IsaacGymHOIDOFPropsTests(unittest.TestCase):
    def test_process_dof_props_uses_robot_config_effort_limits(self):
        env = dict(os.environ)
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        env.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
        script = textwrap.dedent(
            """
            import numpy as np
            import types
            from humanoidverse.simulator.isaacgym.isaacgym_hoi import IsaacGym

            sim = IsaacGym.__new__(IsaacGym)
            sim.num_dof = 2
            sim.device = "cpu"
            sim.dof_names = ["joint_a", "joint_b"]
            sim.env_config = types.SimpleNamespace(
                rewards=types.SimpleNamespace(
                    reward_limit=types.SimpleNamespace(soft_dof_pos_limit=1.0)
                )
            )
            sim.robot_config = types.SimpleNamespace(
                dof_effort_limit_list=[50.0, 5.0],
                dof_armature_prop={"joint_a": 0.1, "joint_b": 0.2},
            )

            dtype = [
                ("lower", "f4"),
                ("upper", "f4"),
                ("velocity", "f4"),
                ("effort", "f4"),
                ("armature", "f4"),
            ]
            props = np.zeros(2, dtype=dtype)
            props["lower"] = [-1.0, -2.0]
            props["upper"] = [1.0, 2.0]
            props["velocity"] = [10.0, 20.0]
            props["effort"] = [35.0, 13.4]

            out = IsaacGym._process_dof_props(sim, props, env_id=0)

            np.testing.assert_allclose(out["effort"], [50.0, 5.0])
            np.testing.assert_allclose(out["armature"], [0.1, 0.2])
            np.testing.assert_allclose(sim.torque_limits.cpu().numpy(), [50.0, 5.0])
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            env=env,
        )
        if result.returncode != 0:
            self.fail(result.stderr or result.stdout)

    def test_process_dof_props_overrides_effort_limits_for_nonzero_env_ids(self):
        env = dict(os.environ)
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        env.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
        script = textwrap.dedent(
            """
            import numpy as np
            import types
            from humanoidverse.simulator.isaacgym.isaacgym_hoi import IsaacGym

            sim = IsaacGym.__new__(IsaacGym)
            sim.num_dof = 2
            sim.device = "cpu"
            sim.dof_names = ["joint_a", "joint_b"]
            sim.env_config = types.SimpleNamespace(
                rewards=types.SimpleNamespace(
                    reward_limit=types.SimpleNamespace(soft_dof_pos_limit=1.0)
                )
            )
            sim.robot_config = types.SimpleNamespace(
                dof_effort_limit_list=[50.0, 5.0],
                dof_armature_prop={"joint_a": 0.1, "joint_b": 0.2},
            )
            sim.torque_limits = None

            dtype = [
                ("lower", "f4"),
                ("upper", "f4"),
                ("velocity", "f4"),
                ("effort", "f4"),
                ("armature", "f4"),
            ]
            props = np.zeros(2, dtype=dtype)
            props["lower"] = [-1.0, -2.0]
            props["upper"] = [1.0, 2.0]
            props["velocity"] = [10.0, 20.0]
            props["effort"] = [35.0, 13.4]

            out = IsaacGym._process_dof_props(sim, props, env_id=1)

            np.testing.assert_allclose(out["effort"], [50.0, 5.0])
            np.testing.assert_allclose(out["armature"], [0.1, 0.2])
            assert sim.torque_limits is None
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            env=env,
        )
        if result.returncode != 0:
            self.fail(result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
