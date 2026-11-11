"""Exercise global curriculum updates when only one rank resets."""
import os
import socket
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from humanx_distributed_curriculum import GlobalResetCurriculum


class FakeEnv:
    def __init__(self, rank):
        self.device = "cpu"
        self.config = SimpleNamespace(
            enable_adaptive_sampling=False, resample_motion_when_training=False,
            termination=SimpleNamespace(terminate_when_motion_far=True, terminate_when_ball_status_far=False),
            termination_curriculum=SimpleNamespace(terminate_when_motion_far_curriculum=True,
                                                   terminate_when_ball_status_far_curriculum=False),
        )
        self.average_episode_length = 10.0 if rank == 0 else 99.0
        self.num_compute_average_epl = 100
        self.use_reward_penalty_curriculum = True
        self.use_reward_limits_dof_pos_curriculum = True
        self.use_reward_limits_dof_vel_curriculum = False
        self.use_reward_limits_torque_curriculum = False
        self.add_noise_currculum = False
        self.penalty_updates = 0
        self.limit_updates = 0
        self.termination_updates = 0
        self.reset_buf = torch.zeros(2, dtype=torch.bool)
        self.last_episode_length_buf = torch.zeros(2)

    def _compute_reward(self):
        return 42

    def _update_average_episode_length(self, env_ids):
        raise AssertionError("local average update must be suppressed")

    def _update_reward_penalty_curriculum(self):
        self.penalty_updates += 1

    def _update_reward_limits_curriculum(self):
        self.limit_updates += 1

    def _update_obs_noise_curriculum(self):
        raise AssertionError("noise curriculum is disabled")

    def _update_terminate_when_motion_far_curriculum(self):
        self.termination_updates += 1

    def _update_terminate_when_ball_status_far_curriculum(self):
        raise AssertionError("ball curriculum is disabled")


def worker(rank, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        env = FakeEnv(rank)
        GlobalResetCurriculum(env)
        if rank == 0:
            env.reset_buf[:] = True
            env.last_episode_length_buf[:] = torch.tensor([10.0, 20.0])
        assert env._compute_reward() == 42
        torch.testing.assert_close(torch.as_tensor(env.average_episode_length), torch.tensor(10.1))
        # A rank-local reset must not apply a second update.
        env._update_average_episode_length(torch.tensor([0]))
        env._update_reward_penalty_curriculum()
        assert (env.penalty_updates, env.limit_updates, env.termination_updates) == (1, 1, 1)
        env.reset_buf[:] = False
        if rank == 1:
            env.reset_buf[0] = True
            env.last_episode_length_buf[0] = 50
        env._compute_reward()
        torch.testing.assert_close(torch.as_tensor(env.average_episode_length), torch.tensor(10.499), atol=1e-5, rtol=0)
        assert (env.penalty_updates, env.limit_updates, env.termination_updates) == (2, 2, 2)
        env.reset_buf[:] = False
        env._compute_reward()
        assert (env.penalty_updates, env.limit_updates, env.termination_updates) == (2, 2, 2)
    finally:
        dist.destroy_process_group()


class DistributedCurriculumTests(unittest.TestCase):
    def test_global_reset_statistics_and_empty_rank(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(worker, args=(port,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
