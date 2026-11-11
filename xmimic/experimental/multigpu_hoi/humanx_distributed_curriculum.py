"""Experimental global reset statistics for the HumanX HOI benchmark.

All ranks must call the reward hook in the same order.  This module is not a
production trainer and intentionally rejects adaptive reference sampling.
"""
from types import MethodType

import torch
import torch.distributed as dist


class GlobalResetCurriculum:
    """Aggregate per-step completed-episode lengths before applying curricula."""

    def __init__(self, env):
        if getattr(env.config, "enable_adaptive_sampling", False):
            raise ValueError("adaptive reference sampling is not synchronized")
        if getattr(env.config, "resample_motion_when_training", False):
            raise ValueError("global motion resampling is not synchronized")
        self.env = env
        self.original_reward = env._compute_reward
        self.methods = {}
        names = (
            "_update_average_episode_length", "_update_reward_penalty_curriculum",
            "_update_reward_limits_curriculum", "_update_obs_noise_curriculum",
            "_update_terminate_when_motion_far_curriculum", "_update_terminate_when_ball_status_far_curriculum",
        )
        for name in names:
            self.methods[name] = getattr(env, name)
            setattr(env, name, lambda *args, **kwargs: None)
        self._broadcast_initial_state()

        def reward_hook(instance):
            result = self.original_reward()
            self.after_reward()
            return result

        env._compute_reward = MethodType(reward_hook, env)

    def _broadcast_initial_state(self):
        names = (
            "average_episode_length", "reward_penalty_scale", "soft_dof_pos_curriculum_value",
            "soft_dof_vel_curriculum_value", "soft_torque_curriculum_value", "current_noise_curriculum_value",
            "terminate_when_motion_far_threshold", "terminate_when_ball_status_far_threshold",
        )
        for name in names:
            if not hasattr(self.env, name):
                continue
            value = getattr(self.env, name)
            tensor = torch.as_tensor(value, device=self.env.device, dtype=torch.float64).clone()
            dist.broadcast(tensor, src=0)
            if isinstance(value, torch.Tensor):
                setattr(self.env, name, tensor.to(dtype=value.dtype))
            else:
                setattr(self.env, name, float(tensor.item()))

    def after_reward(self):
        env = self.env
        done = env.reset_buf > 0
        lengths = env.last_episode_length_buf.to(dtype=torch.float32)
        stats = torch.stack((done.sum(dtype=torch.float32), (lengths * done).sum()))
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        count = int(stats[0].item())
        if count == 0:
            return
        previous = env.average_episode_length
        mean = stats[1] / count
        fraction = count / env.num_compute_average_epl
        env.average_episode_length = previous * (1 - fraction) + mean * fraction
        if env.use_reward_penalty_curriculum:
            self.methods["_update_reward_penalty_curriculum"]()
        if (env.use_reward_limits_dof_pos_curriculum or env.use_reward_limits_dof_vel_curriculum
                or env.use_reward_limits_torque_curriculum):
            self.methods["_update_reward_limits_curriculum"]()
        if env.add_noise_currculum:
            self.methods["_update_obs_noise_curriculum"]()
        termination = env.config.termination
        curriculum = env.config.termination_curriculum
        if termination.terminate_when_motion_far and curriculum.terminate_when_motion_far_curriculum:
            self.methods["_update_terminate_when_motion_far_curriculum"]()
        if termination.terminate_when_ball_status_far and curriculum.terminate_when_ball_status_far_curriculum:
            self.methods["_update_terminate_when_ball_status_far_curriculum"]()

    def verify(self):
        """Assert that the shared scalar curriculum state agrees across ranks."""
        names = (
            "average_episode_length", "reward_penalty_scale", "soft_dof_pos_curriculum_value",
            "soft_dof_vel_curriculum_value", "soft_torque_curriculum_value", "current_noise_curriculum_value",
            "terminate_when_motion_far_threshold", "terminate_when_ball_status_far_threshold",
        )
        values = [torch.as_tensor(getattr(self.env, name), device=self.env.device, dtype=torch.float64)
                  for name in names if hasattr(self.env, name)]
        local = torch.stack(values)
        gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local)
        for other in gathered[1:]:
            torch.testing.assert_close(other, gathered[0], rtol=0, atol=1e-6)
