import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from humanoidverse.agents.ppo.bc_warmstart import (
    bc_expert_actions,
    collect_bc_snapshots,
    train_bc_actor,
)
from humanoidverse.agents.ppo.ppo import PPO


class _FakeEnv:
    def __init__(self, num_envs=2, obs_dim=3, action_dim=2):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.default_dof_pos = torch.tensor([[1.0, -2.0]], dtype=torch.float32).repeat(num_envs, 1)
        self.action_scales = torch.tensor([[0.5, 2.0]], dtype=torch.float32).repeat(num_envs, 1)
        self.expert_dof_pos = torch.zeros(num_envs, action_dim, dtype=torch.float32)
        self.reset_calls = 0
        self.step_calls = 0
        self.step_in_episode = 0
        self.actions_seen = []
        self._set_expert()

    def _label(self):
        base = torch.tensor(
            [float(self.reset_calls), float(self.step_in_episode)],
            dtype=torch.float32,
        )
        return base.unsqueeze(0).repeat(self.num_envs, 1)

    def _obs(self):
        row = torch.tensor(
            [
                float(self.reset_calls),
                float(self.step_in_episode),
                float(self.reset_calls + self.step_in_episode),
            ],
            dtype=torch.float32,
        )
        return {"actor_obs": row.unsqueeze(0).repeat(self.num_envs, 1)}

    def _set_expert(self):
        self.expert_dof_pos = self.default_dof_pos + self._label() * self.action_scales

    def reset_all(self):
        self.reset_calls += 1
        self.step_in_episode = 0
        self._set_expert()
        return self._obs()

    def step(self, actor_state):
        self.actions_seen.append(actor_state["actions"].detach().clone())
        self.step_calls += 1
        self.step_in_episode += 1
        self._set_expert()
        return (
            self._obs(),
            torch.zeros(self.num_envs, dtype=torch.float32),
            torch.zeros(self.num_envs, dtype=torch.bool),
            {"to_log": {}},
        )


class _TinyActor(torch.nn.Module):
    def __init__(self, obs_dim=3, action_dim=2):
        super().__init__()
        self.actor_module = torch.nn.Linear(obs_dim, action_dim)
        torch.nn.init.zeros_(self.actor_module.weight)
        torch.nn.init.zeros_(self.actor_module.bias)

    @property
    def actor(self):
        return self.actor_module


class _ResetMutatingEnv(_FakeEnv):
    def __init__(self, num_envs=2, obs_dim=3, action_dim=2):
        super().__init__(num_envs=num_envs, obs_dim=obs_dim, action_dim=action_dim)
        self.actions = torch.zeros(num_envs, action_dim, dtype=torch.float32)

    def reset_all(self):
        self.actions[:] = 0.0
        return super().reset_all()

    def step(self, actor_state):
        self.actions = actor_state["actions"]
        return super().step(actor_state)


class BCWarmstartTests(unittest.TestCase):
    def test_ppo_config_bc_warmstart_defaults_to_disabled_snapshot_collection(self):
        cfg = OmegaConf.load("humanoidverse/config/algo/ppo.yaml")

        self.assertFalse(bool(cfg.algo.config.bc_warmstart.enabled))
        self.assertEqual(int(cfg.algo.config.bc_warmstart.collect_horizon), 1)
        self.assertEqual(int(cfg.algo.config.bc_warmstart.history_warmup_steps), 4)
        self.assertTrue(bool(cfg.algo.config.bc_warmstart.reset_optimizer_after))

    def test_ppo_maybe_run_bc_warmstart_calls_helper_and_resets_for_ppo(self):
        ppo = object.__new__(PPO)
        ppo.config = SimpleNamespace(
            bc_warmstart=SimpleNamespace(enabled=True, reset_optimizer_after=True)
        )
        ppo.actor = _TinyActor()
        ppo.actor_learning_rate = 0.123
        ppo.actor_optimizer = object()
        ppo.device = "cpu"
        ppo.writer = None
        ppo.log_dir = "/tmp/humanx-bc-test"
        ppo._train_mode = lambda: None
        reset_obs = {"actor_obs": torch.ones(2, 3)}
        ppo.env = SimpleNamespace(reset_all=lambda: reset_obs)
        initial_obs = {"actor_obs": torch.zeros(2, 3)}

        with patch("humanoidverse.agents.ppo.ppo.run_bc_warmstart") as warmstart:
            warmstart.return_value = {"num_samples": 2, "initial_loss": 1.0, "final_loss": 0.5}
            returned_obs = PPO._maybe_run_bc_warmstart(ppo, initial_obs)

        warmstart.assert_called_once()
        self.assertIs(returned_obs["actor_obs"], reset_obs["actor_obs"])
        self.assertIsInstance(ppo.actor_optimizer, torch.optim.Adam)
        self.assertAlmostEqual(ppo.actor_optimizer.param_groups[0]["lr"], 0.123)

    def test_bc_expert_actions_normalizes_reference_dof_targets(self):
        env = _FakeEnv(num_envs=1)
        env.default_dof_pos = torch.tensor([[0.5, -0.5]])
        env.action_scales = torch.tensor([[0.25, 2.0]])
        env.expert_dof_pos = torch.tensor([[1.0, 3.5]])

        actions = bc_expert_actions(env, target_clip=None)

        torch.testing.assert_close(actions, torch.tensor([[2.0, 2.0]]))

    def test_collect_bc_snapshots_uses_reset_snapshots_after_history_warmup(self):
        env = _FakeEnv()
        cfg = SimpleNamespace(
            num_batches=3,
            history_warmup_steps=4,
            collect_horizon=1,
            target_clip=None,
        )

        obs, labels = collect_bc_snapshots(env=env, config=cfg, device="cpu")

        self.assertEqual(env.reset_calls, 3)
        self.assertEqual(env.step_calls, 12)
        self.assertEqual(tuple(obs.shape), (6, 3))
        self.assertEqual(tuple(labels.shape), (6, 2))
        torch.testing.assert_close(obs[:2], torch.tensor([[1.0, 4.0, 5.0], [1.0, 4.0, 5.0]]))
        torch.testing.assert_close(labels[:2], torch.tensor([[1.0, 4.0], [1.0, 4.0]]))
        torch.testing.assert_close(env.actions_seen[0], torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
        torch.testing.assert_close(env.actions_seen[3], torch.tensor([[1.0, 3.0], [1.0, 3.0]]))

    def test_collect_bc_snapshots_leaves_env_reset_buffers_mutable(self):
        env = _ResetMutatingEnv()
        cfg = SimpleNamespace(
            num_batches=2,
            history_warmup_steps=1,
            collect_horizon=1,
            target_clip=None,
        )

        obs, labels = collect_bc_snapshots(env=env, config=cfg, device="cpu")
        reset_obs = env.reset_all()

        self.assertEqual(tuple(obs.shape), (4, 3))
        self.assertEqual(tuple(labels.shape), (4, 2))
        self.assertEqual(tuple(reset_obs["actor_obs"].shape), (2, 3))
        torch.testing.assert_close(env.actions, torch.zeros_like(env.actions))

    def test_train_bc_actor_reduces_supervised_action_loss(self):
        actor = _TinyActor()
        obs = torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 2.0],
                [2.0, 1.0, 3.0],
            ],
            dtype=torch.float32,
        )
        labels = torch.stack((obs[:, 0] + obs[:, 1], obs[:, 2] - obs[:, 0]), dim=-1)
        cfg = SimpleNamespace(epochs=80, minibatch_size=4, lr=0.05, max_grad_norm=1.0)

        metrics = train_bc_actor(actor=actor, actor_obs=obs, labels=labels, config=cfg, device="cpu")

        self.assertGreater(metrics["initial_loss"], metrics["final_loss"] * 10.0)
        with torch.inference_mode():
            pred = actor.actor(obs)
        self.assertLess(torch.nn.functional.mse_loss(pred, labels).item(), 1e-2)


if __name__ == "__main__":
    unittest.main()
