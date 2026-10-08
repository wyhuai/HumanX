import joblib
import time
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from humanoidverse.agents.modules.amp_modules import Discriminator, Normalizer
from humanoidverse.agents.ppo.ppo import PPO as _PPOBase


class AMP_PPO(_PPOBase):
    """PPO with AMP-specific discriminator reward and training losses."""

    def _init_config(self):
        super()._init_config()
        self.disc_learning_rate = self.config.disc_learning_rate
        self.amploss_coef = self.config.amploss_coef
        # Backward-compatible default matches Discriminator.compute_grad_pen default.
        self.disc_grad_penalty_lambda = getattr(self.config, "disc_grad_penalty_lambda", 10)

    def _setup_models_and_optimizer(self):
        super()._setup_models_and_optimizer()

        # AMP discriminator and its optimizer.
        self.disc = Discriminator(
            self.algo_obs_dim_dict,
            self.config.module_dict.Discriminator,
            self.device,
        ).to(self.device)

        params = [
            {"params": self.disc.trunk.parameters(), "weight_decay": 10e-4, "name": "amp_trunk"},
            {"params": self.disc.amp_linear.parameters(), "weight_decay": 10e-2, "name": "amp_head"},
        ]
        self.disc_optimizer = optim.Adam(params, lr=self.disc_learning_rate)

        self.amp_obs_normalizer = Normalizer(
            self.algo_obs_dim_dict,
            self.config.module_dict.amp_normalizer,
            device=self.device,
        )

    def _storage_observation_keys(self):
        keys = super()._storage_observation_keys()
        for obs_key in ("amp_obs", "amp_expert_obs"):
            if obs_key in self.algo_obs_dim_dict and obs_key not in keys:
                keys.append(obs_key)
        return keys

    def _eval_mode(self):
        super()._eval_mode()
        self.disc.eval()

    def _train_mode(self):
        super()._train_mode()
        self.disc.train()

    def load(self, ckpt_path):
        if ckpt_path is None:
            return None

        logger.info(f"Loading checkpoint from {ckpt_path}")
        loaded_dict = torch.load(ckpt_path, map_location=self.device)

        if "actor_model_state_dict" in loaded_dict:
            self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
        else:
            raise KeyError("Missing key 'actor_model_state_dict' in checkpoint")

        if "critic_model_state_dict" in loaded_dict:
            self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])
        else:
            raise KeyError("Missing key 'critic_model_state_dict' in checkpoint")

        # Backward-compatible warm start: AMP can load checkpoints without discriminator.
        if "disc_model_state_dict" in loaded_dict:
            self.disc.load_state_dict(loaded_dict["disc_model_state_dict"])
        else:
            logger.warning(
                "Checkpoint has no 'disc_model_state_dict'. "
                "Initializing discriminator from scratch (AMP warm-start)."
            )

        if self.load_optimizer:
            if "actor_optimizer_state_dict" in loaded_dict:
                self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
                self.actor_learning_rate = loaded_dict["actor_optimizer_state_dict"]["param_groups"][0]["lr"]
            else:
                logger.warning("Checkpoint has no 'actor_optimizer_state_dict'; optimizer will start fresh.")

            if "critic_optimizer_state_dict" in loaded_dict:
                self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
                self.critic_learning_rate = loaded_dict["critic_optimizer_state_dict"]["param_groups"][0]["lr"]
            else:
                logger.warning("Checkpoint has no 'critic_optimizer_state_dict'; optimizer will start fresh.")

            if "disc_optimizer_state_dict" in loaded_dict:
                self.disc_optimizer.load_state_dict(loaded_dict["disc_optimizer_state_dict"])
                self.disc_learning_rate = loaded_dict["disc_optimizer_state_dict"]["param_groups"][0]["lr"]
            else:
                logger.warning("Checkpoint has no 'disc_optimizer_state_dict'; disc optimizer will start fresh.")

            self.set_learning_rate(self.actor_learning_rate, self.critic_learning_rate)
            for param_group in self.disc_optimizer.param_groups:
                param_group["lr"] = self.disc_learning_rate

            logger.info("Optimizer loaded from checkpoint")
            logger.info(f"Actor Learning rate: {self.actor_learning_rate}")
            logger.info(f"Critic Learning rate: {self.critic_learning_rate}")
            logger.info(f"Disc Learning rate: {self.disc_learning_rate}")

        if "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]
        else:
            logger.warning("Checkpoint has no 'iter'; starting from iteration 0.")
            self.current_learning_iteration = 0

        return loaded_dict.get("infos", None)

    def save(self, path, infos=None):
        logger.info(f"Saving checkpoint to {path}")
        torch.save(
            {
                "actor_model_state_dict": self.actor.state_dict(),
                "critic_model_state_dict": self.critic.state_dict(),
                "disc_model_state_dict": self.disc.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "disc_optimizer_state_dict": self.disc_optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

        self.save_logs(ckpt_path=path)

    def save_logs(self, ckpt_path):
        """Write last.pkl so user can resume via algo.config.resume_path."""
        if self.log_dir is None:
            return

        log_data = {}
        log_data["checkpoints_path"] = ckpt_path

        if hasattr(self.env, "terminate_when_motion_far_threshold"):
            log_data["terminate_when_motion_far_threshold"] = self.env.terminate_when_motion_far_threshold
        if hasattr(self.env, "terminate_when_ball_status_far_threshold"):
            log_data["terminate_when_ball_status_far_threshold"] = self.env.terminate_when_ball_status_far_threshold
        if hasattr(self.env, "soft_dof_pos_curriculum_value"):
            log_data["soft_dof_pos_curriculum_value"] = self.env.soft_dof_pos_curriculum_value
        if hasattr(self.env, "soft_dof_vel_curriculum_value"):
            log_data["soft_dof_vel_curriculum_value"] = self.env.soft_dof_vel_curriculum_value
        if hasattr(self.env, "soft_torque_curriculum_value"):
            log_data["soft_torque_curriculum_value"] = self.env.soft_torque_curriculum_value
        if hasattr(self.env, "reward_penalty_scale"):
            log_data["reward_penalty_scale"] = self.env.reward_penalty_scale

        # Keep backward-compatible if periodic save did not run yet.
        log_data["resume_it"] = getattr(self, "log_it", self.current_learning_iteration)
        joblib.dump(log_data, f"{self.log_dir}/last.pkl")

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for _ in range(self.num_steps_per_env):
                policy_state_dict = {}
                policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
                values = self._critic_eval_step(obs_dict).detach()
                policy_state_dict["values"] = values

                # Append observations and policy states.
                for obs_key in obs_dict.keys():
                    if hasattr(self.storage, obs_key):
                        self.storage.update_key(obs_key, obs_dict[obs_key])
                for obs_key in policy_state_dict.keys():
                    self.storage.update_key(obs_key, policy_state_dict[obs_key])

                actor_state = {"actions": policy_state_dict["actions"]}
                obs_dict, rewards, dones, infos = self.env.step(actor_state)

                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                # AMP reward shaping: blend task reward with discriminator-based prior.
                rewards = self.disc.predict_amp_reward(
                    obs_dict["amp_obs"], rewards, normalizer=self.amp_obs_normalizer
                )[0]

                self.episode_env_tensors.add(infos["to_log"])
                rewards_stored = rewards.clone().unsqueeze(1)
                if "time_outs" in infos:
                    rewards_stored += (
                        self.gamma * policy_state_dict["values"] * infos["time_outs"].unsqueeze(1).to(self.device)
                    )

                self.storage.update_key("rewards", rewards_stored)
                self.storage.update_key("dones", dones.unsqueeze(1))
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)

                if self.log_dir is not None:
                    if "episode" in infos:
                        self.ep_infos.append(infos["episode"])
                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time

            returns, advantages = self._compute_returns(
                last_obs_dict=obs_dict,
                policy_state_dict=dict(
                    values=self.storage.query_key("values"),
                    dones=self.storage.query_key("dones"),
                    rewards=self.storage.query_key("rewards"),
                ),
            )
            self.storage.batch_update_data("returns", returns)
            self.storage.batch_update_data("advantages", advantages)

        return obs_dict

    def _init_loss_dict_at_training_step(self):
        loss_dict = {}
        loss_dict["policy_d"] = 0
        loss_dict["expert_d"] = 0
        loss_dict["amp_grad_pen_loss"] = 0
        # Average KL( pi_old || pi_new ) over all minibatch updates in this iteration.
        loss_dict["KL"] = 0
        return loss_dict

    def _update_ppo(self, policy_state_dict, loss_dict):
        actions_batch = policy_state_dict["actions"]
        policy_actions_batch = policy_state_dict["policy_actions"]
        target_values_batch = policy_state_dict["values"]
        advantages_batch = policy_state_dict["advantages"]
        returns_batch = policy_state_dict["returns"]
        old_actions_log_prob_batch = policy_state_dict["actions_log_prob"]
        old_mu_batch = policy_state_dict["action_mean"]
        old_sigma_batch = policy_state_dict["action_sigma"]

        self._actor_act_step(policy_state_dict)
        actions_log_prob_batch = self.actor.get_actions_log_prob(policy_actions_batch)
        value_batch = self._critic_eval_step(policy_state_dict)
        mu_batch = self.actor.action_mean
        sigma_batch = self.actor.action_std
        entropy_batch = self.actor.entropy

        # KL(pi_old || pi_new), used for adaptive LR schedule and convergence diagnosis.
        with torch.inference_mode():
            eps = 1.0e-5
            kl = torch.sum(
                torch.log((sigma_batch + eps) / (old_sigma_batch + eps))
                + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                / (2.0 * torch.square(sigma_batch + eps))
                - 0.5,
                axis=-1,
            )
            kl_mean = torch.mean(kl)

        if self.desired_kl is not None and self.schedule == "adaptive":
            if kl_mean > self.desired_kl * 2.0:
                self.actor_learning_rate = max(1e-5, self.actor_learning_rate / 1.5)
                self.critic_learning_rate = max(1e-5, self.critic_learning_rate / 1.5)
                self.disc_learning_rate = max(1e-5, self.disc_learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.actor_learning_rate = min(1e-2, self.actor_learning_rate * 1.5)
                self.critic_learning_rate = min(1e-2, self.critic_learning_rate * 1.5)
                self.disc_learning_rate = min(1e-2, self.disc_learning_rate * 1.5)

            for param_group in self.actor_optimizer.param_groups:
                param_group["lr"] = self.actor_learning_rate
            for param_group in self.critic_optimizer.param_groups:
                param_group["lr"] = self.critic_learning_rate
            for param_group in self.disc_optimizer.param_groups:
                param_group["lr"] = self.disc_learning_rate

        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        entropy_loss = entropy_batch.mean()
        actor_loss = surrogate_loss - self.entropy_coef * entropy_loss
        critic_loss = self.value_loss_coef * value_loss

        # Discriminator loss.
        amp_policy_state = policy_state_dict["amp_obs"].detach()
        amp_expert_state = policy_state_dict["amp_expert_obs"].detach()
        with torch.no_grad():
            amp_policy_state = self.amp_obs_normalizer.normalize_torch(amp_policy_state, self.device)
            amp_expert_state = self.amp_obs_normalizer.normalize_torch(amp_expert_state, self.device)

        policy_d = self.disc(amp_policy_state)
        expert_d = self.disc(amp_expert_state)
        expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
        policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
        amp_loss = 0.5 * (expert_loss + policy_loss)

        if self.disc_grad_penalty_lambda and float(self.disc_grad_penalty_lambda) > 0:
            grad_pen_loss = self.disc.compute_grad_pen(
                amp_expert_state, lambda_=float(self.disc_grad_penalty_lambda)
            )
        else:
            grad_pen_loss = torch.zeros((), device=self.device)

        disc_loss = self.amploss_coef * amp_loss + self.amploss_coef * grad_pen_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.disc_optimizer.zero_grad()

        actor_loss.backward()
        critic_loss.backward()
        disc_loss.backward()

        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.disc_optimizer.step()

        # Match PPO memory behavior: do not keep stale minibatch distribution refs alive.
        if hasattr(self.actor, "distribution"):
            self.actor.distribution = None

        self.amp_obs_normalizer.update(amp_policy_state.detach())
        self.amp_obs_normalizer.update(amp_expert_state.detach())

        loss_dict["policy_d"] += policy_d.mean().detach().item()
        loss_dict["expert_d"] += expert_d.mean().detach().item()
        loss_dict["amp_grad_pen_loss"] += grad_pen_loss.detach().item()
        loss_dict["KL"] += kl_mean.item()

        return loss_dict
