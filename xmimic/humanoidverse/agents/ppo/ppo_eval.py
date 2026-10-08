import torch
from loguru import logger
import time

from humanoidverse.agents.ppo.ppo import PPO as _PPOBase


class PPO(_PPOBase):
    """PPO evaluation variant (RSS metrics)."""

    def _init_config(self):
        super()._init_config()
        # Backward compatible default: True if not specified.
        self.load_critic = getattr(self.config, "load_critic", True)

    def load(self, ckpt_path):
        if ckpt_path is None:
            return None

        logger.info(f"Loading checkpoint from {ckpt_path}")
        loaded_dict = torch.load(ckpt_path, map_location=self.device)

        self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])

        if self.load_critic:
            self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])

        if self.load_optimizer:
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
            self.actor_learning_rate = loaded_dict["actor_optimizer_state_dict"]["param_groups"][0]["lr"]
            self.critic_learning_rate = loaded_dict["critic_optimizer_state_dict"]["param_groups"][0]["lr"]
            self.set_learning_rate(self.actor_learning_rate, self.critic_learning_rate)
            logger.info("Optimizer loaded from checkpoint")

        self.current_learning_iteration = loaded_dict.get("iter", 0)
        return loaded_dict.get("infos", None)

    @torch.no_grad()
    def evaluate_policy(self):
        self._create_eval_callbacks()
        self._pre_evaluate_policy()

        actor_state = self._create_actor_state()
        step = 0
        eval_start_time = time.time()

        ref_contact = []
        obj_contact = []
        hoi_bodies_contact = []
        ig = []

        obj_pos = []
        basket_pos = []
        has_valid_basket_pos = []
        obj_z = []
        ref_obj_final_z = []

        sr_cfg = getattr(self.env.config, "success_rate", None)
        sr_mode = getattr(sr_cfg, "mode", "badminton") if sr_cfg is not None else "badminton"
        sr_mode = str(sr_mode).lower()
        # hidden backward-compat alias
        if sr_mode == "legacy":
            sr_mode = "badminton"

        # --- MPJPE-style aggregation (treat each env's rollout/episode as one sample) ---
        # We accumulate per-step scalars until the FIRST done for each env, and we DO NOT
        # include steps where done==1 (because many envs reset inside step(), and extras
        # may reflect post-reset states on done steps).
        alive = torch.ones(self.env.num_envs, device=self.device, dtype=torch.bool)
        eh_sum = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.float32)
        eo_sum = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.float32)
        eh_count = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.float32)
        eo_count = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.float32)

        contact_error_rates = []

        self.eval_policy = self._get_inference_policy()
        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        actor_state = self._pre_eval_env_step(actor_state)
        actor_state["dones"] = torch.zeros(self.env.num_envs, device=self.device)

        while actor_state["dones"].sum() < self.env.num_envs and not actor_state.get("stop", False):
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            if actor_state.get("stop", False):
                break
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)
            self._log_eval_step_metrics(
                step=step,
                rewards=actor_state["rewards"],
                dones=actor_state["dones"],
                extras=actor_state["extras"],
                elapsed_time_s=time.time() - eval_start_time,
            )
            step += 1

            extras = actor_state["extras"]
            dones = actor_state["dones"]

            # Per-episode contact error rate (cg_reward != 1) summary from env resets.
            # Only append on steps where resets occurred to avoid duplicating stale values.
            if (
                torch.any(dones)
                and "episode" in extras
                and ("contact_error_rate_per_env" in extras["episode"] or "contact_error_rate" in extras["episode"])
            ):
                if "contact_error_rate_per_env" in extras["episode"]:
                    cer = extras["episode"]["contact_error_rate_per_env"].detach().cpu().reshape(-1)
                    contact_error_rates.append(cer)
                else:
                    cer = extras["episode"]["contact_error_rate"].detach().cpu().reshape(1)
                    contact_error_rates.append(cer)

            # Per-env episode-sample accumulation (MPJPE-style)
            # Only accumulate for envs that are still in their first rollout AND not done this step.
            keep = alive & (dones == 0)
            if keep.any():
                if "Eh" in extras:
                    # Eh: (N, B, 3) absolute position diffs -> L2 per body -> mean over bodies => (N,)
                    eh_step = torch.linalg.norm(extras["Eh"], dim=-1).mean(dim=-1).to(torch.float32)
                    eh_sum[keep] += eh_step[keep]
                    eh_count[keep] += 1.0
                if "Eo" in extras:
                    # Eo: (N, 3) absolute position diffs -> L2 => (N,)
                    eo_step = torch.linalg.norm(extras["Eo"], dim=-1).to(torch.float32)
                    eo_sum[keep] += eo_step[keep]
                    eo_count[keep] += 1.0

            # Stop accumulating for envs once they are done the first time.
            alive = alive & (dones == 0)

            ref_contact.append(extras["ref_contact"].detach().cpu())
            obj_contact.append(extras["obj_contact"].detach().cpu())
            hoi_bodies_contact.append(extras["hoi_bodies_contact"].detach().cpu())
            ig.append(extras["ig_hoi_bodies"].detach().cpu())

            if sr_mode in ("catch2shot", "pick"):
                if "obj_pos" in extras:
                    obj_pos.append(extras["obj_pos"].detach().cpu())
                if "basket_pos" in extras:
                    basket_pos.append(extras["basket_pos"].detach().cpu())
                if "has_valid_basket_pos" in extras:
                    has_valid_basket_pos.append(extras["has_valid_basket_pos"].detach().cpu())
                if "obj_z" in extras:
                    obj_z.append(extras["obj_z"].detach().cpu())
                if "ref_obj_final_z" in extras:
                    ref_obj_final_z.append(extras["ref_obj_final_z"].detach().cpu())

        self._post_evaluate_policy()

        ref_contact = torch.stack(ref_contact, dim=1)
        obj_contact = torch.stack(obj_contact, dim=1)
        hoi_bodies_contact = torch.stack(hoi_bodies_contact, dim=1)
        ig = torch.stack(ig, dim=1)

        success_rate = None

        if sr_mode == "catch2shot":
            if len(obj_pos) == 0 or len(basket_pos) == 0:
                sr_mode = "badminton"
            else:
                obj_pos_t = torch.stack(obj_pos, dim=1)  # (N, T, 3)
                basket_pos_t = torch.stack(basket_pos, dim=1)  # (N, T, 3)
                dist = torch.norm(obj_pos_t - basket_pos_t, dim=-1)  # (N, T)

                obj_contact_flag = obj_contact > 0.5
                contact_cum = torch.cumsum(obj_contact_flag.to(torch.int32), dim=1) > 0

                th = float(getattr(sr_cfg, "basket_dist_threshold", 0.05)) if sr_cfg is not None else 0.05
                success_step = (dist < th) & contact_cum

                if len(has_valid_basket_pos) > 0:
                    valid_basket = torch.stack(has_valid_basket_pos, dim=1)  # (N, T)
                    if not torch.any(valid_basket):
                        sr_mode = "badminton"
                    else:
                        success_env = torch.any(success_step, dim=1)
                        success_rate = success_env.float().mean() * 100.0
                else:
                    success_env = torch.any(success_step, dim=1)
                    success_rate = success_env.float().mean() * 100.0

        if sr_mode == "pick":
            if len(obj_z) == 0 or len(ref_obj_final_z) == 0:
                sr_mode = "badminton"
            else:
                obj_z_t = torch.stack(obj_z, dim=1)  # (N, T)
                ref_final_z_t = torch.stack(ref_obj_final_z, dim=1)  # (N, T)
                ref_final_z = ref_final_z_t[:, 0]
                th = float(getattr(sr_cfg, "pick_height_threshold", 0.05)) if sr_cfg is not None else 0.05
                success_env = torch.any(torch.abs(obj_z_t - ref_final_z.unsqueeze(1)) < th, dim=1)
                success_rate = success_env.float().mean() * 100.0

        if sr_mode == "badminton":
            ig_tol = float(getattr(self.env.config, "ig_tolerance", 0.2))
            ig_flag = torch.any(ig < ig_tol, dim=-1)

            hoi_bodies_contact_flag = torch.any(hoi_bodies_contact, dim=-1)
            obj_contact_flag = obj_contact > 0.5
            ref_contact_flag = ref_contact > 0.5

            bodies2obj = torch.logical_and(hoi_bodies_contact_flag, obj_contact_flag)
            bodies2obj_flag = torch.logical_and(bodies2obj, ref_contact_flag)
            success_rate = torch.any(torch.logical_and(ig_flag, bodies2obj_flag), dim=-1).float().mean() * 100.0

        logger.info("="*80)
        logger.info("RSS EVALUATION RESULTS:")

        # --- MPJPE-style reporting (per-env episode is one sample) ---
        # Per-env episode scalar: time-mean of per-step L2 (and body-mean for humanoid)
        eps = 1e-8
        Eh_per_env = eh_sum / (eh_count + eps)
        Eo_per_env = eo_sum / (eo_count + eps)
        valid_h = eh_count > 0
        valid_o = eo_count > 0

        def _safe_std(x: torch.Tensor, unbiased: bool) -> torch.Tensor:
            if x.numel() == 0:
                return torch.tensor(float("nan"), device=x.device, dtype=torch.float32)
            if unbiased and x.numel() < 2:
                # sample std undefined for n<2; return 0 for logging stability
                return torch.zeros((), device=x.device, dtype=torch.float32)
            return torch.std(x, unbiased=unbiased)

        if valid_h.any():
            xh = Eh_per_env[valid_h]
            logger.info(
                "  Eh_mpJPE_style (per-env episode sample): "
                f"mean = {xh.mean().item():.4f}, "
                f"std_sample(ddof=1) = {_safe_std(xh, unbiased=True).item():.4f}, "
                f"std_pop(ddof=0) = {_safe_std(xh, unbiased=False).item():.4f}, "
                f"n_env = {int(xh.numel())}"
            )
        else:
            logger.info("  Eh_mpJPE_style (per-env episode sample): n_env = 0 (no valid frames)")

        if valid_o.any():
            xo = Eo_per_env[valid_o]
            logger.info(
                "  Eo_mpJPE_style (per-env episode sample): "
                f"mean = {xo.mean().item():.4f}, "
                f"std_sample(ddof=1) = {_safe_std(xo, unbiased=True).item():.4f}, "
                f"std_pop(ddof=0) = {_safe_std(xo, unbiased=False).item():.4f}, "
                f"n_env = {int(xo.numel())}"
            )
        else:
            logger.info("  Eo_mpJPE_style (per-env episode sample): n_env = 0 (no valid frames)")

        if success_rate is None:
            logger.info(f"  Success Rate: N/A (mode={sr_mode})")
        else:
            logger.info(f"  Success Rate: {success_rate.item():.2f}% (mode={sr_mode})")
        if len(contact_error_rates) > 0:
            cer = torch.cat(contact_error_rates)
            logger.info(
                "  Contact Error Rate (cg_reward != 1, windowed): "
                f"mean = {cer.mean().item():.4f}, std = {_safe_std(cer, unbiased=True).item():.4f}"
            )
        logger.info("="*80)
