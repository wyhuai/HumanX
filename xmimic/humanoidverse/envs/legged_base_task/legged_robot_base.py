from time import time
import numpy as np
import matplotlib.pyplot as plt
import os

from humanoidverse.utils.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from humanoidverse.envs.env_utils.general import class_to_dict
from isaac_utils.rotations import get_euler_xyz_in_tensor
from isaac_utils.rotations import quat_apply_yaw, wrap_to_pi
from humanoidverse.envs.base_task.base_task import BaseTask

from humanoidverse.envs.env_utils.history_handler import HistoryHandler
from termcolor import colored
from humanoidverse.utils.helpers import parse_observation
from humanoidverse.utils.hoi_runtime import get_rigid_bodies_per_env
from humanoidverse.utils.mimic_logging import apply_termination_mask, summarize_reward_groups
from humanoidverse.envs.env_utils.visualization import Point

from loguru import logger
import copy

class _P2Quantile:
    """
    Online P² quantile estimator (Jain & Chlamtac, 1985).
    Maintains O(1) state and provides an approximate quantile without storing samples.
    """
    def __init__(self, q: float):
        self.q = float(q)
        self._n = 0
        self._init = []
        # marker positions (n), desired positions (np), increments (dn), heights (h)
        self._npos = None
        self._npos_des = None
        self._dn = None
        self._h = None

    def add(self, x: float):
        x = float(x)
        if self._n < 5:
            self._init.append(x)
            self._n += 1
            if self._n == 5:
                self._init.sort()
                self._h = self._init[:]  # 5 heights
                self._npos = [1, 2, 3, 4, 5]
                q = self.q
                self._npos_des = [1.0, 1.0 + 2.0 * q, 1.0 + 4.0 * q, 3.0 + 2.0 * q, 5.0]
                self._dn = [0.0, q / 2.0, q, (1.0 + q) / 2.0, 1.0]
            return

        # Find cell k
        if x < self._h[0]:
            self._h[0] = x
            k = 0
        elif x < self._h[1]:
            k = 0
        elif x < self._h[2]:
            k = 1
        elif x < self._h[3]:
            k = 2
        elif x < self._h[4]:
            k = 3
        else:
            self._h[4] = x
            k = 3

        # Increment positions of markers above k
        for i in range(k + 1, 5):
            self._npos[i] += 1
        for i in range(5):
            self._npos_des[i] += self._dn[i]

        # Adjust intermediate markers 1..3 (index 1..3)
        for i in (1, 2, 3):
            d = self._npos_des[i] - self._npos[i]
            if (d >= 1.0 and (self._npos[i + 1] - self._npos[i]) > 1) or (d <= -1.0 and (self._npos[i - 1] - self._npos[i]) < -1):
                di = 1 if d > 0 else -1
                # parabolic prediction
                n_im1 = self._npos[i - 1]
                n_i = self._npos[i]
                n_ip1 = self._npos[i + 1]
                h_im1 = self._h[i - 1]
                h_i = self._h[i]
                h_ip1 = self._h[i + 1]

                num = di * (n_i - n_im1 + di) * (h_ip1 - h_i) / (n_ip1 - n_i) + di * (n_ip1 - n_i - di) * (h_i - h_im1) / (n_i - n_im1)
                denom = (n_ip1 - n_im1)
                h_hat = h_i + num / denom

                # If parabolic goes out of bounds, use linear
                if h_im1 < h_hat < h_ip1:
                    self._h[i] = h_hat
                else:
                    self._h[i] = h_i + di * (self._h[i + di] - h_i) / (self._npos[i + di] - n_i)

                self._npos[i] += di

    def value(self) -> float:
        if self._n == 0:
            return 0.0
        if self._n < 5:
            # exact for small n
            s = sorted(self._init)
            idx = int(round((len(s) - 1) * self.q))
            return float(s[max(0, min(idx, len(s) - 1))])
        return float(self._h[2])

class LeggedRobotBase(BaseTask):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self._domain_rand_config()
        self._prepare_reward_function()
        self.history_handler = HistoryHandler(self.num_envs, config.obs.obs_auxiliary, config.obs.obs_dims, device)
        self.is_evaluating = False
        self.init_done = True

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super()._init_buffers()

        self.base_quat = self.simulator.base_quat
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)

        # initialize some data used later on
        self._init_counters()
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions_after_delay = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_pos = torch.zeros_like(self.simulator.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.simulator.dof_vel)
        self.last_root_vel = torch.zeros_like(self.simulator.robot_root_states[:, 7:13])
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        rigid_bodies_per_env = get_rigid_bodies_per_env(self.simulator._rigid_body_state, self.num_envs)
        self.disturbance = torch.zeros(
            self.num_envs,
            rigid_bodies_per_env,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # --- Feet contact/stance bookkeeping (for new foot penalties; backward compatible defaults) ---
        # Contact masks are updated each step in `_pre_compute_observations_callback`.
        num_feet = int(self.feet_indices.shape[0])
        self.feet_contact = torch.zeros(self.num_envs, num_feet, dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_contact_filt = torch.zeros(self.num_envs, num_feet, dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_first_contact = torch.zeros(self.num_envs, num_feet, dtype=torch.bool, device=self.device, requires_grad=False)
        # Internal contact history for the new foot penalties / debug (do NOT reuse `last_contacts`,
        # which some legacy rewards expect to be updated inside reward functions).
        self._feet_last_contact = torch.zeros(self.num_envs, num_feet, dtype=torch.bool, device=self.device, requires_grad=False)
        self._feet_last_contact_filt = torch.zeros(self.num_envs, num_feet, dtype=torch.bool, device=self.device, requires_grad=False)
        # Stance duration accumulator (seconds) for each foot, based on filtered contact.
        self.feet_stance_time = torch.zeros(self.num_envs, num_feet, dtype=torch.float, device=self.device, requires_grad=False)
        # Estimated stance height baseline (world z), per-foot per-env.
        self.feet_stance_z = torch.zeros(self.num_envs, num_feet, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_stance_z_inited = torch.zeros(self.num_envs, num_feet, dtype=torch.bool, device=self.device, requires_grad=False)
        # Frozen peak height (absolute z) captured on air->contact (first contact) for each foot.
        # This avoids reward-order dependence when we reset `feet_air_max_height` upon contact.
        self.feet_air_max_height_on_contact = torch.zeros(
            self.num_envs, num_feet, dtype=torch.float, device=self.device, requires_grad=False
        )
        # Last-step foot vertical velocity (world frame), for touchdown velocity diagnostics (and future rewards).
        self.last_feet_vel_z = torch.zeros(self.num_envs, num_feet, dtype=torch.float, device=self.device, requires_grad=False)
        # Online touchdown velocity stats (so-far) for debug printing.
        self._td_vz_count = 0
        self._td_vz_mean = 0.0
        self._td_vz_max = 0.0
        self._td_vz_p95 = _P2Quantile(0.95)

        # Cache a global robot weight for force normalization.
        # Fallback: assume 35kg (G1) if we cannot query body properties in this runtime.
        self.robot_total_mass = None
        self.robot_weight = 35.0 * 9.81
        try:
            # Prefer BaseTask-owned handles if present; fall back to simulator wrapper if needed.
            gym = getattr(self, "gym", None) or getattr(self.simulator, "gym", None)
            envs = getattr(self, "envs", None) or getattr(self.simulator, "envs", None)
            robot_handles = getattr(self, "robot_handles", None) or getattr(self.simulator, "robot_handles", None)
            if gym is None or envs is None or robot_handles is None:
                raise RuntimeError("Gym/envs/robot_handles not available for mass query")

            props = gym.get_actor_rigid_body_properties(envs[0], robot_handles[0])
            total_mass = 0.0
            for p in props:
                total_mass += float(p.mass)
            self.robot_total_mass = float(total_mass)
            self.robot_weight = max(self.robot_total_mass * 9.81, 1e-6)
        except Exception:
            # Keep fallback (35kg) for max compatibility across simulators/tests.
            self.robot_total_mass = None
            self.robot_weight = 35.0 * 9.81

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.config.robot.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.config.robot.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.config.robot.control.stiffness[dof_name]
                    self.d_gains[i] = self.config.robot.control.damping[dof_name]
                    found = True
                    logger.debug(f"PD gain of joint {name} were defined, setting them to {self.p_gains[i]} and {self.d_gains[i]}")
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.config.robot.control.control_type in ["P", "V"]:
                    logger.warning(f"PD gain of joint {name} were not defined, setting them to zero")
                    raise ValueError(f"PD gain of joint {name} were not defined. Should be defined in the yaml file.")
        self.original_default_dof_pos = self.default_dof_pos.detach().clone().unsqueeze(0)
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0).repeat((self.num_envs, 1))
        self._init_domain_rand_buffers()

        # for reward penalty curriculum
        self.average_episode_length = 0. # num_compute_average_epl last termination episode length
        self.last_episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.num_compute_average_epl = self.config.rewards.num_compute_average_epl

        self.need_to_refresh_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self.add_noise_currculum = self.config.obs.add_noise_currculum
        self.current_noise_curriculum_value = self.config.obs.noise_initial_value

    def _domain_rand_config(self):
        if self.config.domain_rand.push_robots:
            self.push_interval_s = torch.randint(self.config.domain_rand.push_interval_s[0], self.config.domain_rand.push_interval_s[1], (self.num_envs,), device=self.device)
        if self.config.domain_rand.disturb_robots:
            self.disturb_interval_s = torch.randint(self.config.domain_rand.disturb_interval_s[0], self.config.domain_rand.disturb_interval_s[1], (self.num_envs,), device=self.device)
            
    def _init_counters(self):
        self.common_step_counter = 0
        self.push_robot_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        self.push_robot_plot_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        self.disturb_robot_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        self.disturb_robot_plot_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        self.command_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)

    def _update_counters_each_step(self):
        self.common_step_counter  +=1
        self.push_robot_counter[:] += 1
        self.push_robot_plot_counter[:] += 1
        self.disturb_robot_counter[:] += 1
        self.disturb_robot_plot_counter[:] += 1
        self.command_counter[:] += 1

    def _init_domain_rand_buffers(self):
        ######################################### DR related tensors #########################################
        if self.config.domain_rand.randomize_ctrl_delay:
            self.action_queue = torch.zeros(self.num_envs, self.config.domain_rand.ctrl_delay_step_range[1]+1, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.action_delay_idx = torch.randint(self.config.domain_rand.ctrl_delay_step_range[0], 
                                                self.config.domain_rand.ctrl_delay_step_range[1]+1, (self.num_envs,), device=self.device, requires_grad=False)

        # self._link_mass_scale = torch.ones(self.num_envs, len(self.config.robot.randomize_link_body_names), dtype=torch.float, device=self.device, requires_grad=False)
        self._kp_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._kd_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._rfi_lim_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_robot_vel_buf = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.record_push_robot_vel_buf = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.push_robot_ang_vel_buf = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.record_push_robot_ang_vel_buf = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.last_contacts_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_air_max_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        logger.info(colored(f"{self.config.rewards.set_reward} set reward on {self.config.rewards.set_reward_date}", "green"))
        
        self.reward_scales = self.config.rewards.reward_scales
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            logger.info(f"Scale: {key} = {self.reward_scales[key]}")
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt

        self.use_reward_penalty_curriculum = self.config.rewards.reward_penalty_curriculum
        if self.use_reward_penalty_curriculum:
            self.reward_penalty_scale = self.config.rewards.reward_initial_penalty_scale

        logger.info(colored(f"Use Reward Penalty: {self.use_reward_penalty_curriculum}", "green"))
        if self.use_reward_penalty_curriculum:
            logger.info(f"Penalty Reward Names: {self.config.rewards.reward_penalty_reward_names}")
            logger.info(f"Penalty Reward Initial Scale: {self.config.rewards.reward_initial_penalty_scale}")
        
        self.use_reward_limits_dof_pos_curriculum = self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum
        self.use_reward_limits_dof_vel_curriculum = self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum
        self.use_reward_limits_torque_curriculum = self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum

        if self.use_reward_limits_dof_pos_curriculum:
            logger.info(f"Use Reward Limits DOF Curriculum: {self.use_reward_limits_dof_pos_curriculum}")
            logger.info(f"Reward Limits DOF Curriculum Initial Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_initial_limit}")
            logger.info(f"Reward Limits DOF Curriculum Max Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_max_limit}")
            logger.info(f"Reward Limits DOF Curriculum Min Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_min_limit}")
            self.soft_dof_pos_curriculum_value = self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_initial_limit
        
        if self.use_reward_limits_dof_vel_curriculum:
            logger.info(f"Use Reward Limits DOF Vel Curriculum: {self.use_reward_limits_dof_vel_curriculum}")
            logger.info(f"Reward Limits DOF Vel Curriculum Initial Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_initial_limit}")
            logger.info(f"Reward Limits DOF Vel Curriculum Max Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_max_limit}")
            logger.info(f"Reward Limits DOF Vel Curriculum Min Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_min_limit}")
            self.soft_dof_vel_curriculum_value = self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_initial_limit
        
        if self.use_reward_limits_torque_curriculum:
            logger.info(f"Use Reward Limits Torque Curriculum: {self.use_reward_limits_torque_curriculum}")
            logger.info(f"Reward Limits Torque Curriculum Initial Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_initial_limit}")
            logger.info(f"Reward Limits Torque Curriculum Max Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_max_limit}")
            logger.info(f"Reward Limits Torque Curriculum Min Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_min_limit}")
            self.soft_torque_curriculum_value = self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_initial_limit

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))
            # reward episode sums
            self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                                for name in self.reward_scales.keys()}

    def set_is_evaluating(self):
        logger.info("Setting Env is evaluating")
        self.is_evaluating = True
    
    def step(self, actor_state):
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = actor_state["actions"]
        # actions *= 0.0
        self._pre_physics_step(actions)
        self._physics_step()
        self._post_physics_step()

        # if self.episode_length_buf[0] == 1:
        #     import ipdb; ipdb.set_trace()

        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    def _pre_physics_step(self, actions):
        clip_action_limit = self.config.robot.control.action_clip_value
        self.actions = torch.clip(actions, -clip_action_limit, clip_action_limit).to(self.device)

        self.log_dict["action_clip_frac"] = (
                self.actions.abs() == clip_action_limit
            ).sum() / self.actions.numel()

        if self.config.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = self.actions.clone()
            self.actions_after_delay = self.action_queue[torch.arange(self.num_envs), self.action_delay_idx].clone()
        else:
            self.actions_after_delay = self.actions.clone()


    def _physics_step(self):
        self.render()
        for _ in range(self.config.simulator.config.sim.control_decimation):
            self._apply_force_in_physics_step()
            self.simulator.simulate_at_each_physics_step()

    def _apply_force_in_physics_step(self):
        self.torques = self._compute_torques(self.actions_after_delay).view(self.torques.shape)
        self.simulator.apply_torques_at_dof(self.torques)

    def _post_physics_step(self):
        self._refresh_sim_tensors()
        self.episode_length_buf += 1
        # update counters
        self._update_counters_each_step()
        self.last_episode_length_buf = self.episode_length_buf.clone()

        self._pre_compute_observations_callback()
        self._update_tasks_callback()
        # compute observations, rewards, resets, ...
        self._check_termination()
        self._compute_reward()
        # check terminations
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_envs_idx(env_ids)

        # set envs
        refresh_env_ids = self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
        if len(refresh_env_ids) > 0:
            self.simulator.set_actor_root_state_tensor(refresh_env_ids, self.simulator.all_root_states)
            self.simulator.set_dof_state_tensor(refresh_env_ids, self.simulator.dof_state)
            self.need_to_refresh_envs[refresh_env_ids] = False

        self._compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        
        self._post_compute_observations_callback()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.config.normalization.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs)

        for key in self.history_handler.history.keys():
            self.history_handler.add(key, self.hist_obs_dict[key])

        self.extras["to_log"] = self.log_dict
        if self.viewer:
            self._setup_simulator_control()
            self._setup_simulator_next_task()
            # if self.debug_viz:
            #     self._draw_debug_vis()
    
    def _setup_simulator_next_task(self):
        pass

    def _setup_simulator_control(self):
        pass

    def _pre_compute_observations_callback(self):
        # prepare quantities
        self.base_quat[:] = self.simulator.base_quat[:]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 7:10])
        # print("self.base_lin_vel", self.base_lin_vel)
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 10:13])
        # print("self.base_ang_vel", self.base_ang_vel)
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # --- Update feet contact/stance/air-peak states once per step ---
        # Contact force threshold (N) for feet contact; default 1N.
        thr = float(getattr(self.config.rewards, "feet_contact_force_z_threshold", 1.0))
        # Feet world states
        feet_pos_z = self.simulator._rigid_body_pos[:, self.feet_indices, 2]
        feet_vel_z = self.simulator._rigid_body_vel[:, self.feet_indices, 2]
        feet_fz = self.simulator.contact_forces[:, self.feet_indices, 2]

        contact = feet_fz > thr
        # Two-frame filtered contact for robustness, using our own history buffers.
        contact_filt = torch.logical_or(contact, self._feet_last_contact)
        first_contact = torch.logical_and(contact_filt, ~self._feet_last_contact_filt)

        self.feet_contact[:] = contact
        self.feet_contact_filt[:] = contact_filt
        self.feet_first_contact[:] = first_contact

        # Update stance time (use filtered contact to reduce flicker).
        self.feet_stance_time[:] = self.feet_stance_time + float(self.dt) * contact_filt.to(torch.float32)
        self.feet_stance_time[:] = self.feet_stance_time * contact_filt.to(torch.float32)

        # Update internal contact history (keep legacy `last_contacts*` untouched for backward compatibility).
        self._feet_last_contact[:] = contact
        self._feet_last_contact_filt[:] = contact_filt

        # Update stance baseline z for relative clearance.
        # IMPORTANT: use raw `contact` (not contact_filt) so we don't update stance baseline during the
        # "one-step lag" that contact_filt introduces.
        init_mask = torch.logical_and(contact, ~self.feet_stance_z_inited)
        if init_mask.any():
            self.feet_stance_z[init_mask] = feet_pos_z[init_mask]
            self.feet_stance_z_inited[init_mask] = True

        # EMA update while in `contact` (after init); helps absorb tiny contact jitter.
        ema = float(getattr(self.config.rewards, "feet_stance_z_ema", 0.05))
        ema = float(np.clip(ema, 0.0, 1.0))
        ema_mask = torch.logical_and(contact, self.feet_stance_z_inited)
        if ema > 0.0 and ema_mask.any():
            self.feet_stance_z[ema_mask] = (1.0 - ema) * self.feet_stance_z[ema_mask] + ema * feet_pos_z[ema_mask]

        # Track per-swing peak height (absolute z).
        # IMPORTANT: capture the peak at first_contact BEFORE resetting for the next swing.
        self.feet_air_max_height[:] = torch.max(self.feet_air_max_height, feet_pos_z)
        # Clear and freeze per-step peak-on-contact (used only when first_contact is true).
        self.feet_air_max_height_on_contact[:] = 0.0
        if first_contact.any():
            self.feet_air_max_height_on_contact[first_contact] = self.feet_air_max_height[first_contact]
        # Reset swing peak after contact_filt (including first_contact).
        self.feet_air_max_height[:] = self.feet_air_max_height * (~contact_filt)

        # --- Optional debug: print touchdown vertical speed (pre-contact v_z) during evaluation ---
        # We use last-step v_z for first_contact events to approximate pre-impact downward speed.
        if getattr(self, "is_evaluating", False) and bool(getattr(self.config.rewards, "debug_touchdown_velocity", False)):
            # If true, print per touchdown event (can be very verbose with large num_envs).
            print_each = bool(getattr(self.config.rewards, "debug_touchdown_print_each_event", False))
            if first_contact.any():
                idx = first_contact.nonzero(as_tuple=False)  # (K, 2): [env, foot]
                for k in range(idx.shape[0]):
                    env_i = int(idx[k, 0].item())
                    foot_i = int(idx[k, 1].item())
                    v = float((-self.last_feet_vel_z[env_i, foot_i]).clamp_min(0.0).item())

                    # update online mean/max
                    self._td_vz_count += 1
                    delta = v - self._td_vz_mean
                    self._td_vz_mean += delta / float(self._td_vz_count)
                    self._td_vz_max = max(self._td_vz_max, v)
                    self._td_vz_p95.add(v)

                    if print_each:
                        logger.info(
                            f"[TD_VZ] step={int(self.common_step_counter)} env={env_i} foot={foot_i} "
                            f"td_vz={v:.3f} m/s | sofar: n={self._td_vz_count} mean={self._td_vz_mean:.3f} "
                            f"p95={self._td_vz_p95.value():.3f} max={self._td_vz_max:.3f}"
                        )

        # Update last-step foot v_z (for touchdown diagnostics)
        self.last_feet_vel_z[:] = feet_vel_z

    def _update_tasks_callback(self):
        if self.config.domain_rand.push_robots:
            push_robot_env_ids = (self.push_robot_counter == (self.push_interval_s / self.dt).int()).nonzero(as_tuple=False).flatten()
            self.push_robot_counter[push_robot_env_ids] = 0
            self.push_robot_plot_counter[push_robot_env_ids] = 0
            self.push_interval_s[push_robot_env_ids] = torch.randint(self.config.domain_rand.push_interval_s[0], self.config.domain_rand.push_interval_s[1], (len(push_robot_env_ids),), device=self.device, requires_grad=False)
            self._push_robots(push_robot_env_ids)
            
        if self.config.domain_rand.disturb_robots:
            disturb_robot_env_ids = (self.disturb_robot_counter == (self.disturb_interval_s / self.dt).int()).nonzero(as_tuple=False).flatten()
            self.disturb_robot_counter[disturb_robot_env_ids] = 0
            self.disturb_robot_plot_counter[disturb_robot_env_ids] = 0
            self.disturb_interval_s[disturb_robot_env_ids] = torch.randint(self.config.domain_rand.disturb_interval_s[0], self.config.domain_rand.disturb_interval_s[1], (len(disturb_robot_env_ids),), device=self.device, requires_grad=False)
            self._disturbance_robots(disturb_robot_env_ids)
            
    def _disturbance_robots(self, env_ids):
        """ Random add disturbance force to the robots.
        """
        # if len(env_ids) == 0:
        #     return
        disturbance = torch_rand_float(self.config.domain_rand.disturbance_range[0], self.config.domain_rand.disturbance_range[1], (self.num_envs, 3), device=self.device)
        stand_frame_idx = self.episode_length_buf + (self.motion_start_times/self.dt).to(int) > self.stand_ref_episode_length
        disturbance[stand_frame_idx] = torch_rand_float(self.config.domain_rand.disturbance_range_at_stand[0], self.config.domain_rand.disturbance_range_at_stand[1], (stand_frame_idx.sum().item(), 3), device=self.device)

        # self.disturbance[env_ids, self.body_names.index("torso_link"), :] = disturbance[env_ids]
        self.disturbance[env_ids, self.body_names.index("torso_link"), :] = disturbance[env_ids]
        self.simulator.gym.apply_rigid_body_force_tensors(self.simulator.sim, forceTensor=gymtorch.unwrap_tensor(self.disturbance), space=gymapi.CoordinateSpace.LOCAL_SPACE)

    def _post_compute_observations_callback(self):
        self.last_actions[:] = self.actions[:]
        self.last_dof_pos[:] = self.simulator.dof_pos[:]
        self.last_dof_vel[:] = self.simulator.dof_vel[:]
        self.last_root_vel[:] = self.simulator.robot_root_states[:, 7:13]
        # Update legacy feet contact history buffers AFTER reward computation.
        # Many legacy rewards expect `last_contacts` to be previous-step contact state at reward time.
        # We therefore update them here (end of step), and avoid overwriting them for envs that were reset this step.
        if hasattr(self, "feet_contact") and hasattr(self, "feet_contact_filt"):
            # reset_buf is torch.long (0/1). Use an explicit boolean mask.
            keep = (self.reset_buf == 0)
            if keep.any():
                self.last_contacts[keep] = self.feet_contact[keep]
                self.last_contacts_filt[keep] = self.feet_contact_filt[keep]


    def _check_termination(self):
        """ Check if environments need to be reset
        """
        # self.reset_buf = 0
        # self.time_out_buf = 0
        # Note: DO NOT USE FOLLOWING TWO LINES STYLE
        self.reset_buf[:] = 0
        self.time_out_buf[:] = 0

        self._update_reset_buf()
        self._update_timeout_buf()

    def _update_reset_buf(self):
        if self.config.termination.terminate_by_contact:
            reset_buf_contact = torch.any(
                torch.norm(self.simulator.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
                dim=1,
            )
            p = float(getattr(self.config.termination, "terminate_by_contact_p", 1.0))
            # log (will be averaged across steps and shown as Env/* in tensorboard)
            self.log_dict["terminate_by_contact_trigger_frac"] = reset_buf_contact.float().mean()
            # self.log_dict["terminate_by_contact_p"] = torch.tensor(p, dtype=torch.float, device=self.device)
            if p >= 1.0:
                applied = reset_buf_contact
            elif p > 0.0:
                applied = (reset_buf_contact & (torch.rand_like(reset_buf_contact.float()) < p)).bool()
                self.log_dict["terminate_by_contact_applied_frac"] = applied.float().mean()
            else:
                applied = torch.zeros_like(reset_buf_contact, dtype=torch.bool)
                self.log_dict["terminate_by_contact_applied_frac"] = torch.tensor(
                    0.0, dtype=torch.float, device=self.device
                )
            if p >= 1.0:
                self.log_dict["terminate_by_contact_applied_frac"] = applied.float().mean()
            self.reset_buf, self.time_out_buf, _ = apply_termination_mask(
                self.log_dict, self.reset_buf, self.time_out_buf, "contact", reset_buf_contact, applied
            )

        if self.config.termination.terminate_by_gravity:
            # print(self.projected_gravity)
            gravity_mask = torch.any(
                torch.abs(self.projected_gravity[:, 0:1]) > self.config.termination_scales.termination_gravity_x,
                dim=1,
            ) | torch.any(
                torch.abs(self.projected_gravity[:, 1:2]) > self.config.termination_scales.termination_gravity_y,
                dim=1,
            )
            self.reset_buf, self.time_out_buf, _ = apply_termination_mask(
                self.log_dict, self.reset_buf, self.time_out_buf, "gravity", gravity_mask
            )
        if self.config.termination.terminate_by_low_height:
            # import ipdb; ipdb.set_trace()
            low_height_mask = torch.any(
                self.simulator.robot_root_states[:, 2:3] < self.config.termination_scales.termination_min_base_height,
                dim=1,
            )
            self.reset_buf, self.time_out_buf, _ = apply_termination_mask(
                self.log_dict, self.reset_buf, self.time_out_buf, "low_height", low_height_mask
            )

        if self.config.termination.terminate_when_close_to_dof_pos_limit:
            out_of_dof_pos_limits = -(self.simulator.dof_pos - self.simulator.dof_pos_limits_termination[:, 0]).clip(max=0.) # lower limit
            out_of_dof_pos_limits += (self.simulator.dof_pos - self.simulator.dof_pos_limits_termination[:, 1]).clip(min=0.)
            
            out_of_dof_pos_limits = torch.sum(out_of_dof_pos_limits, dim=1)
            # get random number between 0 and 1, if it is smaller than self.config.termination_probality.terminate_when_close_to_dof_pos_limit, apply the termination
            raw_mask = out_of_dof_pos_limits > 0.
            applied_mask = torch.zeros_like(raw_mask, dtype=torch.bool)
            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_dof_pos_limit:
                applied_mask = raw_mask
            self.reset_buf, self.time_out_buf, _ = apply_termination_mask(
                self.log_dict, self.reset_buf, self.time_out_buf, "dof_pos_limit", raw_mask, applied_mask
            )
        
        if self.config.termination.terminate_when_close_to_dof_vel_limit:
            out_of_dof_vel_limits = torch.sum((torch.abs(self.simulator.dof_vel) - self.dof_vel_limits * self.config.termination_scales.termination_close_to_dof_vel_limit).clip(min=0., max=1.), dim=1)
            raw_mask = out_of_dof_vel_limits > 0.
            applied_mask = torch.zeros_like(raw_mask, dtype=torch.bool)
            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_dof_vel_limit:
                applied_mask = raw_mask
            self.reset_buf, self.time_out_buf, _ = apply_termination_mask(
                self.log_dict, self.reset_buf, self.time_out_buf, "dof_vel_limit", raw_mask, applied_mask
            )
        
        if self.config.termination.terminate_when_close_to_torque_limit:
            out_of_torque_limits = torch.sum((torch.abs(self.torques) - self.torque_limits * self.config.termination_scales.termination_close_to_torque_limit).clip(min=0., max=1.), dim=1)
            raw_mask = out_of_torque_limits > 0.
            applied_mask = torch.zeros_like(raw_mask, dtype=torch.bool)
            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_torque_limit:
                applied_mask = raw_mask
            self.reset_buf, self.time_out_buf, _ = apply_termination_mask(
                self.log_dict, self.reset_buf, self.time_out_buf, "torque_limit", raw_mask, applied_mask
            )


    def _update_timeout_buf(self):
        time_limit_mask = self.episode_length_buf > self.max_episode_length
        self.reset_buf, self.time_out_buf, _ = apply_termination_mask(
            self.log_dict, self.reset_buf, self.time_out_buf, "time_limit", time_limit_mask, mark_timeout=True
        )

    def _fill_episode_reward_extras(self, env_ids):
        self.extras["episode"] = {}
        episode_sums_for_envs = {}
        for key in self.episode_sums.keys():
            values = self.episode_sums[key][env_ids]
            episode_sums_for_envs[key] = values
            self.extras["episode"]['rew_' + key] = torch.mean(values) / self.max_episode_length_s

            self.extras["episode"].update(
                summarize_reward_groups(
                    episode_sums=episode_sums_for_envs,
                    reward_scales=self.reward_scales,
                    episode_lengths=self.last_episode_length_buf[env_ids],
                    max_episode_length_s=self.max_episode_length_s,
                    step_dt=self.dt,
                )
            )

        for key in self.episode_sums.keys():
            self.episode_sums[key][env_ids] = 0.

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
            target_states (dict): Dictionary containing lists of target states for the robot
        """
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        self._reset_buffers_callback(env_ids, target_buf)
        self._reset_tasks_callback(env_ids)        # if target_states is not None, reset to target states
        self._reset_robot_states_callback(env_ids, target_states)

        # fill extras
        self._fill_episode_reward_extras(env_ids)
        self.extras["time_outs"] = self.time_out_buf
        # self._refresh_sim_tensors() #wyh


    def _reset_robot_states_callback(self, env_ids, target_states=None):
        # if target_states is not None, reset to target states
        if target_states is not None:
            self._reset_dofs(env_ids, target_states["dof_states"])
            self._reset_root_states(env_ids, target_states["root_states"])
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

    def _reset_tasks_callback(self, env_ids):
        self._episodic_domain_randomization(env_ids)
        if self.use_reward_penalty_curriculum:
            self._update_reward_penalty_curriculum()
        if self.use_reward_limits_dof_pos_curriculum or self.use_reward_limits_dof_vel_curriculum or self.use_reward_limits_torque_curriculum:
            self._update_reward_limits_curriculum()
        if self.add_noise_currculum:
            self._update_obs_noise_curriculum()
    
    def _update_obs_noise_curriculum(self):
        if self.average_episode_length < self.config.obs.soft_dof_pos_curriculum_level_down_threshold:
            self.current_noise_curriculum_value *= (1 - self.config.obs.soft_dof_pos_curriculum_degree)
        elif self.average_episode_length > self.config.rewards.reward_penalty_level_up_threshold:
            self.current_noise_curriculum_value *= (1 + self.config.obs.soft_dof_pos_curriculum_degree)

        self.current_noise_curriculum_value = np.clip(self.current_noise_curriculum_value, 
                                                      self.config.obs.noise_value_min, 
                                                      self.config.obs.noise_value_max)

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        if target_buf is not None:
            self.simulator.dof_pos[env_ids] = target_buf["dof_pos"].to(self.simulator.dof_pos.dtype)
            self.simulator.dof_vel[env_ids] = target_buf["dof_vel"].to(self.simulator.dof_vel.dtype)
            self.base_quat[env_ids] = target_buf["base_quat"].to(self.base_quat.dtype)
            self.base_lin_vel[env_ids] = target_buf["base_lin_vel"].to(self.base_lin_vel.dtype)
            self.base_ang_vel[env_ids] = target_buf["base_ang_vel"].to(self.base_ang_vel.dtype)
            self.projected_gravity[env_ids] = target_buf["projected_gravity"].to(self.projected_gravity.dtype)
            self.torques[env_ids] = target_buf["torques"].to(self.torques.dtype)
            self.actions[env_ids] = target_buf["actions"].to(self.actions.dtype)
            self.last_actions[env_ids] = target_buf["last_actions"].to(self.last_actions.dtype)
            self.last_dof_pos[env_ids] = target_buf["last_dof_pos"].to(self.last_dof_pos.dtype)
            self.last_dof_vel[env_ids] = target_buf["last_dof_vel"].to(self.last_dof_vel.dtype)
            self.episode_length_buf[env_ids] = target_buf["episode_length_buf"].to(self.episode_length_buf.dtype)
            self.reset_buf[env_ids] = target_buf["reset_buf"].to(self.reset_buf.dtype)
            self.time_out_buf[env_ids] = target_buf["time_out_buf"].to(self.time_out_buf.dtype)
            self.feet_air_time[env_ids] = target_buf["feet_air_time"].to(self.feet_air_time.dtype)
            self.last_contacts[env_ids] = target_buf["last_contacts"].to(self.last_contacts.dtype)
            self.last_contacts_filt[env_ids] = target_buf["last_contacts_filt"].to(self.last_contacts_filt.dtype)
            self.feet_air_max_height[env_ids] = target_buf["feet_air_max_height"].to(self.feet_air_max_height.dtype)
            # Backward compatible: older buffers may not exist in target_buf.
            if "feet_stance_z" in target_buf:
                self.feet_stance_z[env_ids] = target_buf["feet_stance_z"].to(self.feet_stance_z.dtype)
            if "feet_stance_z_inited" in target_buf:
                self.feet_stance_z_inited[env_ids] = target_buf["feet_stance_z_inited"].to(self.feet_stance_z_inited.dtype)
            if "feet_stance_time" in target_buf:
                self.feet_stance_time[env_ids] = target_buf["feet_stance_time"].to(self.feet_stance_time.dtype)
            if "feet_air_max_height_on_contact" in target_buf:
                self.feet_air_max_height_on_contact[env_ids] = target_buf["feet_air_max_height_on_contact"].to(
                    self.feet_air_max_height_on_contact.dtype
                )
            if "_feet_last_contact" in target_buf:
                self._feet_last_contact[env_ids] = target_buf["_feet_last_contact"].to(self._feet_last_contact.dtype)
            if "_feet_last_contact_filt" in target_buf:
                self._feet_last_contact_filt[env_ids] = target_buf["_feet_last_contact_filt"].to(self._feet_last_contact_filt.dtype)
            if "last_feet_vel_z" in target_buf:
                self.last_feet_vel_z[env_ids] = target_buf["last_feet_vel_z"].to(self.last_feet_vel_z.dtype)
        else:
            self.actions[env_ids] = 0.
            self.last_actions[env_ids] = 0.
            self.actions_after_delay[env_ids] = 0.
            self.last_dof_pos[env_ids] = 0.
            self.last_dof_vel[env_ids] = 0.
            self.feet_air_time[env_ids] = 0.
            self.episode_length_buf[env_ids] = 0
            # self.reset_buf[env_ids] = 0
            # self.time_out_buf[env_ids] = 0
            self.reset_buf[env_ids] = 1
            self._update_average_episode_length(env_ids)

            self.history_handler.reset(env_ids)
            # Reset stance/air-peak buffers for new foot-penalty bookkeeping.
            self.feet_air_max_height[env_ids] = 0.0
            self.feet_air_max_height_on_contact[env_ids] = 0.0
            self.feet_stance_z_inited[env_ids] = False
            self.feet_stance_z[env_ids] = 0.0
            self.feet_stance_time[env_ids] = 0.0
            self.last_feet_vel_z[env_ids] = 0.0
            self._feet_last_contact[env_ids] = False
            self._feet_last_contact_filt[env_ids] = False
            # Also reset legacy contact-history buffers used by other reward terms / MPPI state.
            self.last_contacts[env_ids] = False
            self.last_contacts_filt[env_ids] = False

    def get_mppi_buffers(self, env_ids):
        """ Get buffers for MPPI
        MPPI algo need to store the buffers to replicate environments
        """
        return {
            "dof_pos": copy.deepcopy(self.simulator.dof_pos[env_ids]),
            "dof_vel": copy.deepcopy(self.simulator.dof_vel[env_ids]),
            "base_quat": copy.deepcopy(self.base_quat[env_ids]),
            "base_lin_vel": copy.deepcopy(self.base_lin_vel[env_ids]),
            "base_ang_vel": copy.deepcopy(self.base_ang_vel[env_ids]),
            "projected_gravity": copy.deepcopy(self.projected_gravity[env_ids]),
            "torques": copy.deepcopy(self.torques[env_ids]),
            "actions": copy.deepcopy(self.actions[env_ids]),
            "last_actions": copy.deepcopy(self.last_actions[env_ids]),
            "last_dof_pos": copy.deepcopy(self.last_dof_pos[env_ids]),
            "last_dof_vel": copy.deepcopy(self.last_dof_vel[env_ids]),
            "episode_length_buf": copy.deepcopy(self.episode_length_buf[env_ids]),
            "reset_buf": copy.deepcopy(self.reset_buf[env_ids]),
            "time_out_buf": copy.deepcopy(self.time_out_buf[env_ids]),
            "feet_air_time": copy.deepcopy(self.feet_air_time[env_ids]),
            "last_contacts": copy.deepcopy(self.last_contacts[env_ids]),
            "last_contacts_filt": copy.deepcopy(self.last_contacts_filt[env_ids]),
            "feet_air_max_height": copy.deepcopy(self.feet_air_max_height[env_ids]),
            "feet_air_max_height_on_contact": copy.deepcopy(self.feet_air_max_height_on_contact[env_ids]),
            "feet_stance_z": copy.deepcopy(self.feet_stance_z[env_ids]),
            "feet_stance_z_inited": copy.deepcopy(self.feet_stance_z_inited[env_ids]),
            "feet_stance_time": copy.deepcopy(self.feet_stance_time[env_ids]),
            "last_feet_vel_z": copy.deepcopy(self.last_feet_vel_z[env_ids]),
            "_feet_last_contact": copy.deepcopy(self._feet_last_contact[env_ids]),
            "_feet_last_contact_filt": copy.deepcopy(self._feet_last_contact_filt[env_ids]),
        }

    def _compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            try:
                assert rew.shape[0] == self.num_envs
            except:
                import ipdb; ipdb.set_trace()
            # penalty curriculum
            if name in self.config.rewards.reward_penalty_reward_names:
                if self.config.rewards.reward_penalty_curriculum:
                    rew *= self.reward_penalty_scale
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.config.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

        if self.use_reward_penalty_curriculum:
            self.log_dict["penalty_scale"] = torch.tensor(self.reward_penalty_scale, dtype=torch.float)
            self.log_dict["average_episode_length"] = self.average_episode_length
        
        if self.use_reward_limits_dof_pos_curriculum:
            self.log_dict["soft_dof_pos_curriculum_value"] = torch.tensor(self.soft_dof_pos_curriculum_value, dtype=torch.float)
        if self.use_reward_limits_dof_vel_curriculum:
            self.log_dict["soft_dof_vel_curriculum_value"] = torch.tensor(self.soft_dof_vel_curriculum_value, dtype=torch.float)
        if self.use_reward_limits_torque_curriculum:
            self.log_dict["soft_torque_curriculum_value"] = torch.tensor(self.soft_torque_curriculum_value, dtype=torch.float)
        
        if self.add_noise_currculum:
            self.log_dict["current_noise_curriculum_value"] = torch.tensor(self.current_noise_curriculum_value, dtype=torch.float)
    def _compute_observations(self):
        """ Computes observations
        """
        # super().compute_observations()
        self.obs_buf_dict_raw = {}
        self.hist_obs_dict = {}
        
        if self.add_noise_currculum:
            noise_extra_scale = self.current_noise_curriculum_value
        else:
            noise_extra_scale = 1.
        self._obs_noise_extra_scale = noise_extra_scale
        self._obs_raw_noisy_cache = {}
        # print("noise_extra_scale", noise_extra_scale)
        # compute Algo observations
        for obs_key, obs_config in self.config.obs.obs_dict.items():
            self.obs_buf_dict_raw[obs_key] = dict()
            # Backward compatible:
            # - default: use `noise_scales` for all obs groups
            # - optional: if `noise_scales_critic` exists, use it only for critic_obs
            noise_scales = self.config.obs.noise_scales
            if (
                obs_key == "critic_obs"
                and hasattr(self.config.obs, "noise_scales_critic")
                and self.config.obs.noise_scales_critic is not None
            ):
                # Merge override on top of default, so missing keys fall back to default noise scales.
                merged = dict(self.config.obs.noise_scales)
                merged.update(dict(self.config.obs.noise_scales_critic))
                noise_scales = merged

            parse_observation(
                self,
                obs_config,
                self.obs_buf_dict_raw[obs_key],
                self.config.obs.obs_scales,
                noise_scales,
                noise_extra_scale,
            )
        
        # Compute history observations
        history_obs_list = self.history_handler.history.keys()
        parse_observation(self, history_obs_list, self.hist_obs_dict, self.config.obs.obs_scales, self.config.obs.noise_scales, noise_extra_scale)
        
        self._post_config_observation_callback()

    def _post_config_observation_callback(self):
        self.obs_buf_dict = dict()
        
        for obs_key, obs_config in self.config.obs.obs_dict.items():
            obs_keys = sorted(obs_config)
            # print("obs_keys", obs_keys)
            self.obs_buf_dict[obs_key] = torch.cat([self.obs_buf_dict_raw[obs_key][key] for key in obs_keys], dim=-1)
    
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # actions_scaled = actions * self.config.robot.control.action_scale
        actions_scaled = actions * self.action_scales
        control_type = self.config.robot.control.control_type
        if control_type=="P":
            torques = self._kp_scale * self.p_gains*(actions_scaled + self.default_dof_pos - self.simulator.dof_pos) - self._kd_scale * self.d_gains*self.simulator.dof_vel
        elif control_type=="V":
            torques = self._kp_scale * self.p_gains*(actions_scaled - self.simulator.dof_vel) - self._kd_scale * self.d_gains*(self.simulator.dof_vel - self.last_dof_vel)/self.sim_dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        if self.config.domain_rand.randomize_torque_rfi:
            torques = torques + (torch.rand_like(torques)*2.-1.) * self.config.domain_rand.rfi_lim * self._rfi_lim_scale * self.torque_limits
        
        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        
        else:
            return torques

    def _create_terrain(self):
        super()._create_terrain()

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        self.gym.clear_lines(self.viewer)
        self._refresh_sim_tensors()

        draw_env_ids = (self.push_robot_plot_counter < 10).nonzero(as_tuple=False).flatten()
        not_draw_env_ids = (self.push_robot_plot_counter >= 10).nonzero(as_tuple=False).flatten()
        self.record_push_robot_vel_buf[not_draw_env_ids] *=0
        self.push_robot_plot_counter[not_draw_env_ids] = 0
        
        for env_id in draw_env_ids:
            push_vel = self.record_push_robot_vel_buf[env_id]
            push_vel = torch.cat([push_vel, torch.zeros(1, device=self.device)])
            push_pos = self.simulator.robot_root_states[env_id, :3]
            push_vel_list = [push_vel]
            push_pos_list = [push_pos]
            push_mag_list = [1]
            push_color_schems = [(0.851, 0.144, 0.07)]
            push_line_widths = [0.03]
            for push_vel, push_pos, push_mag, push_color, push_line_width in zip(push_vel_list, push_pos_list, push_mag_list, push_color_schems, push_line_widths):
                for _ in range(200):
                    gymutil.draw_line(Point(push_pos +torch.rand(3, device=self.device) * push_line_width),
                                        Point(push_pos + push_vel * push_mag),
                                        Point(push_color),
                                        self.gym, self.viewer, self.envs[env_id])


    ################ Curriculum #################

    def _update_average_episode_length(self, env_ids):
        num = len(env_ids)
        current_average_episode_length = torch.mean(self.last_episode_length_buf[env_ids], dtype=torch.float)
        
        self.average_episode_length = self.average_episode_length * (1 - num / self.num_compute_average_epl) + current_average_episode_length * (num / self.num_compute_average_epl)

        
    def _update_reward_penalty_curriculum(self):
        """
        Update the penalty curriculum based on the average episode length.

        If the average episode length is below the penalty level down threshold,
        decrease the penalty scale by a certain level degree.
        If the average episode length is above the penalty level up threshold,
        increase the penalty scale by a certain level degree.
        Clip the penalty scale within the specified range.

        Returns:
            None
        """
        if self.average_episode_length < self.config.rewards.reward_penalty_level_down_threshold:
            self.reward_penalty_scale *= (1 - self.config.rewards.reward_penalty_degree)
        elif self.average_episode_length > self.config.rewards.reward_penalty_level_up_threshold:
            self.reward_penalty_scale *= (1 + self.config.rewards.reward_penalty_degree)

        self.reward_penalty_scale = np.clip(self.reward_penalty_scale, self.config.rewards.reward_min_penalty_scale, self.config.rewards.reward_max_penalty_scale)

    def _update_reward_limits_curriculum(self):
        """
        Update the reward limits curriculum based on the average episode length.
        """
        if self.use_reward_limits_dof_pos_curriculum:
            if self.average_episode_length < self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum_level_down_threshold:
                self.soft_dof_pos_curriculum_value *= (1 + self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum_degree)
            elif self.average_episode_length > self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum_level_up_threshold:
                self.soft_dof_pos_curriculum_value *= (1 - self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum_degree)
            self.soft_dof_pos_curriculum_value = np.clip(self.soft_dof_pos_curriculum_value, 
                                                         self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_min_limit, 
                                                         self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_max_limit)
        
        if self.use_reward_limits_dof_vel_curriculum:
            if self.average_episode_length < self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum_level_down_threshold:
                self.soft_dof_vel_curriculum_value *= (1 + self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum_degree)
            elif self.average_episode_length > self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum_level_up_threshold:
                self.soft_dof_vel_curriculum_value *= (1 - self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum_degree)
            self.soft_dof_vel_curriculum_value = np.clip(self.soft_dof_vel_curriculum_value, 
                                                         self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_min_limit, 
                                                         self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_max_limit)
        
        if self.use_reward_limits_torque_curriculum:
            if self.average_episode_length < self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum_level_down_threshold:
                self.soft_torque_curriculum_value *= (1 + self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum_degree)
            elif self.average_episode_length > self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum_level_up_threshold:
                self.soft_torque_curriculum_value *= (1 - self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum_degree)
            self.soft_torque_curriculum_value = np.clip(self.soft_torque_curriculum_value, 
                                                        self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_min_limit, 
                                                        self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_max_limit)

    #------------ reward functions----------------
    ########################### PENALTY REWARDS ###########################

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_penalty_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_penalty_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.simulator.dof_vel), dim=1)
    
    def _reward_penalty_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.simulator.dof_vel) / self.dt), dim=1)
    
    def _reward_penalty_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_penalty_self_collision(self):
        """
        Best-effort approximation of OmniRetarget's self-collision penalty:
        binary penalty per (selected) body when net contact force magnitude exceeds a threshold.

        Note: Isaac Gym exposes net contact forces but does not separate self-collision from
        environment/object contacts in this codebase. We therefore apply it to
        `penalised_contact_indices` (configured via robot.penalize_contacts_on), which is a
        reasonable proxy for "undesired contacts" in upright humanoid tracking.
        """
        thr = float(getattr(self.config.rewards, "self_collision_force_threshold", 11.0))
        idx = getattr(self, "penalised_contact_indices", None)
        if idx is None or idx.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        forces = torch.norm(self.simulator.contact_forces[:, idx, :], dim=-1)  # (N, K)
        return (forces > thr).to(torch.float32).sum(dim=-1)

    def _reward_penalty_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    ######################## LIMITS REWARDS #########################

    def _reward_limits_dof_pos(self):
        # Penalize dof positions too close to the limit

        if self.use_reward_limits_dof_pos_curriculum:
            m = (self.simulator.hard_dof_pos_limits[:, 0] + self.simulator.hard_dof_pos_limits[:, 1]) / 2
            r = self.simulator.hard_dof_pos_limits[:, 1] - self.simulator.hard_dof_pos_limits[:, 0]
            lower_soft_limit = m - 0.5 * r * self.soft_dof_pos_curriculum_value
            upper_soft_limit = m + 0.5 * r * self.soft_dof_pos_curriculum_value
        else:
            lower_soft_limit = self.simulator.dof_pos_limits[:, 0]
            upper_soft_limit = self.simulator.dof_pos_limits[:, 1]
        out_of_limits = -(self.simulator.dof_pos - lower_soft_limit).clip(max=0.) # lower limit
        out_of_limits += (self.simulator.dof_pos - upper_soft_limit).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_limits_dof_vel(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        if self.use_reward_limits_dof_vel_curriculum:
            return torch.sum((torch.abs(self.simulator.dof_vel) - self.dof_vel_limits * self.soft_dof_vel_curriculum_value).clip(min=0., max=1.), dim=1)
        else:
            return torch.sum((torch.abs(self.simulator.dof_vel) - self.dof_vel_limits * self.config.rewards.reward_limit.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_limits_torque(self):
        # penalize torques too close to the limit
        if self.use_reward_limits_torque_curriculum:
            return torch.sum((torch.abs(self.torques) - self.torque_limits * self.soft_torque_curriculum_value).clip(min=0., max=1.), dim=1)
        else:
            return torch.sum((torch.abs(self.torques) - self.torque_limits * self.config.rewards.reward_limit.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_penalty_slippage(self):
        # assert self.simulator._rigid_body_vel.shape[1] == 20
        foot_vel = self.simulator._rigid_body_vel[:, self.feet_indices]
        return torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)

    def _reward_feet_max_height_for_this_air(self):
        """
        Backward compatible reward term used in some configs.
        NOTE: Contact filtering + feet peak tracking are updated once per step in `_pre_compute_observations_callback`
        to avoid reward-order side effects.
        """
        # Fallback for very old tasks/tests that might not have the new buffers.
        if not hasattr(self, "feet_first_contact") or not hasattr(self, "feet_air_max_height_on_contact"):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        desired = float(getattr(self.config.rewards, "desired_feet_max_height_for_this_air", 0.2))
        # reward only on first contact with the ground (per-foot), sum across feet
        from_air_to_contact = self.feet_first_contact
        rew = torch.clamp_min(desired - self.feet_air_max_height_on_contact, 0.0) * from_air_to_contact.to(torch.float32)
        return torch.sum(rew, dim=1)

    def _reward_penalty_both_feet_air(self):
        """
        Penalize double-float (both feet simultaneously not in contact).
        Uses filtered contact to reduce flicker.
        """
        if not hasattr(self, "feet_contact_filt"):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        both_air = (~self.feet_contact_filt).all(dim=1)
        return both_air.to(torch.float32)
    
    def _reward_feet_heading_alignment(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]

        forward_left_feet = quat_apply(left_quat, self.forward_vec)
        heading_left_feet = torch.atan2(forward_left_feet[:, 1], forward_left_feet[:, 0])
        forward_right_feet = quat_apply(right_quat, self.forward_vec)
        heading_right_feet = torch.atan2(forward_right_feet[:, 1], forward_right_feet[:, 0])


        root_forward = quat_apply(self.base_quat, self.forward_vec)
        heading_root = torch.atan2(root_forward[:, 1], root_forward[:, 0])

        heading_diff_left = torch.abs(wrap_to_pi(heading_left_feet - heading_root))
        heading_diff_right = torch.abs(wrap_to_pi(heading_right_feet - heading_root))
        
        return heading_diff_left + heading_diff_right
    
    def _reward_penalty_feet_ori(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5 
    
    def _episodic_domain_randomization(self, env_ids):
        """ Update scale of Kp, Kd, rfi lim"""
        if len(env_ids) == 0:
            return
        if self.config.domain_rand.randomize_pd_gain:
            self._kp_scale[env_ids] = torch_rand_float(self.config.domain_rand.kp_range[0], self.config.domain_rand.kp_range[1], (len(env_ids), self.num_dofs), device=self.device)
            self._kd_scale[env_ids] = torch_rand_float(self.config.domain_rand.kd_range[0], self.config.domain_rand.kd_range[1], (len(env_ids), self.num_dofs), device=self.device)
    
        if self.config.domain_rand.randomize_rfi_lim:
            self._rfi_lim_scale[env_ids] = torch_rand_float(self.config.domain_rand.rfi_lim_range[0], self.config.domain_rand.rfi_lim_range[1], (len(env_ids), self.num_dofs), device=self.device)

        if self.config.domain_rand.randomize_ctrl_delay:            
            # self.action_queue[env_ids] = 0.delay:
            self.action_queue[env_ids] *= 0.
            # self.action_queue[env_ids] = 0.
            self.action_delay_idx[env_ids] = torch.randint(self.config.domain_rand.ctrl_delay_step_range[0], 
                                            self.config.domain_rand.ctrl_delay_step_range[1]+1, (len(env_ids),), device=self.device, requires_grad=False)

        if self.config.domain_rand.randomize_default_pos:
            # Default joint position offsets.
            # Backward compatible: if `default_pos_range_by_dof_name` is missing, apply a uniform range to all DOFs.
            lo = float(self.config.domain_rand.default_pos_range[0])
            hi = float(self.config.domain_rand.default_pos_range[1])
            num_dofs = int(getattr(self, "num_dofs", getattr(self, "num_dof")))

            offsets = torch_rand_float(lo, hi, (len(env_ids), num_dofs), device=self.device)

            by_name = getattr(self.config.domain_rand, "default_pos_range_by_dof_name", None)
            if by_name is not None:
                # `by_name` may be an OmegaConf DictConfig; iterate keys for max compatibility.
                for dof_name in list(by_name.keys()):
                    try:
                        dof_idx = self.dof_names.index(dof_name)
                    except Exception:
                        continue
                    rng = by_name[dof_name]
                    if rng is None or len(rng) != 2:
                        continue
                    dlo = float(rng[0])
                    dhi = float(rng[1])
                    offsets[:, dof_idx] = torch_rand_float(dlo, dhi, (len(env_ids), 1), device=self.device).squeeze(-1)

            self.default_dof_pos[env_ids] = self.original_default_dof_pos + offsets

    def _push_robots(self, env_ids):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        max_vel = self.config.domain_rand.max_push_vel_xyz
        self.push_robot_vel_buf[env_ids] = torch_rand_float(-max_vel, max_vel, (len(env_ids), 3), device=str(self.device))  # lin vel x/y
        self.record_push_robot_vel_buf[env_ids] = self.push_robot_vel_buf[env_ids].clone()
        self.simulator.robot_root_states[env_ids, 7:10] += self.push_robot_vel_buf[env_ids]
        # self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.simulator.all_root_states))

        max_ang_vel = self.config.domain_rand.max_push_ang_vel_xyz
        self.push_robot_ang_vel_buf[env_ids] = torch_rand_float(-max_ang_vel, max_ang_vel, (len(env_ids), 3), device=str(self.device))  # lin vel x/y
        self.record_push_robot_ang_vel_buf[env_ids] = self.push_robot_ang_vel_buf[env_ids].clone()
        self.simulator.robot_root_states[env_ids, 10:13] += self.push_robot_ang_vel_buf[env_ids]

    ############ TERRAIN AND COMMANDS

    ################ ENV CALLBACKS #################

    def _reset_dofs(self, env_ids, target_state=None):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.
        If target_state is not None, reset to target_state

        Args:
            env_ids (List[int]): Environemnt ids
            target_state (Tensor): Target state
        """
        if target_state is not None:
            self.simulator.dof_pos[env_ids] = target_state[..., 0]
            self.simulator.dof_vel[env_ids] = target_state[..., 1]
        else:
            self.simulator.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=str(self.device))
            # self.simulator.dof_pos[env_ids] = self.default_dof_pos
            # import ipdb; ipdb.set_trace()
            
            self.simulator.dof_vel[env_ids] = 0.


    def _reset_root_states(self, env_ids, target_root_states=None):
        """ Resets ROOT states position and velocities of selected environmments
            if target_root_states is not None, reset to target_root_states
        Args:
            env_ids (List[int]): Environemnt ids
            target_root_states (Tensor): Target root states
        """
        if target_root_states is not None:
            self.simulator.robot_root_states[env_ids] = target_root_states
            self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]

        else:
            # base position
            if self.custom_origins:
                self.simulator.robot_root_states[env_ids] = self.base_init_state
                self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]
                self.simulator.robot_root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=str(self.device)) # xy position within 1m of the center
            else:
                self.simulator.robot_root_states[env_ids] = self.base_init_state
                self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]
            # base velocities
            
            self.simulator.robot_root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=str(self.device)) # [7:10]: lin vel, [10:13]: ang vel


    def _plot_domain_rand_params(self):
        raise NotImplementedError

    ######################### Observations #########################
    def _get_obs_base_pos_z(self,):
        return self.simulator.robot_root_states[:, 2:3]
    
    def _get_obs_feet_contact_force(self,):
        return self.simulator.contact_forces[:, self.feet_indices, :].view(self.num_envs, -1)
          
    
    def _get_obs_base_lin_vel(self,):
        return self.base_lin_vel
    
    def _get_obs_base_ang_vel(self,):
        return self.base_ang_vel
    
    def _get_obs_projected_gravity(self,):
        return self.projected_gravity
    
    def _get_obs_dof_pos(self,):
        return self.simulator.dof_pos - self.default_dof_pos
    
    def _get_obs_dof_vel(self,):
        return self.simulator.dof_vel
    
    def _get_obs_history(self,):
        assert "history" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    def _get_obs_short_history(self,):
        assert "short_history" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['short_history']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    def _get_obs_long_history(self,):
        assert "long_history" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['long_history']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    def _get_obs_actions(self,):
        return self.actions
