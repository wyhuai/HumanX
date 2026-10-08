import torch
import numpy as np
from pathlib import Path
from isaacgym.torch_utils import *
from torch import Tensor
import os
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from isaac_utils.rotations import (
    my_quat_rotate,
    calc_heading_quat_inv,
    calc_heading_quat,
    quat_mul,
    quat_conjugate,
    quat_rotate_inverse,
    xyzw_to_wxyz,
    wxyz_to_xyzw,
    get_euler_xyz_in_tensor,
    quat_from_euler_xyz,
)
# from isaacgym import gymtorch, gymapi, gymutil
from humanoidverse.envs.env_utils.visualization import Point

from humanoidverse.utils.motion_lib.skeleton import SkeletonTree

from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot

from termcolor import colored
from loguru import logger

from scipy.spatial.transform import Rotation as sRot
import joblib

class LeggedRobotMotionTracking(LeggedRobotBase):
    def __init__(self, config, device):
        self.init_done = False
        self.debug_viz = True
        
        super().__init__(config, device)
        self._init_motion_lib()
        self._init_motion_extend()
        self._init_tracking_config()

        self.init_done = True
        self.debug_viz = True

        self.obs_reduction_timer = 2000*24

        self._init_save_motion()

        if self.config.use_teleop_control:
            self.teleop_marker_coords = torch.zeros(self.num_envs, 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import Float64MultiArray
            self.node = Node("motion_tracking")
            self.teleop_sub = self.node.create_subscription(Float64MultiArray, "vision_pro_data", self.teleop_callback, 1)

        if self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum:
            self.terminate_when_motion_far_threshold = self.config.termination_curriculum.terminate_when_motion_far_initial_threshold
            logger.info(f"Terminate when motion far threshold: {self.terminate_when_motion_far_threshold}")

        else:
            self.terminate_when_motion_far_threshold = self.config.termination_scales.termination_motion_far_threshold
            logger.info(f"Terminate when motion far threshold: {self.terminate_when_motion_far_threshold}")



        

    def teleop_callback(self, msg):
        self.teleop_marker_coords = torch.tensor(msg.data, device=self.device)

    def _init_save_motion(self):
        if "save_motion" in self.config:
            self.save_motion = self.config.save_motion
            if self.save_motion:
                os.makedirs(Path(self.config.ckpt_dir) / "motions", exist_ok = True)

                
                if hasattr(self.config, 'dump_motion_name'):
                    self.save_motion_dir = Path(self.config.ckpt_dir) / "motions" / (str(self.config.eval_timestamp) + "_" + self.config.dump_motion_name)
                else:
                    self.save_motion_dir = Path(self.config.ckpt_dir) / "motions" / f"{self.config.save_note}_{self.config.eval_timestamp}"
                self.save_motion = True
                self.num_augment_joint = len(self.config.robot.motion.extend_config)
                self.motions_for_saving = {'root_trans_offset':[], 'pose_aa':[], 'dof':[], 'root_rot':[], 'actor_obs':[], 'action':[], 'terminate':[],
                                            'root_lin_vel':[], 'root_ang_vel':[], 'dof_vel':[]}
                self.motion_times_buf = []
                self.start_save = False

        else:
            self.save_motion = False

    def _init_motion_lib(self):
        self.config.robot.motion.step_dt = self.dt
        self._motion_lib = MotionLibRobot(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
        if self.is_evaluating:
            self._motion_lib.load_motions(random_sample=False)
            # self._motion_lib.load_motions(random_sample=True)
        else:
            self._motion_lib.load_motions(random_sample=True)
            
        # res = self._motion_lib.get_motion_state(self.motion_ids, self.motion_times, offset=self.env_origins)
        res = self._resample_motion_times(torch.arange(self.num_envs))
        self.motion_dt = self._motion_lib._motion_dt
        self.motion_start_idx = 0
        self.num_motions = self._motion_lib._num_unique_motions

    def _init_tracking_config(self):
        if "motion_tracking_link" in self.config.robot.motion:
            self.motion_tracking_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.motion_tracking_link]
        if "lower_body_link" in self.config.robot.motion:
            self.lower_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.lower_body_link]
        if "upper_body_link" in self.config.robot.motion:
            self.upper_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.upper_body_link]
        if self.config.resample_motion_when_training:
            self.resample_time_interval = np.ceil(self.config.resample_time_interval_s / self.dt)
        
    def _init_motion_extend(self):
        if "extend_config" in self.config.robot.motion:
            extend_parent_ids, extend_pos, extend_rot = [], [], []
            for extend_config in self.config.robot.motion.extend_config:
                extend_parent_ids.append(self.simulator._body_list.index(extend_config["parent_name"]))
                # extend_parent_ids.append(self.simulator.find_rigid_body_indice(extend_config["parent_name"]))
                extend_pos.append(extend_config["pos"])
                extend_rot.append(extend_config["rot"])
                self.simulator._body_list.append(extend_config["joint_name"])

            self.extend_body_parent_ids = torch.tensor(extend_parent_ids, device=self.device, dtype=torch.long)
            self.extend_body_pos_in_parent = torch.tensor(extend_pos).repeat(self.num_envs, 1, 1).to(self.device)
            self.extend_body_rot_in_parent_wxyz = torch.tensor(extend_rot).repeat(self.num_envs, 1, 1).to(self.device)
            self.extend_body_rot_in_parent_xyzw = self.extend_body_rot_in_parent_wxyz[:, :, [1, 2, 3, 0]]
            self.num_extend_bodies = len(extend_parent_ids)

            self.marker_coords = torch.zeros(self.num_envs, 
                                         self.num_bodies + self.num_extend_bodies, 
                                         3, 
                                         dtype=torch.float, 
                                         device=self.device, 
                                         requires_grad=False) # extend
            
            self.ref_body_pos_extend = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.dif_global_body_pos = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)

    def start_compute_metrics(self):
        self.compute_metrics = True
        self.start_idx = 0
    
    def forward_motion_samples(self):
        pass
    
    def _init_buffers(self):
        super()._init_buffers()
        self.vr_3point_marker_coords = torch.zeros(self.num_envs, 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.realtime_vr_keypoints_pos = torch.zeros(3, 3, dtype=torch.float, device=self.device, requires_grad=False) # hand, hand, head
        self.realtime_vr_keypoints_vel = torch.zeros(3, 3, dtype=torch.float, device=self.device, requires_grad=False) # hand, hand, head
        self.motion_ids = torch.arange(self.num_envs).to(self.device)
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        
    def _init_domain_rand_buffers(self):
        super()._init_domain_rand_buffers()
        self.ref_episodic_offset = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

    def _reset_tasks_callback(self, env_ids):
        if len(env_ids) == 0:
            return
        super()._reset_tasks_callback(env_ids)
        # env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._resample_motion_times(env_ids) # need to resample before reset root states
        if self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum:
            self._update_terminate_when_motion_far_curriculum()
    
    def _update_terminate_when_motion_far_curriculum(self):
        assert self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum
        if self.average_episode_length < self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_down_threshold:
            self.terminate_when_motion_far_threshold *= (1 + self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        elif self.average_episode_length > self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_up_threshold:
            self.terminate_when_motion_far_threshold *= (1 - self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        self.terminate_when_motion_far_threshold = np.clip(self.terminate_when_motion_far_threshold, 
                                                         self.config.termination_curriculum.terminate_when_motion_far_threshold_min, 
                                                         self.config.termination_curriculum.terminate_when_motion_far_threshold_max)
        

    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        if self.config.resample_motion_when_training:
            if self.common_step_counter % self.resample_time_interval == 0:
                logger.info(f"Resampling motion at step {self.common_step_counter}")
                self.resample_motion()

    def set_is_evaluating(self):
        super().set_is_evaluating()

    def _check_termination(self):
        super()._check_termination()
        if self.config.termination.terminate_when_motion_far:
            reset_buf_motion_far = torch.any(torch.norm(self.dif_global_body_pos, dim=-1) > self.terminate_when_motion_far_threshold, dim=-1)
            self.reset_buf |= reset_buf_motion_far
            # log current motion far threshold
            if self.config.termination_curriculum.terminate_when_motion_far_curriculum:
                self.log_dict["terminate_when_motion_far_threshold"] = torch.tensor(self.terminate_when_motion_far_threshold, dtype=torch.float)

    def _update_timeout_buf(self):
        super()._update_timeout_buf()
        if self.config.termination.terminate_when_motion_end:
            current_time = (self.episode_length_buf) * self.dt + self.motion_start_times
            self.time_out_buf |= current_time > self.motion_len

    def next_task(self):
        # This function is only called when evaluating
        self.motion_start_idx += self.num_envs
        if self.motion_start_idx >= self.num_motions:
            self.motion_start_idx = 0
        self._motion_lib.load_motions(random_sample=False, start_idx=self.motion_start_idx)
        self.reset_all()

    def _resample_motion_times(self, env_ids):
        if len(env_ids) == 0:
            return
        self.motion_len[env_ids] = self._motion_lib.get_motion_length(self.motion_ids[env_ids])
        if self.is_evaluating and not self.config.enforce_randomize_motion_start_eval:
            self.motion_start_times[env_ids] = torch.zeros(len(env_ids), dtype=torch.float32, device=self.device)#wyh
            # self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        else:
            self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        # self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        # offset = self.env_origins
        # motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset= offset)
        # motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)

    def resample_motion(self):
        self._motion_lib.load_motions(random_sample=True)
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))


    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        
        offset = self.env_origins
        B = self.motion_ids.shape[0]
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)

        ref_body_pos_extend = motion_res["rg_pos_t"]
        self.ref_body_pos_extend[:] = ref_body_pos_extend # for visualization and analysis
        ref_body_vel_extend = motion_res["body_vel_t"] # [num_envs, num_markers, 3]
        self.ref_body_rot_extend = ref_body_rot_extend = motion_res["rg_rot_t"] # [num_envs, num_markers, 4]
        ref_body_ang_vel_extend = motion_res["body_ang_vel_t"] # [num_envs, num_markers, 3]
        ref_joint_pos = motion_res["dof_pos"] # [num_envs, num_dofs]
        ref_joint_vel = motion_res["dof_vel"] # [num_envs, num_dofs]


        ################### EXTEND Rigid body POS #####################
        rotated_pos_in_parent = my_quat_rotate(
            self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
            self.extend_body_pos_in_parent.reshape(-1, 3)
        )
        extend_curr_pos = my_quat_rotate(
            self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
            rotated_pos_in_parent
        ).view(self.num_envs, -1, 3) + self.simulator._rigid_body_pos[:, self.extend_body_parent_ids]
        self._rigid_body_pos_extend = torch.cat([self.simulator._rigid_body_pos, extend_curr_pos], dim=1)

        ################### EXTEND Rigid body Rotation #####################
        extend_curr_rot = quat_mul(self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
                                    self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
                                    w_last=True).view(self.num_envs, -1, 4)
        self._rigid_body_rot_extend = torch.cat([self.simulator._rigid_body_rot, extend_curr_rot], dim=1)
        
        ################### EXTEND Rigid Body Angular Velocity #####################
        self._rigid_body_ang_vel_extend = torch.cat([self.simulator._rigid_body_ang_vel, self.simulator._rigid_body_ang_vel[:, self.extend_body_parent_ids]], dim=1)
    
        ################### EXTEND Rigid Body Linear Velocity #####################
        self._rigid_body_ang_vel_global = self.simulator._rigid_body_ang_vel[:, self.extend_body_parent_ids]
        angular_velocity_contribution = torch.cross(self._rigid_body_ang_vel_global, self.extend_body_pos_in_parent.view(self.num_envs, -1, 3), dim=2)
        extend_curr_vel = self.simulator._rigid_body_vel[:, self.extend_body_parent_ids] + angular_velocity_contribution.view(self.num_envs, -1, 3)
        self._rigid_body_vel_extend = torch.cat([self.simulator._rigid_body_vel, extend_curr_vel], dim=1)

        ################### Compute differences #####################

        ## diff compute - kinematic position
        self.dif_global_body_pos = ref_body_pos_extend - self._rigid_body_pos_extend
        # import ipdb; ipdb.set_trace()
        ## diff compute - kinematic rotation
        self.dif_global_body_rot = ref_body_rot_extend - self._rigid_body_rot_extend
        ## diff compute - kinematic velocity
        self.dif_global_body_vel = ref_body_vel_extend - self._rigid_body_vel_extend
        ## diff compute - kinematic angular velocity
        
        self.dif_global_body_ang_vel = ref_body_ang_vel_extend - self._rigid_body_ang_vel_extend
        # ang_vel_reward = self._reward_teleop_body_ang_velocity_extend()



        
        ## diff compute - kinematic joint position
        self.dif_joint_angles = ref_joint_pos - self.simulator.dof_pos
        ## diff compute - kinematic joint velocity
        self.dif_joint_velocities = ref_joint_vel - self.simulator.dof_vel

        



        # marker_coords for visualization
        self.marker_coords[:] = ref_body_pos_extend.reshape(B, -1, 3)

        env_batch_size = self.simulator._rigid_body_pos.shape[0]
        num_rigid_bodies = self.simulator._rigid_body_pos.shape[1]

        heading_inv_rot = calc_heading_quat_inv(self.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        # expand to (B*num_rigid_bodies, 4) for fatser computation in jit
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(1).expand(-1, num_rigid_bodies+self.num_extend_bodies, -1).reshape(-1, 4)

        heading_rot = calc_heading_quat(self.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        heading_rot_expand = heading_rot.unsqueeze(1).expand(-1, num_rigid_bodies, -1).reshape(-1, 4)

        dif_global_body_pos_for_obs_compute = ref_body_pos_extend.view(env_batch_size, -1, 3) - self._rigid_body_pos_extend.view(env_batch_size, -1, 3)
        dif_local_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), dif_global_body_pos_for_obs_compute.view(-1, 3))
        
        self._obs_dif_local_rigid_body_pos = dif_local_body_pos_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)

        global_ref_rigid_body_pos = ref_body_pos_extend.view(env_batch_size, -1, 3) - self.simulator.robot_root_states[:, :3].view(env_batch_size, 1, 3)  # preserves the body position
        local_ref_rigid_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_rigid_body_pos.view(-1, 3))
        self._obs_local_ref_rigid_body_pos = local_ref_rigid_body_pos_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)

        global_ref_body_vel = ref_body_vel_extend.view(env_batch_size, -1, 3)
        local_ref_rigid_body_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_body_vel.view(-1, 3))

        self._obs_local_ref_rigid_body_vel = local_ref_rigid_body_vel_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)

        ######################VR 3 point ########################
        if not self.config.use_teleop_control:
            ref_vr_3point_pos = ref_body_pos_extend.view(env_batch_size, -1, 3)[:, self.motion_tracking_id, :]
        else:
            ref_vr_3point_pos = self.teleop_marker_coords
        vr_2root_pos = (ref_vr_3point_pos - self.simulator.robot_root_states[:, 0:3].view(env_batch_size, 1, 3))
        heading_inv_rot_vr = heading_inv_rot.repeat(3,1)
        self._obs_vr_3point_pos = my_quat_rotate(heading_inv_rot_vr.view(-1, 4), vr_2root_pos.view(-1, 3)).view(env_batch_size, -1)
        #################### Deepmimic phase ###################### 

        self._ref_motion_length = self._motion_lib.get_motion_length(self.motion_ids)
        self._ref_motion_phase = motion_times / self._ref_motion_length
        if not (torch.all(self._ref_motion_phase >= 0) and torch.all(self._ref_motion_phase <= 1.05)): # hard coded 1.05 because +1 will exceed 1
            max_phase = self._ref_motion_phase.max()
            # import ipdb; ipdb.set_trace()
        self._ref_motion_phase = self._ref_motion_phase.unsqueeze(1)
        # print(f"ref_motion_phase: {self._ref_motion_phase[0].item():.2f}")
        # print(f"ref_motion_length: {self._ref_motion_length[0].item():.2f}")

        # #wyh
        # if self.obs_reduction_timer > 0:
        #     self.obs_reduction_timer += -1
        # obs_reduction_scale = (self.obs_reduction_timer//24)*(1./2000.)
        # self._ref_motion_phase = self._ref_motion_phase*obs_reduction_scale #*0 #test
        
        self._log_motion_tracking_info()

    def _compute_reward(self):
        super()._compute_reward()
        self.extras["ref_body_pos_extend"] = self.ref_body_pos_extend.clone()
        self.extras["ref_body_rot_extend"] = self.ref_body_rot_extend.clone()

    def _log_motion_tracking_info(self):
        upper_body_diff = self.dif_global_body_pos[:, self.upper_body_id, :]
        lower_body_diff = self.dif_global_body_pos[:, self.lower_body_id, :]
        vr_3point_diff = self.dif_global_body_pos[:, self.motion_tracking_id, :]
        joint_pos_diff = self.dif_joint_angles

        upper_body_diff_norm = upper_body_diff.norm(dim=-1).mean()
        lower_body_diff_norm = lower_body_diff.norm(dim=-1).mean()
        vr_3point_diff_norm = vr_3point_diff.norm(dim=-1).mean()
        joint_pos_diff_norm = joint_pos_diff.norm(dim=-1).mean()

        self.log_dict["upper_body_diff_norm"] = upper_body_diff_norm
        self.log_dict["lower_body_diff_norm"] = lower_body_diff_norm
        self.log_dict["vr_3point_diff_norm"] = vr_3point_diff_norm
        self.log_dict["joint_pos_diff_norm"] = joint_pos_diff_norm
        

    def _draw_debug_vis(self):
        self.simulator.clear_lines()
        self._refresh_sim_tensors()

        for env_id in range(self.num_envs):
            if not self.config.use_teleop_control:
                # draw marker joints
                for pos_id, pos_joint in enumerate(self.marker_coords[env_id]): # idx 0 torso (duplicate with 11)
                    if self.config.robot.motion.visualization.customize_color:
                        color_inner = self.config.robot.motion.visualization.marker_joint_colors[pos_id % len(self.config.robot.motion.visualization.marker_joint_colors)]
                    else:
                        color_inner = (0.3, 0.3, 0.3)
                    color_inner = tuple(color_inner)

                    # import ipdb; ipdb.set_trace()
                    self.simulator.draw_sphere(pos_joint, 0.04, color_inner, env_id, pos_id)


            else:
                # draw teleop joints
                for pos_id, pos_joint in enumerate(self.teleop_marker_coords[env_id]):
                    self.simulator.draw_sphere(pos_joint, 0.04, (0.851, 0.144, 0.07), env_id, pos_id)

    def _reset_root_states(self, env_ids):
        # reset root states according to the reference motion
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins: # trimesh
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
            offset = self.env_origins
            motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
            self.simulator.robot_root_states[env_ids, :3] = motion_res['root_pos'][env_ids]
            # self.robot_root_states[env_ids, 2] += 0.04 # in case under the terrain
            if self.config.simulator.config.name == 'isaacgym':
                self.simulator.robot_root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
            elif self.config.simulator.config.name == 'isaacsim':
                self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(motion_res['root_rot'][env_ids])
            elif self.config.simulator.config.name == 'genesis':
                self.simulator.robot_root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
                raise NotImplementedError
            self.simulator.robot_root_states[env_ids, 7:10] = motion_res['root_vel'][env_ids]
            self.simulator.robot_root_states[env_ids, 10:13] = motion_res['root_ang_vel'][env_ids]
            

        else:
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
            offset = self.env_origins
            motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)


            root_pos_noise = self.config.init_noise_scale.root_pos * self.config.noise_to_initial_level
            root_rot_noise = self.config.init_noise_scale.root_rot * 3.14 / 180 * self.config.noise_to_initial_level
            # optional: yaw-only init noise (degree in config, converted to rad here). Keep backward compatible.
            root_rot_yaw_noise = getattr(self.config.init_noise_scale, "root_rot_yaw", 0.0) * 3.14 / 180 * self.config.noise_to_initial_level
            root_vel_noise = self.config.init_noise_scale.root_vel * self.config.noise_to_initial_level
            root_ang_vel_noise = self.config.init_noise_scale.root_ang_vel * self.config.noise_to_initial_level

            root_pos = motion_res['root_pos'][env_ids]
            root_rot = motion_res['root_rot'][env_ids]
            root_vel = motion_res['root_vel'][env_ids]
            root_ang_vel = motion_res['root_ang_vel'][env_ids]

            # self.simulator.robot_root_states[env_ids, :3] = root_pos + torch.randn_like(root_pos) * root_pos_noise
            # if self.config.simulator.config.name == 'isaacgym':
            #     self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            # elif self.config.simulator.config.name == 'isaacsim':
            #     self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True))
            # elif self.config.simulator.config.name == 'genesis':
            #     self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            # elif self.config.simulator.config.name == 'mujoco':
            #     self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            # else:
            #     raise NotImplementedError
            # self.simulator.robot_root_states[env_ids, 7:10] = root_vel + torch.randn_like(root_vel) * root_vel_noise
            # self.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel + torch.randn_like(root_ang_vel) * root_ang_vel_noise

            #wyh
            self.simulator.robot_root_states[env_ids, :3] = root_pos + self.p_noise(root_pos.shape, p=self.config.init_noise_p, scale=root_pos_noise) #wyh

            # keep existing root_rot init noise behavior per-simulator, then optionally stack yaw-only noise
            if self.config.simulator.config.name == 'isaacgym':
                noisy_root_rot = self.p_noise_rotate(root_rot, root_rot_noise)
                if root_rot_yaw_noise != 0.0:
                    noisy_root_rot = quat_mul(self.small_random_yaw_quaternions(root_rot.shape[0], root_rot_yaw_noise), noisy_root_rot, w_last=True)
                self.simulator.robot_root_states[env_ids, 3:7] = noisy_root_rot
            elif self.config.simulator.config.name == 'isaacsim':
                noisy_root_rot = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
                if root_rot_yaw_noise != 0.0:
                    noisy_root_rot = quat_mul(self.small_random_yaw_quaternions(root_rot.shape[0], root_rot_yaw_noise), noisy_root_rot, w_last=True)
                self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(noisy_root_rot)
            elif self.config.simulator.config.name == 'genesis':
                noisy_root_rot = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
                if root_rot_yaw_noise != 0.0:
                    noisy_root_rot = quat_mul(self.small_random_yaw_quaternions(root_rot.shape[0], root_rot_yaw_noise), noisy_root_rot, w_last=True)
                self.simulator.robot_root_states[env_ids, 3:7] = noisy_root_rot
            elif self.config.simulator.config.name == 'mujoco':
                noisy_root_rot = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
                if root_rot_yaw_noise != 0.0:
                    noisy_root_rot = quat_mul(self.small_random_yaw_quaternions(root_rot.shape[0], root_rot_yaw_noise), noisy_root_rot, w_last=True)
                self.simulator.robot_root_states[env_ids, 3:7] = noisy_root_rot
            else:
                raise NotImplementedError
            self.simulator.robot_root_states[env_ids, 7:10] = root_vel + self.p_noise(root_vel.shape, p=self.config.init_noise_p, scale=root_vel_noise)
            self.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel + self.p_noise(root_ang_vel.shape, p=self.config.init_noise_p, scale=root_ang_vel_noise)



    def p_noise(self, shape, p, scale): #wyh
        mask = torch.rand(shape[0]).to('cuda') < p
        # noise = (torch.rand(shape).to('cuda') * 2 - 1) * scale
        noise = torch.randn(shape).to('cuda') * scale
        return mask.unsqueeze(1) * noise

    def p_noise_rotate(self, init_obj_rot, max_radians): #wyh
        rand_angles = self.p_noise(init_obj_rot.shape, p=self.config.init_noise_p, scale=max_radians)
        roll, pitch, yaw = rand_angles[:,0], rand_angles[:,1], rand_angles[:,2]
        current_euler = quat_to_euler(init_obj_rot)  # (N, 3)
        roll_cur, pitch_cur, yaw_cur = current_euler[:,0], current_euler[:,1], current_euler[:,2]
        roll_new, pitch_new, yaw_new = roll_cur + roll, pitch_cur + pitch, yaw_cur + yaw
        new_quat = quat_from_euler_xyz(roll_new, pitch_new, yaw_new)
        return new_quat

    def small_random_quaternions(self, n, max_angle):
            axis = torch.randn((n, 3), device=self.device)
            axis = axis / torch.norm(axis, dim=1, keepdim=True)  # Normalize axis
            angles = max_angle * torch.rand((n, 1), device=self.device)
            
            # Convert angle-axis to quaternion
            sin_half_angle = torch.sin(angles / 2)
            cos_half_angle = torch.cos(angles / 2)
            
            q = torch.cat([sin_half_angle * axis, cos_half_angle], dim=1)  
            return q

    def small_random_yaw_quaternions(self, n, max_yaw):
            """Yaw-only random rotation quaternion (xyzw, w last).
            yaw ~ Uniform(-max_yaw, +max_yaw) in radians.
            """
            yaw = (torch.rand((n, 1), device=self.device) * 2.0 - 1.0) * max_yaw
            sin_half = torch.sin(yaw / 2.0)
            cos_half = torch.cos(yaw / 2.0)
            zeros = torch.zeros_like(sin_half)
            q = torch.cat([zeros, zeros, sin_half, cos_half], dim=1)
            return q

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """

        motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
        offset = self.env_origins
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)

        dof_pos_noise = self.config.init_noise_scale.dof_pos * self.config.noise_to_initial_level
        dof_vel_noise = self.config.init_noise_scale.dof_vel * self.config.noise_to_initial_level
        dof_pos = motion_res['dof_pos'][env_ids]
        dof_vel = motion_res['dof_vel'][env_ids]
        self.simulator.dof_pos[env_ids] = dof_pos + torch.randn_like(dof_pos) * dof_pos_noise
        self.simulator.dof_vel[env_ids] = dof_vel + torch.randn_like(dof_vel) * dof_vel_noise


    def _post_physics_step(self):
        super()._post_physics_step()
        
        if self.save_motion:    
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times

            if (len(self.motions_for_saving['dof'])) > self.config.save_total_steps:
                for k, v in self.motions_for_saving.items():
                    self.motions_for_saving[k] = torch.stack(v[3:]).transpose(0,1).numpy()
                
                self.motions_for_saving['motion_times'] = torch.stack(self.motion_times_buf[3:]).transpose(0,1).numpy()
                
                dump_data = {}
                num_motions = self.num_envs 
                keys_to_save = self.motions_for_saving.keys()

                for i in range(num_motions):
                    motion_key = f"motion{i}" 
                    dump_data[motion_key] = {
                        key: self.motions_for_saving[key][i] for key in keys_to_save
                    }
                    dump_data[motion_key]['fps'] = 1 / self.dt
    
                joblib.dump(dump_data, f'{self.save_motion_dir}.pkl')
                
                print(colored(f"Saved motion data to {self.save_motion_dir}.pkl", 'green'))
                import sys
                sys.exit()

            root_trans = self.simulator.robot_root_states[:, 0:3].cpu()
            if self.config.simulator.config.name == "isaacgym":
                root_rot = self.simulator.robot_root_states[:, 3:7].cpu() # xyzw
            elif self.config.simulator.config.name == "isaacsim":
                root_rot = self.simulator.robot_root_states[:, [4, 5, 6, 3]].cpu() # wxyz to xyzw   
            elif self.config.simulator.config.name == "genesis":
                root_rot = self.simulator.robot_root_states[:,  3:7].cpu() # xyzw
            else:
                raise NotImplementedError
            root_rot_vec = torch.from_numpy(sRot.from_quat(root_rot.numpy()).as_rotvec()).float()
            dof = self.simulator.dof_pos.cpu()
            # T, num_env, J, 3
            # print(self._motion_lib.mesh_parsers.dof_axis)
            pose_aa = torch.cat([root_rot_vec[:, None, :], self._motion_lib.mesh_parsers.dof_axis * dof[:, :, None], torch.zeros((self.num_envs, self.num_augment_joint, 3))], axis = 1)
            self.motions_for_saving['root_trans_offset'].append(root_trans)
            self.motions_for_saving['root_rot'].append(root_rot)
            self.motions_for_saving['dof'].append(dof)
            self.motions_for_saving['pose_aa'].append(pose_aa)
            self.motions_for_saving['action'].append(self.actions.cpu())
            self.motions_for_saving['actor_obs'].append(self.obs_buf_dict['actor_obs'].cpu())
            self.motions_for_saving['terminate'].append(self.reset_buf.cpu())
            
            self.motions_for_saving['dof_vel'].append(self.simulator.dof_vel.cpu())
            self.motions_for_saving['root_lin_vel'].append(self.simulator.robot_root_states[:, 7:10].cpu())
            self.motions_for_saving['root_ang_vel'].append(self.simulator.robot_root_states[:, 10:13].cpu())
            
            self.motion_times_buf.append(motion_times.cpu())

            self.start_save = True

    # ############################################################
        
    def _get_obs_dif_local_rigid_body_pos(self):
        return self._obs_dif_local_rigid_body_pos
    
    def _get_obs_local_ref_rigid_body_pos(self):
        return self._obs_local_ref_rigid_body_pos
    
    def _get_obs_ref_motion_phase(self):
        # print(self._ref_motion_phase)
        return self._ref_motion_phase #wyh
    
    def _get_obs_vr_3point_pos(self):
        return self._obs_vr_3point_pos

    ######################### Observations #########################
    def _get_obs_history_actor(self,):
        assert "history_actor" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_actor']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    def _get_obs_history_critic(self,):
        assert "history_critic" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_critic']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    ###############################################################

    def _reward_teleop_body_position_extend(self):
        # Optional: BeyondMimic-style anchor-centered tracking for body positions.
        # Backward compatible: default False -> keep existing global tracking.
        use_anchor_centered = bool(getattr(self.config.rewards, "anchor_centered_body_position", False))
        dif_body_pos = self._get_anchor_centered_dif_global_body_pos() if use_anchor_centered else self.dif_global_body_pos

        upper_ids = self.upper_body_id
        lower_ids = self.lower_body_id
        if use_anchor_centered:
            anchor_idx = self._get_anchor_body_index()
            upper_ids = [i for i in upper_ids if i != anchor_idx]
            lower_ids = [i for i in lower_ids if i != anchor_idx]

        upper_body_diff = dif_body_pos[:, upper_ids, :] if len(upper_ids) > 0 else None
        lower_body_diff = dif_body_pos[:, lower_ids, :] if len(lower_ids) > 0 else None

        if upper_body_diff is None:
            diff_body_pos_dist_upper = torch.zeros(self.num_envs, device=self.device)
        else:
            diff_body_pos_dist_upper = (upper_body_diff**2).mean(dim=-1).mean(dim=-1)

        if lower_body_diff is None:
            diff_body_pos_dist_lower = torch.zeros(self.num_envs, device=self.device)
        else:
            diff_body_pos_dist_lower = (lower_body_diff**2).mean(dim=-1).mean(dim=-1)

        r_body_pos_upper = torch.exp(-diff_body_pos_dist_upper / self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos)
        r_body_pos_lower = torch.exp(-diff_body_pos_dist_lower / self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)
        r_body_pos = r_body_pos_lower * self.config.rewards.teleop_body_pos_lowerbody_weight + r_body_pos_upper * self.config.rewards.teleop_body_pos_upperbody_weight
    
        return r_body_pos

    def _reward_teleop_anchor_position(self):
        """
        Optional BeyondMimic-style anchor global position tracking reward.
        Backward compatible: only has effect if reward_scales.teleop_anchor_position != 0.
        """
        anchor_idx = self._get_anchor_body_index()
        anchor_diff = self.dif_global_body_pos[:, anchor_idx, :]
        diff_dist = (anchor_diff ** 2).mean(dim=-1)
        sigma = getattr(self.config.rewards.reward_tracking_sigma, "teleop_anchor_pos",
                        self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)
        return torch.exp(-diff_dist / sigma)

    def _reward_teleop_anchor_rotation(self):
        """
        Optional BeyondMimic-style anchor global orientation tracking reward.
        Backward compatible: only has effect if reward_scales.teleop_anchor_rotation != 0.
        """
        anchor_idx = self._get_anchor_body_index()
        anchor_rot_diff = self.dif_global_body_rot[:, anchor_idx, :]
        diff_dist = (anchor_rot_diff ** 2).mean(dim=-1)
        sigma = getattr(self.config.rewards.reward_tracking_sigma, "teleop_anchor_rot",
                        self.config.rewards.reward_tracking_sigma.teleop_body_rot)
        return torch.exp(-diff_dist / sigma)

    # ---------------- Anchor-centered tracking helpers ----------------
    def _get_anchor_body_name(self) -> str:
        name = getattr(self.config.rewards, "anchor_body_name", None)
        if name:
            return str(name)
        name = getattr(getattr(self.config.robot, "motion", None), "base_link", None)
        if name:
            return str(name)
        name = getattr(self.config.robot, "torso_name", None)
        if name:
            return str(name)
        return "torso_link"

    def _get_anchor_body_index(self) -> int:
        idx = getattr(self, "_anchor_body_idx_cached", None)
        if idx is not None:
            return int(idx)
        anchor_name = self._get_anchor_body_name()
        try:
            idx = self.simulator._body_list.index(anchor_name)
        except Exception:
            idx = self.simulator.find_rigid_body_indice(anchor_name)
        self._anchor_body_idx_cached = int(idx)
        return int(idx)

    def _get_anchor_centered_dif_global_body_pos(self) -> torch.Tensor:
        """
        Compute BeyondMimic-style anchor-centered desired body positions (world frame),
        then return (desired - current) for all bodies in `_rigid_body_pos_extend`.

        Reference: [BeyondMimic paper](https://arxiv.org/html/2508.08241v4).
        """
        cached = getattr(self, "_anchor_centered_desired_body_pos_cache", None)
        cached_step = getattr(self, "_anchor_centered_desired_body_pos_cache_step", None)
        step = int(self.common_step_counter) if hasattr(self, "common_step_counter") else (
            int(self.episode_length_buf[0].item()) if hasattr(self, "episode_length_buf") else None
        )
        if cached is not None and cached_step == step:
            desired = cached
        else:
            anchor_idx = self._get_anchor_body_index()
            ref_pos = self.ref_body_pos_extend
            ref_rot = self.ref_body_rot_extend
            curr_pos = self._rigid_body_pos_extend
            curr_rot = self._rigid_body_rot_extend

            pA_ref = ref_pos[:, anchor_idx, :]
            pA_cur = curr_pos[:, anchor_idx, :]
            qA_ref = ref_rot[:, anchor_idx, :]
            qA_cur = curr_rot[:, anchor_idx, :]

            q_rel = quat_mul(qA_cur, quat_conjugate(qA_ref, w_last=True), w_last=True)
            q_rel = q_rel / torch.norm(q_rel, dim=-1, keepdim=True).clamp(min=1e-9)
            rpy = get_euler_xyz_in_tensor(q_rel)
            yaw = rpy[:, 2]
            zeros = torch.zeros_like(yaw)
            q_yaw_delta = quat_from_euler_xyz(zeros, zeros, yaw)
            q_yaw_delta = q_yaw_delta / torch.norm(q_yaw_delta, dim=-1, keepdim=True).clamp(min=1e-9)

            p_delta = torch.stack([pA_cur[:, 0], pA_cur[:, 1], pA_ref[:, 2]], dim=-1)

            offset = ref_pos - pA_ref.unsqueeze(1)
            B = offset.shape[1]
            q_rep = q_yaw_delta.unsqueeze(1).expand(-1, B, -1).reshape(-1, 4)
            rot_offset = my_quat_rotate(q_rep, offset.reshape(-1, 3)).reshape(-1, B, 3)
            desired = p_delta.unsqueeze(1) + rot_offset

            desired[:, anchor_idx, :] = pA_cur

            self._anchor_centered_desired_body_pos_cache = desired
            self._anchor_centered_desired_body_pos_cache_step = step

        return desired - self._rigid_body_pos_extend

    def _get_anchor_centered_dif_global_body_rot(self) -> torch.Tensor:
        """
        Compute BeyondMimic-style anchor-centered desired body orientations (world frame),
        then return quaternion error (desired * inv(current)) for all bodies in `_rigid_body_rot_extend`.
        Reference: [BeyondMimic paper](https://arxiv.org/html/2508.08241v4).
        """
        cached = getattr(self, "_anchor_centered_desired_body_rot_cache", None)
        cached_step = getattr(self, "_anchor_centered_desired_body_rot_cache_step", None)
        step = int(self.common_step_counter) if hasattr(self, "common_step_counter") else (
            int(self.episode_length_buf[0].item()) if hasattr(self, "episode_length_buf") else None
        )
        if cached is not None and cached_step == step:
            desired_rot = cached
        else:
            anchor_idx = self._get_anchor_body_index()
            ref_rot = self.ref_body_rot_extend
            curr_rot = self._rigid_body_rot_extend

            qA_ref = ref_rot[:, anchor_idx, :]
            qA_cur = curr_rot[:, anchor_idx, :]

            q_rel = quat_mul(qA_cur, quat_conjugate(qA_ref, w_last=True), w_last=True)
            q_rel = q_rel / torch.norm(q_rel, dim=-1, keepdim=True).clamp(min=1e-9)
            rpy = get_euler_xyz_in_tensor(q_rel)
            yaw = rpy[:, 2]
            zeros = torch.zeros_like(yaw)
            q_yaw_delta = quat_from_euler_xyz(zeros, zeros, yaw)
            q_yaw_delta = q_yaw_delta / torch.norm(q_yaw_delta, dim=-1, keepdim=True).clamp(min=1e-9)

            B = ref_rot.shape[1]
            q_rep = q_yaw_delta.unsqueeze(1).expand(-1, B, -1).reshape(-1, 4)
            desired_rot = quat_mul(q_rep, ref_rot.reshape(-1, 4), w_last=True).reshape(-1, B, 4)

            desired_rot[:, anchor_idx, :] = curr_rot[:, anchor_idx, :]

            self._anchor_centered_desired_body_rot_cache = desired_rot
            self._anchor_centered_desired_body_rot_cache_step = step

        return quat_mul(desired_rot, quat_conjugate(self._rigid_body_rot_extend, w_last=True), w_last=True)
    
    def _reward_teleop_vr_3point(self):
        use_anchor_centered = bool(getattr(self.config.rewards, "anchor_centered_body_position", False))
        dif_body_pos = self._get_anchor_centered_dif_global_body_pos() if use_anchor_centered else self.dif_global_body_pos
        vr_3point_diff = dif_body_pos[:, self.motion_tracking_id, :]
        vr_3point_dist = (vr_3point_diff**2).mean(dim=-1).mean(dim=-1)
        r_vr_3point = torch.exp(-vr_3point_dist / self.config.rewards.reward_tracking_sigma.teleop_vr_3point_pos)
        return r_vr_3point

    def _reward_teleop_body_position_feet(self):

        use_anchor_centered = bool(getattr(self.config.rewards, "anchor_centered_body_position", False))
        dif_body_pos = self._get_anchor_centered_dif_global_body_pos() if use_anchor_centered else self.dif_global_body_pos
        feet_diff = dif_body_pos[:, self.feet_indices, :]
        feet_dist = (feet_diff**2).mean(dim=-1).mean(dim=-1)
        r_feet = torch.exp(-feet_dist / self.config.rewards.reward_tracking_sigma.teleop_feet_pos)
        return r_feet
    
    def _reward_teleop_body_rotation_extend(self):
        use_anchor_centered = bool(getattr(self.config.rewards, "anchor_centered_body_position", False))
        rotation_diff = self._get_anchor_centered_dif_global_body_rot() if use_anchor_centered else self.dif_global_body_rot

        if use_anchor_centered:
            anchor_idx = self._get_anchor_body_index()
            all_ids = list(range(rotation_diff.shape[1]))
            ids = [i for i in all_ids if i != anchor_idx]
            if len(ids) > 0:
                rotation_diff = rotation_diff[:, ids, :]

        diff_body_rot_dist = (rotation_diff**2).mean(dim=-1).mean(dim=-1)
        r_body_rot = torch.exp(-diff_body_rot_dist / self.config.rewards.reward_tracking_sigma.teleop_body_rot)
        return r_body_rot

    def _reward_teleop_body_velocity_extend(self):
        velocity_diff = self.dif_global_body_vel    
        diff_body_vel_dist = (velocity_diff**2).mean(dim=-1).mean(dim=-1)
        r_body_vel = torch.exp(-diff_body_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_body_vel)
        return r_body_vel
    
    def _reward_teleop_body_ang_velocity_extend(self):
        ang_velocity_diff = self.dif_global_body_ang_vel
        diff_body_ang_vel_dist = (ang_velocity_diff**2).mean(dim=-1).mean(dim=-1)
        r_body_ang_vel = torch.exp(-diff_body_ang_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_body_ang_vel)
        return r_body_ang_vel

    def _reward_teleop_joint_position(self):
        joint_pos_diff = self.dif_joint_angles
        diff_joint_pos_dist = (joint_pos_diff**2).mean(dim=-1)
        r_joint_pos = torch.exp(-diff_joint_pos_dist / self.config.rewards.reward_tracking_sigma.teleop_joint_pos)
        return r_joint_pos
    
    def _reward_teleop_joint_velocity(self):
        joint_vel_diff = self.dif_joint_velocities
        diff_joint_vel_dist = (joint_vel_diff**2).mean(dim=-1)
        r_joint_vel = torch.exp(-diff_joint_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_joint_vel)
        return r_joint_vel
    
    def setup_visualize_entities(self):
        if self.debug_viz and self.config.simulator.config.name == "genesis":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.simulator.add_visualize_entities(num_visualize_markers)
        elif self.debug_viz and self.config.simulator.config.name == "mujoco":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.simulator.add_visualize_entities(num_visualize_markers)
        else:
            pass


@torch.jit.script
def quat_to_euler(q: Tensor) -> Tensor:
    """
    Convert a normalized quaternion to Euler angles (roll, pitch, yaw).
    Args:
        q (Tensor): Quaternion tensor with shape (..., 4), assuming last dim is [x, y, z, w].
    Returns:
        Tensor: Euler-angle tensor with shape (..., 3), ordered as [roll, pitch, yaw] in radians.
    """
    # Extract quaternion components.
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Compute cross terms.
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    # Clamp numeric range to prevent asin input from exceeding [-1, 1].
    sinp_clamped = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp_clamped)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    # Stack Euler angles.
    euler = torch.stack((roll, pitch, yaw), dim=-1)  # shape: (..., 3)

    return euler