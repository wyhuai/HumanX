from isaacgym import gymtorch, gymapi, gymutil
import torch
import numpy as np
from pathlib import Path
import os
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.utils.torch_utils import quat_rotate_inverse as torch_utils_quat_rotate_inverse
from humanoidverse.utils.mimic_logging import (
    apply_termination_mask,
    summarize_contact_error_metrics,
)
from humanoidverse.utils.torch_utils import quat_apply 
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
    wrap_to_pi,
    quaternion_to_matrix,
    slerp,
)

# from isaacgym import gymtorch, gymapi, gymutil
from humanoidverse.envs.env_utils.visualization import Point

from humanoidverse.utils.motion_lib.skeleton import SkeletonTree

from humanoidverse.utils.motion_lib.motion_lib_robot_HOI import MotionLibRobotHOI
from humanoidverse.utils.hoi_runtime import (
    apply_xy_only_init_noise,
    build_hoi_body_contact_force,
    get_hoi_object_contact,
    support_top_to_world_center,
)
from humanoidverse.utils.motion_lib.motion_utils.support import parked_support_center
from humanoidverse.utils.motion_data_control import apply_motion_frame_to_state_tensors

from termcolor import colored
from loguru import logger

from scipy.spatial.transform import Rotation as sRot
import joblib
import copy
from typing import Optional

SMPL_REFERENCE_JOINT_NAME_TO_INDEX = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    "left_thumb": 22,
    "right_thumb": 23,
}

class LeggedRobotMotionTrackingHOI(LeggedRobotBase):
    def _get_actor_ids_by_env_ids(self, env_ids: torch.Tensor, actor_index: int) -> torch.Tensor:
        """Convert env_ids -> actor ids in flattened `all_root_states`.
        Important: IsaacGym root state tensor is indexed by *actor* (not asset), so we must use actors_per_env.
        """
        actors_per_env = int(self.simulator.all_root_states.shape[0] // self.num_envs)
        return (env_ids.to(torch.long) * actors_per_env + int(actor_index)).to(torch.long)

    def _park_support_asset_states(self, env_ids: torch.Tensor):
        if not hasattr(self.simulator, "support_root_state"):
            return
        parked_center = parked_support_center(self.env_origins[env_ids], dz=100.0)
        self.simulator.support_root_state[env_ids, :3] = parked_center
        self.simulator.support_root_state[env_ids, 3:7] = torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            device=self.device,
            dtype=self.simulator.support_root_state.dtype,
        ).unsqueeze(0).repeat(env_ids.shape[0], 1)
        self.simulator.support_root_state[env_ids, 7:13] = 0.0

    def _reset_support_asset_states(self, env_ids: torch.Tensor, motion_res_obj: dict):
        if not hasattr(self.simulator, "support_root_state"):
            return
        self._park_support_asset_states(env_ids)
        need_support = motion_res_obj.get("need_support", None)
        support_top_pos = motion_res_obj.get("support_top_pos", None)
        if need_support is None or support_top_pos is None:
            return
        if need_support.ndim > 1:
            need_support = need_support.squeeze(-1)
        support_env_ids = env_ids[need_support[env_ids].to(torch.bool)]
        if support_env_ids.numel() == 0:
            return
        support_center = support_top_to_world_center(
            support_top_pos[support_env_ids],
            self.env_origins[support_env_ids],
            thickness=float(getattr(self.simulator, "support_box_thickness", 0.02)),
        )
        self.simulator.support_root_state[support_env_ids, :3] = support_center
        self.simulator.support_root_state[support_env_ids, 3:7] = torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            device=self.device,
            dtype=self.simulator.support_root_state.dtype,
        ).unsqueeze(0).repeat(support_env_ids.shape[0], 1)
        self.simulator.support_root_state[support_env_ids, 7:13] = 0.0

    def __init__(self, config, device):
        self.init_done = False
        self.debug_viz = True #False
        
        # Early init for packet loss flag (needed before super().__init__ calls _setup_robot_body_indices)
        pl_cfg = getattr(config, "packet_loss", None)
        self._packet_loss_enabled = pl_cfg is not None and getattr(pl_cfg, "enable", False)
        
        super().__init__(config, device)
        self._visualize_motion_data_only = bool(
            getattr(self.config, "visualize_motion_data_only", False)
        )
        if self._visualize_motion_data_only:
            logger.info("HOI motion-data visualization mode enabled")
        # Optional: per-env freeze flag for the object (ball). Backward compatible default False.
        self._freeze_obj_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # Store per-env frozen root state (13-dim). Only rows where _freeze_obj_buf=True are meaningful.
        self._frozen_obj_root_state = torch.zeros((self.num_envs, 13), device=self.device, dtype=torch.float32)
        # Per-env reset-time state delay offsets (seconds). Used to intentionally misalign initial states.
        # Backward compatible: default zeros -> no effect unless enable_state_delay_init is True.
        self._state_delay_init_offset_s_robot = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._state_delay_init_offset_s_object = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._init_motion_lib()
        self._init_human_joint_debug_buffers()
        self._init_motion_extend()
        self._init_reference_command_config()
        self._init_tracking_config()

        self.init_done = True
        self.debug_viz = True #False
        self.obs_reduction_timer = 2000*24

        self._init_save_motion()
        self.setup_ind_for_amp()

        # Only compute OmniRetarget minimal proprioceptive obs when it is actually used by obs config.
        # This keeps backward compatible behavior for configs that do not include these keys, and avoids
        # unnecessary per-step alloc/churn which can accumulate CUDA reserved memory over long runs.
        omni_keys = {
            "ref_dof_pos",
            "ref_dof_vel",
            "pelvis_pos_err",
            "pelvis_rot_err",
            "pelvis_rot_err_rot6d",
        }
        used_keys = set()
        try:
            for group in getattr(self.config.obs, "obs_dict", {}).values():
                for k in list(group):
                    if isinstance(k, str):
                        used_keys.add(k[:-4] if k.endswith("_raw") else k)
            for aux in getattr(self.config.obs, "obs_auxiliary", {}).values():
                for k in getattr(aux, "keys", lambda: aux.keys())():
                    if isinstance(k, str):
                        used_keys.add(k)
        except Exception:
            used_keys = omni_keys
        self._enable_omniretarget_min_obs = len(omni_keys.intersection(used_keys)) > 0

        if self.config.use_teleop_control:
            self.teleop_marker_coords = torch.zeros(self.num_envs, 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import Float64MultiArray
            self.node = Node("motion_tracking")
            self.teleop_sub = self.node.create_subscription(Float64MultiArray, "vision_pro_data", self.teleop_callback, 1)

        #motion termination curriculum
        if self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum:
            self.terminate_when_motion_far_threshold = self.config.termination_curriculum.terminate_when_motion_far_initial_threshold
            logger.info(f"Terminate when motion far threshold: {self.terminate_when_motion_far_threshold}")

        else:
            self.terminate_when_motion_far_threshold = self.config.termination_scales.termination_motion_far_threshold
            logger.info(f"Terminate when motion far threshold: {self.terminate_when_motion_far_threshold}")

        #object interaction termination curriculum
        if self.config.termination.terminate_when_ball_status_far and self.config.termination_curriculum.terminate_when_ball_status_far_curriculum:
            self.terminate_when_ball_status_far_threshold = self.config.termination_curriculum.terminate_when_ball_status_far_initial_threshold
            logger.info(f"Terminate when ball status far threshold: {self.terminate_when_ball_status_far_threshold}")
        else:
            self.terminate_when_ball_status_far_threshold = self.config.termination_scales.termination_ball_status_far_threshold
            logger.info(f"Terminate when ball status far threshold: {self.terminate_when_ball_status_far_threshold}")

        # object rotation termination (optional; backward compatible)
        if getattr(self.config.termination, "terminate_when_ball_rotation_far", False):
            self.terminate_when_ball_rotation_far_threshold = self.config.termination_scales.termination_ball_rotation_far_threshold
            self.terminate_when_ball_rotation_far_consider_yaw = getattr(
                self.config.termination, "terminate_when_ball_rotation_far_consider_yaw", True
            )
            logger.info(f"Terminate when ball rotation far threshold (rad): {self.terminate_when_ball_rotation_far_threshold}")
            logger.info(f"Terminate when ball rotation far consider yaw: {self.terminate_when_ball_rotation_far_consider_yaw}")
            
        #ig termination
        self.terminate_when_ig_status_far_threshold = self.config.termination_scales.termination_ig_status_far_threshold
        logger.info(f"Terminate when ig status far threshold: {self.terminate_when_ig_status_far_threshold}")
        
        #contact conditioned ig termination
        # Backward compatible: if contact-conditioned threshold is not provided, fall back to the generic ig threshold.
        self.contact_conditioned_ig_terminate_threshold = getattr(
            self.config.termination_scales,
            "contact_conditioned_ig_terminate_threshold",
            self.terminate_when_ig_status_far_threshold,
        )
        logger.info(
            f"Contact Conditioned IG termination threshold: {self.contact_conditioned_ig_terminate_threshold}"
        )
        
        #terminate when ball z far
        if self.config.termination.terminate_when_ball_z_far:
            self.terminate_when_ball_z_far_threshold = self.config.termination_scales.termination_ball_z_far_threshold
            logger.info(f"Terminate when ball z far threshold: {self.terminate_when_ball_z_far_threshold}")

        # BeyondMimic-style termination (optional; keep disabled for OmniRetarget termination alignment)
        self._bm_anchor_body_name = str(getattr(self.config.termination_scales, "bm_anchor_body_name", "torso_link"))
        self._bm_end_effector_body_names = list(getattr(self.config.termination_scales, "bm_end_effector_body_names", []))
        # BeyondMimic termination thresholds (from paper text):
        # - position (z) error threshold: |e_{p,z,b}| > 0.25 m (anchor OR any end-effector)
        # - anchor orientation error threshold: ||e_R,anchor|| > 0.8 rad, where ||e_R|| is rotation angle
        # Keep backward compatible with older config keys:
        # - bm_pos_threshold (falls back to bm_pos_z_threshold)
        # - bm_rp_threshold (falls back to bm_rot_threshold)
        self._bm_pos_z_threshold = float(
            getattr(self.config.termination_scales, "bm_pos_z_threshold",
                    getattr(self.config.termination_scales, "bm_pos_threshold", 0.25))
        )
        self._bm_rot_threshold = float(
            getattr(self.config.termination_scales, "bm_rot_threshold",
                    getattr(self.config.termination_scales, "bm_rp_threshold", 0.8))
        )

        # Resolve body indices (best-effort; ignore missing names for backward compatibility)
        try:
            self._bm_anchor_body_idx = int(self.simulator._body_list.index(self._bm_anchor_body_name))
        except Exception:
            self._bm_anchor_body_idx = int(getattr(self, "torso_index", 0))

        bm_ee_indices = []
        for n in self._bm_end_effector_body_names:
            try:
                bm_ee_indices.append(int(self.simulator._body_list.index(str(n))))
            except Exception:
                pass
        self._bm_end_effector_body_indices = bm_ee_indices

    def _sample_visualization_motion_state(self):
        motion_time = self.episode_length_buf * self.dt + self.motion_start_times
        return self._motion_lib.get_motion_state(
            self.motion_ids,
            motion_time,
            motion_time,
            offset=self.env_origins,
        )

    def _apply_motion_frame_to_sim(self, env_ids=None, motion_res=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif env_ids.numel() == 0:
            return

        if motion_res is None:
            motion_res = self._sample_visualization_motion_state()

        apply_motion_frame_to_state_tensors(
            env_ids=env_ids,
            simulator_name=self.config.simulator.config.name,
            robot_root_states=self.simulator.robot_root_states,
            object_root_state=self.simulator.object_root_state,
            dof_pos=self.simulator.dof_pos,
            dof_vel=self.simulator.dof_vel,
            motion_res=motion_res,
        )

        robot_actor_ids = self._get_actor_ids_by_env_ids(env_ids, self.simulator.robot_asset_idx)
        ball_actor_ids = self._get_actor_ids_by_env_ids(env_ids, self.simulator.ball_asset_idx)
        actor_ids = torch.cat([robot_actor_ids, ball_actor_ids], dim=0)
        self.simulator.set_actor_root_state_tensor(actor_ids, self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(robot_actor_ids, self.simulator.dof_state)
        self._refresh_sim_tensors()

    def _apply_force_in_physics_step(self):
        if self._visualize_motion_data_only:
            self.torques.zero_()
            self.simulator.apply_torques_at_dof(self.torques)
            return
        super()._apply_force_in_physics_step()
            
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
        rsi_enabled = getattr(self.config, "enable_rsi", False)
        adaptive_sampling_enabled = getattr(self.config, "enable_adaptive_sampling", False)
        
        # If RSI is disabled, disable adaptive sampling automatically.
        if not rsi_enabled and adaptive_sampling_enabled:
            logger.warning("RSI is disabled but adaptive sampling is enabled. Disabling adaptive sampling.")
            adaptive_sampling_enabled = False
            
        setattr(self.config.robot.motion, "enable_adaptive_sampling", adaptive_sampling_enabled)
        
        # Pass adaptive sampling config from env.config to motion_lib
        self._motion_lib = MotionLibRobotHOI(
            self.config.robot.motion, 
            num_envs=self.num_envs, 
            device=self.device,
            adaptive_sampling_cfg=self.config if adaptive_sampling_enabled else None
        )
        if self.is_evaluating:
            self._motion_lib.load_motions(random_sample=False)
        else:
            self._motion_lib.load_motions(random_sample=True)

        # Optional: separate motion source for AMP expert observations.
        # Backward-compatible: if not set, AMP expert obs uses the same reference motion as tracking.
        self._amp_motion_lib = None
        amp_motion_file = getattr(self.config.robot.motion, "amp_motion_file", None)
        if amp_motion_file is not None:
            amp_motion_cfg = copy.deepcopy(self.config.robot.motion)
            amp_motion_cfg.motion_file = amp_motion_file
            amp_motion_cfg.step_dt = self.dt
            self._amp_motion_lib = MotionLibRobotHOI(
                amp_motion_cfg,
                num_envs=self.num_envs,
                device=self.device,
                adaptive_sampling_cfg=None,  # keep AMP expert sampling simple/independent
            )
            if self.is_evaluating:
                self._amp_motion_lib.load_motions(random_sample=False)
            else:
                self._amp_motion_lib.load_motions(random_sample=True)
            logger.info(f"AMP expert motions loaded from amp_motion_file={amp_motion_file}")
            
        # Print initial motion IDs for all environments
        actual_motion_ids = self._motion_lib._curr_motion_ids.cpu().numpy()
        motion_keys = self._motion_lib.curr_motion_keys
        max_print = 10
        if self.num_envs <= max_print:
            logger.info(f"Initialized {self.num_envs} envs with motion_ids={actual_motion_ids}, motion_keys={motion_keys.tolist()}")
        else:
            logger.info(f"Initialized {self.num_envs} envs: first {max_print} motion_ids={actual_motion_ids[:max_print]}..., "
                       f"first {max_print} motion_keys={motion_keys[:max_print].tolist()}...")
            
        # res = self._motion_lib.get_motion_state(self.motion_ids, self.motion_times, self.motion_start_times, offset=self.env_origins)
        env_ids_all = torch.arange(self.num_envs, device=self.device)
        res = self._resample_motion_times(env_ids_all)
        self.motion_dt = self._motion_lib._motion_dt
        self.motion_start_idx = 0
        self.num_motions = self._motion_lib._num_unique_motions
        self.motion_stand_frame = self._motion_lib._motion_stand_frame
        self.stand_ref_episode_length = torch.round((self.motion_stand_frame) * self.motion_dt / self.dt).int()

    def _init_tracking_config(self):
        if "motion_tracking_link" in self.config.robot.motion:
            self.motion_tracking_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.motion_tracking_link]
        if "extended_hand_branch_link" in self.config.robot.motion:
            self.extended_hand_branch_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.extended_hand_branch_link]
        if "lower_body_link" in self.config.robot.motion:
            self.lower_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.lower_body_link]
        if "upper_body_link" in self.config.robot.motion:
            self.upper_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.upper_body_link]
        if self.config.resample_motion_when_training:
            self.resample_time_interval = np.ceil(self.config.resample_time_interval_s / self.dt)
            
        #contact conditioned ig terminate
        if getattr(self.config.termination, 'contact_conditioned_ig_termination', False):
            self.selected_conditioned_contact_bodies_id = [self.simulator._body_list.index(link) for link in self.config.termination_misc.selected_contact_conditioned_ig_termination_bodies]
        
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

        # AMP expert reference buffers (local frame). Always initialize to avoid missing attributes
        # before the first _pre_compute_observations_callback().
        total_bodies = int(self.num_bodies + getattr(self, "num_extend_bodies", 0))
        self._obs_local_amp_expert_rigid_body_pos = torch.zeros(
            self.num_envs, total_bodies * 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self._obs_local_amp_expert_rigid_body_vel = torch.zeros(
            self.num_envs, total_bodies * 3, dtype=torch.float, device=self.device, requires_grad=False
        )

        # Setup hand link indices for packet loss (must be after extend bodies are added to _body_list)
        if self._packet_loss_enabled:
            try:
                self.left_hand_link_idx = self.simulator._body_list.index("left_hand_link")
                self.right_hand_link_idx = self.simulator._body_list.index("right_hand_link")
            except ValueError:
                # Fallback to wrist yaw links if extended hand links not found
                self.left_hand_link_idx = self.simulator._body_list.index("left_wrist_yaw_link")
                self.right_hand_link_idx = self.simulator._body_list.index("right_wrist_yaw_link")

    def _init_reference_command_config(self):
        """Resolve task-level future reference command config from obs yaml."""
        ref_cfg = getattr(getattr(self.config, "obs", None), "reference_command", None)
        self._reference_command_enabled = ref_cfg is not None
        if not self._reference_command_enabled:
            self._reference_command_body_indices = torch.zeros(0, dtype=torch.long, device=self.device)
            self._reference_command_human_joint_indices = torch.zeros(0, dtype=torch.long, device=self.device)
            self._reference_command_future_steps = torch.zeros(0, dtype=torch.long, device=self.device)
            self._reference_command_include_ball_contact = False
            self._reference_command_include_human_global_orient_quat = False
            self._reference_command_include_root_rot = False
            self._reference_command_use_human_smpl = False
            self._reference_command_body_names = ()
            self._reference_command_root_body_index = 0
            self._obs_reference_command = torch.zeros((self.num_envs, 0), dtype=torch.float32, device=self.device)
            return

        body_names = list(getattr(ref_cfg, "body_names", []))
        human_smpl_joint_names = list(getattr(ref_cfg, "human_smpl_joint_names", []))
        future_steps = list(getattr(ref_cfg, "future_steps", []))
        include_ball_contact = bool(getattr(ref_cfg, "include_ball_contact", True))
        include_human_global_orient_quat = bool(
            getattr(ref_cfg, "include_human_global_orient_quat", False)
        )
        include_root_rot = bool(getattr(ref_cfg, "include_root_rot", False))
        if len(body_names) > 0 and len(human_smpl_joint_names) > 0:
            raise ValueError(
                "obs.reference_command.body_names and human_smpl_joint_names are mutually exclusive"
            )
        if len(body_names) == 0 and len(human_smpl_joint_names) == 0:
            raise ValueError(
                "obs.reference_command must specify either body_names or human_smpl_joint_names"
            )
        if len(future_steps) == 0:
            raise ValueError("obs.reference_command.future_steps must not be empty")

        body_indices = []
        human_joint_indices = []
        root_body_index = 0
        use_human_smpl = len(human_smpl_joint_names) > 0
        if use_human_smpl:
            for joint_name in human_smpl_joint_names:
                if joint_name not in SMPL_REFERENCE_JOINT_NAME_TO_INDEX:
                    raise ValueError(
                        f"reference_command human_smpl_joint '{joint_name}' is unknown. "
                        f"Available: {sorted(SMPL_REFERENCE_JOINT_NAME_TO_INDEX.keys())}"
                    )
                human_joint_indices.append(SMPL_REFERENCE_JOINT_NAME_TO_INDEX[joint_name])
        else:
            for body_name in body_names:
                if body_name not in self.simulator._body_list:
                    raise ValueError(
                        f"reference_command body '{body_name}' not found in simulator._body_list. "
                        f"Available examples: {self.simulator._body_list[:10]}"
                    )
                body_indices.append(self.simulator._body_list.index(body_name))
            if include_root_rot:
                root_body_name = getattr(ref_cfg, "root_rot_body_name", body_names[0])
                if root_body_name not in self.simulator._body_list:
                    raise ValueError(
                        f"reference_command root_rot_body_name '{root_body_name}' not found in simulator._body_list. "
                        f"Available examples: {self.simulator._body_list[:10]}"
                    )
                root_body_index = self.simulator._body_list.index(root_body_name)

        self._reference_command_body_indices = torch.tensor(
            body_indices, dtype=torch.long, device=self.device, requires_grad=False
        )
        self._reference_command_human_joint_indices = torch.tensor(
            human_joint_indices, dtype=torch.long, device=self.device, requires_grad=False
        )
        self._reference_command_future_steps = torch.tensor(
            future_steps, dtype=torch.long, device=self.device, requires_grad=False
        )
        self._reference_command_include_ball_contact = include_ball_contact
        self._reference_command_include_human_global_orient_quat = include_human_global_orient_quat
        self._reference_command_include_root_rot = include_root_rot
        self._reference_command_use_human_smpl = use_human_smpl
        self._reference_command_body_names = tuple(body_names)
        self._reference_command_root_body_index = root_body_index

        cmd_dim = 3 * (
            len(human_joint_indices) if use_human_smpl else len(body_indices)
        ) * len(future_steps)
        if use_human_smpl and include_human_global_orient_quat:
            cmd_dim += 6 * len(future_steps)
        if include_ball_contact:
            cmd_dim += len(future_steps)
        if not use_human_smpl and include_root_rot:
            cmd_dim += 6 * len(future_steps)
        self._obs_reference_command = torch.zeros(
            (self.num_envs, cmd_dim), dtype=torch.float32, device=self.device, requires_grad=False
        )

    def _build_reference_command(self, now_time: torch.Tensor, heading_inv_rot: torch.Tensor):
        """Build sparse future body reference + future ball-contact command on GPU."""
        if not getattr(self, "_reference_command_enabled", False):
            return

        motion_ids = self.motion_ids
        future_steps = self._reference_command_future_steps
        num_future = future_steps.shape[0]
        body_indices = self._reference_command_body_indices
        human_joint_indices = self._reference_command_human_joint_indices
        num_bodies = body_indices.shape[0]

        future_times = now_time.unsqueeze(-1) + future_steps.to(now_time.dtype).unsqueeze(0) * self.dt
        flat_motion_ids = motion_ids.unsqueeze(-1).expand(-1, num_future).reshape(-1)
        flat_future_times = future_times.reshape(-1)

        motion_len = self._motion_lib._motion_lengths[flat_motion_ids]
        num_frames = self._motion_lib._motion_num_frames[flat_motion_ids]
        motion_dt = self._motion_lib._motion_dt[flat_motion_ids]
        frame_idx0, frame_idx1, blend = self._motion_lib._calc_frame_blend(
            flat_future_times, motion_len, num_frames, motion_dt
        )
        global_idx0 = frame_idx0 + self._motion_lib.length_starts[flat_motion_ids]
        global_idx1 = frame_idx1 + self._motion_lib.length_starts[flat_motion_ids]

        parts = []
        if self._reference_command_use_human_smpl:
            blend_local = blend.view(-1, 1, 1)
            human_smpl_reference = self._motion_lib.human_smpl_joints_local_centered
            human_joint_pos0 = human_smpl_reference[global_idx0][:, human_joint_indices, :]
            human_joint_pos1 = human_smpl_reference[global_idx1][:, human_joint_indices, :]
            future_human_joint_pos = (
                (1.0 - blend_local) * human_joint_pos0 + blend_local * human_joint_pos1
            )
            parts.append(future_human_joint_pos.view(self.num_envs, -1))

            if self._reference_command_include_human_global_orient_quat:
                blend_rot = blend.view(-1, 1)
                human_root_rot0 = self._motion_lib.human_global_orient_quat[global_idx0]
                human_root_rot1 = self._motion_lib.human_global_orient_quat[global_idx1]
                future_human_root_rot = slerp(
                    human_root_rot0, human_root_rot1, blend_rot
                ).view(self.num_envs, num_future, 4)
                parts.append(
                    self._motion_lib._smpl_anchor_orientation_local_rot6d(
                        self.base_quat,
                        self._reference_command_init_base_quat_robot,
                        getattr(
                            self,
                            "_reference_command_init_human_ref_quat",
                            self._reference_command_init_ref_quat,
                        ),
                        future_human_root_rot,
                    )
                )
        else:
            ref_body_source = self._motion_lib.gts_t if hasattr(self._motion_lib, "gts_t") else self._motion_lib.gts
            body_pos0 = ref_body_source[global_idx0][:, body_indices, :]
            body_pos1 = ref_body_source[global_idx1][:, body_indices, :]
            blend_world = blend.view(-1, 1, 1)
            future_body_pos = (1.0 - blend_world) * body_pos0 + blend_world * body_pos1
            future_body_pos = future_body_pos.view(self.num_envs, num_future, num_bodies, 3)
            future_body_pos = future_body_pos + self.env_origins[:, None, None, :]
            future_body_pos_local = self._world_pos_to_heading_local(
                future_body_pos, heading_inv_rot=heading_inv_rot
            ).view(self.num_envs, -1)
            parts.append(future_body_pos_local)

        if self._reference_command_include_ball_contact:
            future_ball_contact = self._motion_lib.obj_contact[global_idx0]
            if future_ball_contact.ndim > 1:
                future_ball_contact = future_ball_contact.squeeze(-1)
            future_ball_contact = future_ball_contact.to(dtype=torch.float32).view(self.num_envs, num_future)
            parts.append(future_ball_contact)

        if (
            not self._reference_command_use_human_smpl
            and getattr(self, "_reference_command_include_root_rot", False)
        ):
            ref_rot_source = self._motion_lib.grs_t if hasattr(self._motion_lib, "grs_t") else self._motion_lib.grs
            root_body_index = int(getattr(self, "_reference_command_root_body_index", 0))
            root_rot0 = ref_rot_source[global_idx0][:, root_body_index, :]
            root_rot1 = ref_rot_source[global_idx1][:, root_body_index, :]
            future_root_rot = slerp(root_rot0, root_rot1, blend.view(-1, 1))
            heading_inv_rot_flat = (
                heading_inv_rot.unsqueeze(1)
                .expand(-1, num_future, -1)
                .reshape(-1, 4)
            )
            future_root_rot_local = quat_mul(
                heading_inv_rot_flat,
                future_root_rot,
                w_last=True,
            ).view(self.num_envs, num_future, 4)
            future_root_rot6d = quaternion_to_matrix(
                xyzw_to_wxyz(future_root_rot_local)
            )[..., :2].reshape(self.num_envs, -1)
            parts.append(future_root_rot6d)

        self._obs_reference_command[:] = torch.cat(parts, dim=-1)

    def _world_pos_to_heading_local(
        self,
        world_pos: torch.Tensor,
        heading_inv_rot: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert world positions to the current robot heading-local frame.

        The first dimension must be num_envs. Any extra dimensions are preserved.
        """
        if heading_inv_rot is None:
            heading_inv_rot = calc_heading_quat_inv(self.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        root_pos = self.simulator.robot_root_states[:, :3]
        extra_shape = world_pos.shape[1:-1]
        flat_world = world_pos.reshape(self.num_envs, -1, 3)
        rel_pos = flat_world - root_pos.unsqueeze(1)
        heading_expand = heading_inv_rot.unsqueeze(1).expand(-1, rel_pos.shape[1], -1)
        local_flat = my_quat_rotate(heading_expand.reshape(-1, 4), rel_pos.reshape(-1, 3))
        return local_flat.view(self.num_envs, *extra_shape, 3)

    def _heading_local_pos_to_world_for_env(self, local_pos: torch.Tensor, env_id: int) -> torch.Tensor:
        """Convert heading-local positions for one env back to world frame."""
        robot_quat = self.simulator.robot_root_states[env_id:env_id + 1, 3:7].clone()
        heading_inv_rot = calc_heading_quat_inv(robot_quat, w_last=True)
        heading_rot = quat_conjugate(heading_inv_rot, w_last=True)
        extra_shape = local_pos.shape[:-1]
        flat_local = local_pos.reshape(-1, 3)
        heading_expand = heading_rot.expand(flat_local.shape[0], -1)
        world_flat = my_quat_rotate(heading_expand, flat_local) + self.simulator.robot_root_states[env_id, :3]
        return world_flat.view(*extra_shape, 3)

    def _apply_init_obj_pos_offset(self, obj_pos: torch.Tensor) -> torch.Tensor:
        obj_pos_offset = getattr(self.config, "init_obj_pos_offset", None)
        if obj_pos_offset is None:
            return obj_pos
        obj_pos_offset = torch.as_tensor(list(obj_pos_offset), device=self.device, dtype=obj_pos.dtype)
        if obj_pos_offset.numel() != 3:
            raise ValueError(f"init_obj_pos_offset must have 3 values, got {obj_pos_offset.numel()}")
        return obj_pos + obj_pos_offset.view(1, 3)

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
        # Optional: independent timeline for AMP expert sampling when robot.motion.amp_motion_file is set.
        # Backward compatible: if not used, it stays zeros and AMP falls back to tracking reference.
        self.amp_motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        # Optional per-motion scalar broadcasted as per-frame observation
        self._obs_apex_height = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        # For debug visualization: world-space basket position from motion (num_envs, 3)
        self._debug_basket_pos_world = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device, requires_grad=False)
        # Flag to indicate if basket_pos is valid (non-zero) for each env
        self._has_valid_basket_pos = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self._debug_human_joints_world = torch.zeros(self.num_envs, 0, 3, dtype=torch.float32, device=self.device, requires_grad=False)
        self._has_valid_human_joints = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self._reference_command_init_base_quat_robot = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device, requires_grad=False
        )
        self._reference_command_init_base_quat_robot[:, 3] = 1.0
        self._reference_command_init_ref_quat = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device, requires_grad=False
        )
        self._reference_command_init_ref_quat[:, 3] = 1.0
        self._reference_command_init_human_ref_quat = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device, requires_grad=False
        )
        self._reference_command_init_human_ref_quat[:, 3] = 1.0

        # RSI adaptive sampling buffers
        if getattr(self.config, "enable_rsi", False):
            self._rsi_episode_return = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
            self._rsi_episode_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        
        if "multiplicative_reward" in self.config.rewards.keys():
            self.multiplicative_rewards = self.config.rewards.multiplicative_reward
        else:
            self.multiplicative_rewards = []

        if "object_related_rewards" in self.config.rewards.keys():
            self.object_related_rewards = self.config.rewards.object_related_rewards
        else:
            self.object_related_rewards = []

        # AMP expert buffers (filled in _pre_compute_observations_callback)
        self.amp_expert_dof_pos = torch.zeros_like(self.simulator.dof_pos, device=self.device)
        self.amp_expert_dof_vel = torch.zeros_like(self.simulator.dof_vel, device=self.device)

        # OmniRetarget-style minimal proprioceptive obs buffers (filled in _pre_compute_observations_callback)
        # Keep backward compatible: only used when corresponding obs keys appear in config.
        self._obs_ref_dof_pos = torch.zeros_like(self.simulator.dof_pos, device=self.device)
        self._obs_ref_dof_vel = torch.zeros_like(self.simulator.dof_vel, device=self.device)
        self._obs_pelvis_pos_err = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device, requires_grad=False)
        self._obs_pelvis_rot_err = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device, requires_grad=False)
        # Rot6D orientation error (paper uses Rot6D)
        self._obs_pelvis_rot_err_rot6d = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device, requires_grad=False)

        # Per-episode contact error tracking (cg_reward != 1) within the cg time window.
        self._contact_error_count = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self._forbidden_contact_error_count = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self._object_contact_mismatch_error_count = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self._missed_object_contact_error_count = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self._unexpected_object_contact_error_count = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self._contact_window_count = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)

        #setup action scale
        action_scales = []
        for jn in self.config.robot.dof_names:
            j_id = self.config.robot.dof_names.index(jn)
            e = self.config.robot.dof_effort_limit_list[j_id]
            s = self.p_gains[j_id].detach().cpu().item()
            action_scales.append( self.config.robot.control.action_scale * e / s)  
        action_scales = torch.tensor(action_scales, dtype=torch.float32, requires_grad=False, device=self.device)
        self.action_scales = action_scales.unsqueeze(0)

        # packet loss simulation buffers
        self._init_packet_loss_buffers()

    def _init_human_joint_debug_buffers(self):
        num_human_joints = int(getattr(self._motion_lib, "human_joints_num", 0))
        self._debug_human_joints_world = torch.zeros(
            self.num_envs,
            num_human_joints,
            3,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self._has_valid_human_joints = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )

    def _capture_reference_command_heading_state(
        self,
        env_ids,
        init_base_quat_robot,
        init_ref_quat,
        init_human_ref_quat=None,
    ):
        self._reference_command_init_base_quat_robot[env_ids] = init_base_quat_robot.to(
            self._reference_command_init_base_quat_robot.dtype
        )
        self._reference_command_init_ref_quat[env_ids] = init_ref_quat.to(
            self._reference_command_init_ref_quat.dtype
        )
        if init_human_ref_quat is None:
            init_human_ref_quat = init_ref_quat
        self._reference_command_init_human_ref_quat[env_ids] = init_human_ref_quat.to(
            self._reference_command_init_human_ref_quat.dtype
        )

    def _sample_human_root_quat_at_times(
        self,
        motion_ids: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        flat_motion_ids = motion_ids.reshape(-1)
        flat_times = times.reshape(-1)
        motion_len = self._motion_lib._motion_lengths[flat_motion_ids]
        num_frames = self._motion_lib._motion_num_frames[flat_motion_ids]
        motion_dt = self._motion_lib._motion_dt[flat_motion_ids]
        frame_idx0, frame_idx1, blend = self._motion_lib._calc_frame_blend(
            flat_times,
            motion_len,
            num_frames,
            motion_dt,
        )
        global_idx0 = frame_idx0 + self._motion_lib.length_starts[flat_motion_ids]
        global_idx1 = frame_idx1 + self._motion_lib.length_starts[flat_motion_ids]
        return slerp(
            self._motion_lib.human_global_orient_quat[global_idx0],
            self._motion_lib.human_global_orient_quat[global_idx1],
            blend.view(-1, 1),
        )

    @staticmethod
    def _get_reference_command_alignment_base_quat(clean_root_rot, noisy_root_rot=None):
        """Choose the clean reference root for source-frame alignment.

        The source-alignment term should correct cross-dataset heading differences only.
        It must not absorb training-time initialization noise, otherwise part of the
        intended initial tracking error is canceled out.
        """
        return clean_root_rot

    def _init_packet_loss_buffers(self):
        """Initialize buffers for object pose packet loss simulation."""
        pl_cfg = getattr(self.config, "packet_loss", None)
        self._packet_loss_enabled = pl_cfg is not None and getattr(pl_cfg, "enable", False)
        if self._packet_loss_enabled:
            self.packet_loss_timer = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device, requires_grad=False)
            self.packet_loss_dist_threshold = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
            # Track the active rule/mode when the current loss event started.
            # This supports "once you cross the range, do not continue previous packet loss state".
            # 0: near  (dist_to_hand_center < per-env threshold)
            # 1: far   (dist_to_hand_center >= per-env threshold)
            # 2: very_far (dist_to_hand_center > dist_threshold_very_far) [optional]
            # 3: ground_abs_dist (abs(obj_global_z) > ground_abs_dist_threshold) [optional, highest priority]
            self.packet_loss_mode_active = torch.full(
                (self.num_envs,), -1, dtype=torch.int8, device=self.device, requires_grad=False
            )

    def _init_domain_rand_buffers(self):
        super()._init_domain_rand_buffers()
        self.ref_episodic_offset = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

    def _reset_tasks_callback(self, env_ids):
        if len(env_ids) == 0:
            return
        super()._reset_tasks_callback(env_ids)
        if getattr(self.config, "enable_rsi", False):
            self._rsi_pre_reset(env_ids)

        self._resample_motion_times(env_ids)
            
        if self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum:
            self._update_terminate_when_motion_far_curriculum()
        if self.config.termination.terminate_when_ball_status_far and self.config.termination_curriculum.terminate_when_ball_status_far_curriculum:
            self._update_terminate_when_ball_status_far_curriculum()

        # reset packet loss state for reset envs
        if self._packet_loss_enabled:
            pl_cfg = self.config.packet_loss
            th_range = pl_cfg.dist_threshold_range
            self.packet_loss_dist_threshold[env_ids] = torch.rand(len(env_ids), device=self.device) * (th_range[1] - th_range[0]) + th_range[0]
            self.packet_loss_timer[env_ids] = 0
            self.packet_loss_mode_active[env_ids] = -1

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        # Let base class clear episode_length_buf, history, etc. first.
        super()._reset_buffers_callback(env_ids, target_buf)

        # Reset offsets for these envs (avoid leaking previous episode's offsets).
        self._state_delay_init_offset_s_robot[env_ids] = 0.0
        self._state_delay_init_offset_s_object[env_ids] = 0.0

        # If restoring from a snapshot buffer, keep offsets at 0 for determinism unless user explicitly stores them.
        if target_buf is not None:
            return

        if not getattr(self.config, "enable_state_delay_init", False):
            return

        # Which entities should be delayed at reset time: "both" (default), "robot", or "object".
        target = str(getattr(self.config, "state_delay_init_target", "both")).lower()
        if target not in ("both", "robot", "object"):
            target = "both"

        apply_mask = torch.rand((env_ids.shape[0],), device=self.device) < float(self.config.state_delay_init_prob)
        if not torch.any(apply_mask):
            return

        low_f = int(self.config.state_delay_init_interval_f[0])
        high_f = int(self.config.state_delay_init_interval_f[1])
        # randint high is exclusive; config interval is in frames and is treated as inclusive.
        delay_frames = torch.randint(low=low_f, high=high_f + 1, size=(int(apply_mask.sum()),), device=self.device)
        delay_s = delay_frames.to(torch.float32) * float(self.dt)

        sel_env_ids = env_ids[apply_mask]
        if target in ("both", "robot"):
            self._state_delay_init_offset_s_robot[sel_env_ids] = delay_s
        if target in ("both", "object"):
            self._state_delay_init_offset_s_object[sel_env_ids] = delay_s

    def _rsi_pre_reset(self, env_ids):
        if len(env_ids) == 0:
            return
        episode_steps = self._rsi_episode_step[env_ids]
        valid_mask = episode_steps > 0 #no valid episodes (e.g., initial reset before training starts)
        
        # Compute average reward for completed episodes (if any)
        if valid_mask.any():
            avg_reward = torch.zeros_like(episode_steps, dtype=torch.float32)
            episode_steps_float = episode_steps[valid_mask].to(torch.float32)
            episode_steps_float = torch.clamp(episode_steps_float, min=1.0)
            avg_reward[valid_mask] = self._rsi_episode_return[env_ids][valid_mask] / episode_steps_float
            
            # Update frame rewards in motion library using current episode's motion info
            motion_ids = self.motion_ids[env_ids][valid_mask]
            start_times = self.motion_start_times[env_ids][valid_mask]
            self._motion_lib.update_frame_reward(motion_ids, start_times, avg_reward[valid_mask])
        
        # Always clear RSI buffers for the resetting envs
        self._rsi_episode_step[env_ids] = 0
        self._rsi_episode_return[env_ids] = 0.0

    
    def _update_terminate_when_motion_far_curriculum(self):
        
        assert self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum
        
        if self.average_episode_length < self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_down_threshold:
            self.terminate_when_motion_far_threshold *= (1 + self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree)
            
        elif self.average_episode_length > self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_up_threshold:
            self.terminate_when_motion_far_threshold *= (1 - self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        self.terminate_when_motion_far_threshold = np.clip(self.terminate_when_motion_far_threshold, 
                                                         self.config.termination_curriculum.terminate_when_motion_far_threshold_min, 
                                                         self.config.termination_curriculum.terminate_when_motion_far_threshold_max)
        
    def _update_terminate_when_ball_status_far_curriculum(self):
        
        assert self.config.termination.terminate_when_ball_status_far and self.config.termination_curriculum.terminate_when_ball_status_far_curriculum
        
        if self.average_episode_length < self.config.termination_curriculum.terminate_when_ball_status_far_curriculum_level_down_threshold:
            self.terminate_when_ball_status_far_threshold *= (1 + self.config.termination_curriculum.terminate_when_ball_status_far_curriculum_degree)
            
        elif self.average_episode_length > self.config.termination_curriculum.terminate_when_ball_status_far_curriculum_level_up_threshold:
            self.terminate_when_ball_status_far_threshold *= (1 - self.config.termination_curriculum.terminate_when_ball_status_far_curriculum_degree)
            
        self.terminate_when_ball_status_far_threshold = np.clip(self.terminate_when_ball_status_far_threshold, 
                                                         self.config.termination_curriculum.terminate_when_ball_status_far_threshold_min, 
                                                         self.config.termination_curriculum.terminate_when_ball_status_far_threshold_max)
    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        if self.config.resample_motion_when_training:
            if self.common_step_counter % self.resample_time_interval == 0:
                logger.info(f"Resampling motion at step {self.common_step_counter}")
                self.resample_motion()

    def set_is_evaluating(self):
        super().set_is_evaluating()
        # track per-env ball apex during evaluation
        self._ball_max_z = torch.full((self.num_envs,), -1e9, device=self.device)

    def _check_termination(self):
        if self._visualize_motion_data_only:
            self.reset_buf[:] = 0
            self.time_out_buf[:] = 0
            self._update_timeout_buf()
            return

        super()._check_termination()
        #motion
        if self.config.termination.terminate_when_motion_far:
            reset_buf_motion_far = torch.any(torch.norm(self.dif_global_body_pos, dim=-1) > self.terminate_when_motion_far_threshold, dim=-1)
            self.reset_buf, self.time_out_buf, _ = apply_termination_mask(
                self.log_dict, self.reset_buf, self.time_out_buf, "motion_far", reset_buf_motion_far
            )
            # log current motion far threshold
            if self.config.termination_curriculum.terminate_when_motion_far_curriculum:
                self.log_dict["terminate_when_motion_far_threshold"] = torch.tensor(self.terminate_when_motion_far_threshold, dtype=torch.float)

        # BeyondMimic-style termination (anchor + end-effector), optional.
        # Defaults off to preserve existing behavior.
        if bool(getattr(self.config.termination, "terminate_when_bm_anchor_error", False)):
            aidx = int(getattr(self, "_bm_anchor_body_idx", getattr(self, "torso_index", 0)))

            # BeyondMimic: position error uses ONLY z component (height) deviation.
            anchor_z_far = torch.abs(self.dif_global_body_pos[:, aidx, 2]) > self._bm_pos_z_threshold

            # BeyondMimic: orientation error uses ||log(R_des R^T)||, which equals rotation angle.
            ref_q = self.ref_body_rot_extend[:, aidx, :]
            cur_q = self._rigid_body_rot_extend[:, aidx, :]
            q_err = quat_mul(ref_q, quat_conjugate(cur_q, w_last=True), w_last=True)
            q_err = q_err / torch.norm(q_err, dim=-1, keepdim=True).clamp(min=1e-9)
            w_abs = torch.abs(q_err[:, 3]).clamp(max=1.0)
            angle = 2.0 * torch.acos(w_abs)  # [0, pi]
            anchor_rot_far = angle > self._bm_rot_threshold

            self.reset_buf, self.time_out_buf, _ = apply_termination_mask(
                self.log_dict,
                self.reset_buf,
                self.time_out_buf,
                "bm_anchor_error",
                anchor_z_far | anchor_rot_far,
            )

        if bool(getattr(self.config.termination, "terminate_when_bm_end_effector_error", False)) and len(getattr(self, "_bm_end_effector_body_indices", [])) > 0:
            ee_idx = torch.tensor(self._bm_end_effector_body_indices, device=self.device, dtype=torch.long)
            ee_z_far = torch.any(torch.abs(self.dif_global_body_pos[:, ee_idx, 2]) > self._bm_pos_z_threshold, dim=-1)
            # BeyondMimic text does not require end-effector orientation termination.
            self.reset_buf, self.time_out_buf, _ = apply_termination_mask(
                self.log_dict, self.reset_buf, self.time_out_buf, "bm_end_effector_error", ee_z_far
            )

        #object status
        if self.config.termination.terminate_when_ball_status_far:
            reset_buf_ball_status_far = torch.any(torch.norm(self.dif_global_ball_pos, dim=-1).unsqueeze(-1) > self.terminate_when_ball_status_far_threshold, dim=-1)
            # reset_buf_ball_status_far = torch.any(torch.abs(self.dif_global_ball_pos[:,0:1]) > 0.2, dim=-1) #wyh 
            # self.reset_buf |= reset_buf_ball_status_far
            # print(reset_buf_ball_status_far)
            # print(self.curr_ball_position, self.ref_object_root_pos, self.dif_global_ball_pos[:,0:1])
            # log current motion far threshold

            p = self.config.termination.terminate_when_ball_status_far_p
            window_mask = self.episode_length_buf + (self.motion_start_times/self.dt).to(int) < self.stand_ref_episode_length
            raw_ball_status_far = reset_buf_ball_status_far & window_mask
            applied_ball_status_far = (
                reset_buf_ball_status_far & (torch.rand_like(reset_buf_ball_status_far.float()) < p)
            )
            applied_ball_status_far = applied_ball_status_far.bool() & window_mask
            self.reset_buf, self.time_out_buf, _ = apply_termination_mask(
                self.log_dict,
                self.reset_buf,
                self.time_out_buf,
                "ball_status_far",
                raw_ball_status_far,
                applied_ball_status_far,
            )

            if self.config.termination_curriculum.terminate_when_ball_status_far_curriculum:
                self.log_dict["terminate_when_ball_status_far_threshold"] = torch.tensor(self.terminate_when_ball_status_far_threshold, dtype=torch.float)

        # object rotation (relative quaternion angle) termination
        if getattr(self.config.termination, "terminate_when_ball_rotation_far", False):
            # dif_global_ball_rot is xyzw (w last), representing relative rotation: q_ref * conj(q_curr)
            q = self.dif_global_ball_rot
            if not getattr(self, "terminate_when_ball_rotation_far_consider_yaw", True):
                q = self._remove_yaw_from_rel_quat(q)
            q = q / torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-9)
            w_abs = torch.abs(q[:, 3]).clamp(max=1.0)
            rel_angle = 2.0 * torch.acos(w_abs)  # [0, pi]

            reset_buf_ball_rot_far = rel_angle > self.terminate_when_ball_rotation_far_threshold
            p = getattr(self.config.termination, "terminate_when_ball_rotation_far_p", 1.0)
            window_mask = self.episode_length_buf + (self.motion_start_times/self.dt).to(int) < self.stand_ref_episode_length
            raw_ball_rot_far = reset_buf_ball_rot_far & window_mask
            applied_ball_rot_far = (
                reset_buf_ball_rot_far & (torch.rand_like(reset_buf_ball_rot_far.float()) < p)
            )
            applied_ball_rot_far = applied_ball_rot_far.bool() & window_mask
            self.reset_buf, self.time_out_buf, _ = apply_termination_mask(
                self.log_dict,
                self.reset_buf,
                self.time_out_buf,
                "ball_rotation_far",
                raw_ball_rot_far,
                applied_ball_rot_far,
            )
                
        if self.config.termination.terminate_by_low_obj_height:
            low_obj_height_mask = torch.any(
                self.simulator.object_pos[:, 2:3] < self.config.termination_scales.termination_min_obj_height, dim=1
            )
            self.reset_buf, self.time_out_buf, _ = apply_termination_mask(
                self.log_dict, self.reset_buf, self.time_out_buf, "low_obj_height", low_obj_height_mask
            )
        
        if self.config.termination.terminate_when_ball_z_far:
            ball_z_far_mask = (
                torch.abs(self.simulator.object_pos[:, 2] - self.ref_object_root_pos[:, 2])
                > self.terminate_when_ball_z_far_threshold
            )
            self.reset_buf, self.time_out_buf, _ = apply_termination_mask(
                self.log_dict, self.reset_buf, self.time_out_buf, "ball_z_far", ball_z_far_mask
            )

        if self.config.termination.terminate_when_heading_deviate:
            #quat
            curr_heading = torch_utils_quat_rotate_inverse(self.base_quat, self.forward_vec)
            ref_heading = torch_utils_quat_rotate_inverse(self.ref_root_rot, self.forward_vec)
            heading_deviation = torch.abs(curr_heading[:, 0] - ref_heading[:, 0])

            #euler
            # curr_heading = self.rpy[:, 2]
            # ref_heading = get_euler_xyz_in_tensor(self.ref_root_rot)[:, 2]
            # heading_deviation = torch.abs(curr_heading - ref_heading)
            
            heading_far_mask = heading_deviation > self.config.termination_scales.termination_heading_deviate_scale
            self.reset_buf, self.time_out_buf, _ = apply_termination_mask(
                self.log_dict, self.reset_buf, self.time_out_buf, "heading_deviate", heading_far_mask
            )
            

        if self.config.termination.termination_body_height:
            curr_body_z_error = self.dif_global_body_pos[:, self.torso_index, 2] # ref-result
            reset_buf_body_height = (curr_body_z_error > self.config.termination_scales.termination_body_height_scale)
            p = self.config.termination.termination_body_height_p
            applied_body_height = (reset_buf_body_height & (torch.rand_like(reset_buf_body_height.float()) < p)).bool()
            self.reset_buf, self.time_out_buf, _ = apply_termination_mask(
                self.log_dict,
                self.reset_buf,
                self.time_out_buf,
                "body_height",
                reset_buf_body_height,
                applied_body_height,
            )

        if self.config.termination.terminate_when_ig_status_far:
            reset_buf_ig_status_far = torch.any(torch.norm(self.dif_ig_extended_hand, dim=-1).unsqueeze(-1) > self.terminate_when_ig_status_far_threshold, dim=-1)
            reset_buf_ig_status_far = torch.any(reset_buf_ig_status_far, dim=-1)
            # self.reset_buf |= reset_buf_ig_status_far

            p = self.config.termination.terminate_when_ig_status_far_p
            window_mask = self.episode_length_buf + (self.motion_start_times/self.dt).to(int) < self.stand_ref_episode_length
            raw_ig_status_far = reset_buf_ig_status_far & window_mask
            applied_ig_status_far = (
                reset_buf_ig_status_far & (torch.rand_like(reset_buf_ig_status_far.float()) < p)
            )
            applied_ig_status_far = applied_ig_status_far.bool() & window_mask
            self.reset_buf, self.time_out_buf, _ = apply_termination_mask(
                self.log_dict,
                self.reset_buf,
                self.time_out_buf,
                "ig_status_far",
                raw_ig_status_far,
                applied_ig_status_far,
            )
            
        if self.config.termination.contact_conditioned_ig_termination:
            bodies_id = self.selected_conditioned_contact_bodies_id
            reset_buf_ig_status_far = torch.any(
                torch.norm(self.dif_ig_full[:, bodies_id], dim=-1).unsqueeze(-1)
                > self.contact_conditioned_ig_terminate_threshold,
                dim=-1,
            )
            reset_buf_ig_status_far = torch.any(reset_buf_ig_status_far, dim=-1)
            
            contact_conditioned_ig_status = torch.logical_and(reset_buf_ig_status_far, self.ref_obj_contact_status.bool())
            # self.reset_buf |= reset_buf_ig_status_far

            p = self.config.termination.contact_conditioned_ig_termination_p  
            applied_contact_conditioned_ig_status = (
                contact_conditioned_ig_status & (torch.rand_like(contact_conditioned_ig_status.float()) < p)
            ).bool()
            self.reset_buf, self.time_out_buf, _ = apply_termination_mask(
                self.log_dict,
                self.reset_buf,
                self.time_out_buf,
                "contact_conditioned_ig",
                contact_conditioned_ig_status,
                applied_contact_conditioned_ig_status,
            )
            
    def _update_timeout_buf(self):
        super()._update_timeout_buf()
        if self.config.termination.terminate_when_motion_end:
            current_time = (self.episode_length_buf) * self.dt + self.motion_start_times
            motion_end_mask = current_time > self.motion_len
            self.reset_buf, self.time_out_buf, _ = apply_termination_mask(
                self.log_dict, self.reset_buf, self.time_out_buf, "motion_end", motion_end_mask, mark_timeout=True
            )

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
        
        # Check whether a custom motion start time/frame is requested.
        if hasattr(self.config, 'use_custom_motion_start') and self.config.use_custom_motion_start:
            if getattr(self.config, "motion_start_time", None) is not None:
                # Use the specified start time (seconds).
                self.motion_start_times[env_ids] = torch.full((len(env_ids),), self.config.motion_start_time, dtype=torch.float32, device=self.device)
            elif getattr(self.config, "motion_start_frame", None) is not None:
                # Use the specified start frame index and convert it to time.
                motion_dt = self._motion_lib._motion_dt[self.motion_ids[env_ids]]
                start_time = self.config.motion_start_frame * motion_dt
                self.motion_start_times[env_ids] = start_time.to(self.device)
            else:
                # Neither motion_start_time nor motion_start_frame is set: raise an error.
                raise ValueError("use_custom_motion_start is True but neither motion_start_time nor motion_start_frame is specified in config")
        elif self.config.enable_rsi: #wyh
            if self.is_evaluating and not self.config.enforce_randomize_motion_start_eval:
                # self.motion_start_times[env_ids] = torch.zeros(len(env_ids), dtype=torch.float32, device=self.device) #+ 1.2
                self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
            else:
                self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        else:
            self.motion_start_times[env_ids] = torch.zeros(len(env_ids), dtype=torch.float32, device=self.device) 

        # AMP expert motion timeline is sampled independently when amp_motion_file is provided.
        if getattr(self, "_amp_motion_lib", None) is not None:
            try:
                self.amp_motion_start_times[env_ids] = self._amp_motion_lib.sample_time(self.motion_ids[env_ids])
            except Exception:
                # Fallback: deterministic start at 0 if sampling is unavailable.
                self.amp_motion_start_times[env_ids] = torch.zeros(len(env_ids), dtype=torch.float32, device=self.device)
        
        # self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        # offset = self.env_origins
        # motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset= offset)
        # motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)

    def resample_motion(self):
        self._motion_lib.load_motions(random_sample=True)
        if getattr(self, "_amp_motion_lib", None) is not None:
            self._amp_motion_lib.load_motions(random_sample=True)
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))


    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        
        offset = self.env_origins
        B = self.motion_ids.shape[0]
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times # next frames so +1
        now_time = self.episode_length_buf * self.dt + self.motion_start_times 
        # motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, now_time, offset=offset)
        #robot status
        ref_body_pos_extend = motion_res["rg_pos_t"]
        self.ref_body_pos_extend[:] = ref_body_pos_extend # for visualization and analysis
        ref_body_vel_extend = motion_res["body_vel_t"] # [num_envs, num_markers, 3]
        self.ref_body_rot_extend = ref_body_rot_extend = motion_res["rg_rot_t"] # [num_envs, num_markers, 4]
        ref_body_ang_vel_extend = motion_res["body_ang_vel_t"] # [num_envs, num_markers, 3]
        ref_joint_pos = motion_res["dof_pos"] # [num_envs, num_dofs]
        ref_joint_vel = motion_res["dof_vel"] # [num_envs, num_dofs]
        
        self.expert_dof_pos = ref_joint_pos
        self.expert_dof_vel = ref_joint_vel

        # AMP expert reference (can be from a different motion_file than tracking).
        # If _amp_motion_lib is not provided, fall back to tracking reference (backward compatible).
        if getattr(self, "_amp_motion_lib", None) is not None:
            amp_motion_times = (self.episode_length_buf + 1) * self.dt + self.amp_motion_start_times
            amp_now_time = self.episode_length_buf * self.dt + self.amp_motion_start_times
            amp_motion_res = self._amp_motion_lib.get_motion_state(self.motion_ids, amp_motion_times, amp_now_time, offset=offset)
            amp_ref_body_pos_extend = amp_motion_res["rg_pos_t"]
            amp_ref_body_vel_extend = amp_motion_res["body_vel_t"]
            self.amp_expert_dof_pos = amp_motion_res["dof_pos"]
            self.amp_expert_dof_vel = amp_motion_res["dof_vel"]
        else:
            amp_ref_body_pos_extend = ref_body_pos_extend
            amp_ref_body_vel_extend = ref_body_vel_extend
            self.amp_expert_dof_pos = self.expert_dof_pos
            self.amp_expert_dof_vel = self.expert_dof_vel
        
        # Root reference (world frame). Used by termination and minimal proprioceptive obs.
        self.ref_root_pos = motion_res["root_pos"]
        self.ref_root_rot = motion_res["root_rot"]
        ref_basket_pos = motion_res["basket_pos"]
        self.skill_label = motion_res["skill_label"]
        
        ref_origin_pos = motion_res["ref_origin_pos"]
        self.ref_origin_rot = motion_res["ref_origin_rot"]

        # Cache world basket position for debug visualization.
        # Note: motion_lib already applies env origin offset when offset is provided.
        if ref_basket_pos.ndim == 1:
            ref_basket_pos = ref_basket_pos.unsqueeze(0)
        self._debug_basket_pos_world[:] = ref_basket_pos
        # Check if basket_pos is valid (non-zero). If data doesn't have basket_pos, motion_lib returns zeros.
        self._has_valid_basket_pos[:] = ref_basket_pos.norm(dim=-1) > 1e-6
        human_joints = motion_res.get("human_joints", None)
        human_joints_valid = motion_res.get("human_joints_valid", None)
        human_root_rot = motion_res.get("human_root_rot", None)
        human_root_rot_valid = motion_res.get("human_root_rot_valid", None)
        if (
            human_joints is not None
            and human_joints.ndim == 3
            and human_joints.shape[1] == self._debug_human_joints_world.shape[1]
        ):
            self._debug_human_joints_world[:] = human_joints
            if human_root_rot is not None and human_root_rot_valid is not None:
                aligned_human_joints = self._motion_lib._align_human_joints_heading_to_target(
                    human_joints,
                    human_root_rot,
                    self.ref_root_rot,
                )
                valid_root_rot = human_root_rot_valid.to(torch.bool).view(-1)
                self._debug_human_joints_world[valid_root_rot] = aligned_human_joints[valid_root_rot]
            if human_joints_valid is None:
                self._has_valid_human_joints[:] = human_joints.shape[1] > 0
            else:
                self._has_valid_human_joints[:] = human_joints_valid.to(torch.bool)
        else:
            self._has_valid_human_joints[:] = False

        # Optional per-motion scalar (e.g. jumpshot apex height). Broadcast as (num_envs, 1).
        apex_height = motion_res.get("apex_height", None)
        if apex_height is None:
            self._obs_apex_height[:] = 0.0
        else:
            if apex_height.ndim == 1:
                apex_height = apex_height.unsqueeze(-1)
            self._obs_apex_height[:] = apex_height
        
        #object status
        ref_object_root_pos = motion_res["curr_obj_pos"]  # [num_envs, (xyz)]
        ref_object_root_rot = motion_res["curr_obj_rot"]  # [num_envs, (xyzw)]
        ref_object_root_vel = motion_res["curr_obj_vel"]  # [num_envs, (xyz)]
        self.curr_ball_position = self.simulator.object_pos.clone() # [num_envs, (xyz)]
        # update and print apex z during evaluation
        if getattr(self, "is_evaluating", False):
            curr_z = self.curr_ball_position[:, 2]
            new_max = torch.maximum(self._ball_max_z, curr_z)
            increased = new_max > self._ball_max_z + 1e-6
            if torch.any(increased):
                # print per-env new max and global max for quick inspection
                max_vals = new_max[increased].detach().cpu().tolist()
                env_ids = torch.nonzero(increased, as_tuple=False).flatten().detach().cpu().tolist()
                for eid, val in zip(env_ids, max_vals):
                    print(f"[Eval] Env {eid} ball z apex updated: {val:.4f} m")
                print(f"[Eval] Global ball z apex: {new_max.max().item():.4f} m")
            self._ball_max_z = new_max
        
        self.target_ball_position = motion_res["next_obj_pos"]  # [num_envs, (xyz)]
        self.ref_object_root_pos = ref_object_root_pos.detach().clone()
        self.ref_object_root_rot = ref_object_root_rot.detach().clone()
        self.ref_object_root_vel = ref_object_root_vel.detach().clone()
        
        ref_obj_contact = motion_res["ball_contact"] 
        ref_lfoot_contact = motion_res["lfoot_contact"] 
        ref_rfoot_contact = motion_res["rfoot_contact"] 
        ref_ig = ref_body_pos_extend[:, self.hoi_interaction_bodies_indices, :] -  ref_object_root_pos.unsqueeze(1)
        ref_ig_extended_hand = ref_body_pos_extend[:, self.extended_hand_branch_id, :] -  ref_object_root_pos.unsqueeze(1)
        
        ref_ig_full = ref_body_pos_extend -  ref_object_root_pos.unsqueeze(1)

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

        ################### Interaction #####################
        self.ig = self._rigid_body_pos_extend[:, self.hoi_interaction_bodies_indices, :] - self.curr_ball_position.unsqueeze(1)
        self.ig_extended_hand = self._rigid_body_pos_extend[:, self.extended_hand_branch_id, :] - self.curr_ball_position.unsqueeze(1)
        
        full_ig = self._rigid_body_pos_extend - self.curr_ball_position.unsqueeze(1)
        ################### Contact #####################
        #robot bodies contact
        # self.bodies_contact = torch.any(torch.norm(self.simulator.contact_forces[:, self.hoi_contact_indices, :], dim=-1) > 0.1, dim=1).to(torch.float32)
        # self.object_contact = torch.any(torch.norm(self.simulator.contact_forces[:, -2:-1, :], dim=-1) > 0.1, dim=1).to(torch.float32)
        bodies_contact = torch.any(torch.abs(self.simulator.contact_forces[:, self.hoi_contact_indices, :].clone()) > 0.1, dim=-1)
        self.bodies_contact = torch.any(bodies_contact, dim=-1, keepdim=False).to(torch.float32)
        
        no_bodies_contact = torch.all(torch.abs(self.simulator.contact_forces[:, self.hoi_no_contact_indices, :].clone()) < 0.1, dim=-1)
        self.no_bodies_contact = 1.0 - torch.all(no_bodies_contact, dim=-1, keepdim=False).to(torch.float32)
        
        # wyh
        ball_rigid_body_idx = int(getattr(self.simulator, "ball_rigid_body_idx", self.num_bodies))
        self.object_contact = get_hoi_object_contact(
            self.simulator.contact_forces,
            ball_rigid_body_idx=ball_rigid_body_idx,
        )
        
        # Log body names when contact occurs.
        if hasattr(self.config, 'enable_contact_logging') and self.config.enable_contact_logging:
            self._log_contact_bodies()
        
        self.bodies_contact_force = build_hoi_body_contact_force(
            self.simulator.contact_forces,
            ball_rigid_body_idx=ball_rigid_body_idx,
            hoi_no_contact_indices=self.hoi_no_contact_indices,
        ).view(self.num_envs, -1)
        
        self.bodies_contact_force = self.bodies_contact_force / (torch.norm(self.bodies_contact_force, dim=-1, keepdim=True) + 1e-6)
        self.bodies_contact_force = self.bodies_contact_force.view(self.num_envs, -1)
        

        # if self.bodies_contact_force[:,15]>0.:
        #     print(self.bodies_contact_force[:,15])
        
        # force sensor
        # sensor_tensor = self.simulator.gym.acquire_force_sensor_tensor(self.simulator.sim)
        # self.simulator.gym.refresh_force_sensor_tensor(self.simulator.sim)
        # force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
        
        # s_ = self.simulator.force_sensor_step
        # self.bodies_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # self.bodies_contact_force = []
        # for i in range(len(self.hoi_contact_indices)):
        #     bodies_sensor_forces = force_sensor_readings[i::s_, :3].unsqueeze(1)
        #     self.bodies_contact_force.append(bodies_sensor_forces.squeeze(1))
        #     self.bodies_contact |= torch.any(torch.norm(bodies_sensor_forces, dim=-1) > 1.0, dim=1)
        
        # self.bodies_contact_force = torch.cat(self.bodies_contact_force, dim=-1)
        # self.bodies_contact = self.bodies_contact.to(torch.float32)
        
        #ball contact
        # sensor_forces = force_sensor_readings[i+1::s_, :3].unsqueeze(1)
        # self.object_contact = torch.any(torch.norm(sensor_forces, dim=-1) > 1.0, dim=1).to(torch.float32)
        
        ################### Compute differences #####################

        ## diff compute - kinematic position
        self.dif_global_body_pos = ref_body_pos_extend - self._rigid_body_pos_extend
        # import ipdb; ipdb.set_trace()
        ## diff compute - kinematic rotation
        self.dif_global_body_rot =  quat_mul(ref_body_rot_extend, quat_conjugate(self._rigid_body_rot_extend, w_last=True), w_last=True) #ref_body_rot_extend - self._rigid_body_rot_extend
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

        #compute heading
        heading_inv_rot = calc_heading_quat_inv(self.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        self._build_reference_command(now_time, heading_inv_rot)
        # expand to (B*num_rigid_bodies, 4) for fatser computation in jit
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(1).expand(-1, num_rigid_bodies+self.num_extend_bodies, -1).reshape(-1, 4)

        # heading_rot = calc_heading_quat(self.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        # heading_rot_expand = heading_rot.unsqueeze(1).expand(-1, num_rigid_bodies, -1).reshape(-1, 4)

        dif_global_body_pos_for_obs_compute = ref_body_pos_extend.view(env_batch_size, -1, 3) - self._rigid_body_pos_extend.view(env_batch_size, -1, 3)
        dif_local_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), dif_global_body_pos_for_obs_compute.view(-1, 3))
        
        self._obs_dif_local_rigid_body_pos = dif_local_body_pos_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
        
        ################### Compute Local Status #####################
        #robot
        num_bodies = self._rigid_body_pos_extend.shape[1]
        
        #compute local ref bodies pos
        global_ref_rigid_body_pos = ref_body_pos_extend.view(env_batch_size, -1, 3) - self.simulator.robot_root_states[:, :3].view(env_batch_size, 1, 3)  # preserves the body position
        local_ref_rigid_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_rigid_body_pos.view(-1, 3))
        self._obs_local_ref_rigid_body_pos = local_ref_rigid_body_pos_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
        #compute local ref bodies vel
        global_ref_body_vel = ref_body_vel_extend.view(env_batch_size, -1, 3) - self.simulator.robot_root_states[:, 7:10].view(env_batch_size, 1, 3)
        local_ref_rigid_body_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_body_vel.view(-1, 3))
        self._obs_local_ref_rigid_body_vel = local_ref_rigid_body_vel_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)

        #compute local AMP expert ref bodies pos/vel (may be from a different motion source)
        global_amp_ref_rigid_body_pos = amp_ref_body_pos_extend.view(env_batch_size, -1, 3) - self.simulator.robot_root_states[:, :3].view(env_batch_size, 1, 3)
        local_amp_ref_rigid_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_amp_ref_rigid_body_pos.view(-1, 3))
        self._obs_local_amp_expert_rigid_body_pos = local_amp_ref_rigid_body_pos_flat.view(env_batch_size, -1)
        global_amp_ref_body_vel = amp_ref_body_vel_extend.view(env_batch_size, -1, 3) - self.simulator.robot_root_states[:, 7:10].view(env_batch_size, 1, 3)
        local_amp_ref_rigid_body_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_amp_ref_body_vel.view(-1, 3))
        self._obs_local_amp_expert_rigid_body_vel = local_amp_ref_rigid_body_vel_flat.view(env_batch_size, -1)

        #compute local bodies pos
        global_keybodies_pos = self._rigid_body_pos_extend.view(env_batch_size, -1, 3) -  self.simulator.robot_root_states[:, :3].view(env_batch_size, 1, 3)
        local_keybodies_pos = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_keybodies_pos.view(-1, 3))
        self._obs_local_keybodies_pos = local_keybodies_pos.view(env_batch_size, num_bodies, -1) # (num_envs, num_rigid_bodies*3)
        
        #compute local bodies vel
        global_keybodies_vel = self._rigid_body_vel_extend.view(env_batch_size, -1, 3) - self.simulator.robot_root_states[:, 7:10].view(env_batch_size, 1, 3)
        local_keybodies_vel = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_keybodies_vel.view(-1, 3))
        self._obs_local_keybodies_vel = local_keybodies_vel.view(env_batch_size, num_bodies, -1) # (num_envs, num_rigid_bodies*3)
        
        #compute local rot
        global_keybodies_rot = self._rigid_body_rot_extend.view(env_batch_size, -1, 4) 
        local_keybodies_rot = quat_mul(heading_inv_rot_expand.view(-1, 4), global_keybodies_rot.view(-1, 4), w_last=True)
        self._obs_local_keybodies_rot = local_keybodies_rot.view(env_batch_size, num_bodies, -1) # (num_envs, num_rigid_bodies)
        
        #compute local ang vel
        global_keybodies_ang_vel = self._rigid_body_ang_vel_extend.view(env_batch_size, -1, 3)
        local_keybodies_ang_vel = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_keybodies_ang_vel.view(-1, 3))
        self._obs_local_keybodies_ang_vel = local_keybodies_ang_vel.view(env_batch_size, num_bodies, -1) # (num_envs, num_rigid_bodies)
        
        #compute local_ref_origin_pos_xy
        global_ref_origin_pos = ref_origin_pos.clone() 
        global_ref_origin_pos[:, :2] = global_ref_origin_pos[:, :2] - self.simulator.robot_root_states[:, :2] # preserves the body position
        local_ref_origin_pos_flat = my_quat_rotate(heading_inv_rot.view(-1, 4), global_ref_origin_pos.view(-1, 3))
        self._obs_local_ref_origin_pos = local_ref_origin_pos_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
        
        
        #object
        #compute ref basket pos
        global_ref_basket_pos = ref_basket_pos.unsqueeze(1) - self.simulator.robot_root_states[:, :3].view(env_batch_size, 1, 3)
        local_ref_basket_pos = my_quat_rotate(heading_inv_rot, global_ref_basket_pos.view(-1, 3))
        self.ref_basket_pos = local_ref_basket_pos.view(env_batch_size, -1)# (num_envs, 3) 
        
        #compute object local pos
        self._obs_local_ball_root_pos = self._world_pos_to_heading_local(
            self.simulator.object_pos, heading_inv_rot=heading_inv_rot
        ).view(env_batch_size, -1)  # (num_envs, 3)

        self._obs_local_curr_ball_position_wo_packet_loss = self._obs_local_ball_root_pos.clone() # (num_envs, 3)
        # if self.obs_reduction_timer > 0:
        #     self.obs_reduction_timer += -1
        # obs_reduction_scale = (self.obs_reduction_timer//24)*(1./2000.)
        # self._obs_local_ball_root_pos = local_ball_root_pos.view(env_batch_size, -1)*obs_reduction_scale
        
        #compute object local vel
        global_ball_root_vel = self.simulator.object_vel.view(env_batch_size, 1, 3) - self.simulator.robot_root_states[:, 7:10].view(env_batch_size, 1, 3)
        local_ball_root_vel = my_quat_rotate(heading_inv_rot, global_ball_root_vel.view(-1, 3))
        self._obs_local_ball_root_vel = local_ball_root_vel.view(env_batch_size, -1) # (num_envs, 3)
        
        #compute object local rot
        global_ball_root_rot = self.simulator.object_rot.view(env_batch_size, 1, 4)
        local_ball_root_rot = quat_mul(heading_inv_rot, global_ball_root_rot.view(-1, 4), w_last=True)
        self._obs_local_ball_root_rot = local_ball_root_rot.view(env_batch_size, -1) # (num_envs, 4)
        
        #compute object local ang vel
        global_ball_root_ang_vel = self.simulator.object_ang_vel.view(env_batch_size, 1, 3)
        local_ball_root_ang_vel = my_quat_rotate(heading_inv_rot, global_ball_root_ang_vel.view(-1, 3))
        self._obs_local_ball_root_ang_vel = local_ball_root_ang_vel.view(env_batch_size, -1) # (num_envs, 3)
        
        #compute target object local pos
        global_target_ball_root_pos = self.target_ball_position.view(env_batch_size, 1, 3) - self.simulator.robot_root_states[:, :3].view(env_batch_size, 1, 3)
        local_target_ball_root_pos = my_quat_rotate(heading_inv_rot, global_target_ball_root_pos.view(-1, 3))
        self._obs_local_target_ball_root_pos = local_target_ball_root_pos.view(env_batch_size, -1) # (num_envs, 3)
        
        #compute ref object root pos
        global_ref_obj_root_pos = ref_object_root_pos.view(env_batch_size, 1, 3) - self.simulator.robot_root_states[:, :3].view(env_batch_size, 1, 3)
        local_ref_object_root_pos = my_quat_rotate(heading_inv_rot, global_ref_obj_root_pos.view(-1, 3))
        self._obs_local_ref_object_root_pos = local_ref_object_root_pos.view(env_batch_size, -1) # (num_envs, 3)
        
        #compute pd error wyh
        # actions_scaled = self.actions * self.config.robot.control.action_scale
        actions_scaled = self.actions * self.action_scales
        self._obs_pd_error = (actions_scaled + self.default_dof_pos - self.simulator.dof_pos)

        #compute phase wyh
        # self._obs_phase = self.episode_length_buf * self.dt + self.motion_start_times
        #compute phase wyh
        if self.config.clip_phase_to_standframe:
            self._obs_phase = torch.where(
                self.episode_length_buf + (self.motion_start_times/self.dt).to(int) < self.stand_ref_episode_length,
                self.episode_length_buf * self.dt + self.motion_start_times,
                self.stand_ref_episode_length * self.dt)
        else:
            self._obs_phase = self.episode_length_buf * self.dt + self.motion_start_times
        self._obs_phase = self._obs_phase.unsqueeze(dim=-1)

        ######################VR 3 point ########################
        if not self.config.use_teleop_control:
            ref_vr_3point_pos = ref_body_pos_extend.view(env_batch_size, -1, 3)[:, self.motion_tracking_id, :]
        else:
            ref_vr_3point_pos = self.teleop_marker_coords
        vr_2root_pos = (ref_vr_3point_pos - self.simulator.robot_root_states[:, 0:3].view(env_batch_size, 1, 3))
        heading_inv_rot_vr = heading_inv_rot.repeat(3,1)
        self._obs_vr_3point_pos = my_quat_rotate(heading_inv_rot_vr.view(-1, 4), vr_2root_pos.view(-1, 3)).view(env_batch_size, -1)
        
        #################### object Reward ###################### 
        
        #object status
        self.dif_global_ball_pos = ref_object_root_pos - self.curr_ball_position
        #object rotation
        self.dif_global_ball_rot = quat_mul(ref_object_root_rot, quat_conjugate(self.simulator.object_rot, w_last=True), w_last=True)
        #object velocity
        self.dif_global_ball_vel = ref_object_root_vel - self.simulator.object_vel
        
        #interaction graph
        self.dif_ig = ref_ig - self.ig
        self.dif_ig_extended_hand = ref_ig_extended_hand - self.ig_extended_hand
        
        self.dif_ig_full = ref_ig_full - full_ig
        
        # waist rotation
        ref_waist = ref_joint_pos[:, 12:14]
        self.dif_waist_dof = ref_waist - self.simulator.dof_pos[:, 12:14]
        
        #contact graph
        self.bodies_contact_status = self.bodies_contact - ref_obj_contact 
        self.no_bodies_contact_status = self.no_bodies_contact - torch.zeros_like(ref_obj_contact).to(torch.float32)
        self.object_contact_status = self.object_contact - ref_obj_contact 

        self.ref_obj_contact_status = ref_obj_contact
        #foot contact graph
        self.lfoot_contact_status = (self.simulator.contact_forces[:, self.lfoot_indice, 2] > 1.).to(torch.float32) - ref_lfoot_contact
        self.rfoot_contact_status = (self.simulator.contact_forces[:, self.rfoot_indice, 2] > 1.).to(torch.float32) - ref_rfoot_contact
        
        # update packet loss status
        if self._packet_loss_enabled:
            self._update_packet_loss_status()

        # ---------------- OmniRetarget minimal proprioceptive obs ----------------
        # Compute ONLY when obs config needs it (otherwise this is wasted work and alloc churn).
        if getattr(self, "_enable_omniretarget_min_obs", False):
            # Reference joint pos/vel + pelvis (root) pos/orientation error.
            # Position error is expressed in heading-local frame for yaw invariance.
            # NOTE: reuse heading_inv_rot computed above in this function.
            self._obs_ref_dof_pos[:] = self.expert_dof_pos - self.default_dof_pos
            self._obs_ref_dof_vel[:] = self.expert_dof_vel

            pelvis_pos_err_global = self.ref_root_pos - self.simulator.robot_root_states[:, 0:3]
            self._obs_pelvis_pos_err[:] = my_quat_rotate(heading_inv_rot, pelvis_pos_err_global)

            # quaternion error: q_ref * conj(q_curr), xyzw (w last)
            self._obs_pelvis_rot_err[:] = quat_mul(
                self.ref_root_rot,
                quat_conjugate(self.base_quat, w_last=True),
                w_last=True,
            )

            # Rot6D error (take first 2 columns of rotation matrix).
            # Note: quaternion_to_matrix expects wxyz, while our quats are xyzw.
            q_err_wxyz = xyzw_to_wxyz(self._obs_pelvis_rot_err)
            R = quaternion_to_matrix(q_err_wxyz)  # (N, 3, 3)
            self._obs_pelvis_rot_err_rot6d[:] = R[:, :, 0:2].reshape(self.num_envs, 6)

        self._log_motion_tracking_info()

    # def _compute_reward(self):
    #     super()._compute_reward()
    #     self.extras["ref_body_pos_extend"] = self.ref_body_pos_extend.clone()
    #     self.extras["ref_body_rot_extend"] = self.ref_body_rot_extend.clone()


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
            # only use object_related_rewards when current time < stand_ref_episode_length
            if name in self.object_related_rewards:
                rew = torch.where(
                    self.episode_length_buf + (self.motion_start_times/self.dt).to(int) < self.stand_ref_episode_length,
                    rew,
                    rew*0.
                )
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.config.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

        # Contact error tracking (cg_reward != 1) within the same time window as object_related_rewards.
        window_mask = (
            self.episode_length_buf + (self.motion_start_times / self.dt).to(int) < self.stand_ref_episode_length
        )
        if torch.any(window_mask):
            forbidden_contact_error = (self.no_bodies_contact_status != 0).to(torch.float32)
            object_contact_mismatch_error = (self.object_contact_status != 0).to(torch.float32)
            missed_object_contact_error = (self.object_contact_status < 0).to(torch.float32)
            unexpected_object_contact_error = (self.object_contact_status > 0).to(torch.float32)
            contact_error = torch.maximum(forbidden_contact_error, object_contact_mismatch_error)
            window_mask_f = window_mask.to(torch.float32)
            self._contact_error_count += contact_error * window_mask_f
            self._forbidden_contact_error_count += forbidden_contact_error * window_mask_f
            self._object_contact_mismatch_error_count += object_contact_mismatch_error * window_mask_f
            self._missed_object_contact_error_count += missed_object_contact_error * window_mask_f
            self._unexpected_object_contact_error_count += unexpected_object_contact_error * window_mask_f
            self._contact_window_count += window_mask_f

        #update reward cirriculum
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

        self.extras["ref_body_pos_extend"] = self.ref_body_pos_extend.clone()
        self.extras["ref_body_rot_extend"] = self.ref_body_rot_extend.clone()


    # def _compute_reward(self):
    #     """ Compute rewards
    #         Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
    #         adds each terms to the episode sums and to the total reward
    #     """
    #     self.rew_buf[:] = 0.
    #     # mul_rew_buf = 1. if len(self.multiplicative_rewards) > 0 else 0.
    #     mul_rew_buf = 0. if len(self.multiplicative_rewards) > 0 else 0.
    #     contact_rew = 1.
        
    #     for i in range(len(self.reward_functions)):
    #         name = self.reward_names[i]
    #         rew = self.reward_functions[i]() * self.reward_scales[name]
    #         try:
    #             assert rew.shape[0] == self.num_envs
    #         except:
    #             import ipdb; ipdb.set_trace()
    #         # penalty curriculum
    #         if name in self.config.rewards.reward_penalty_reward_names:
    #             if self.config.rewards.reward_penalty_curriculum:
    #                 rew *= self.reward_penalty_scale
    #         if name in self.multiplicative_rewards:
    #             mul_rew_buf += rew
    #         else:
    #             self.rew_buf += rew
    #         if name == 'contact_graph':
    #             contact_rew = rew
    #         self.episode_sums[name] += rew
    #     if self.config.rewards.only_positive_rewards:
    #         self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
    #     # add termination reward after clipping
    #     if "termination" in self.reward_scales:
    #         rew = self._reward_termination() * self.reward_scales["termination"]
    #         self.rew_buf += rew
    #         self.episode_sums["termination"] += rew
            
    #     #update reward buffers
    #     self.rew_buf[:] += mul_rew_buf*contact_rew
        
    #     #update reward cirriculum

    #     if self.use_reward_penalty_curriculum:
    #         self.log_dict["penalty_scale"] = torch.tensor(self.reward_penalty_scale, dtype=torch.float)
    #         self.log_dict["average_episode_length"] = self.average_episode_length
        
    #     if self.use_reward_limits_dof_pos_curriculum:
    #         self.log_dict["soft_dof_pos_curriculum_value"] = torch.tensor(self.soft_dof_pos_curriculum_value, dtype=torch.float)
    #     if self.use_reward_limits_dof_vel_curriculum:
    #         self.log_dict["soft_dof_vel_curriculum_value"] = torch.tensor(self.soft_dof_vel_curriculum_value, dtype=torch.float)
    #     if self.use_reward_limits_torque_curriculum:
    #         self.log_dict["soft_torque_curriculum_value"] = torch.tensor(self.soft_torque_curriculum_value, dtype=torch.float)
        
    #     if self.add_noise_currculum:
    #         self.log_dict["current_noise_curriculum_value"] = torch.tensor(self.current_noise_curriculum_value, dtype=torch.float)

    #     self.extras["ref_body_pos_extend"] = self.ref_body_pos_extend.clone()
    #     self.extras["ref_body_rot_extend"] = self.ref_body_rot_extend.clone()
        
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

    def _update_packet_loss_status(self):
        """Update packet loss timer based on object state and configured rules."""
        pl_cfg = self.config.packet_loss

        # compute distance between object and hand center
        left_hand_pos = self._rigid_body_pos_extend[:, self.left_hand_link_idx, :]
        right_hand_pos = self._rigid_body_pos_extend[:, self.right_hand_link_idx, :]
        hand_center = (left_hand_pos + right_hand_pos) * 0.5
        # Keep packet-loss trigger consistent with the "object observation" source.
        # If `obs.use_projectile_as_obj_obs` is enabled and projectile state exists, treat projectile as object.
        use_proj = bool(getattr(getattr(self.config, "obs", None), "use_projectile_as_obj_obs", False))
        use_proj = use_proj and bool(getattr(getattr(self.config, "robot", None), "proj", False))
        use_proj = use_proj and hasattr(self.simulator, "_proj_states")
        if use_proj:
            obj_pos = self.simulator._proj_states[:, 0, 0:3]
        else:
            obj_pos = self.curr_ball_position
        dist_to_hands = torch.norm(obj_pos - hand_center, dim=-1)

        # --- determine current mode (priority: ground_abs_dist > very_far > near/far) ---
        # Base zone by per-env threshold (backward compatible)
        is_near = dist_to_hands < self.packet_loss_dist_threshold
        mode = torch.where(
            is_near,
            torch.zeros(self.num_envs, device=self.device, dtype=torch.int64),
            torch.ones(self.num_envs, device=self.device, dtype=torch.int64),
        )

        # Case 1 (optional): very-far by hand-center distance (should be configured non-overlap with near/far threshold range)
        dist_th_very_far = getattr(pl_cfg, "dist_threshold_very_far", None)
        if dist_th_very_far is not None:
            is_very_far = dist_to_hands > float(dist_th_very_far)
            mode = torch.where(is_very_far, torch.full_like(mode, 2), mode)

        # Case 2 (optional, highest priority): abs global z exceeds threshold (may overlap with other cases)
        ground_abs_th = getattr(pl_cfg, "ground_abs_dist_threshold", None)
        if ground_abs_th is not None:
            abs_ground_dist = torch.abs(obj_pos[:, 2])
            is_ground_far = abs_ground_dist > float(ground_abs_th)
            mode = torch.where(is_ground_far, torch.full_like(mode, 3), mode)

        # decrement active timers
        active_mask = self.packet_loss_timer > 0
        self.packet_loss_timer[active_mask] -= 1

        # Do not continue across ranges: if mode changed while active, terminate immediately
        zone_mismatch = active_mask & (mode.to(self.packet_loss_mode_active.dtype) != self.packet_loss_mode_active)
        if zone_mismatch.any():
            self.packet_loss_timer[zone_mismatch] = 0

        # idle envs (including those just terminated above) may start a new event this step
        self.packet_loss_timer.clamp_(min=0)
        idle_mask = self.packet_loss_timer == 0
        if idle_mask.any():
            # Backward compatible defaults:
            # - If new case configs are absent, only near/far are used.
            prob_near = float(getattr(pl_cfg, "prob_near", 0.0))
            prob_far = float(getattr(pl_cfg, "prob_far", 0.0))
            _p_vf = getattr(pl_cfg, "prob_very_far", prob_far)
            _p_gd = getattr(pl_cfg, "prob_ground", prob_far)
            prob_very_far = float(prob_far if _p_vf is None else _p_vf)
            prob_ground = float(prob_far if _p_gd is None else _p_gd)

            # Probability per env based on mode
            prob_full = torch.full((self.num_envs,), prob_far, device=self.device, dtype=torch.float32)
            prob_full[mode == 0] = prob_near
            prob_full[mode == 2] = prob_very_far
            prob_full[mode == 3] = prob_ground

            trigger = (torch.rand(self.num_envs, device=self.device) < prob_full) & idle_mask
            if trigger.any():
                tidx = trigger.nonzero(as_tuple=True)[0]
                tmode = mode[tidx]

                len_near = getattr(pl_cfg, "len_range_near", [1, 1])
                len_far = getattr(pl_cfg, "len_range_far", [1, 1])
                _len_vf = getattr(pl_cfg, "len_range_very_far", len_far)
                _len_gd = getattr(pl_cfg, "len_range_ground", len_far)
                len_very_far = len_far if _len_vf is None else _len_vf
                len_ground = len_far if _len_gd is None else _len_gd

                duration = torch.zeros((tidx.numel(),), dtype=torch.int64, device=self.device)
                m0 = tmode == 0
                m1 = tmode == 1
                m2 = tmode == 2
                m3 = tmode == 3
                if m0.any():
                    duration[m0] = torch.randint(len_near[0], len_near[1] + 1, (int(m0.sum().item()),), device=self.device)
                if m1.any():
                    duration[m1] = torch.randint(len_far[0], len_far[1] + 1, (int(m1.sum().item()),), device=self.device)
                if m2.any():
                    duration[m2] = torch.randint(len_very_far[0], len_very_far[1] + 1, (int(m2.sum().item()),), device=self.device)
                if m3.any():
                    duration[m3] = torch.randint(len_ground[0], len_ground[1] + 1, (int(m3.sum().item()),), device=self.device)

                self.packet_loss_timer[tidx] = duration
                self.packet_loss_mode_active[tidx] = tmode.to(self.packet_loss_mode_active.dtype)
        
    def _draw_sphere_safe(self, pos, radius, color, env_id, pos_id=None):
        try:
            if pos_id is not None:
                self.simulator.draw_sphere(pos, radius, color, env_id, pos_id)
            else:
                self.simulator.draw_sphere(pos, radius, color, env_id)
        except TypeError:
            # Fallback for simulators whose draw_sphere does not accept pos_id
            self.simulator.draw_sphere(pos, radius, color, env_id)

    def _update_sim_object_contact_color(self, env_id: int):
        if not bool(getattr(self.config, "visualize_sim_object_contact", False)):
            return
        if self.config.simulator.config.name != "isaacgym":
            return
        if not hasattr(self, "object_contact"):
            return
        if not hasattr(self.simulator, "ball_handles") or not hasattr(self.simulator, "envs"):
            return

        contact_active = bool((self.object_contact[env_id].reshape(-1)[0] > 0.5).item())
        color = gymapi.Vec3(1.8, 0.0, 0.0) if contact_active else gymapi.Vec3(1.5, 1.5, 1.5)
        self.simulator.gym.set_rigid_body_color(
            self.simulator.envs[env_id],
            self.simulator.ball_handles[env_id],
            0,
            gymapi.MESH_VISUAL,
            color,
        )

    def _draw_tracking_target_vis(self, env_id: int):
        if not bool(getattr(self.config, "visualize_tracking_targets", False)):
            return
        if not getattr(self, "_reference_command_enabled", False):
            return
        if getattr(self, "_reference_command_use_human_smpl", False):
            return

        body_indices = getattr(self, "_reference_command_body_indices", None)
        future_steps = getattr(self, "_reference_command_future_steps", None)
        if body_indices is None or future_steps is None or body_indices.numel() == 0 or future_steps.numel() == 0:
            return

        colors = [
            (1.0, 0.85, 0.05),  # root / pelvis
            (1.0, 0.15, 0.75),  # left wrist
            (0.1, 0.55, 1.0),   # right wrist
        ]
        contact_color = (1.0, 0.0, 0.0)
        body_names = getattr(self, "_reference_command_body_names", ())
        current_contact = False
        if hasattr(self, "ref_obj_contact_status"):
            current_contact = bool((self.ref_obj_contact_status[env_id].reshape(-1)[0] > 0.5).item())

        # Current-frame reference target.
        if hasattr(self, "ref_body_pos_extend"):
            for body_i, body_idx in enumerate(body_indices):
                color = colors[body_i % len(colors)]
                body_name = body_names[body_i] if body_i < len(body_names) else ""
                if current_contact and "wrist" in body_name:
                    color = contact_color
                pos = self.ref_body_pos_extend[env_id, int(body_idx.item())]
                self._draw_sphere_safe(pos, 0.0325, color, env_id)

        # Future reference command stored in actor obs: [T, B, xyz] then optional contact bits.
        num_future = int(future_steps.numel())
        num_bodies = int(body_indices.numel())
        pos_dim = num_future * num_bodies * 3
        if self._obs_reference_command.shape[1] < pos_dim:
            return

        local_future = self._obs_reference_command[env_id, :pos_dim].view(num_future, num_bodies, 3)
        world_future = self._heading_local_pos_to_world_for_env(local_future, env_id)
        future_contact = None
        contact_dim = pos_dim + num_future
        if getattr(self, "_reference_command_include_ball_contact", False) and self._obs_reference_command.shape[1] >= contact_dim:
            future_contact = self._obs_reference_command[env_id, pos_dim:contact_dim] > 0.5
        for t in range(num_future):
            radius = 0.0175 if t == 0 else 0.012
            for body_i in range(num_bodies):
                color = colors[body_i % len(colors)]
                body_name = body_names[body_i] if body_i < len(body_names) else ""
                if future_contact is not None and bool(future_contact[t].item()) and "wrist" in body_name:
                    color = contact_color
                self._draw_sphere_safe(world_future[t, body_i], radius, color, env_id)
 
    def _draw_debug_vis(self):
        self.simulator.clear_lines()
        self._refresh_sim_tensors()

        for env_id in range(self.num_envs):
            visualize_debug_object_markers = bool(getattr(self.config, "visualize_debug_object_markers", True))
            self._update_sim_object_contact_color(env_id)

            # if not self.config.use_teleop_control:
            #     # draw marker joints
            #     for pos_id, pos_joint in enumerate(self.marker_coords[env_id]): # idx 0 torso (duplicate with 11)
            #         if self.config.robot.motion.visualization.customize_color:
            #             color_inner = self.config.robot.motion.visualization.marker_joint_colors[pos_id % len(self.config.robot.motion.visualization.marker_joint_colors)]
            #         else:
            #             color_inner = (0.3, 0.3, 0.3)
            #         color_inner = tuple(color_inner)

            #         # import ipdb; ipdb.set_trace()
            #         self._draw_sphere_safe(pos_joint, 0.04, color_inner, env_id, pos_id)


            # else:
            #     # draw teleop joints
            #     for pos_id, pos_joint in enumerate(self.teleop_marker_coords[env_id]):
            #         self._draw_sphere_safe(pos_joint, 0.04, (0.851, 0.144, 0.07), env_id, pos_id)

            # # Draw hand center for visualization (if hand indices available)
            # if hasattr(self, "left_hand_link_idx") and hasattr(self, "right_hand_link_idx"):
            #     left_hand_pos = self._rigid_body_pos_extend[env_id, self.left_hand_link_idx, :]
            #     right_hand_pos = self._rigid_body_pos_extend[env_id, self.right_hand_link_idx, :]
            #     print(left_hand_pos, right_hand_pos)
            #     hand_center = (left_hand_pos + right_hand_pos) * 0.5
            #     # left/right hands
            #     self._draw_sphere_safe(left_hand_pos, 0.06, (1.,1.,1.), env_id)   # green-ish 0.2, 1.0, 0.2
            #     self._draw_sphere_safe(right_hand_pos, 0.06, (1.,1.,1.), env_id)  # blue-ish 0.2, 0.2, 1.0
            #     self._draw_sphere_safe(hand_center, 0.06, (0.0, 0.8, 1.0), env_id)

            # Draw packet loss indicator (red sphere at ball position when packet loss active)
            if visualize_debug_object_markers and self._packet_loss_enabled and self.packet_loss_timer[env_id] > 0:
                # Backward compatible: default draws at simulator.object_pos (ball).
                # If `obs.use_projectile_as_obj_obs` is enabled and projectile state is available,
                # draw at projectile position to match the object observation source.
                use_proj = bool(getattr(getattr(self.config, "obs", None), "use_projectile_as_obj_obs", False))
                use_proj = use_proj and bool(getattr(getattr(self.config, "robot", None), "proj", False))
                use_proj = use_proj and hasattr(self.simulator, "_proj_states")

                if use_proj:
                    obj_pos = self.simulator._proj_states[env_id, 0, 0:3]
                else:
                    obj_pos = self.simulator.object_pos[env_id]
                self._draw_sphere_safe(obj_pos, 0.15, (1.0, 0.0, 0.0), env_id)

            # Draw basket position from motion (green sphere) - only if data has valid basket_pos
            if visualize_debug_object_markers and self._has_valid_basket_pos[env_id]:
                basket_pos = self._debug_basket_pos_world[env_id]
                self._draw_sphere_safe(basket_pos, 0.12, (0.1, 1.0, 0.1), env_id)

            if self._has_valid_human_joints[env_id]:
                for human_joint in self._debug_human_joints_world[env_id]:
                    self._draw_sphere_safe(human_joint, 0.025, (1.0, 0.55, 0.15), env_id)

            self._draw_tracking_target_vis(env_id)

            # Draw local_ref_ball_position (cyan sphere) - convert from local to world coordinates
            if visualize_debug_object_markers and hasattr(self, '_obs_local_ref_object_root_pos'):
                local_ref_ball_pos = self._obs_local_ref_object_root_pos[env_id]
                # Convert local to world: need heading rotation and robot root position
                # Add batch dimension for calc_heading_quat_inv (expects [batch, 4])
                robot_quat = self.simulator.robot_root_states[env_id:env_id+1, 3:7].clone()
                heading_inv_rot = calc_heading_quat_inv(robot_quat, w_last=True)
                heading_rot = quat_conjugate(heading_inv_rot, w_last=True)
                robot_root_pos = self.simulator.robot_root_states[env_id, :3]
                # Transform: world_pos = rotate(local_pos) + robot_root_pos
                world_ref_ball_pos = my_quat_rotate(heading_rot, local_ref_ball_pos.unsqueeze(0)).squeeze(0) + robot_root_pos
                self._draw_sphere_safe(world_ref_ball_pos, 0.10, (0.0, 1.0, 1.0), env_id)  # cyan color

    def _update_proj(self):

        if not hasattr(self, '_proj_follow_mouse'):
            self._proj_follow_mouse = True
            self._proj_follow_z = 1.0  # Default z height
            self.proj_speed = 5

        if self._proj_follow_mouse:
                cam_pose = self.simulator.gym.get_viewer_camera_transform(self.viewer, None)
                cam_fwd = cam_pose.r.rotate(gymapi.Vec3(0, 0, 1))

                spawn = cam_pose.p
                vel = cam_fwd * self.proj_speed

                angvel = 1.57 - 3.14 * np.random.random(3)

                self.simulator._proj_states[..., 0] = spawn.x + cam_fwd.x*2.5
                self.simulator._proj_states[..., 1] = spawn.y + cam_fwd.y*2.5
                self.simulator._proj_states[..., 2] = spawn.z + cam_fwd.z*2.5
                self.simulator._proj_states[..., 7] = 0
                self.simulator._proj_states[..., 8] = 0
                self.simulator._proj_states[..., 9] = 0

                self.simulator.gym.set_actor_root_state_tensor_indexed(self.simulator.sim, gymtorch.unwrap_tensor(self.simulator.all_root_states),
                                                        gymtorch.unwrap_tensor(self.simulator._proj_actor_ids),
                                                        len(self.simulator._proj_actor_ids))


        # mouse control
        evts = list(self.simulator.gym.query_viewer_action_events(self.viewer))
        for evt in evts:
            if evt.action == "zero_command":
                self._proj_follow_mouse = True
            elif evt.action == "reset" and evt.value > 0:
                self.simulator.gym.set_sim_rigid_body_states(self.simulator.sim, self.simulator._proj_states, gymapi.STATE_ALL)
            elif evt.action == "left_command" and evt.value > 0:
                self.proj_speed += -1
            elif evt.action == "right_command" and evt.value > 0:
                self.proj_speed += 1
            elif (evt.action == "space_shoot" or evt.action == "mouse_shoot") and evt.value > 0:
                if evt.action == "mouse_shoot":
                    pos = self.simulator.gym.get_viewer_mouse_position(self.viewer)
                    window_size = self.simulator.gym.get_viewer_size(self.viewer)
                    xcoord = round(pos.x * window_size.x)
                    ycoord = round(pos.y * window_size.y)
                    print(f"Fired projectile with mouse at coords: {xcoord} {ycoord}")

                cam_pose = self.simulator.gym.get_viewer_camera_transform(self.viewer, None)
                cam_fwd = cam_pose.r.rotate(gymapi.Vec3(0, 0, 1))

                spawn = cam_pose.p
                # self.proj_speed = 25
                vel = cam_fwd * self.proj_speed

                # Print launch velocity every time we shoot a projectile (split XY and Z)
                v_xy = float(np.sqrt(vel.x * vel.x + vel.y * vel.y))
                v_z = float(vel.z)
                pos_z = float(spawn.z + cam_fwd.z*2.5)
                print(
                    f"[Projectile] launch v_xy={v_xy:.3f}, pos_z={pos_z:.3f}, v_z={v_z:.3f}, "
                    f"(vel=({vel.x:.3f}, {vel.y:.3f}, {vel.z:.3f}), proj_speed={self.proj_speed})"
                )

                angvel = 1.57 - 3.14 * np.random.random(3)

                self.simulator._proj_states[..., 0] = spawn.x + cam_fwd.x*2.5
                self.simulator._proj_states[..., 1] = spawn.y + cam_fwd.y*2.5
                self.simulator._proj_states[..., 2] = spawn.z + cam_fwd.z*2.5
                self.simulator._proj_states[..., 7] = vel.x
                self.simulator._proj_states[..., 8] = vel.y
                self.simulator._proj_states[..., 9] = vel.z

                self.simulator.gym.set_actor_root_state_tensor_indexed(self.simulator.sim, gymtorch.unwrap_tensor(self.simulator.all_root_states),
                                                        gymtorch.unwrap_tensor(self.simulator._proj_actor_ids),
                                                        len(self.simulator._proj_actor_ids))
                self._proj_follow_mouse = False

        return



    def _hoi_post_physics_step(self):

        if self.config.robot.proj:
            self._update_proj()


        self._refresh_sim_tensors()
        self.episode_length_buf += 1
        if self._visualize_motion_data_only:
            self._apply_motion_frame_to_sim()
        # update counters
        self._update_counters_each_step()
        self.last_episode_length_buf = self.episode_length_buf.clone()

        self._pre_compute_observations_callback()
        self._update_tasks_callback()
        # compute observations, rewards, resets, ...
        self._check_termination()
        self._compute_reward()
        if getattr(self.config, "enable_rsi", False):
            self._rsi_episode_return += self.rew_buf
            self._rsi_episode_step += 1
        # check terminations
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_envs_idx(env_ids)

        # set envs
        refresh_env_ids = self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
        if len(refresh_env_ids) > 0:
            # NOTE: must use actor ids (actors_per_env), NOT `num_of_asset` (asset count).
            robot_actor_ids = self._get_actor_ids_by_env_ids(refresh_env_ids, self.simulator.robot_asset_idx)
            ball_actor_ids = self._get_actor_ids_by_env_ids(refresh_env_ids, self.simulator.ball_asset_idx)
            actor_ids = [robot_actor_ids, ball_actor_ids]
            if hasattr(self.simulator, "support_asset_idx"):
                actor_ids.append(self._get_actor_ids_by_env_ids(refresh_env_ids, self.simulator.support_asset_idx))
            asset_set_id = torch.cat(actor_ids, dim=0).long().detach()
            #set states
            self.simulator.set_actor_root_state_tensor(asset_set_id, self.simulator.all_root_states)
            self.simulator.set_dof_state_tensor(robot_actor_ids, self.simulator.dof_state)
            #update refresh buffers
            self.need_to_refresh_envs[refresh_env_ids] = False
            # Critical: many HOI observations are cached in `_pre_compute_observations_callback()`.
            # Since reset happens after the first pre-compute, we must refresh tensors and pre-compute again
            # to avoid feeding "fallen" (pre-reset) cached obs into policy on the first post-reset step.
            self._refresh_sim_tensors()
            self._pre_compute_observations_callback()

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
            if self.debug_viz:
                self._draw_debug_vis()

    def reset_all(self):
        """ Reset all robots"""
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))
        # #set robot
        # robot_asset_set_id = torch.arange(self.num_envs, device=self.device) * self.simulator.num_of_asset + self.simulator.robot_asset_idx
        # robot_asset_set_id = robot_asset_set_id.long().detach()
        # self.simulator.set_actor_root_state_tensor(robot_asset_set_id, self.simulator.all_root_states)
        # self.simulator.set_dof_state_tensor(robot_asset_set_id, self.simulator.dof_state)
        # #set object
        # ball_asset_set_id = torch.arange(self.num_envs, device=self.device) * self.simulator.num_of_asset + self.simulator.ball_asset_idx
        # ball_asset_set_id = ball_asset_set_id.long().detach()
        # self.simulator.set_actor_root_state_tensor(ball_asset_set_id, self.simulator.all_root_states)
        
        #set robot & object
        # all actors in all envs
        asset_set_id = torch.arange(int(self.simulator.all_root_states.shape[0]), device=self.device)
        self.simulator.set_actor_root_state_tensor(asset_set_id, self.simulator.all_root_states)
        robot_actor_ids = self._get_actor_ids_by_env_ids(torch.arange(self.num_envs, device=self.device), self.simulator.robot_asset_idx)
        self.simulator.set_dof_state_tensor(robot_actor_ids, self.simulator.dof_state)

        # self._refresh_env_idx_tensors(torch.arange(self.num_envs, device=self.device))
        actions = torch.zeros(self.num_envs, self.dim_actions, device=self.device, requires_grad=False)
        actor_state = {}
        actor_state["actions"] = actions
        obs_dict, _, _, _ = self.step(actor_state)
        return obs_dict
           
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
        if self._visualize_motion_data_only and target_states is None:
            self._apply_motion_frame_to_sim(env_ids=env_ids)
        # self._reset_object_states_callback(env_ids)
        
        # fill extras
        self._fill_episode_reward_extras(env_ids)
        # Contact error rates within the same cg/object-related time window.
        self.extras["episode"].update(
            summarize_contact_error_metrics(
                window_count=self._contact_window_count[env_ids],
                total_error_count=self._contact_error_count[env_ids],
                forbidden_contact_error_count=self._forbidden_contact_error_count[env_ids],
                object_contact_mismatch_error_count=self._object_contact_mismatch_error_count[env_ids],
                missed_object_contact_error_count=self._missed_object_contact_error_count[env_ids],
                unexpected_object_contact_error_count=self._unexpected_object_contact_error_count[env_ids],
            )
        )
        self._contact_error_count[env_ids] = 0.0
        self._forbidden_contact_error_count[env_ids] = 0.0
        self._object_contact_mismatch_error_count[env_ids] = 0.0
        self._missed_object_contact_error_count[env_ids] = 0.0
        self._unexpected_object_contact_error_count[env_ids] = 0.0
        self._contact_window_count[env_ids] = 0.0
        self.extras["time_outs"] = self.time_out_buf
        # self._refresh_sim_tensors()
        
    def _reset_object_states_callback(self, env_ids, target_states=None):
        # if target_states is not None, reset to target states
        if target_states is not None:
            raise NotImplementedError
        else:
            self._reset_object_states(env_ids)
            
    def _reset_object_states(self, env_ids): #wyh
        
        motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
        offset = self.env_origins
        now_time = self.episode_length_buf * self.dt + self.motion_start_times 
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, now_time, offset=offset)
        self._reset_support_asset_states(env_ids, motion_res)

        obj_pos_noise = self.config.init_noise_scale.obj_pos * self.config.noise_to_initial_level
        root_pos = motion_res['curr_obj_pos'][env_ids]# + 1
        root_rot = motion_res['curr_obj_rot'][env_ids]
        root_vel = motion_res['curr_obj_vel'][env_ids]
        root_ang_vel = motion_res['curr_obj_ang_vel'][env_ids]
        root_pos = self._apply_init_obj_pos_offset(root_pos)

        self.simulator.object_root_state[env_ids, :3] = apply_xy_only_init_noise(root_pos, obj_pos_noise)
        self.simulator.object_root_state[env_ids, 3:7] = root_rot
        if getattr(self.config, "zero_obj_vel_init", False):
            self.simulator.object_root_state[env_ids, 7:10] = 0.0
        else:
            self.simulator.object_root_state[env_ids, 7:10] = root_vel

        if getattr(self.config, "zero_obj_ang_vel_init", False):
            self.simulator.object_root_state[env_ids, 10:13] = 0.0
        else:
            self.simulator.object_root_state[env_ids, 10:13] = root_ang_vel
        
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
            now_time = self.episode_length_buf * self.dt + self.motion_start_times 
            offset = self.env_origins
            motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, now_time, offset=offset)
            root_pos = motion_res['root_pos'][env_ids]
            root_rot = motion_res['root_rot'][env_ids]
            root_pos_offset = getattr(self.config, "init_root_pos_offset", None)
            if root_pos_offset is not None:
                root_pos_offset = torch.as_tensor(list(root_pos_offset), device=self.device, dtype=root_pos.dtype)
                if root_pos_offset.numel() != 3:
                    raise ValueError(f"init_root_pos_offset must have 3 values, got {root_pos_offset.numel()}")
                root_pos = root_pos + root_pos_offset.view(1, 3)

            yaw_offset_deg = float(getattr(self.config, "init_root_yaw_offset_deg", 0.0) or 0.0)
            if yaw_offset_deg != 0.0:
                zeros = torch.zeros(root_rot.shape[0], device=self.device, dtype=root_rot.dtype)
                yaw = torch.full_like(zeros, yaw_offset_deg * np.pi / 180.0)
                root_rot = quat_mul(quat_from_euler_xyz(zeros, zeros, yaw), root_rot, w_last=True)

            self.simulator.robot_root_states[env_ids, :3] = root_pos
            # self.robot_root_states[env_ids, 2] += 0.04 # in case under the terrain
            if self.config.simulator.config.name == 'isaacgym':
                self.simulator.robot_root_states[env_ids, 3:7] = root_rot
            elif self.config.simulator.config.name == 'isaacsim':
                self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(root_rot)
            elif self.config.simulator.config.name == 'genesis':
                self.simulator.robot_root_states[env_ids, 3:7] = root_rot
                raise NotImplementedError
            self._capture_reference_command_heading_state(
                env_ids,
                root_rot,
                motion_res['root_rot'][env_ids],
                self._sample_human_root_quat_at_times(
                    self.motion_ids[env_ids],
                    now_time[env_ids],
                ),
            )
            self.simulator.robot_root_states[env_ids, 7:10] = torch.zeros_like(motion_res['root_vel'][env_ids]) #motion_res['root_vel'][env_ids]
            self.simulator.robot_root_states[env_ids, 10:13] = torch.zeros_like(motion_res['root_ang_vel'][env_ids]) #motion_res['root_ang_vel'][env_ids]
            

        else:
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
            now_time = self.episode_length_buf * self.dt + self.motion_start_times 
            offset = self.env_origins

            # Apply reset-time state delay offsets (generated in _reset_buffers_callback).
            # Delay direction: subtract time -> sample an EARLIER motion phase for initialization.
            # This makes the simulated initial state "lag behind" the reference timeline used during rollout.
            off_r = self._state_delay_init_offset_s_robot[env_ids]
            off_o = self._state_delay_init_offset_s_object[env_ids]

            # sample motion state (robot / object can be decoupled)
            if torch.any(off_r != off_o):
                motion_times_r = motion_times.clone()
                now_time_r = now_time.clone()
                motion_times_o = motion_times.clone()
                now_time_o = now_time.clone()

                motion_times_r[env_ids] = torch.clamp(motion_times_r[env_ids] - off_r, min=0.0)
                now_time_r[env_ids] = torch.clamp(now_time_r[env_ids] - off_r, min=0.0)
                motion_times_o[env_ids] = torch.clamp(motion_times_o[env_ids] - off_o, min=0.0)
                now_time_o[env_ids] = torch.clamp(now_time_o[env_ids] - off_o, min=0.0)

                motion_res_robot = self._motion_lib.get_motion_state(self.motion_ids, motion_times_r, now_time_r, offset=offset)
                motion_res_obj = self._motion_lib.get_motion_state(self.motion_ids, motion_times_o, now_time_o, offset=offset)
            else:
                motion_times[env_ids] = torch.clamp(motion_times[env_ids] - off_r, min=0.0)
                now_time[env_ids] = torch.clamp(now_time[env_ids] - off_r, min=0.0)
                motion_res_robot = motion_res_obj = self._motion_lib.get_motion_state(self.motion_ids, motion_times, now_time, offset=offset)
            self._reset_support_asset_states(env_ids, motion_res_obj)

            #robot
            root_pos_noise = self.config.init_noise_scale.root_pos * self.config.noise_to_initial_level
            root_rot_noise = self.config.init_noise_scale.root_rot * 3.14 / 180 * self.config.noise_to_initial_level
            # optional: yaw-only init noise (degree in config, converted to rad here). Keep backward compatible.
            root_rot_yaw_noise = getattr(self.config.init_noise_scale, "root_rot_yaw", 0.0) * 3.14 / 180 * self.config.noise_to_initial_level
            root_vel_noise = self.config.init_noise_scale.root_vel * self.config.noise_to_initial_level
            root_ang_vel_noise = self.config.init_noise_scale.root_ang_vel * self.config.noise_to_initial_level

            root_pos = motion_res_robot['root_pos'][env_ids]
            root_rot = motion_res_robot['root_rot'][env_ids]
            root_vel = motion_res_robot['root_vel'][env_ids]
            root_ang_vel = motion_res_robot['root_ang_vel'][env_ids]

            root_pos_offset = getattr(self.config, "init_root_pos_offset", None)
            if root_pos_offset is not None:
                root_pos_offset = torch.as_tensor(list(root_pos_offset), device=self.device, dtype=root_pos.dtype)
                if root_pos_offset.numel() != 3:
                    raise ValueError(f"init_root_pos_offset must have 3 values, got {root_pos_offset.numel()}")
                root_pos = root_pos + root_pos_offset.view(1, 3)

            self.simulator.robot_root_states[env_ids, :3] = root_pos + torch.randn_like(root_pos) * root_pos_noise

            # apply rotation noises (root_rot: full 3D small rotation; root_rot_yaw: yaw-only), both can coexist
            noisy_root_rot = root_rot
            if root_rot_noise != 0.0:
                noisy_root_rot = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), noisy_root_rot, w_last=True)
            if root_rot_yaw_noise != 0.0:
                noisy_root_rot = quat_mul(self.small_random_yaw_quaternions(root_rot.shape[0], root_rot_yaw_noise), noisy_root_rot, w_last=True)
            yaw_offset_deg = float(getattr(self.config, "init_root_yaw_offset_deg", 0.0) or 0.0)
            if yaw_offset_deg != 0.0:
                zeros = torch.zeros(root_rot.shape[0], device=self.device, dtype=root_rot.dtype)
                yaw = torch.full_like(zeros, yaw_offset_deg * np.pi / 180.0)
                noisy_root_rot = quat_mul(quat_from_euler_xyz(zeros, zeros, yaw), noisy_root_rot, w_last=True)

            # # Add different yaw rotations to different environments using linspace
            # if len(env_ids) > 1:
            #     # Create linearly spaced yaw angles from 0 to 2π for different environments
            #     yaw_angles = torch.linspace(-3.14159/2, 3.14159/2, len(env_ids), device=self.device)
            #     # Create quaternions for yaw rotations
            #     yaw_quats = torch.zeros(len(env_ids), 4, device=self.device)
            #     yaw_quats[:, 3] = torch.cos(yaw_angles / 2)  # w component
            #     yaw_quats[:, 2] = torch.sin(yaw_angles / 2)  # z component (yaw axis)
            #     # Apply yaw rotation to the noisy root rotation
            #     noisy_root_rot = quat_mul(yaw_quats, noisy_root_rot, w_last=True)

            if self.config.simulator.config.name == 'isaacgym':
                self.simulator.robot_root_states[env_ids, 3:7] = noisy_root_rot
            elif self.config.simulator.config.name == 'isaacsim':
                self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(noisy_root_rot)
            elif self.config.simulator.config.name == 'genesis':
                self.simulator.robot_root_states[env_ids, 3:7] = noisy_root_rot
            elif self.config.simulator.config.name == 'mujoco':
                self.simulator.robot_root_states[env_ids, 3:7] = noisy_root_rot
            else:
                raise NotImplementedError
            align_base_quat = self._get_reference_command_alignment_base_quat(
                clean_root_rot=root_rot,
                noisy_root_rot=noisy_root_rot,
            )
            self._capture_reference_command_heading_state(
                env_ids,
                align_base_quat,
                root_rot,
                self._sample_human_root_quat_at_times(
                    self.motion_ids[env_ids],
                    now_time[env_ids],
                ),
            )
            
            
            #wyh
            if self.config.zero_root_vel_init:
                self.simulator.robot_root_states[env_ids, 7:10] = torch.zeros_like(root_vel) + torch.randn_like(root_vel) * root_vel_noise #root_vel + torch.randn_like(root_vel) * root_vel_noise
                self.simulator.robot_root_states[env_ids, 10:13] = torch.zeros_like(root_ang_vel) + torch.randn_like(root_ang_vel) * root_ang_vel_noise #root_ang_vel + torch.randn_like(root_ang_vel) * root_ang_vel_noise
                
            else:
                self.simulator.robot_root_states[env_ids, 7:10] = root_vel + torch.randn_like(root_vel) * root_vel_noise
                self.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel + torch.randn_like(root_ang_vel) * root_ang_vel_noise

            obj_pos_noise = self.config.init_noise_scale.obj_pos * self.config.noise_to_initial_level
            obj_root_pos = motion_res_obj['curr_obj_pos'][env_ids] 
            obj_root_rot = motion_res_obj['curr_obj_rot'][env_ids]
            obj_root_vel = motion_res_obj['curr_obj_vel'][env_ids]
            obj_root_ang_vel = motion_res_obj['curr_obj_ang_vel'][env_ids]
            obj_root_pos = self._apply_init_obj_pos_offset(obj_root_pos)

            self.simulator.object_root_state[env_ids, :3] = apply_xy_only_init_noise(obj_root_pos, obj_pos_noise)
            # bias = torch.zeros_like(obj_pos_noise_tensor)
            # bias[:, 0] = 1.0
            # bias[:, 1] = 0.
            # self.simulator.object_root_state[env_ids, :3] += bias
            # ###########
            # # Sample object position in a semicircle in the direction of robot's initial heading
            # robot_root_pos = self.simulator.robot_root_states[env_ids, :3]
            # robot_root_pos[:, 2] = obj_root_pos[:, 2]
            # robot_root_quat = self.simulator.robot_root_states[env_ids, 3:7]
            # # Get robot's heading direction
            # forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(len(env_ids), 1)
            # robot_forward = quat_apply(robot_root_quat, forward_vec)
            # robot_heading = torch.atan2(robot_forward[:, 1], robot_forward[:, 0])  # [batch_size]
            # # Generate random angles in semicircle relative to robot heading: [heading - π/2, heading + π/2]
            # angles = torch.rand(len(env_ids), device=self.device) * torch.pi - torch.pi / 2 + robot_heading  # [-π/2 + heading, π/2 + heading]
            # radii = torch.sqrt(torch.rand(len(env_ids), device=self.device)) * 3.
            # # Convert polar to cartesian coordinates
            # x_offset = radii * torch.cos(angles)
            # y_offset = radii * torch.sin(angles)
            # z_offset = torch.zeros_like(x_offset)  # Keep z unchanged
            # circular_offset = torch.stack([x_offset, y_offset, z_offset], dim=1)
            # self.simulator.object_root_state[env_ids, :3] = robot_root_pos + circular_offset
            # ############

            # self.simulator.object_root_state[env_ids, 0] +=  1
            self.simulator.object_root_state[env_ids, 3:7] = obj_root_rot
            if getattr(self.config, "zero_obj_vel_init", False): # wyh
                self.simulator.object_root_state[env_ids, 7:10] = 0 
            else:
                self.simulator.object_root_state[env_ids, 7:10] = obj_root_vel
            if getattr(self.config, "zero_obj_ang_vel_init", False):
                self.simulator.object_root_state[env_ids, 10:13] = 0.0
            else:
                self.simulator.object_root_state[env_ids, 10:13] = obj_root_ang_vel

            # Optional: per-env object freeze (actor-level).
            # If motion data provides `obj_freeze` (0/1), we freeze those envs by forcing root state each physics step.
            obj_freeze = motion_res_obj.get("obj_freeze", None)
            if obj_freeze is None:
                self._freeze_obj_buf[env_ids] = False
            else:
                if obj_freeze.ndim > 1:
                    obj_freeze = obj_freeze.squeeze(-1)
                self._freeze_obj_buf[env_ids] = obj_freeze[env_ids].to(torch.bool)

            freeze_env_ids = env_ids[self._freeze_obj_buf[env_ids]]
            if freeze_env_ids.numel() > 0:
                # For frozen envs: no init noise, and zero velocities.
                freeze_obj_pos = self._apply_init_obj_pos_offset(motion_res_obj["curr_obj_pos"][freeze_env_ids])
                self.simulator.object_root_state[freeze_env_ids, :3] = freeze_obj_pos
                self.simulator.object_root_state[freeze_env_ids, 3:7] = motion_res_obj["curr_obj_rot"][freeze_env_ids]
                self.simulator.object_root_state[freeze_env_ids, 7:13] = 0.0
                self._frozen_obj_root_state[freeze_env_ids] = self.simulator.object_root_state[freeze_env_ids].clone()

            # Sync frozen actors to simulator (global list) so physics-step hook can enforce it.
            # Backward compatible across simulator backends: only IsaacGym HOI simulator implements these methods.
            if hasattr(self.simulator, "set_frozen_object") and hasattr(self.simulator, "clear_frozen_object"):
                all_freeze_env_ids = self._freeze_obj_buf.nonzero(as_tuple=False).flatten()
                if all_freeze_env_ids.numel() == 0:
                    self.simulator.clear_frozen_object()
                else:
                    ball_actor_ids = self._get_actor_ids_by_env_ids(all_freeze_env_ids, self.simulator.ball_asset_idx)
                    frozen_states = self._frozen_obj_root_state[all_freeze_env_ids]
                    self.simulator.set_frozen_object(ball_actor_ids, frozen_states)
            
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
        now_time = self.episode_length_buf * self.dt + self.motion_start_times 
        offset = self.env_origins
        # Apply robot reset-time delay to DOFs as well (keeps robot state internally consistent).
        off_r = self._state_delay_init_offset_s_robot[env_ids]
        motion_times[env_ids] = torch.clamp(motion_times[env_ids] - off_r, min=0.0)
        now_time[env_ids] = torch.clamp(now_time[env_ids] - off_r, min=0.0)
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, now_time, offset=offset)

        dof_pos_noise = self.config.init_noise_scale.dof_pos * self.config.noise_to_initial_level
        dof_vel_noise = self.config.init_noise_scale.dof_vel * self.config.noise_to_initial_level
        dof_pos = motion_res['dof_pos'][env_ids]
        dof_vel = motion_res['dof_vel'][env_ids]
        self.simulator.dof_pos[env_ids] = dof_pos + torch.randn_like(dof_pos) * dof_pos_noise #* 10 #wyh
        if self.config.zero_root_vel_init:
            self.simulator.dof_vel[env_ids] = torch.zeros_like(dof_vel) + torch.randn_like(dof_vel) * dof_vel_noise #dof_vel + torch.randn_like(dof_vel) * dof_vel_noise
        else:
            self.simulator.dof_vel[env_ids] = dof_vel + torch.randn_like(dof_vel) * dof_vel_noise


    def _post_physics_step(self):
        # super()._post_physics_step()
        self._hoi_post_physics_step()
        
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
            
    def _setup_robot_body_indices(self):
        
        #setup feet indices
        feet_names = [s for s in self.body_names if self.config.robot.foot_name in s] 
        
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.simulator.find_rigid_body_indice(feet_names[i])

        #wyh
        lfoot_name = self.config.robot.left_foot_name
        rfoot_name = self.config.robot.right_foot_name
        self.lfoot_indice = torch.zeros(len(lfoot_name), dtype=torch.long, device=self.device, requires_grad=False)
        self.rfoot_indice = torch.zeros(len(rfoot_name), dtype=torch.long, device=self.device, requires_grad=False)
        self.lfoot_indice = self.simulator.find_rigid_body_indice(lfoot_name[0])
        self.rfoot_indice = self.simulator.find_rigid_body_indice(rfoot_name[0])

        #setup knee indices
        knee_names = [s for s in self.body_names if self.config.robot.knee_name in s]
        
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.simulator.find_rigid_body_indice(knee_names[i])
        #setup penalized contact indices
        penalized_contact_names = []
        for name in self.config.robot.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
            
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            
            self.penalised_contact_indices[i] = self.simulator.find_rigid_body_indice(penalized_contact_names[i])
        #setup termination contact indices
        termination_contact_names = []
        for name in self.config.robot.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])
            
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.simulator.find_rigid_body_indice(termination_contact_names[i])
        
        #setup key bodies indices
        key_body_names = []
        for name in self.config.robot.key_body_names:
            key_body_names.extend([s for s in self.body_names if name in s])
        self.key_body_indices = torch.zeros(len(key_body_names), dtype=torch.long, device=self.device, requires_grad=False)
        
        for i in range(len(key_body_names)):
            self.key_body_indices[i] = self.simulator.find_rigid_body_indice(key_body_names[i])
            
        #setup hoi contact indices
        hoi_contact_names = []
        for name in self.config.robot.hoi_contact_bodies:
            hoi_contact_names.extend([s for s in self.body_names if name in s])
            
        self.hoi_contact_indices = torch.zeros(len(hoi_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hoi_contact_names)):
            self.hoi_contact_indices[i] = self.simulator.find_rigid_body_indice(hoi_contact_names[i])
            
        #setup hoi no contact indices
        hoi_no_contact_names = []
        for name in self.config.robot.hoi_no_contact_bodies:
            hoi_no_contact_names.extend([s for s in self.body_names if name in s])
            
        self.hoi_no_contact_indices = torch.zeros(len(hoi_no_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hoi_no_contact_names)):
            self.hoi_no_contact_indices[i] = self.simulator.find_rigid_body_indice(hoi_no_contact_names[i])
        
        #setup hoi contact indices
        hoi_interaction_bodies_name = []
        for name in self.config.robot.hoi_interaction_bodies:
            hoi_interaction_bodies_name.extend([s for s in self.body_names if name in s])
        
        self.hoi_interaction_bodies_indices = torch.zeros(len(hoi_interaction_bodies_name), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hoi_interaction_bodies_name)):
            self.hoi_interaction_bodies_indices[i] = self.simulator.find_rigid_body_indice(hoi_interaction_bodies_name[i])
        
        #misc
        if self.config.robot.has_upper_body_dof:
            # maintain upper/lower dof idxs
            self.upper_dof_names = self.config.robot.upper_dof_names
            self.lower_dof_names = self.config.robot.lower_dof_names
            self.upper_dof_indices = [self.dof_names.index(dof) for dof in self.upper_dof_names]
            self.lower_dof_indices = [self.dof_names.index(dof) for dof in self.lower_dof_names]
            self.waist_dof_indices = [self.dof_names.index(dof) for dof in self.config.robot.waist_dof_names]

        if self.config.robot.has_torso:
            self.torso_name = self.config.robot.torso_name
            self.torso_index = self.simulator.find_rigid_body_indice(self.torso_name)

    def setup_ind_for_amp(self):
        #amp selected keypos
        amp_keypos_selected_names = []
        for name in self.config.robot.amp_keypos_selected:
            amp_keypos_selected_names.extend([s for s in self.simulator._body_list if name in s])
            
        self.amp_selected_keypos_indices = torch.tensor(
            [self.simulator._body_list.index(pos) for pos in amp_keypos_selected_names],
            dtype = torch.long,
            device = self.device,
            requires_grad = False,
        )
        
        #amp selected dof
        amp_dof_selected_names = []
        for name in self.config.robot.amp_dof_selected:
            amp_dof_selected_names.extend([s for s in self.dof_names if name in s])
            
        self.amp_selected_dof_indices = torch.tensor(
            [self.dof_names.index(dof) for dof in amp_dof_selected_names],
            dtype = torch.long,
            device = self.device,
            requires_grad = False,
        )

    # ############################################################
        
    def _get_obs_dif_local_rigid_body_pos(self):
        return self._obs_dif_local_rigid_body_pos
    
    def _get_obs_local_ref_rigid_body_pos(self):
        return self._obs_local_ref_rigid_body_pos
        
    def _get_obs_vr_3point_pos(self):
        return self._obs_vr_3point_pos

    # -------- OmniRetarget minimal proprioceptive obs getters --------
    def _get_obs_ref_dof_pos(self):
        return self._obs_ref_dof_pos

    def _get_obs_ref_dof_vel(self):
        return self._obs_ref_dof_vel

    def _get_obs_pelvis_pos_err(self):
        return self._obs_pelvis_pos_err

    def _get_obs_pelvis_rot_err(self):
        return self._obs_pelvis_rot_err

    def _get_obs_pelvis_rot_err_rot6d(self):
        return self._obs_pelvis_rot_err_rot6d
    
    def _get_obs_local_curr_ball_position(self):
        # Optional: treat projectile as the "object" observation fed to policy/critic.
        # Backward-compatible: only enabled when `obs.use_projectile_as_obj_obs: true`.
        use_proj = bool(getattr(getattr(self.config, "obs", None), "use_projectile_as_obj_obs", False))
        use_proj = use_proj and bool(getattr(getattr(self.config, "robot", None), "proj", False))
        use_proj = use_proj and hasattr(self.simulator, "_proj_states")

        if use_proj:
            # print("projectile position", self.simulator._proj_states[:, 0, 0:3])
            # self.simulator._proj_states: (num_envs, num_projectors, 13) view into all_root_states
            proj_pos_world = self.simulator._proj_states[:, 0, 0:3]
            obs = self._world_pos_to_heading_local(proj_pos_world)
        else:
            obs = self._obs_local_ball_root_pos.clone()

        if self._packet_loss_enabled:
            mask = (self.packet_loss_timer > 0).unsqueeze(-1)
            obs = torch.where(mask, torch.zeros_like(obs), obs)
        return obs
    
    def _get_obs_local_curr_ball_position_wo_packet_loss(self):
        return self._obs_local_curr_ball_position_wo_packet_loss

    def _get_obs_local_ball_root_vel(self):
        use_proj = bool(getattr(getattr(self.config, "obs", None), "use_projectile_as_obj_obs", False))
        use_proj = use_proj and bool(getattr(getattr(self.config, "robot", None), "proj", False))
        use_proj = use_proj and hasattr(self.simulator, "_proj_states")

        if use_proj:
            proj_vel_world = self.simulator._proj_states[:, 0, 7:10]
            heading_inv_rot = calc_heading_quat_inv(self.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
            global_vel = proj_vel_world - self.simulator.robot_root_states[:, 7:10]
            obs = my_quat_rotate(heading_inv_rot, global_vel)
        else:
            obs = self._obs_local_ball_root_vel.clone()

        if self._packet_loss_enabled:
            mask = (self.packet_loss_timer > 0).unsqueeze(-1)
            obs = torch.where(mask, torch.zeros_like(obs), obs)
        return obs

    def _get_obs_local_ball_root_rot(self):
        use_proj = bool(getattr(getattr(self.config, "obs", None), "use_projectile_as_obj_obs", False))
        use_proj = use_proj and bool(getattr(getattr(self.config, "robot", None), "proj", False))
        use_proj = use_proj and hasattr(self.simulator, "_proj_states")

        if use_proj:
            proj_rot_world = self.simulator._proj_states[:, 0, 3:7]
            heading_inv_rot = calc_heading_quat_inv(self.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
            obs = quat_mul(heading_inv_rot, proj_rot_world, w_last=True)
        else:
            obs = self._obs_local_ball_root_rot.clone()

        if self._packet_loss_enabled:
            mask = (self.packet_loss_timer > 0).unsqueeze(-1)
            obs = torch.where(mask, torch.zeros_like(obs), obs)
        return obs

    def _get_obs_local_ball_root_ang_vel(self):
        use_proj = bool(getattr(getattr(self.config, "obs", None), "use_projectile_as_obj_obs", False))
        use_proj = use_proj and bool(getattr(getattr(self.config, "robot", None), "proj", False))
        use_proj = use_proj and hasattr(self.simulator, "_proj_states")

        if use_proj:
            proj_angvel_world = self.simulator._proj_states[:, 0, 10:13]
            heading_inv_rot = calc_heading_quat_inv(self.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
            obs = my_quat_rotate(heading_inv_rot, proj_angvel_world)
        else:
            obs = self._obs_local_ball_root_ang_vel.clone()

        if self._packet_loss_enabled:
            mask = (self.packet_loss_timer > 0).unsqueeze(-1)
            obs = torch.where(mask, torch.zeros_like(obs), obs)
        return obs
    
    def _get_obs_pd_error(self):
        return self._obs_pd_error

    def _get_obs_phase(self):
        return self._obs_phase
    
    def _get_obs_basket_pos(self):
        return self.ref_basket_pos
    
    def _get_obs_skill_label(self):
        return self.skill_label

    def _get_obs_apex_height(self):
        return self._obs_apex_height
    
    def _get_obs_local_target_ball_position(self):
        return self._obs_local_target_ball_root_pos
    
    def _get_obs_local_ref_ball_position(self):
        # Optional: treat projectile as the "object" observation (same switch as local_curr_ball_position).
        use_proj = bool(getattr(getattr(self.config, "obs", None), "use_projectile_as_obj_obs", False))
        use_proj = use_proj and bool(getattr(getattr(self.config, "robot", None), "proj", False))
        use_proj = use_proj and hasattr(self.simulator, "_proj_states")

        if use_proj:
            proj_pos_world = self.simulator._proj_states[:, 0, 0:3]
            heading_inv_rot = calc_heading_quat_inv(self.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
            global_pos = proj_pos_world - self.simulator.robot_root_states[:, :3]
            return my_quat_rotate(heading_inv_rot, global_pos)

        return self._obs_local_ref_object_root_pos

    def _get_obs_reference_command(self):
        return self._obs_reference_command

    def _get_obs_curr_interaction_graph(self):
        return self.ig.view(self.num_envs, -1)
        
    def _get_obs_local_keybodies_pos(self):
        return self._obs_local_keybodies_pos[:, self.key_body_indices, :].view(self.num_envs, -1)
    
    def _get_obs_local_keybodies_vel(self):
        return self._obs_local_keybodies_vel[:, self.key_body_indices, :].view(self.num_envs, -1)
    
    def _get_obs_local_keybodies_rot(self):
        return self._obs_local_keybodies_rot[:, self.key_body_indices, :].view(self.num_envs, -1)
    
    def _get_obs_local_keybodies_ang_vel(self):
        return self._obs_local_keybodies_ang_vel[:, self.key_body_indices, :].view(self.num_envs, -1)
        
    def _get_obs_diff_ig(self):
        return self.dif_ig.view(self.num_envs, -1)
    
    def _get_obs_bodies_contact_force(self):
        return self.bodies_contact_force
    
    def _get_obs_local_amp_body_pos(self):
        return self._obs_local_keybodies_pos.view(self.num_envs, -1, 3)[:, self.amp_selected_keypos_indices].view(self.num_envs, -1)
    
    def _get_obs_local_amp_body_vel(self):
        return self._obs_local_keybodies_vel.view(self.num_envs, -1, 3)[:, self.amp_selected_keypos_indices].view(self.num_envs, -1)
    
    def _get_obs_amp_dof_pos(self):
        return (self.simulator.dof_pos - self.default_dof_pos)[:, self.amp_selected_dof_indices]
    
    def _get_obs_amp_dof_vel(self):
        return self.simulator.dof_vel[:, self.amp_selected_dof_indices]
    
    def _get_obs_local_amp_expert_body_pos(self):
        return self._obs_local_amp_expert_rigid_body_pos.view(self.num_envs, -1, 3)[:, self.amp_selected_keypos_indices].view(self.num_envs, -1)
    
    def _get_obs_local_amp_expert_body_vel(self):
        return self._obs_local_amp_expert_rigid_body_vel.view(self.num_envs, -1, 3)[:, self.amp_selected_keypos_indices].view(self.num_envs, -1)
    
    def _get_obs_amp_expert_dof_pos(self):
        return (self.amp_expert_dof_pos - self.default_dof_pos)[:, self.amp_selected_dof_indices]
    
    def _get_obs_amp_expert_dof_vel(self):
        return self.amp_expert_dof_vel[:, self.amp_selected_dof_indices]
    
    def _get_obs_local_ref_origin_pos_xy(self):
        return self._obs_local_ref_origin_pos[:, :2]
    
    def _get_obs_dif_ref_heading(self):
        
        root_forward = quat_apply(self.base_quat, self.forward_vec)
        heading_root = torch.atan2(root_forward[:, 1], root_forward[:, 0])
        
        ref_forward = quat_apply(self.ref_origin_rot, self.forward_vec)
        heading_ref = torch.atan2(ref_forward[:, 1], ref_forward[:, 0])
        
        heading_dif = wrap_to_pi(heading_root - heading_ref)
        
        return heading_dif.unsqueeze(1)

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
    
    def _get_obs_history_teacher(self,):
        assert "history_teacher" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_teacher']
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
    
    def _get_obs_history_amp(self,):
        assert "history_amp" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_amp']
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    def _get_obs_history_expert_amp(self,):
        assert "history_expert_amp" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_expert_amp']
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    ###############################################################
    def _reward_SM1_body_position_extend(self):
        upper_body_diff = self.dif_global_body_pos[:, self.upper_body_id, :]
        lower_body_diff = self.dif_global_body_pos[:, self.lower_body_id, :]

        diff_body_pos_dist_upper = (upper_body_diff**2).mean(dim=-1).mean(dim=-1)
        diff_body_pos_dist_lower = (lower_body_diff**2).mean(dim=-1).mean(dim=-1)

        r_body_pos_upper = torch.exp(-diff_body_pos_dist_upper / self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos)
        r_body_pos_lower = torch.exp(-diff_body_pos_dist_lower / self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)
        r_body_pos = r_body_pos_lower * r_body_pos_upper 
    
        return r_body_pos
    
    def _reward_teleop_body_position_extend(self):
        # Optional: BeyondMimic-style anchor-centered tracking for body positions.
        # Backward compatible: default False -> keep existing global tracking.
        use_anchor_centered = bool(getattr(self.config.rewards, "anchor_centered_body_position", False))
        dif_body_pos = self._get_anchor_centered_dif_global_body_pos() if use_anchor_centered else self.dif_global_body_pos

        upper_ids = self.upper_body_id
        lower_ids = self.lower_body_id
        if use_anchor_centered:
            # Typically anchor is torso/base. Exclude it from body-position tracking to avoid re-introducing
            # global drift penalties through the anchor itself (anchor tracking can be added separately if desired).
            anchor_idx = self._get_anchor_body_index()
            upper_ids = [i for i in upper_ids if i != anchor_idx]
            lower_ids = [i for i in lower_ids if i != anchor_idx]

        upper_body_diff = dif_body_pos[:, upper_ids, :] if len(upper_ids) > 0 else None
        lower_body_diff = dif_body_pos[:, lower_ids, :] if len(lower_ids) > 0 else None

        # If an id list becomes empty (unlikely), fall back to neutral reward 1.0 for that part.
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
        # dif_global_body_pos is (ref - curr) in world frame
        anchor_diff = self.dif_global_body_pos[:, anchor_idx, :]  # (N, 3)
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
        # dif_global_body_rot is relative quaternion error (ref * inv(curr)) in xyzw
        anchor_rot_diff = self.dif_global_body_rot[:, anchor_idx, :]  # (N, 4)
        diff_dist = (anchor_rot_diff ** 2).mean(dim=-1)
        sigma = getattr(self.config.rewards.reward_tracking_sigma, "teleop_anchor_rot",
                        self.config.rewards.reward_tracking_sigma.teleop_body_rot)
        return torch.exp(-diff_dist / sigma)

    # ---------------- Anchor-centered tracking helpers ----------------
    def _get_anchor_body_name(self) -> str:
        # Priority: rewards.anchor_body_name > robot.motion.base_link > robot.torso_name
        name = getattr(self.config.rewards, "anchor_body_name", None)
        if name:
            return str(name)
        name = getattr(getattr(self.config.robot, "motion", None), "base_link", None)
        if name:
            return str(name)
        name = getattr(self.config.robot, "torso_name", None)
        if name:
            return str(name)
        # Last resort (common in this repo)
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

        Backward compatible: only used when rewards.anchor_centered_body_position=True.

        Reference: BeyondMimic anchor transform (yaw-aligned, height-preserving) in
        [BeyondMimic paper](https://arxiv.org/html/2508.08241v4).
        """
        # Cache the full desired positions for this step (optional micro-optimization)
        cached = getattr(self, "_anchor_centered_desired_body_pos_cache", None)
        cached_step = getattr(self, "_anchor_centered_desired_body_pos_cache_step", None)
        step = int(self.common_step_counter) if hasattr(self, "common_step_counter") else (
            int(self.episode_length_buf[0].item()) if hasattr(self, "episode_length_buf") else None
        )
        if cached is not None and cached_step == step:
            desired = cached
        else:
            anchor_idx = self._get_anchor_body_index()
            ref_pos = self.ref_body_pos_extend  # (N, B, 3), world
            ref_rot = self.ref_body_rot_extend  # (N, B, 4), xyzw (w last)
            curr_pos = self._rigid_body_pos_extend  # (N, B, 3), world
            curr_rot = self._rigid_body_rot_extend  # (N, B, 4), xyzw (w last)

            pA_ref = ref_pos[:, anchor_idx, :]
            pA_cur = curr_pos[:, anchor_idx, :]
            qA_ref = ref_rot[:, anchor_idx, :]
            qA_cur = curr_rot[:, anchor_idx, :]

            # yaw(qA_cur * inv(qA_ref))
            q_rel = quat_mul(qA_cur, quat_conjugate(qA_ref, w_last=True), w_last=True)
            q_rel = q_rel / torch.norm(q_rel, dim=-1, keepdim=True).clamp(min=1e-9)
            rpy = get_euler_xyz_in_tensor(q_rel)  # roll, pitch, yaw
            yaw = rpy[:, 2]
            zeros = torch.zeros_like(yaw)
            q_yaw_delta = quat_from_euler_xyz(zeros, zeros, yaw)  # xyzw
            q_yaw_delta = q_yaw_delta / torch.norm(q_yaw_delta, dim=-1, keepdim=True).clamp(min=1e-9)

            # p_delta = [pA_cur.x, pA_cur.y, pA_ref.z]  (height-preserving)
            p_delta = torch.stack([pA_cur[:, 0], pA_cur[:, 1], pA_ref[:, 2]], dim=-1)

            # desired for all bodies: p_delta + R_yaw_delta * (p_ref - pA_ref)
            offset = ref_pos - pA_ref.unsqueeze(1)
            B = offset.shape[1]
            q_rep = q_yaw_delta.unsqueeze(1).expand(-1, B, -1).reshape(-1, 4)
            rot_offset = my_quat_rotate(q_rep, offset.reshape(-1, 3)).reshape(-1, B, 3)
            desired = p_delta.unsqueeze(1) + rot_offset

            # Anchor body itself: by default do NOT impose global tracking through this path.
            # (If needed, add a separate anchor-global reward term.)
            desired[:, anchor_idx, :] = pA_cur

            self._anchor_centered_desired_body_pos_cache = desired
            self._anchor_centered_desired_body_pos_cache_step = step

        return desired - self._rigid_body_pos_extend

    def _get_anchor_centered_dif_global_body_rot(self) -> torch.Tensor:
        """
        Compute BeyondMimic-style anchor-centered desired body orientations (world frame),
        then return quaternion error (desired * inv(current)) for all bodies in `_rigid_body_rot_extend`.

        We apply a yaw-only re-anchoring rotation derived from the anchor's current vs reference yaw.
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
            ref_rot = self.ref_body_rot_extend  # (N, B, 4), xyzw
            curr_rot = self._rigid_body_rot_extend  # (N, B, 4), xyzw

            qA_ref = ref_rot[:, anchor_idx, :]
            qA_cur = curr_rot[:, anchor_idx, :]

            # yaw(qA_cur * inv(qA_ref))
            q_rel = quat_mul(qA_cur, quat_conjugate(qA_ref, w_last=True), w_last=True)
            q_rel = q_rel / torch.norm(q_rel, dim=-1, keepdim=True).clamp(min=1e-9)
            rpy = get_euler_xyz_in_tensor(q_rel)
            yaw = rpy[:, 2]
            zeros = torch.zeros_like(yaw)
            q_yaw_delta = quat_from_euler_xyz(zeros, zeros, yaw)  # xyzw
            q_yaw_delta = q_yaw_delta / torch.norm(q_yaw_delta, dim=-1, keepdim=True).clamp(min=1e-9)

            B = ref_rot.shape[1]
            q_rep = q_yaw_delta.unsqueeze(1).expand(-1, B, -1).reshape(-1, 4)
            desired_rot = quat_mul(q_rep, ref_rot.reshape(-1, 4), w_last=True).reshape(-1, B, 4)

            # Anchor body itself: do not enforce global tracking through this path.
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
    
    def _reward_ball_position(self):

        ball_pos_diff = (self.dif_global_ball_pos**2).mean(dim=-1)
        ball_pos_diff_rew = torch.exp(-ball_pos_diff / self.config.rewards.reward_tracking_sigma.ball_position)
        return ball_pos_diff_rew
    
    def _remove_yaw_from_rel_quat(self, q_rel: torch.Tensor) -> torch.Tensor:
        """
        Remove yaw (rotation about world Z axis) from a *relative* quaternion (xyzw, w last).
        This is used to optionally ignore yaw error in ball rotation reward/termination.
        """
        q = q_rel / torch.norm(q_rel, dim=-1, keepdim=True).clamp(min=1e-9)
        rpy = get_euler_xyz_in_tensor(q)  # (N, 3): roll, pitch, yaw
        roll = rpy[:, 0]
        pitch = rpy[:, 1]
        zero_yaw = torch.zeros_like(roll)
        q_no_yaw = quat_from_euler_xyz(roll, pitch, zero_yaw)
        q_no_yaw = q_no_yaw / torch.norm(q_no_yaw, dim=-1, keepdim=True).clamp(min=1e-9)
        return q_no_yaw

    def _reward_ball_rotation(self):
        consider_yaw = getattr(self.config.rewards, "ball_rotation_consider_yaw", True)
        ball_rot_diff = self.dif_global_ball_rot if consider_yaw else self._remove_yaw_from_rel_quat(self.dif_global_ball_rot)
        diff_ball_rot_dist = (ball_rot_diff**2).mean(dim=-1)
        r_ball_rot = torch.exp(-diff_ball_rot_dist / self.config.rewards.reward_tracking_sigma.ball_rotation)
        return r_ball_rot
    
    def _reward_ball_velocity(self):
        ball_vel_diff = self.dif_global_ball_vel
        diff_ball_vel_dist = (ball_vel_diff**2).mean(dim=-1)
        r_ball_vel = torch.exp(-diff_ball_vel_dist / self.config.rewards.reward_tracking_sigma.ball_velocity)
        return r_ball_vel

    def _reward_penalty_ball_spin_axis_alignment(self):
        """
        Penalty when ball spin axis (angular velocity direction) is
        (1) not perpendicular to linear velocity, or
        (2) not parallel to the ground plane (i.e., has non-zero world-Z component).

        Returns a non-negative scalar per env. Use a negative scale in reward_scales.
        Backward compatible: if scale is 0, it has no effect.
        """
        eps = 1e-6
        omega = self.simulator.object_ang_vel  # (num_envs, 3), world frame
        v = self.simulator.object_vel          # (num_envs, 3), world frame

        omega_norm = torch.norm(omega, dim=-1, keepdim=True)  # (N,1)
        v_norm = torch.norm(v, dim=-1, keepdim=True)          # (N,1)

        axis = omega / omega_norm.clamp(min=eps)  # spin axis direction
        v_hat = v / v_norm.clamp(min=eps)

        # Want axis ⟂ v_hat  => dot = 0
        dot_perp = torch.sum(axis * v_hat, dim=-1).abs()
        # Use linear penalty to be more sensitive to small errors (vs square which shrinks in 0~1).
        perp_pen = dot_perp

        # Want axis ∥ ground plane => axis·z = 0  (world up is +Z)
        dot_up = axis[:, 2].abs()
        plane_pen = dot_up

        # If no spin, axis is undefined -> no penalty. If no linear velocity, skip perp term.
        has_spin = (omega_norm.squeeze(-1) > eps)
        has_vel = (v_norm.squeeze(-1) > eps)

        perp_pen = perp_pen * (has_spin & has_vel).to(perp_pen.dtype)
        plane_pen = plane_pen * has_spin.to(plane_pen.dtype)
        penalty = perp_pen + plane_pen

        # If ball is close to the hands center (< 10cm), do not penalize.
        # Hand link indices are prepared in init (used by packet loss logic).
        if hasattr(self, "left_hand_link_idx") and hasattr(self, "right_hand_link_idx"):
            pos_src = getattr(self, "_rigid_body_pos_extend", None)
            if pos_src is None:
                pos_src = getattr(self.simulator, "_rigid_body_pos", None)
            if pos_src is not None and pos_src.ndim == 3:
                li = int(self.left_hand_link_idx)
                ri = int(self.right_hand_link_idx)
                if pos_src.shape[1] > max(li, ri):
                    left_hand_pos = pos_src[:, li, :]
                    right_hand_pos = pos_src[:, ri, :]
                    hand_center = (left_hand_pos + right_hand_pos) * 0.5
                    dist_to_hands_center = torch.norm(self.simulator.object_pos - hand_center, dim=-1)
                    penalty = penalty * (dist_to_hands_center >= 0.10).to(penalty.dtype)

        # print(penalty, self.simulator.object_rot, omega)
        return penalty
    
    def _reward_ball_spin_axis_alignment(self):
        """
        Reward for the *desired* condition:
        ball spin axis (angular velocity direction) should be
        (1) perpendicular to linear velocity direction, and
        (2) parallel to the ground plane (zero world-Z component).

        Exponential form (tracking-style): reward = exp(-cost/sigma), in (0, 1].
        Use a positive scale in reward_scales.
        Notes:
        - If no spin (|omega| ~ 0), axis is undefined -> reward = 0 (no free reward).
        - If no linear velocity (|v| ~ 0), ignore the perpendicular-to-v term (cost contribution = 0.0).
        - If ball is close to hands center (< 10cm), reward is set to 0 (same gating as penalty).
        """
        eps = 1e-6
        omega = self.simulator.object_ang_vel  # (num_envs, 3), world frame
        v = self.simulator.object_vel          # (num_envs, 3), world frame

        omega_norm = torch.norm(omega, dim=-1, keepdim=True)
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        has_spin = (omega_norm.squeeze(-1) > eps)
        has_vel = (v_norm.squeeze(-1) > eps)

        axis = omega / omega_norm.clamp(min=eps)  # (N,3)
        v_hat = v / v_norm.clamp(min=eps)

        # cost components in [0, 1]
        dot_perp = torch.sum(axis * v_hat, dim=-1).abs()  # want 0
        dot_up = axis[:, 2].abs()                         # want 0

        cost_perp = dot_perp * has_vel.to(dot_perp.dtype)
        cost_plane = dot_up
        cost = cost_perp + cost_plane  # in [0, 2]

        sigma = float(self.config.rewards.reward_tracking_sigma.ball_spin_axis_alignment)
        reward = torch.exp(-cost / max(sigma, 1e-6))
        reward = reward * has_spin.to(reward.dtype)  # no free reward when no spin

        # Same near-hands gating as penalty: if close (<10cm), do not reward.
        if hasattr(self, "left_hand_link_idx") and hasattr(self, "right_hand_link_idx"):
            pos_src = getattr(self, "_rigid_body_pos_extend", None)
            if pos_src is None:
                pos_src = getattr(self.simulator, "_rigid_body_pos", None)
            if pos_src is not None and pos_src.ndim == 3:
                li = int(self.left_hand_link_idx)
                ri = int(self.right_hand_link_idx)
                if pos_src.shape[1] > max(li, ri):
                    left_hand_pos = pos_src[:, li, :]
                    right_hand_pos = pos_src[:, ri, :]
                    hand_center = (left_hand_pos + right_hand_pos) * 0.5
                    dist_to_hands_center = torch.norm(self.simulator.object_pos - hand_center, dim=-1)
                    reward = reward * (dist_to_hands_center >= 0.10).to(reward.dtype)

        return reward
    
    def _reward_interaction_graph(self):

        e_ig = (self.dif_ig**2).mean(dim=-1).mean(dim=-1)
        ig_rew = torch.exp(-e_ig / self.config.rewards.reward_tracking_sigma.interaction_graph)
        
        return ig_rew
    
    def _reward_contact_graph(self):

        # e_cg_bodies = torch.abs(self.bodies_contact_status)
        e_cg_no_bodies = torch.abs(self.no_bodies_contact_status)
        e_cg_no_bodies = torch.exp(-(e_cg_no_bodies) / self.config.rewards.reward_tracking_sigma.contact_graph)
        e_cg_obj = torch.abs(self.object_contact_status)
        e_cg_obj = torch.exp(-(e_cg_obj) / self.config.rewards.reward_tracking_sigma.contact_graph)
        
        cg_rew = e_cg_no_bodies * e_cg_obj

        return cg_rew

    def _reward_contact_graph_feet(self):
   
        e_lfoot_contact = torch.abs(self.lfoot_contact_status)
        lfoot_cg_rew = torch.exp(-(e_lfoot_contact) / self.config.rewards.reward_tracking_sigma.contact_graph_feet)
        e_rfoot_contact = torch.abs(self.rfoot_contact_status)
        rfoot_cg_rew = torch.exp(-(e_rfoot_contact) / self.config.rewards.reward_tracking_sigma.contact_graph_feet)
        return lfoot_cg_rew * rfoot_cg_rew

    def _reward_ig_extended_hand(self):
        e_ig = (self.dif_ig_extended_hand**2).mean(dim=-1).mean(dim=-1)
        ig_rew = torch.exp(-e_ig / self.config.rewards.reward_tracking_sigma.ig_extended_hand)
        
        return ig_rew
    
    # def _reward_waist_dof(self):

    #     joint_pos_diff = self.dif_waist_dof
    #     diff_joint_pos_dist = (joint_pos_diff**2).mean(dim=-1)
    #     r_waist_dof = torch.exp(-diff_joint_pos_dist / self.config.rewards.reward_tracking_sigma.waist_dof)
    #     return r_waist_dof

    def _reward_waist_dof(self): # wyh, for penalize waist dof

        waist_dof = self.simulator.dof_pos[:, 12:14].clone()
        r_waist_dof = (waist_dof**2).mean(dim=-1)
        return r_waist_dof
    
    def _reward_hip_yaw_dof(self): # wyh, for penalize hip yaw dof
        hip_yaw_dof = self.simulator.dof_pos[:, [2, 8]].clone()  # left_hip_yaw_joint (2), right_hip_yaw_joint (8)
        r_hip_yaw_dof = (hip_yaw_dof**2).mean(dim=-1)
        return r_hip_yaw_dof

    def _log_contact_bodies(self):
        """Log body names when contact occurs (checks only x/y force components)."""
        # Check whether logging should happen based on the configured frequency.
        log_frequency = getattr(self.config, 'log_contact_frequency', 1)
        if not hasattr(self, '_contact_log_counter'):
            self._contact_log_counter = 0
        self._contact_log_counter += 1
        if self._contact_log_counter % log_frequency != 0:
            return
            
        # Get the contact-force threshold.
        contact_threshold = getattr(self.config, 'contact_force_threshold', 0.1)
        
        # Get contact forces for all bodies.
        all_contact_forces = self.simulator.contact_forces  # [num_envs, num_bodies, 3]
        
        # Iterate over environments.
        for env_idx in range(self.num_envs):
            contact_bodies = []
            
            # Check contact force for all bodies (x/y components only).
            for body_idx in range(all_contact_forces.shape[1]):
                contact_force = all_contact_forces[env_idx, body_idx, :]
                # Compute magnitude using only x/y force components.
                horizontal_force = torch.norm(contact_force[:2]).item()  # x/y only
                
                if horizontal_force > contact_threshold:
                    # Get the body name.
                    if body_idx < len(self.body_names):
                        body_name = self.body_names[body_idx]
                        # Filter out bodies that should not be logged.
                        if body_name in ['left_wrist_yaw_link', 'right_wrist_yaw_link']:
                            continue
                        # Report horizontal force magnitude (optionally full force vector).
                        force_vector = contact_force.cpu().numpy()
                        contact_bodies.append(f"{body_name}(horizontal_force:{horizontal_force:.3f}") #, vector:[{force_vector[0]:.3f},{force_vector[1]:.3f},{force_vector[2]:.3f}])
            
            # Print info if there is any contact.
            if contact_bodies:
                frame_info = f"Frame {self.episode_length_buf[env_idx].item():.0f}"
                env_info = f"Env {env_idx}"
                contact_info = ", ".join(contact_bodies)
                print(f"[{frame_info}] {env_info} - Horizontal contact bodies: {contact_info}")
    
    def setup_visualize_entities(self):
        if self.debug_viz and self.config.simulator.config.name == "genesis":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.simulator.add_visualize_entities(num_visualize_markers)
        elif self.debug_viz and self.config.simulator.config.name == "mujoco":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.simulator.add_visualize_entities(num_visualize_markers)
        else:
            pass
