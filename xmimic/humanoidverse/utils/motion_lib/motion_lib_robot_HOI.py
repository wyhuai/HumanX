from humanoidverse.utils.motion_lib.motion_lib_base import *
from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
from humanoidverse.utils.motion_lib.motion_utils.support import derive_support_top_pose_from_motion
from humanoidverse.utils.smpl_reference import resolve_smpl_joints_local_centered

class MotionLibRobotHOI(MotionLibBase):
    def __init__(self, motion_lib_cfg, num_envs, device, adaptive_sampling_cfg=None):
        super().__init__(motion_lib_cfg = motion_lib_cfg, num_envs = num_envs, device = device)
        self.mesh_parsers = Humanoid_Batch(motion_lib_cfg)
        self._adaptive_sampling_cfg = adaptive_sampling_cfg
        return
    
    # def load_motions(self, 
    #                  random_sample=True, 
    #                  start_idx=0, 
    #                  max_len=-1, 
    #                  target_heading = None):
    #     # import ipdb; ipdb.set_trace()

    #     motions = []
    #     _motion_lengths = []
    #     _motion_fps = []
    #     _motion_dt = []
    #     _motion_num_frames = []
    #     _motion_bodies = []
    #     _motion_aa = []
    #     has_action = False
    #     _motion_actions = []
        
    #     _obj_status = []
        
    #     if flags.real_traj:
    #         self.q_gts, self.q_grs, self.q_gavs, self.q_gvs = [], [], [], []

    #     total_len = 0.0
    #     self.num_joints = len(self.skeleton_tree.node_names)
    #     num_motion_to_load = self.num_envs

    #     if random_sample:
    #         sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_motion_to_load, replacement=True).to(self._device)
    #     else:
    #         sample_idxes = torch.remainder(torch.arange(num_motion_to_load) + start_idx, self._num_unique_motions ).to(self._device)

    #     self._curr_motion_ids = sample_idxes
    #     self.curr_motion_keys = self._motion_data_keys[sample_idxes.cpu()]
    #     # self._sampling_batch_prob = self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()

    #     logger.info(f"Loading {num_motion_to_load} motions...")
    #     logger.info(f"Sampling motion: {sample_idxes[:5]}, ....")
    #     logger.info(f"Current motion keys: {self.curr_motion_keys[:5]}, ....")

    #     motion_data_list = self._motion_data_list[sample_idxes.cpu().numpy()]
    #     res_acc = self.load_motion_with_skeleton(motion_data_list, self.fix_height, target_heading, max_len)
    #     for f in track(range(len(res_acc)), description="Loading motions..."):
    #         motion_file_data, curr_motion, curr_obj_status = res_acc[f]
    #         motion_fps = curr_motion.fps
    #         curr_dt = 1.0 / motion_fps
    #         num_frames = curr_motion.global_rotation.shape[0]
    #         curr_len = 1.0 / motion_fps * (num_frames - 1)

    #         if "beta" in motion_file_data:
    #             _motion_aa.append(motion_file_data['pose_aa'].reshape(-1, self.num_joints * 3))
    #             _motion_bodies.append(curr_motion.gender_beta)
    #         else:
    #             _motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
    #             _motion_bodies.append(torch.zeros(17))

    #         _motion_fps.append(motion_fps)
    #         _motion_dt.append(curr_dt)
    #         _motion_num_frames.append(num_frames)
    #         motions.append(curr_motion)
    #         _motion_lengths.append(curr_len)
    #         if self.has_action:
    #             _motion_actions.append(curr_motion.action)
            
    #         if flags.real_traj:
    #             self.q_gts.append(curr_motion.quest_motion['quest_trans'])
    #             self.q_grs.append(curr_motion.quest_motion['quest_rot'])
    #             self.q_gavs.append(curr_motion.quest_motion['global_angular_vel'])
    #             self.q_gvs.append(curr_motion.quest_motion['linear_vel'])
                
    #         _obj_status.append(curr_obj_status)
                
    #         del curr_motion
        
    #     self._motion_lengths = torch.tensor(_motion_lengths, device=self._device, dtype=torch.float32)
    #     self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32)
    #     self._motion_bodies = torch.stack(_motion_bodies).to(self._device).type(torch.float32)
    #     self._motion_aa = torch.tensor(np.concatenate(_motion_aa), device=self._device, dtype=torch.float32)

    #     self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)
    #     self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device)
    #     # import ipdb; ipdb.set_trace()
    #     if self.has_action:
    #         self._motion_actions = torch.cat(_motion_actions, dim=0).float().to(self._device)
    #     self._num_motions = len(motions)
        
    #     #robot properties
    #     self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
    #     self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
    #     self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float().to(self._device)
    #     self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(self._device)
    #     self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(self._device)
    #     self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
    #     self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float().to(self._device)
    #     self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)
        
    #     if "global_translation_extend" in motions[0].__dict__:
    #         self.gts_t = torch.cat([m.global_translation_extend for m in motions], dim=0).float().to(self._device)
    #         self.grs_t = torch.cat([m.global_rotation_extend for m in motions], dim=0).float().to(self._device)
    #         self.gvs_t = torch.cat([m.global_velocity_extend for m in motions], dim=0).float().to(self._device)
    #         self.gavs_t = torch.cat([m.global_angular_velocity_extend for m in motions], dim=0).float().to(self._device)
        
    #     if "dof_pos" in motions[0].__dict__:
    #         self.dof_pos = torch.cat([m.dof_pos for m in motions], dim=0).float().to(self._device)
    #     # import ipdb; ipdb.set_trace()
    #     if flags.real_traj:
    #         self.q_gts = torch.cat(self.q_gts, dim=0).float().to(self._device)
    #         self.q_grs = torch.cat(self.q_grs, dim=0).float().to(self._device)
    #         self.q_gavs = torch.cat(self.q_gavs, dim=0).float().to(self._device)
    #         self.q_gvs = torch.cat(self.q_gvs, dim=0).float().to(self._device)
        
    #     lengths = self._motion_num_frames
    #     lengths_shifted = lengths.roll(1)
    #     lengths_shifted[0] = 0
    #     self.length_starts = lengths_shifted.cumsum(0)
    #     self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)
    #     motion = motions[0]
    #     self.num_bodies = self.num_joints
        
    #     #object properties
    #     self.obj_gts = torch.cat([m.obj_pos for m in _obj_status], dim = 0 ).float().to(self._device)
    #     self.obj_grs = torch.cat([m.obj_rot_quat for m in _obj_status], dim = 0 ).float().to(self._device)
    #     self.obj_contact = torch.cat([m.contact for m in _obj_status], dim = 0 ).long().to(self._device)
    #     self.lfoot_contact = torch.cat([m.lfoot_contact for m in _obj_status], dim = 0 ).long().to(self._device)
    #     self.rfoot_contact = torch.cat([m.rfoot_contact for m in _obj_status], dim = 0 ).long().to(self._device)
    #     self.obj_gvs = torch.cat([m.obj_vel for m in _obj_status], dim = 0 ).float().to(self._device)
    #     self.obj_grvs = torch.cat([m.obj_ang_vel for m in _obj_status], dim = 0 ).float().to(self._device)
        
    #     num_motions = self.num_motions()
    #     total_len = self.get_total_length()
    #     logger.info(f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")
    #     return motions

    def load_motions(self, 
                     random_sample=True, 
                     start_idx=0, 
                     max_len=-1, 
                     target_heading = None):
        # import ipdb; ipdb.set_trace()

        motions = []
        _motion_lengths = []
        _motion_stand_frame = []
        _motion_fps = []
        _motion_dt = []
        _motion_num_frames = []
        _motion_bodies = []
        _motion_aa = []
        _motion_apex_height = []
        has_action = False
        _motion_actions = []
        
        _obj_status = []
        
        if flags.real_traj:
            self.q_gts, self.q_grs, self.q_gavs, self.q_gvs = [], [], [], []

        total_len = 0.0
        self.num_joints = len(self.skeleton_tree.node_names)
        num_motion_to_load = self.num_envs

        if random_sample:
            sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_motion_to_load, replacement=True).to(self._device)
        else:
            sample_idxes = torch.remainder(torch.arange(num_motion_to_load) + start_idx, self._num_unique_motions ).to(self._device)

        # Hard code: force-select one specific sample.
        # Uncomment the two lines below and set desired_motion_idx to hard-code a sample.
        # NOTE: index follows the original order in the pickle (numeric order), not alphabetical.
        # desired_motion_idx = 2  # Example: select index 0 sample ('badminton_front_sample_0')
        # sample_idxes = torch.full((num_motion_to_load,), desired_motion_idx, dtype=torch.long, device=self._device)

        self._curr_motion_ids = sample_idxes
        self.curr_motion_keys = self._motion_data_keys[sample_idxes.cpu()]
        # self._sampling_batch_prob = self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()

        logger.info(f"Loading {num_motion_to_load} motions...")
        logger.info(f"Sampling motion: {sample_idxes[:5]}, ....")
        logger.info(f"Current motion keys: {self.curr_motion_keys[:5]}, ....")

        sample_idxes_np = sample_idxes.cpu().numpy()
        if max_len == -1:
            unique_sample_idxes, inverse_idxes = self._deduplicate_motion_indices(sample_idxes_np)
            if len(unique_sample_idxes) < len(sample_idxes_np):
                logger.info(
                    f"Deduplicated motion loading from {len(sample_idxes_np)} sampled slots "
                    f"to {len(unique_sample_idxes)} unique motions."
                )
            motion_data_list = self._motion_data_list[unique_sample_idxes]
            unique_res_acc = self.load_motion_with_skeleton(motion_data_list, self.fix_height, target_heading, max_len)
            res_acc = [unique_res_acc[int(unique_pos)] for unique_pos in inverse_idxes]
        else:
            motion_data_list = self._motion_data_list[sample_idxes_np]
            res_acc = self.load_motion_with_skeleton(motion_data_list, self.fix_height, target_heading, max_len)
        for f in track(range(len(res_acc)), description="Loading motions..."):
            motion_file_data, curr_motion, curr_obj_status = res_acc[f]
            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion.global_rotation.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)
            curr_stand_frame = motion_file_data.get('stand_frame', None)
            if curr_stand_frame is None:
                curr_stand_frame = num_frames - 1
            # Optional per-motion scalar (e.g. jumpshot apex height). Keep backward compatible with old pkls.
            _motion_apex_height.append(float(motion_file_data.get("apex_height", 0.0)))

            if "beta" in motion_file_data:
                _motion_aa.append(curr_motion.normalized_pose_aa.reshape(-1, self.num_joints * 3).cpu().numpy())
                _motion_bodies.append(curr_motion.gender_beta)
            else:
                _motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
                _motion_bodies.append(torch.zeros(17))

            _motion_fps.append(motion_fps)
            _motion_dt.append(curr_dt)
            _motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            _motion_lengths.append(curr_len)
            _motion_stand_frame.append(curr_stand_frame)
            if self.has_action:
                _motion_actions.append(curr_motion.action)
            
            if flags.real_traj:
                self.q_gts.append(curr_motion.quest_motion['quest_trans'])
                self.q_grs.append(curr_motion.quest_motion['quest_rot'])
                self.q_gavs.append(curr_motion.quest_motion['global_angular_vel'])
                self.q_gvs.append(curr_motion.quest_motion['linear_vel'])
                
            _obj_status.append(curr_obj_status)
                
            del curr_motion
        
        self._motion_lengths = torch.tensor(_motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32)
        self._motion_stand_frame = torch.tensor(_motion_stand_frame, device=self._device, dtype=torch.float32)
        # Per-motion constant, returned as (B, 1) in get_motion_state()
        self._motion_apex_height = torch.tensor(_motion_apex_height, device=self._device, dtype=torch.float32).unsqueeze(-1)
        self._motion_bodies = torch.stack(_motion_bodies).to(self._device).type(torch.float32)
        self._motion_aa = torch.tensor(np.concatenate(_motion_aa), device=self._device, dtype=torch.float32)

        self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device)
        # import ipdb; ipdb.set_trace()
        if self.has_action:
            self._motion_actions = torch.cat(_motion_actions, dim=0).float().to(self._device)
        self._num_motions = len(motions)
        
        #robot properties
        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float().to(self._device)
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(self._device)
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float().to(self._device)
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)
        
        self.ref_origin_pos = torch.cat([m.ref_origin_pos for m in motions], dim=0).float().to(self._device)
        self.ref_origin_rot = torch.cat([m.ref_origin_rot for m in motions], dim=0).float().to(self._device)
        
        if "global_translation_extend" in motions[0].__dict__:
            self.gts_t = torch.cat([m.global_translation_extend for m in motions], dim=0).float().to(self._device)
            self.grs_t = torch.cat([m.global_rotation_extend for m in motions], dim=0).float().to(self._device)
            self.gvs_t = torch.cat([m.global_velocity_extend for m in motions], dim=0).float().to(self._device)
            self.gavs_t = torch.cat([m.global_angular_velocity_extend for m in motions], dim=0).float().to(self._device)
        
        if "dof_pos" in motions[0].__dict__:
            self.dof_pos = torch.cat([m.dof_pos for m in motions], dim=0).float().to(self._device)
        # import ipdb; ipdb.set_trace()
        if flags.real_traj:
            self.q_gts = torch.cat(self.q_gts, dim=0).float().to(self._device)
            self.q_grs = torch.cat(self.q_grs, dim=0).float().to(self._device)
            self.q_gavs = torch.cat(self.q_gavs, dim=0).float().to(self._device)
            self.q_gvs = torch.cat(self.q_gvs, dim=0).float().to(self._device)
        
        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)
        motion = motions[0]
        self.num_bodies = self.num_joints
        
        #object properties
        self.obj_gts = torch.cat([m.obj_pos for m in _obj_status], dim = 0 ).float().to(self._device)
        self.obj_grs = torch.cat([m.obj_rot_quat for m in _obj_status], dim = 0 ).float().to(self._device)
        self.obj_contact = torch.cat([m.contact for m in _obj_status], dim = 0 ).long().to(self._device)
        self.lfoot_contact = torch.cat([m.lfoot_contact for m in _obj_status], dim = 0 ).long().to(self._device)
        self.rfoot_contact = torch.cat([m.rfoot_contact for m in _obj_status], dim = 0 ).long().to(self._device)
        self.obj_gvs = torch.cat([m.obj_vel for m in _obj_status], dim = 0 ).float().to(self._device)
        self.obj_grvs = torch.cat([m.obj_ang_vel for m in _obj_status], dim = 0 ).float().to(self._device)
        # Optional flag: 1 means freeze object in simulator (keep pose fixed).
        # Backward compatible: if not provided by data, it's all zeros.
        self.obj_freeze = torch.cat([m.obj_freeze for m in _obj_status], dim=0).long().to(self._device)
        self.basket_pos = torch.cat([m.basket_pos for m in _obj_status], dim=0).float().to(self._device)
        self.skill_label = torch.cat([m.skill_label for m in _obj_status], dim=0).float().to(self._device)
        self._motion_need_support = torch.stack([m.need_support for m in _obj_status], dim=0).long().to(self._device)
        self._motion_support_top_pos = torch.stack([m.support_top_pos for m in _obj_status], dim=0).float().to(self._device)
        self.human_joints = torch.cat([m.human_joints for m in _obj_status], dim=0).float().to(self._device)
        self.human_joints_valid = torch.cat([m.human_joints_valid for m in _obj_status], dim=0).bool().to(self._device)
        self.human_root_rot = torch.cat([m.human_root_rot for m in _obj_status], dim=0).float().to(self._device)
        self.human_root_rot_valid = torch.cat([m.human_root_rot_valid for m in _obj_status], dim=0).bool().to(self._device)
        self.human_smpl_joints_local = torch.cat([m.human_smpl_joints_local for m in _obj_status], dim=0).float().to(self._device)
        self.human_smpl_joints_local_centered = torch.cat([m.human_smpl_joints_local_centered for m in _obj_status], dim=0).float().to(self._device)
        human_smpl_pose = [m.human_smpl_pose for m in _obj_status]
        if all(pose is not None for pose in human_smpl_pose):
            self.human_smpl_pose = torch.cat(human_smpl_pose, dim=0).float().to(self._device)
        self.human_global_orient_quat = torch.cat([m.human_global_orient_quat for m in _obj_status], dim=0).float().to(self._device)
        self.human_joints_num = int(self.human_joints.shape[1]) if self.human_joints.ndim == 3 else 0
        
        num_motions = self.num_motions()
        total_len = self.get_total_length()
        logger.info(f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")
        self._init_adaptive_sampling_buffers()
        return motions

    def _init_adaptive_sampling_buffers(self):
        adaptive_sampling_enabled = getattr(self.m_cfg, "enable_adaptive_sampling", False)
        has_adaptive_params = (hasattr(self.m_cfg, "adaptive_sampling_alpha") or 
                              hasattr(self.m_cfg, "adaptive_sampling_eps"))

        if not adaptive_sampling_enabled and not has_adaptive_params:
            self._frame_reward_sum = None
            self._frame_reward_count = None
            # Ensure logging-related attributes exist even when disabled
            self._last_motion_0_rewards = None
            self._last_motion_0_num_frames = None
            self._last_weights = None
            self._last_sampled_indices = None
            return
        
        # Build mapping for unique motions
        # For each unique motion, compute its frame range in a compact buffer
        unique_motion_ids = self._curr_motion_ids.unique(sorted=True)
        num_unique = len(unique_motion_ids)
        
        # Get frame count for each unique motion (using first occurrence)
        unique_motion_frames = []
        for uid in unique_motion_ids:
            # Find first env that uses this unique motion
            first_env_idx = (self._curr_motion_ids == uid).nonzero(as_tuple=True)[0][0]
            num_frames = int(self._motion_num_frames[first_env_idx].item())
            unique_motion_frames.append(num_frames)
        
        # Build cumulative index for unique motions
        unique_motion_frames_tensor = torch.tensor(unique_motion_frames, device=self._device)
        lengths_shifted = unique_motion_frames_tensor.roll(1)
        lengths_shifted[0] = 0
        self._unique_length_starts = lengths_shifted.cumsum(0)  # Start index for each unique motion
        self._unique_motion_num_frames = unique_motion_frames_tensor
        self._unique_motion_ids_list = unique_motion_ids  # Maps index -> unique motion id
        
        # Allocate space based on unique motions only
        total_unique_frames = int(unique_motion_frames_tensor.sum().item())
        self._frame_reward_sum = torch.zeros(total_unique_frames, dtype=torch.float32, device=self._device)
        self._frame_reward_count = torch.zeros(total_unique_frames, dtype=torch.float32, device=self._device)
        
        # Get adaptive sampling parameters from env config (passed via adaptive_sampling_cfg)
        if self._adaptive_sampling_cfg is not None:
            self._adaptive_eps = getattr(self._adaptive_sampling_cfg, "adaptive_sampling_eps", 1e-4)
            self._adaptive_alpha = getattr(self._adaptive_sampling_cfg, "adaptive_sampling_alpha", 1.0)
            self._adaptive_ema_beta = getattr(self._adaptive_sampling_cfg, "adaptive_sampling_ema_beta", 0.9)
        else:
            # Fallback to default values
            self._adaptive_eps = 1e-4
            self._adaptive_alpha = 1.0
            self._adaptive_ema_beta = 0.9

        # Pre/post-stand phase reward upper bounds for scaling
        self._phase_upper_pre = None
        self._phase_upper_post = None
        if self._adaptive_sampling_cfg is not None:
            rewards_cfg = getattr(self._adaptive_sampling_cfg, "rewards", None)
            reward_scales = getattr(rewards_cfg, "reward_scales", None) if rewards_cfg is not None else None
            if reward_scales:
                penalty_names = set(getattr(rewards_cfg, "reward_penalty_reward_names", []))
                object_related = set(getattr(rewards_cfg, "object_related_rewards", []))

                def _to_float(val):
                    if isinstance(val, (int, float)):
                        return float(val)
                    if hasattr(val, "item"):
                        try:
                            return float(val.item())
                        except Exception:
                            pass
                    return float(val)

                try:
                    items_iter = reward_scales.items()
                except AttributeError:
                    items_iter = []

                total_positive = 0.0
                total_object = 0.0
                for name, scale in items_iter:
                    try:
                        scale_val = _to_float(scale)
                    except Exception:
                        continue
                    if name in penalty_names or name == "termination" or scale_val <= 0.0:
                        continue
                    
                    # Handle function upper bounds
                    if name == "teleop_body_position_extend":
                        # This reward has upper bound = lower_weight + upper_weight (default 1.0 + 1.0 = 2.0)
                        lower_weight = getattr(rewards_cfg, "teleop_body_pos_lowerbody_weight", 1.0)
                        upper_weight = getattr(rewards_cfg, "teleop_body_pos_upperbody_weight", 1.0)
                        func_upper_bound = _to_float(lower_weight) + _to_float(upper_weight)
                        contribution = scale_val * func_upper_bound
                    else:
                        # Most rewards have upper bound = 1 (exp functions)
                        contribution = scale_val * 1.0
                    
                    assert contribution > 0.0, f"Contribution {name} is not positive: {contribution}"
                    total_positive += contribution
                    if name in object_related:
                        total_object += contribution

                if total_positive > 0.0:
                    post_raw = total_positive - total_object
                    if post_raw <= 0.0:
                        post_raw = total_positive
                    pre_upper = max(total_positive, 1e-6)
                    post_upper = max(post_raw, 1e-6)
                    self._phase_upper_pre = torch.tensor(pre_upper, dtype=torch.float32, device=self._device)
                    self._phase_upper_post = torch.tensor(post_upper, dtype=torch.float32, device=self._device)
        
        # Cache for TensorBoard logging
        self._last_motion_0_rewards = None
        self._last_motion_0_num_frames = None
        self._last_weights = None
        self._last_sampled_indices = None
        
        logger.info(f"[RSI] Initialized adaptive sampling for {num_unique} unique motions, "
                   f"total {total_unique_frames} frames")

    def _get_frame_scores(self, motion_ids, frame_offsets):
        """Get average reward scores for given frames.
        
        Args:
            motion_ids: Motion instance indices (per env), shape [N, M]
            frame_offsets: Frame offsets within each motion, shape [N, M]
        
        Returns:
            Average rewards per frame (reward_sum / reward_count)
        """
        # The motion_ids tensor is [N, M], but all values in a given row are the same instance id.
        # We only need one motion id per row (per environment) to find the correct unique motion.
        motion_instance_ids_per_row = motion_ids[:, 0]  # Shape [N]
        
        # Map motion instance ids to unique motion ids
        unique_motion_ids_per_row = self._curr_motion_ids[motion_instance_ids_per_row]  # Shape [N]
        
        # Find position of each unique_motion_id in self._unique_motion_ids_list
        # This gives us the index to use with _unique_length_starts
        unique_indices_per_row = torch.searchsorted(self._unique_motion_ids_list, unique_motion_ids_per_row)  # Shape [N]
        
        # Get the starting offset for each row's unique motion
        length_starts_per_row = self._unique_length_starts[unique_indices_per_row]  # Shape [N]
        
        # Add the starting offset to all frame offsets in each row via broadcasting
        # [N, M] + [N, 1] -> [N, M]
        flat_idx = frame_offsets + length_starts_per_row.unsqueeze(1)
        
        # Get stats and compute average
        reward_sum = self._frame_reward_sum[flat_idx]
        reward_count = self._frame_reward_count[flat_idx]
        avg = torch.where(reward_count > 0, reward_sum / reward_count, torch.zeros_like(reward_sum))
        return avg

    def update_frame_reward(self, motion_ids, start_times, avg_rewards):
        """Update frame-level reward statistics for unique motions.
        
        Args:
            motion_ids: Motion instance indices (per env)
            start_times: Start times for each motion instance
            avg_rewards: Average rewards for each episode
        """
        if self._frame_reward_sum is None or len(motion_ids) == 0:
            return
        motion_ids = motion_ids.to(self._device)
        start_times = start_times.to(self._device)
        avg_rewards = avg_rewards.to(self._device)

        # Map motion instance ids to unique motion ids
        unique_motion_ids = self._curr_motion_ids[motion_ids]
        unique_indices = torch.searchsorted(self._unique_motion_ids_list, unique_motion_ids)

        dt = self._motion_dt[motion_ids]
        start_frames = torch.floor(start_times / dt).long().clamp(min=0)
        num_frames = self._motion_num_frames[motion_ids]
        start_frames = torch.min(start_frames, num_frames - 1)

        # Use unique motion indexing
        frame_indices = start_frames + self._unique_length_starts[unique_indices]

        # If EMA is enabled, update only the touched indices with EMA; untouched indices are not decayed
        if self._adaptive_ema_beta is not None and float(self._adaptive_ema_beta) > 0.0:
            beta = float(self._adaptive_ema_beta)
            # Aggregate duplicates per index before EMA update
            unique_idx, inverse = torch.unique(frame_indices, return_inverse=True)
            agg_sum = torch.zeros(unique_idx.shape[0], dtype=torch.float32, device=self._device)
            agg_cnt = torch.zeros_like(agg_sum)
            agg_sum.scatter_add_(0, inverse, avg_rewards)
            agg_cnt.scatter_add_(0, inverse, torch.ones_like(avg_rewards, dtype=torch.float32))

            old_sum = self._frame_reward_sum[unique_idx]
            old_cnt = self._frame_reward_count[unique_idx]
            new_sum = old_sum * (1.0 - beta) + beta * agg_sum
            new_cnt = old_cnt * (1.0 - beta) + beta * agg_cnt
            self._frame_reward_sum[unique_idx] = new_sum
            self._frame_reward_count[unique_idx] = new_cnt
        else:
            # Default: simple accumulation
            self._frame_reward_sum.index_add_(0, frame_indices, avg_rewards)
            self._frame_reward_count.index_add_(0, frame_indices, torch.ones_like(avg_rewards))
        
        # # [DEBUG] Log reward updates
        # logger.info(f"[RSI] Updated {len(motion_ids)} frames | "
        #            f"Frames: {start_frames[:5].tolist()}... | "
        #            f"Avg rewards: {avg_rewards[:5].tolist()}... | "
        #            f"Total frames with data: {(self._frame_reward_count > 0).sum().item()}")

    def sample_time(self, motion_ids, truncate_time=None):
        """Sample starting times for motions using adaptive sampling based on frame rewards.
        
        Args:
            motion_ids: Tensor of motion indices to sample from
            truncate_time: Optional time to truncate motion length
            
        Returns:
            Sampled starting times for each motion
        """
        if self._frame_reward_sum is None:
            return super().sample_time(motion_ids, truncate_time)

        motion_ids = motion_ids.to(self._device)
        dt = self._motion_dt[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        max_frames = int(num_frames.max().item())
        
        # Create frame offset grid: [num_motions, max_frames]
        frame_offsets = torch.arange(max_frames, device=self._device).unsqueeze(0).repeat(len(motion_ids), 1)
        mask = (frame_offsets < num_frames.unsqueeze(1))
        # NOTE: we must guard invalid offsets BEFORE indexing into frame-level buffers.
        # Otherwise, short motions will cause GPU index out-of-bounds inside _get_frame_scores.
        frame_offsets_safe = torch.where(mask, frame_offsets, torch.zeros_like(frame_offsets))
        
        # Get average reward scores for all frames
        avg_scores = self._get_frame_scores(
            motion_ids.unsqueeze(1).expand_as(frame_offsets_safe),
            frame_offsets_safe,
        )

        if self._phase_upper_pre is not None and self._phase_upper_post is not None:
            stand_frames = self._motion_stand_frame[motion_ids].long().unsqueeze(1)
            phase_upper = torch.where(
                frame_offsets < stand_frames,
                self._phase_upper_pre,
                self._phase_upper_post
            )
            phase_upper = torch.clamp(phase_upper, min=1e-6)
            avg_scores = avg_scores / phase_upper

        avg_scores = torch.where(mask, avg_scores, torch.zeros_like(avg_scores))
        
        # Compute adaptive sampling weights: lower reward → higher weight
        min_scores = avg_scores.amin(dim=1, keepdim=True)
        normalized_scores = avg_scores - min_scores
        weights = self._adaptive_eps + torch.exp(-self._adaptive_alpha * normalized_scores)
        
        # Normalize weights to form valid probability distribution
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        weights_sum = torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8)
        weights = weights / weights_sum
        # Keep invalid (masked) frames at exactly 0 probability.
        # Only clamp valid frames to avoid sampling padded/out-of-range frame indices.
        weights = torch.where(mask, torch.clamp(weights, min=1e-8), torch.zeros_like(weights))
        # Re-normalize after clamping (since clamping changes the sum).
        weights_sum = torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8)
        weights = weights / weights_sum
        
        # Sample frame indices based on computed weights
        # weights shape: [num_motions, max_frames], sample one frame index per motion
        probs = torch.distributions.Categorical(probs=weights)
        sampled_indices = probs.sample()  # shape: [num_motions]
        sampled_times = sampled_indices.to(torch.float32) * dt  # shape: [num_motions]
        
        # Cache for TensorBoard logging (will be logged at end of iteration)
        if len(motion_ids) > 0:
            # Cache first UNIQUE motion's frame-by-frame rewards
            unique_motion_0_id = self._unique_motion_ids_list[0]
            unique_motion_0_start = int(self._unique_length_starts[0].item())
            unique_motion_0_frames = int(self._unique_motion_num_frames[0].item())
            
            unique_motion_0_rewards = torch.where(
                self._frame_reward_count[unique_motion_0_start:unique_motion_0_start + unique_motion_0_frames] > 0,
                self._frame_reward_sum[unique_motion_0_start:unique_motion_0_start + unique_motion_0_frames] / 
                self._frame_reward_count[unique_motion_0_start:unique_motion_0_start + unique_motion_0_frames],
                torch.zeros(unique_motion_0_frames, device=self._device)
            )
            
            # Cache for TensorBoard logging
            total_unique_frames = int(self._frame_reward_sum.shape[0])
            visited_mask = self._frame_reward_count > 0
            
            self._last_motion_0_rewards = unique_motion_0_rewards.detach().cpu()
            self._last_motion_0_num_frames = unique_motion_0_frames
            
            # weights: current sampling weights for first motion (as example)
            self._last_weights = weights[0].detach().cpu()
            
            # sampled_indices: all environments' sampled frame indices
            self._last_sampled_indices = sampled_indices.detach().cpu()
            
            # # [DEBUG] Log sampling statistics
            # frames_with_data = visited_mask.sum().item()
            # unique_motion_0_visited = visited_mask[unique_motion_0_start:unique_motion_0_start + unique_motion_0_frames].sum().item()
            # logger.info(f"[RSI] Sampled {len(motion_ids)} envs | "
            #            f"Sample frames: {sampled_indices[:5].tolist()}... | "
            #            f"Sample times: {sampled_times[:5].tolist()}... | "
            #            f"Total unique frames with data: {frames_with_data}/{total_unique_frames} | "
            #            f"Unique motion 0 visited: {unique_motion_0_visited}/{unique_motion_0_frames} | "
            #            f"Unique motion 0 reward range: [{unique_motion_0_rewards[unique_motion_0_rewards > 0].min().item() if (unique_motion_0_rewards > 0).any() else 0:.3f}, {unique_motion_0_rewards.max().item():.3f}]")

        if truncate_time is not None:
            motion_len = self._motion_lengths[motion_ids]
            motion_len = torch.clamp(motion_len - truncate_time, min=0.0)
            sampled_times = torch.min(sampled_times, motion_len)

        return sampled_times
    
    def log_adaptive_sampling_to_tensorboard(self, writer, global_step):
        """Log cached adaptive sampling distributions to TensorBoard.
        
        Should be called at the end of each training iteration.
        
        Logged items:
        - motion_0_curves: Frame-by-frame reward and weight curves for first motion
        - sampled_indices_distribution: Histogram of actually sampled frame indices
        
        Args:
            writer: TensorBoard SummaryWriter
            global_step: Current training iteration number
        """
        if getattr(self, "_last_motion_0_rewards", None) is None:
            return
        
        # Log frame-by-frame rewards for Motion 0 as custom image
        if self._last_motion_0_rewards is not None:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Plot 1: Frame rewards
                frame_indices = torch.arange(self._last_motion_0_num_frames)
                rewards = self._last_motion_0_rewards.numpy()
                ax1.plot(frame_indices, rewards, 'b-', linewidth=1.5)
                ax1.set_xlabel('Frame Index')
                ax1.set_ylabel('Average Reward')
                ax1.set_title('Motion 0: Frame-by-Frame Average Rewards')
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Sampling weights
                weights = self._last_weights.numpy()
                # _last_weights is cached from the padded sampling grid (max_frames),
                # which can be longer than the true motion length. Align lengths for plotting.
                x = frame_indices.numpy()
                n = min(len(x), len(weights))
                ax2.plot(x[:n], weights[:n], 'r-', linewidth=1.5)
                ax2.set_xlabel('Frame Index')
                ax2.set_ylabel('Sampling Weight')
                ax2.set_title('Motion 0: Adaptive Sampling Weights (Lower Reward → Higher Weight)')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Convert to image and log
                fig.canvas.draw()
                import numpy as np
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                writer.add_image('AdaptiveSampling/motion_0_curves', 
                                img, global_step=global_step, dataformats='HWC')
                plt.close(fig)
            except ImportError:
                pass  # matplotlib not available
        
        # Log sampled indices distribution
        writer.add_histogram('AdaptiveSampling/sampled_indices_distribution', 
                            self._last_sampled_indices.float(), global_step=global_step)
        
        # Compute and log weights entropy (measure of sampling diversity)
        weights_entropy = -(self._last_weights * torch.log(self._last_weights + 1e-8)).sum().item()
        writer.add_scalar('AdaptiveSampling/weights_entropy', 
                         weights_entropy, global_step=global_step)
        
        # Log frame coverage statistics
        if self._frame_reward_count is not None:
            frames_with_data = (self._frame_reward_count > 0).sum().item()
            total_frames = self._frame_reward_count.numel()
            coverage = frames_with_data / max(total_frames, 1)
            writer.add_scalar('AdaptiveSampling/frame_coverage', 
                             coverage, global_step=global_step)
            writer.add_scalar('AdaptiveSampling/frames_with_data', 
                             frames_with_data, global_step=global_step)
    
    def get_motion_state(self, motion_ids, motion_times, curr_time, offset=None):
        
        #base info
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        #get current frame
        now_time = curr_time.clone()
        phase = now_time / motion_len
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        now_time[now_time < 0] = 0

        # We intentionally avoid using a separate "now_idx" for object states.
        # All signals are stored in concatenated global buffers, and we already have the
        # per-motion global indices (pf0l/pf1l + blend_prev) corresponding to curr_time.
        # Using these keeps robot/object aligned and avoids any local-vs-global indexing bugs.
        
        #get blend frames
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]
        
        #get prev blend frames
        pre_frame_idx0, pre_frame_idx1, blend_prev = self._calc_frame_blend(now_time, motion_len, num_frames, dt)
        pf0l = pre_frame_idx0 + self.length_starts[motion_ids]
        pf1l = pre_frame_idx1 + self.length_starts[motion_ids]

        if "dof_pos" in self.__dict__:
            local_rot0 = self.dof_pos[f0l]
            local_rot1 = self.dof_pos[f1l]
        else:
            local_rot0 = self.lrs[f0l]
            local_rot1 = self.lrs[f1l]
            
        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)
        blend_exp = blend.unsqueeze(-1)
        
        blend_prev = blend_prev.unsqueeze(-1)
        blend_prev_exp = blend_prev.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]  # ZL: apply offset

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1

        if "dof_pos" in self.__dict__: # Robot Joints
            dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
            dof_pos = (1.0 - blend) * local_rot0 + blend * local_rot1
        else:
            dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1
            local_rot = slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
            dof_pos = self._local_rotation_to_dof_smpl(local_rot)

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = slerp(rb_rot0, rb_rot1, blend_exp)
        return_dict = {}
        
        if "gts_t" in self.__dict__:
            rg_pos_t0 = self.gts_t[f0l]
            rg_pos_t1 = self.gts_t[f1l]
            
            rg_rot_t0 = self.grs_t[f0l]
            rg_rot_t1 = self.grs_t[f1l]
            
            body_vel_t0 = self.gvs_t[f0l]
            body_vel_t1 = self.gvs_t[f1l]
            
            body_ang_vel_t0 = self.gavs_t[f0l]
            body_ang_vel_t1 = self.gavs_t[f1l]
            if offset is None:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1  
            else:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1 + offset[..., None, :]
            rg_rot_t = slerp(rg_rot_t0, rg_rot_t1, blend_exp)
            body_vel_t = (1.0 - blend_exp) * body_vel_t0 + blend_exp * body_vel_t1
            body_ang_vel_t = (1.0 - blend_exp) * body_ang_vel_t0 + blend_exp * body_ang_vel_t1
        else:
            rg_pos_t = rg_pos
            rg_rot_t = rb_rot
            body_vel_t = body_vel
            body_ang_vel_t = body_ang_vel
        
        if flags.real_traj:
            q_body_ang_vel0, q_body_ang_vel1 = self.q_gavs[f0l], self.q_gavs[f1l]
            q_rb_rot0, q_rb_rot1 = self.q_grs[f0l], self.q_grs[f1l]
            q_rg_pos0, q_rg_pos1 = self.q_gts[f0l, :], self.q_gts[f1l, :]
            q_body_vel0, q_body_vel1 = self.q_gvs[f0l], self.q_gvs[f1l]

            q_ang_vel = (1.0 - blend_exp) * q_body_ang_vel0 + blend_exp * q_body_ang_vel1
            q_rb_rot = slerp(q_rb_rot0, q_rb_rot1, blend_exp)
            q_rg_pos = (1.0 - blend_exp) * q_rg_pos0 + blend_exp * q_rg_pos1
            q_body_vel = (1.0 - blend_exp) * q_body_vel0 + blend_exp * q_body_vel1
            
            rg_pos[:, self.track_idx] = q_rg_pos
            rb_rot[:, self.track_idx] = q_rb_rot
            body_vel[:, self.track_idx] = q_body_vel
            body_ang_vel[:, self.track_idx] = q_ang_vel
            
        #object part
        blend_exp_obj = blend_exp.squeeze(-1)
        blend_prev_obj = blend_prev_exp.squeeze(-1)
        
        #pos
        pre_obj_pos0 = self.obj_gts[pf0l]
        pre_obj_pos1 = self.obj_gts[pf1l]
        if offset is None:
            # obj_pos_now = self.obj_gts[now_idx]
            obj_pos_now = (1.0 - blend_prev_obj) * pre_obj_pos0 + blend_prev_obj * pre_obj_pos1  # ZL: apply offset
        else:
            # obj_pos_now = self.obj_gts[now_idx] + offset
            obj_pos_now = (1.0 - blend_prev_obj) * pre_obj_pos0 + blend_prev_obj * pre_obj_pos1 + offset  # ZL: apply offset
    
        obj_pos0 = self.obj_gts[f0l]
        obj_pos1 = self.obj_gts[f1l]
        
        if offset is None:
            next_obj_pos = (1.0 - blend_exp_obj) * obj_pos0 + blend_exp_obj * obj_pos1  # ZL: apply offset
        else:
            next_obj_pos = (1.0 - blend_exp_obj) * obj_pos0 + blend_exp_obj * obj_pos1 + offset  # ZL: apply offset
            
        #basket pos
        if offset is not None:
            basket_pos = self.basket_pos[f0l] + offset
        else:
            basket_pos = self.basket_pos[f0l]
        
        #skill label
        skill_label = self.skill_label[f0l]
        unique_skill_labels = torch.unique(skill_label, dim=0)
        
        #ref origin pos
        if offset is not None:
            ref_origin_pos = self.ref_origin_pos[f0l] + offset
        else:   
            ref_origin_pos = self.ref_origin_pos[f0l]
            
        #ref origin rot
        ref_origin_rot = self.ref_origin_rot[f0l]

        # Per-motion scalar (broadcasted for each frame)
        if hasattr(self, "_motion_apex_height"):
            apex_height = self._motion_apex_height[motion_ids]
        else:
            apex_height = torch.zeros((motion_ids.shape[0], 1), device=self._device, dtype=torch.float32)
        need_support = self._motion_need_support[motion_ids].squeeze(-1)
        support_top_pos = self._motion_support_top_pos[motion_ids]
        
        # rot (interpolated at curr_time, consistent with curr_obj_pos)
        pre_obj_rot0 = self.obj_grs[pf0l].unsqueeze(1)
        pre_obj_rot1 = self.obj_grs[pf1l].unsqueeze(1)
        obj_rot_now = slerp(pre_obj_rot0, pre_obj_rot1, blend_prev_exp).squeeze(1)
        
        obj_rot0 = self.obj_grs[f0l].unsqueeze(1)
        obj_rot1 = self.obj_grs[f1l].unsqueeze(1)
        obj_rot = slerp(obj_rot0, obj_rot1, blend_exp)
        obj_rot = obj_rot.squeeze(1)
        
        # vel (interpolated at curr_time)
        pre_obj_vel0 = self.obj_gvs[pf0l]
        pre_obj_vel1 = self.obj_gvs[pf1l]
        obj_vel_now = (1.0 - blend_prev_obj) * pre_obj_vel0 + blend_prev_obj * pre_obj_vel1
        
        pre_obj_ang_vel0 = self.obj_grvs[pf0l]
        pre_obj_ang_vel1 = self.obj_grvs[pf1l]
        obj_angular_vel_now = (1.0 - blend_prev_obj) * pre_obj_ang_vel0 + blend_prev_obj * pre_obj_ang_vel1
        
        # contact (discrete; use floor frame at curr_time)
        obj_contact_status = self.obj_contact[pf0l].squeeze(-1)
        lfoot_contact_status = self.lfoot_contact[pf0l].squeeze(-1)
        rfoot_contact_status = self.rfoot_contact[pf0l].squeeze(-1)
        obj_freeze_status = self.obj_freeze[pf0l].squeeze(-1)
        human_joints_valid = self.human_joints_valid[pf0l].squeeze(-1)
        human_root_rot_valid = self.human_root_rot_valid[pf0l].squeeze(-1)

        if self.human_joints_num > 0:
            human_joints0 = self.human_joints[f0l]
            human_joints1 = self.human_joints[f1l]
            human_root_rot0 = self.human_root_rot[f0l]
            human_root_rot1 = self.human_root_rot[f1l]
            if offset is None:
                human_joints = (1.0 - blend_exp) * human_joints0 + blend_exp * human_joints1
            else:
                human_joints = (
                    (1.0 - blend_exp) * human_joints0
                    + blend_exp * human_joints1
                    + offset[..., None, :]
                )
            human_root_rot = slerp(human_root_rot0, human_root_rot1, blend_exp)
        else:
            human_joints = torch.zeros((motion_ids.shape[0], 0, 3), device=self._device, dtype=torch.float32)
            human_root_rot = torch.zeros((motion_ids.shape[0], 4), device=self._device, dtype=torch.float32)
            human_root_rot[:, 3] = 1.0

        return_dict.update({
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "motion_aa": self._motion_aa[f0l],
            "motion_bodies": self._motion_bodies[motion_ids],
            "rg_pos": rg_pos,
            "rb_rot": rb_rot,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "rg_pos_t": rg_pos_t,
            "rg_rot_t": rg_rot_t,
            "body_vel_t": body_vel_t,
            "body_ang_vel_t": body_ang_vel_t,
            "curr_obj_pos": obj_pos_now,
            "curr_obj_rot": obj_rot_now,
            "curr_obj_vel": obj_vel_now,
            "curr_obj_ang_vel": obj_angular_vel_now,
            "next_obj_pos": next_obj_pos,
            "next_obj_rot": obj_rot,
            "ball_contact": obj_contact_status,
            "obj_freeze": obj_freeze_status,
            "lfoot_contact": lfoot_contact_status,
            "rfoot_contact": rfoot_contact_status,
            "human_joints": human_joints,
            "human_joints_valid": human_joints_valid,
            "human_root_rot": human_root_rot,
            "human_root_rot_valid": human_root_rot_valid,
            "basket_pos": basket_pos,
            "skill_label": skill_label,
            "unique_skill_labels": unique_skill_labels,
            "apex_height": apex_height,
            "need_support": need_support,
            "support_top_pos": support_top_pos,
            "ref_origin_pos": ref_origin_pos,
            "ref_origin_rot": ref_origin_rot,
        })
        return return_dict
    

    def load_motion_with_skeleton(self,
                                  motion_data_list,
                                  fix_height,
                                  target_heading,
                                  max_len):
        # loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        res = {}
        for f in track(range(len(motion_data_list)), description="Loading motions..."):
            curr_file = motion_data_list[f]
            motion_id = f
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                curr_file = joblib.load(curr_file)[key]
                motion_id = key

            seq_len = curr_file['root_trans_offset'].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len

            trans = to_torch(curr_file['root_trans_offset']).clone()[start:end]
            raw_pose_aa = curr_file['pose_aa'][start:end]
            base_joints = len(self.skeleton_tree.node_names)
            extend_cfg = getattr(self.m_cfg, "extend_config", None)
            try:
                extend_count = len(extend_cfg) if extend_cfg is not None else 0
            except Exception:
                extend_count = 0
            try:
                # Preferred motion format is (T, base_joints, 3). Legacy (T, base_joints + extend_count, 3)
                # remains readable temporarily, but the trailing extended-joint slots are deprecated and ignored.
                pose_aa = self._normalize_pose_aa_shape(raw_pose_aa, base_joints, extend_count, motion_id)
            except ValueError as exc:
                logger.warning(f"Skip motion {motion_id}: {exc}")
                continue
            # import ipdb; ipdb.set_trace()
            if "action" in curr_file.keys():
                self.has_action = True

            dt = 1/curr_file['fps']

            B, J, N = pose_aa.shape

            if not target_heading is None:
                start_root_rot = sRot.from_rotvec(pose_aa[0, 0])
                heading_inv_rot = sRot.from_quat(calc_heading_quat_inv(torch.from_numpy(start_root_rot.as_quat()[None, ])))
                heading_delta = sRot.from_quat(target_heading) * heading_inv_rot 
                pose_aa[:, 0] = torch.tensor((heading_delta * sRot.from_rotvec(pose_aa[:, 0])).as_rotvec())

                trans = torch.matmul(trans, torch.from_numpy(heading_delta.as_matrix().squeeze().T))

            if self.mesh_parsers is not None:
                # trans, trans_fix = MotionLibRobot.fix_trans_height(pose_aa, trans, mesh_parsers, fix_height_mode = fix_height)
                curr_motion = self.mesh_parsers.fk_batch(pose_aa[None, ], trans[None, ], return_full= True, dt = dt)
                curr_motion = EasyDict({k: v.squeeze() if torch.is_tensor(v) else v for k, v in curr_motion.items() })
                curr_motion.normalized_pose_aa = pose_aa
                # add "action" to curr_motion
                if self.has_action:
                    curr_motion.action = to_torch(curr_file['action']).clone()[start:end]
                res[f] = (curr_file, curr_motion)
            else:
                logger.error("No mesh parser found")
                
            #object status
            try:
                obj_status = {
                    "obj_pos": curr_file['obj_pos'],
                    "contact": curr_file['contact'],
                }
                obj_status = EasyDict({k: v if torch.is_tensor(v) else torch.from_numpy(v) for k, v in obj_status.items() })
                Len_M = obj_status["obj_pos"].shape[0]
                obj_status["obj_rot_quat"] = self._required_obj_rot_quat_tensor(curr_file, motion_id, length=Len_M)
                obj_status["lfoot_contact"] = self._optional_contact_tensor(curr_file, "lfoot_contact", Len_M)
                obj_status["rfoot_contact"] = self._optional_contact_tensor(curr_file, "rfoot_contact", Len_M)
            except ValueError as exc:
                logger.warning(f"Skip motion {motion_id}: {exc}")
                res.pop(f, None)
                continue

            # Optional per-frame/per-motion freeze flag for the object.
            # Backward compatible: default all False.
            freeze_key = None
            if "freeze_obj" in curr_file:
                freeze_key = "freeze_obj"
            elif "obj_freeze" in curr_file:
                freeze_key = "obj_freeze"

            if freeze_key is None:
                obj_status["obj_freeze"] = torch.zeros((Len_M, 1), dtype=torch.long)
            else:
                v = curr_file[freeze_key]
                if torch.is_tensor(v):
                    t = v
                else:
                    # could be numpy array or python scalar/bool
                    t = torch.from_numpy(v) if hasattr(v, "shape") else torch.tensor(v)
                t = t.to(dtype=torch.long).view(-1, 1)
                if t.shape[0] == 1 and Len_M > 1:
                    t = t.expand(Len_M, 1).contiguous()
                obj_status["obj_freeze"] = t

            motion_need_support = bool(curr_file.get("need_support", False))
            obj_status["need_support"] = torch.tensor([1 if motion_need_support else 0], dtype=torch.long)
            if motion_need_support:
                support_top_pos = derive_support_top_pose_from_motion(
                    obj_status["obj_pos"],
                    obj_status["contact"],
                    window_size=int(curr_file.get("support_window_size", 5)),
                    static_threshold=float(curr_file.get("support_static_threshold", 5e-3)),
                    z_offset=float(curr_file.get("support_z_offset", 0.1)),
                )
            else:
                support_top_pos = np.zeros(3, dtype=np.float32)
            obj_status["support_top_pos"] = torch.from_numpy(np.asarray(support_top_pos, dtype=np.float32))
            
            # Object velocity: prefer explicit data fields, otherwise fall back to finite differences.
            obj_status['obj_vel'] = self._optional_sequence_tensor_or_none(
                curr_file,
                "obj_vel",
                Len_M,
                trailing_shape=(3,),
            )
            if obj_status['obj_vel'] is None:
                obj_status['obj_vel'] = self._compute_velocity(obj_status['obj_pos'], curr_file['fps'])

            obj_status['obj_ang_vel'] = self._optional_sequence_tensor_or_none(
                curr_file,
                "obj_ang_vel",
                Len_M,
                trailing_shape=(3,),
            )
            if obj_status['obj_ang_vel'] is None:
                obj_status['obj_ang_vel'] = self._compute_angular_velocity_from_quat(obj_status['obj_rot_quat'], curr_file['fps'])
            
            #misc
            
            #basket pos
            if "basket_pos" in curr_file.keys():
                obj_status['basket_pos'] = torch.from_numpy(curr_file['basket_pos']).float()
            else:
                Len_M = obj_status['obj_vel'].shape[0]
                obj_status['basket_pos'] = torch.zeros((Len_M, 3)).float()
            
            #skill label
            if "skill_label" in curr_file.keys():
                obj_status['skill_label'] = torch.from_numpy(curr_file["skill_label"]).float()
            else:
                Len_M = obj_status['obj_vel'].shape[0]
                obj_status['skill_label'] = torch.zeros((Len_M, 32)).float()

            obj_status["human_smpl_joints_local"] = self._optional_sequence_tensor(
                curr_file,
                "human_smpl_joints_local",
                Len_M,
                trailing_shape=(24, 3),
            )[start:end]
            explicit_centered_smpl_joints = self._optional_sequence_tensor_or_none(
                curr_file,
                "human_smpl_joints_local_centered",
                Len_M,
                trailing_shape=(24, 3),
            )
            if explicit_centered_smpl_joints is None:
                explicit_centered_smpl_joints = self._optional_sequence_tensor_or_none(
                    curr_file,
                    "smpl_joints_local_centered",
                    Len_M,
                    trailing_shape=(24, 3),
                )
            if explicit_centered_smpl_joints is not None:
                explicit_centered_smpl_joints = explicit_centered_smpl_joints[start:end]
            obj_status["human_smpl_joints_local_centered"] = resolve_smpl_joints_local_centered(
                obj_status["human_smpl_joints_local"],
                explicit_centered_smpl_joints,
            )
            human_smpl_pose = self._optional_sequence_tensor_or_none(
                curr_file,
                "human_smpl_pose",
                Len_M,
                trailing_shape=(63,),
            )
            if human_smpl_pose is None:
                human_smpl_pose = self._optional_sequence_tensor_or_none(
                    curr_file,
                    "smpl_pose",
                    Len_M,
                    trailing_shape=(63,),
                )
            if human_smpl_pose is not None:
                human_smpl_pose = human_smpl_pose[start:end]
            obj_status["human_smpl_pose"] = human_smpl_pose
            human_joints_local = self._optional_sequence_tensor(
                curr_file,
                "human_joints",
                Len_M,
                trailing_shape=(24, 3),
            )[start:end]
            human_adjusted_transl = self._optional_sequence_tensor(
                curr_file,
                "human_adjusted_transl",
                Len_M,
                trailing_shape=(3,),
            )[start:end]
            Len_H = human_joints_local.shape[0]
            obj_status["human_joints"] = human_joints_local
            if "human_joints" in curr_file and "human_adjusted_transl" in curr_file:
                human_root_z_offset = float(getattr(self.m_cfg, "human_openxr_z_offset", 1.7))
                obj_status["human_joints"] = self._human_joints_world_from_local(
                    human_joints_local,
                    human_adjusted_transl,
                    z_offset=human_root_z_offset,
                )
            human_root_rot = self._optional_sequence_tensor(
                curr_file,
                "human_global_orient_quat",
                Len_M,
                trailing_shape=(4,),
            )[start:end]
            if "human_global_orient_quat" in curr_file:
                obj_status["human_global_orient_quat"] = human_root_rot[:, [1, 2, 3, 0]]
                obj_status["human_root_rot"] = human_root_rot[:, [1, 2, 3, 0]]
            else:
                obj_status["human_global_orient_quat"] = torch.zeros((Len_H, 4), dtype=torch.float32)
                obj_status["human_global_orient_quat"][:, 3] = 1.0
                obj_status["human_root_rot"] = torch.zeros((Len_H, 4), dtype=torch.float32)
                obj_status["human_root_rot"][:, 3] = 1.0
            obj_status["human_joints_valid"] = torch.full(
                (Len_H, 1),
                1 if "human_joints" in curr_file and "human_adjusted_transl" in curr_file else 0,
                dtype=torch.bool,
            )
            obj_status["human_root_rot_valid"] = torch.full(
                (Len_H, 1),
                1 if "human_global_orient_quat" in curr_file else 0,
                dtype=torch.bool,
            )
            
            #ref origin pos
            # NOTE: pose_aa/trans are cropped by [start:end] above; keep ref_origin_* aligned to the same cropped length.
            Len_M = int(end - start)
            if "ref_origin_pos" in curr_file.keys():
                v = curr_file["ref_origin_pos"]
                v = v.detach().cpu().numpy() if torch.is_tensor(v) else v
                v = np.asarray(v)
                if v.shape[0] == 1 and Len_M > 1:
                    v = np.repeat(v, Len_M, axis=0)
                elif v.shape[0] >= end:
                    v = v[start:end]
                else:
                    # Fallback: if provided length is unexpected, treat it as constant.
                    v0 = v[0:1].copy()
                    v = np.repeat(v0, Len_M, axis=0)
                res[f][1]["ref_origin_pos"] = torch.from_numpy(v).float()
            else:
                v0 = np.asarray(curr_file["root_trans_offset"])[0:1, :].copy()
                v = np.repeat(v0, Len_M, axis=0)
                res[f][1]["ref_origin_pos"] = torch.from_numpy(v).float()
                
            #ref origin rot
            if "ref_origin_rot" in curr_file.keys():
                v = curr_file["ref_origin_rot"]
                v = v.detach().cpu().numpy() if torch.is_tensor(v) else v
                v = np.asarray(v)
                if v.shape[0] == 1 and Len_M > 1:
                    v = np.repeat(v, Len_M, axis=0)
                elif v.shape[0] >= end:
                    v = v[start:end]
                else:
                    v0 = v[0:1].copy()
                    v = np.repeat(v0, Len_M, axis=0)
                res[f][1]["ref_origin_rot"] = torch.from_numpy(v).float()
            else:
                v0 = np.asarray(curr_file["root_rot"])[0:1, :].copy()
                v = np.repeat(v0, Len_M, axis=0)
                res[f][1]["ref_origin_rot"] = torch.from_numpy(v).float()
            
            #save data
            res[f] += (obj_status, )
            
        return res
    
    
    def _compute_velocity(self, positions, fps):
        velocity = (positions[1:, :].clone() - positions[:-1, :].clone()) * fps
        # velocity = torch.cat((torch.zeros((1, positions.shape[-1])), velocity), dim=0)#.to(self._device)
        velocity = torch.cat((velocity[0:1, :], velocity), dim=0)
        return velocity
