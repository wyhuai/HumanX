import glob
import os.path as osp
import numpy as np
import joblib
import torch
import random
import warnings

from humanoidverse.utils.motion_lib.motion_utils.flags import flags
from enum import Enum
from humanoidverse.utils.motion_lib.skeleton import SkeletonTree
from pathlib import Path
from easydict import EasyDict
from loguru import logger
from rich.progress import track

from isaac_utils.rotations import(
    quat_angle_axis,
    quat_inverse,
    quat_mul_norm,
    get_euler_xyz,
    normalize_angle,
    slerp,
    quat_to_exp_map,
    quat_to_angle_axis,
    quat_mul,
    quat_conjugate,
    calc_heading_quat,
    calc_heading_quat_inv,
    my_quat_rotate,
    quaternion_to_matrix,
    xyzw_to_wxyz,
)

class FixHeightMode(Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2

class MotionlibMode(Enum):
    file = 1
    directory = 2


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)

class MotionLibBase():
    _pose_aa_deprecation_warned = False
    _obj_rot_euler_deprecation_warned = False

    def __init__(self, motion_lib_cfg, num_envs, device):
        self.m_cfg = motion_lib_cfg
        self._sim_fps = 1/self.m_cfg.get("step_dt", 1/50)
        
        self.num_envs = num_envs
        self._device = device
        self.mesh_parsers = None
        self.has_action = False
        skeleton_file = Path(self.m_cfg.asset.assetRoot) / self.m_cfg.asset.assetFileName
        self.skeleton_tree = SkeletonTree.from_mjcf(skeleton_file)
        logger.info(f"Loaded skeleton from {skeleton_file}")
        logger.info(f"Loading motion data from {self.m_cfg.motion_file}...")
        self.load_data(self.m_cfg.motion_file)
        self.setup_constants(fix_height = False,  multi_thread = False)
        if flags.real_traj:
            self.track_idx = self._motion_data_load[next(iter(self._motion_data_load))].get("track_idx", [19, 24, 29])
        return

    @staticmethod
    def _deduplicate_motion_indices(sample_idxes):
        sample_idxes = np.asarray(sample_idxes, dtype=np.int64)
        unique_idxes, first_idxes, inverse_sorted = np.unique(
            sample_idxes,
            return_index=True,
            return_inverse=True,
        )
        first_order = np.argsort(first_idxes)
        remap = np.empty_like(first_order)
        remap[first_order] = np.arange(len(first_order))
        inverse_idxes = remap[inverse_sorted]
        return unique_idxes[first_order], inverse_idxes
        
    def load_data(self, motion_file, min_length=-1, im_eval = False):
        if osp.isfile(motion_file):
            self.mode = MotionlibMode.file
            self._motion_data_load = joblib.load(motion_file)
        else:
            self.mode = MotionlibMode.directory
            self._motion_data_load = glob.glob(osp.join(motion_file, "**", "*.pkl"), recursive=True)
        data_list = self._motion_data_load
        if self.mode == MotionlibMode.file:
            if min_length != -1:
                # filtering the data by the length of the motion
                data_list = {k: v for k, v in list(self._motion_data_load.items()) if len(v['pose_quat_global']) >= min_length}
            elif im_eval:
                # sorting the data by the length of the motion
                data_list = {item[0]: item[1] for item in sorted(self._motion_data_load.items(), key=lambda entry: len(entry[1]['pose_quat_global']), reverse=True)}
            else:
                data_list = self._motion_data_load
            self._motion_data_list = np.array(list(data_list.values()))
            self._motion_data_keys = np.array(list(data_list.keys()))
        else:
            self._motion_data_list = np.array(self._motion_data_load)
            self._motion_data_keys = np.array(self._motion_data_load)
        
        self._num_unique_motions = len(self._motion_data_list)
        if self.mode == MotionlibMode.directory:
            if len(self._motion_data_load) == 0:
                raise FileNotFoundError(f"No .pkl files found in motion_file directory: {motion_file}")
            self._motion_data_load = joblib.load(self._motion_data_load[0]) # set self._motion_data_load to a sample of the data 
        logger.info(f"Loaded {self._num_unique_motions} motions")

    def setup_constants(self, fix_height = FixHeightMode.full_fix, multi_thread = True):
        self.fix_height = fix_height
        self.multi_thread = multi_thread
        
        #### Termination history
        self._curr_motion_ids = None
        self._termination_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._success_rate = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions  # For use in sampling batches

    @classmethod
    def _normalize_pose_aa_shape(cls, pose_aa, base_joints, extend_count, motion_id):
        """Normalize raw pose_aa to the preferred base-joint-only shape (T, base_joints, 3).

        Preferred input shape is `(T, base_joints, 3)`. Legacy `(T, base_joints + extend_count, 3)`
        is still accepted temporarily, but the trailing extended-joint slots are ignored and therefore
        deprecated.
        """
        expected_joints = base_joints + extend_count
        pose_aa = to_torch(pose_aa).clone()

        if pose_aa.dim() == 2:
            width = pose_aa.shape[1]
            if width == base_joints * 3:
                pose_aa = pose_aa.view(pose_aa.shape[0], base_joints, 3)
            elif extend_count > 0 and width == expected_joints * 3:
                pose_aa = pose_aa.view(pose_aa.shape[0], expected_joints, 3)
            else:
                raise ValueError(
                    f"pose_aa shape {tuple(pose_aa.shape)} is invalid; expected "
                    f"(T, {base_joints * 3}) or (T, {expected_joints * 3})"
                )
        elif pose_aa.dim() != 3:
            raise ValueError(
                f"pose_aa dim {pose_aa.dim()} is invalid; expected 2D flattened or 3D axis-angle tensor"
            )

        joints = pose_aa.shape[1]
        if joints == base_joints:
            return pose_aa

        if extend_count > 0 and joints == expected_joints:
            if not cls._pose_aa_deprecation_warned:
                warnings.warn(
                    f"pose_aa shape (T, {expected_joints}, 3) is deprecated and will be removed in a future "
                    f"release; the last {extend_count} extended-joint slots are ignored during training. "
                    f"Please migrate motion '{motion_id}' to (T, {base_joints}, 3).",
                    FutureWarning,
                    stacklevel=2,
                )
                cls._pose_aa_deprecation_warned = True
            return pose_aa[:, :base_joints, :]

        # Some older converters wrote a partial set of trailing extended-joint
        # slots. Accept those files only when the extra slots are empty, since
        # they carry no kinematic information and can be safely discarded.
        if joints > base_joints and torch.count_nonzero(pose_aa[:, base_joints:, :]) == 0:
            if not cls._pose_aa_deprecation_warned:
                warnings.warn(
                    f"pose_aa shape (T, {joints}, 3) has empty trailing joint slots; "
                    f"trimming to the {base_joints} base joints for motion '{motion_id}'.",
                    FutureWarning,
                    stacklevel=2,
                )
                cls._pose_aa_deprecation_warned = True
            return pose_aa[:, :base_joints, :]

        if joints < base_joints:
            raise ValueError(
                f"pose_aa joints {joints} < base joint count {base_joints}"
            )

        raise ValueError(
            f"pose_aa shape {tuple(pose_aa.shape)} is invalid; expected joint count "
            f"{base_joints} or deprecated legacy count {expected_joints}"
        )

    @staticmethod
    def _optional_contact_tensor(curr_file, key, length, default_value=0.5):
        """Load an optional per-frame contact tensor as shape (T, 1).

        Missing contact labels default to `default_value`. Existing labels are preserved.
        """
        if key not in curr_file:
            return torch.full((length, 1), float(default_value), dtype=torch.float32)

        value = curr_file[key]
        tensor = value if torch.is_tensor(value) else torch.from_numpy(np.asarray(value))
        tensor = tensor.to(dtype=torch.float32).view(-1, 1)
        if tensor.shape[0] == 1 and length > 1:
            tensor = tensor.expand(length, 1).contiguous()
        return tensor

    @staticmethod
    def _optional_sequence_tensor(curr_file, key, length, trailing_shape, default_value=0.0):
        """Load an optional per-frame float tensor with shape (T, *trailing_shape)."""
        trailing_shape = tuple(trailing_shape)
        target_shape = (int(length), *trailing_shape)
        if key not in curr_file:
            return torch.full(target_shape, float(default_value), dtype=torch.float32)

        value = curr_file[key]
        tensor = value if torch.is_tensor(value) else torch.from_numpy(np.asarray(value))
        tensor = tensor.to(dtype=torch.float32).reshape(-1, *trailing_shape)
        if tensor.shape[0] == 1 and length > 1:
            tensor = tensor.expand(target_shape).contiguous()
        return tensor

    @staticmethod
    def _optional_sequence_tensor_or_none(curr_file, key, length, trailing_shape):
        """Load an optional per-frame float tensor, or return None when the key is missing."""
        if key not in curr_file:
            return None

        trailing_shape = tuple(trailing_shape)
        target_shape = (int(length), *trailing_shape)
        value = curr_file[key]
        tensor = value if torch.is_tensor(value) else torch.from_numpy(np.asarray(value))
        tensor = tensor.to(dtype=torch.float32).reshape(-1, *trailing_shape)
        if tensor.shape[0] == 1 and length > 1:
            tensor = tensor.expand(target_shape).contiguous()
        return tensor

    @staticmethod
    def _openxr_to_isaac_translation(transl):
        """Convert OpenXR tracking-space translation to Isaac's world-axis convention.

        OpenXR uses a right-handed frame with +X right, +Y up, +Z backward.
        HumanX/Isaac uses +X forward, +Y left, +Z up, so the axis remap is:
        `(x, y, z)_openxr -> (-z, -x, y)_isaac`.
        """
        tensor = transl if torch.is_tensor(transl) else torch.from_numpy(np.asarray(transl))
        tensor = tensor.to(dtype=torch.float32).view(-1, 3)
        return torch.stack((-tensor[:, 2], -tensor[:, 0], tensor[:, 1]), dim=-1)

    @classmethod
    def _human_joints_world_from_local(cls, human_joints, human_adjusted_transl, z_offset=0.0):
        joints = human_joints if torch.is_tensor(human_joints) else torch.from_numpy(np.asarray(human_joints))
        joints = joints.to(dtype=torch.float32)
        transl_isaac = cls._openxr_to_isaac_translation(human_adjusted_transl)
        z_offset_tensor = torch.as_tensor(z_offset, dtype=torch.float32, device=transl_isaac.device)
        if z_offset_tensor.numel() == 1:
            transl_isaac[:, 2] = transl_isaac[:, 2] + z_offset_tensor
        else:
            transl_isaac[:, 2] = transl_isaac[:, 2] + z_offset_tensor.view(-1)
        root_joint = joints[:, 0:1, :]
        return joints - root_joint + transl_isaac.unsqueeze(1)

    @staticmethod
    def _align_human_joints_heading_to_target(human_joints_world, source_root_rot, target_root_rot):
        joints = (
            human_joints_world
            if torch.is_tensor(human_joints_world)
            else torch.from_numpy(np.asarray(human_joints_world))
        )
        joints = joints.to(dtype=torch.float32)
        source_rot = (
            source_root_rot
            if torch.is_tensor(source_root_rot)
            else torch.from_numpy(np.asarray(source_root_rot))
        ).to(dtype=torch.float32)
        target_rot = (
            target_root_rot
            if torch.is_tensor(target_root_rot)
            else torch.from_numpy(np.asarray(target_root_rot))
        ).to(dtype=torch.float32)
        batch_size = joints.shape[0]
        source_rot = source_rot.reshape(batch_size, -1, 4)[:, 0, :]
        target_rot = target_rot.reshape(batch_size, -1, 4)[:, 0, :]
        source_heading = calc_heading_quat(source_rot, w_last=True)
        target_heading = calc_heading_quat(target_rot, w_last=True)
        delta_heading = quat_mul(
            target_heading,
            quat_conjugate(source_heading, w_last=True),
            w_last=True,
        )
        root = joints[:, 0:1, :]
        offsets = joints - root
        delta_expand = delta_heading.unsqueeze(1).expand(-1, offsets.shape[1], -1).reshape(-1, 4)
        rotated_offsets = my_quat_rotate(delta_expand, offsets.reshape(-1, 3)).view_as(offsets)
        return rotated_offsets + root

    @staticmethod
    def _smpl_anchor_orientation_local_rot6d(
        base_quat_current,
        init_base_quat_robot,
        init_ref_quat,
        global_orient_quat,
    ):
        base_current = (
            base_quat_current
            if torch.is_tensor(base_quat_current)
            else torch.from_numpy(np.asarray(base_quat_current))
        ).to(dtype=torch.float32)
        init_base = (
            init_base_quat_robot
            if torch.is_tensor(init_base_quat_robot)
            else torch.from_numpy(np.asarray(init_base_quat_robot))
        ).to(dtype=torch.float32)
        init_ref = (
            init_ref_quat
            if torch.is_tensor(init_ref_quat)
            else torch.from_numpy(np.asarray(init_ref_quat))
        ).to(dtype=torch.float32)
        global_orient = (
            global_orient_quat
            if torch.is_tensor(global_orient_quat)
            else torch.from_numpy(np.asarray(global_orient_quat))
        ).to(dtype=torch.float32)

        batch_size = base_current.shape[0]
        base_current = base_current.reshape(batch_size, -1, 4)[:, 0, :]
        init_base = init_base.reshape(batch_size, -1, 4)[:, 0, :]
        init_ref = init_ref.reshape(batch_size, -1, 4)[:, 0, :]
        global_orient = global_orient.reshape(batch_size, -1, 4)

        apply_delta_heading = quat_mul(
            calc_heading_quat(init_base, w_last=True),
            calc_heading_quat_inv(init_ref, w_last=True),
            w_last=True,
        )
        apply_delta_heading = apply_delta_heading.unsqueeze(1).expand(-1, global_orient.shape[1], -1)
        base_current_inv = quat_conjugate(base_current, w_last=True).unsqueeze(1).expand_as(apply_delta_heading)
        q_rel = quat_mul(
            base_current_inv.reshape(-1, 4),
            quat_mul(
                apply_delta_heading.reshape(-1, 4),
                global_orient.reshape(-1, 4),
                w_last=True,
            ),
            w_last=True,
        ).reshape(base_current.shape[0], global_orient.shape[1], 4)

        rot_mats = quaternion_to_matrix(xyzw_to_wxyz(q_rel))
        rot6d = rot_mats[..., :2].reshape(base_current.shape[0], -1)
        return rot6d

    @classmethod
    def _required_obj_rot_quat_tensor(cls, curr_file, motion_id, length=None):
        """Load required object quaternion sequence as shape (T, 4), xyzw.

        Legacy `obj_rot_euler` is ignored if present and triggers a one-shot deprecation warning.
        """
        if "obj_rot_quat" not in curr_file:
            raise ValueError(f"motion {motion_id} is missing required field 'obj_rot_quat'")

        if "obj_rot_euler" in curr_file and not cls._obj_rot_euler_deprecation_warned:
            warnings.warn(
                "obj_rot_euler is deprecated and ignored; object angular velocity is now derived "
                "directly from obj_rot_quat. Please migrate your motion data to provide obj_rot_quat only.",
                FutureWarning,
                stacklevel=2,
            )
            cls._obj_rot_euler_deprecation_warned = True

        value = curr_file["obj_rot_quat"]
        tensor = value if torch.is_tensor(value) else torch.from_numpy(np.asarray(value))
        tensor = tensor.to(dtype=torch.float32).view(-1, 4)
        if length is not None and tensor.shape[0] == 1 and length > 1:
            tensor = tensor.expand(length, 4).contiguous()
        return tensor

    @staticmethod
    def _compute_angular_velocity_from_quat(quats, fps):
        """Compute angular velocity from quaternion sequence in xyzw format."""
        quats = to_torch(quats).clone().to(dtype=torch.float32).view(-1, 4)
        if quats.shape[0] == 0:
            return torch.zeros((0, 3), dtype=torch.float32)
        if quats.shape[0] == 1:
            return torch.zeros((1, 3), dtype=torch.float32, device=quats.device)

        delta = quat_mul_norm(
            quats[1:],
            quat_inverse(quats[:-1], w_last=True),
            w_last=True,
        )
        angle, axis = quat_angle_axis(delta, w_last=True)
        ang_vel = axis * angle.unsqueeze(-1) * float(fps)
        return torch.cat([ang_vel[:1], ang_vel], dim=0)

    def get_motion_actions(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]
        # import ipdb; ipdb.set_trace()
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        action = self._motion_actions[f0l]
        return action

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

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
        })
        return return_dict
    
    def load_motions(self, 
                     random_sample=True, 
                     start_idx=0, 
                     max_len=-1, 
                     target_heading = None):
        # import ipdb; ipdb.set_trace()

        motions = []
        _motion_lengths = []
        _motion_fps = []
        _motion_dt = []
        _motion_num_frames = []
        _motion_bodies = []
        _motion_aa = []
        has_action = False
        _motion_actions = []
        
        if flags.real_traj:
            self.q_gts, self.q_grs, self.q_gavs, self.q_gvs = [], [], [], []

        total_len = 0.0
        self.num_joints = len(self.skeleton_tree.node_names)
        num_motion_to_load = self.num_envs

        if random_sample:
            sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_motion_to_load, replacement=True).to(self._device)
        else:
            sample_idxes = torch.remainder(torch.arange(num_motion_to_load) + start_idx, self._num_unique_motions ).to(self._device)

        self._curr_motion_ids = sample_idxes
        self.curr_motion_keys = self._motion_data_keys[sample_idxes.cpu()]
        # self._sampling_batch_prob = self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()

        logger.info(f"Loading {num_motion_to_load} motions...")
        logger.info(f"Sampling motion: {sample_idxes[:5]}, ....")
        logger.info(f"Current motion keys: {self.curr_motion_keys[:5]}, ....")

        motion_data_list = self._motion_data_list[sample_idxes.cpu().numpy()]
        res_acc = self.load_motion_with_skeleton(motion_data_list, self.fix_height, target_heading, max_len)
        for f in track(range(len(res_acc)), description="Loading motions..."):
            motion_file_data, curr_motion = res_acc[f]
            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion.global_rotation.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

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
            if self.has_action:
                _motion_actions.append(curr_motion.action)
            
            if flags.real_traj:
                self.q_gts.append(curr_motion.quest_motion['quest_trans'])
                self.q_grs.append(curr_motion.quest_motion['quest_rot'])
                self.q_gavs.append(curr_motion.quest_motion['global_angular_vel'])
                self.q_gvs.append(curr_motion.quest_motion['linear_vel'])
                
            del curr_motion
        
        self._motion_lengths = torch.tensor(_motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32)
        self._motion_bodies = torch.stack(_motion_bodies).to(self._device).type(torch.float32)
        self._motion_aa = torch.tensor(np.concatenate(_motion_aa), device=self._device, dtype=torch.float32)

        self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device)
        # import ipdb; ipdb.set_trace()
        if self.has_action:
            self._motion_actions = torch.cat(_motion_actions, dim=0).float().to(self._device)
        self._num_motions = len(motions)
        
        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float().to(self._device)
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(self._device)
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float().to(self._device)
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)
        
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
        
        num_motions = self.num_motions()
        total_len = self.get_total_length()
        logger.info(f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")
        return motions

    def fix_trans_height(self, pose_aa, trans, fix_height_mode):
        if fix_height_mode == FixHeightMode.no_fix:
            return trans, 0
        with torch.no_grad():
            mesh_obj = self.mesh_parsers.mesh_fk(pose_aa[None, :1], trans[None, :1])
            height_diff = np.asarray(mesh_obj.vertices)[..., 2].min()
            trans[..., 2] -= height_diff
            
            return trans, height_diff

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
        return res
    

    def num_motions(self):
        return self._num_motions


    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion_num_steps(self, motion_ids=None):
        if motion_ids is None:
            return (self._motion_num_frames * self._sim_fps / self._motion_fps).ceil().int()
        else:
            return (self._motion_num_frames[motion_ids] * self._sim_fps / self._motion_fps).ceil().int()

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time
        motion_time = phase * motion_len
        # motion_time = phase * motion_len * 0.3 # wyh, only for flyshot!!!
        return motion_time.to(self._device)

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]


    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0) # clip blend to be within 0 and 1
        
        return frame_idx0, frame_idx1, blend


    def _get_num_bodies(self):
        return self.num_bodies


    def _local_rotation_to_dof_smpl(self, local_rot):
        B, J, _ = local_rot.shape
        dof_pos = quat_to_exp_map(local_rot[:, 1:])
        return dof_pos.reshape(B, -1)
