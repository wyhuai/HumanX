import sys
import os
import re
from loguru import logger
from isaacgym import gymtorch, gymapi
import torch
from humanoidverse.utils.torch_utils import to_torch, torch_rand_float
import numpy as np
import xml.etree.ElementTree as ET
import cv2
import joblib
from datetime import datetime
from rich.progress import Progress
from humanoidverse.simulator.isaacgym.isaacgym import IsaacGym as IsaacGymBase
from humanoidverse.utils.hoi_runtime import get_hoi_object_rigid_body_indices
from pathlib import Path
from typing import Optional

PERTURB_PROJECTORS = [
    ["small", 60],
]

class IsaacGym(IsaacGymBase):
    _OBJECT_URDF_BY_PREFIX = {
        "bin": "bin/bin.urdf",
        "basketball": "basketball/ball.urdf",
        "bmaster": "basketball/ball.urdf",
        "speed_capped": "basketball/ball.urdf",
        "football": "football/ball.urdf",
        "shuttlecock": "shuttlecock/ball.urdf",
        "box": "box/box.urdf",
        # Backward-compatible prefix for older cube-named motions.
        "cube": "box/box.urdf",
        "cuboid": "cuboid/cuboid.urdf",
    }

    def __init__(self, config, device):
        super().__init__(config, device)
        self.simulator_config = config.simulator.config
        self.robot_config = config.robot
        self.visualize_viewer = False
        # Optional: freeze object (ball) for a subset of envs by overwriting root state each physics step.
        # This is actor-level (per-env) control, independent of asset_options.fix_base_link.
        self._frozen_object_actor_ids_long = None   # (N,) long indices into flattened all_root_states
        self._frozen_object_actor_ids_int32 = None  # (N,) int32 for IsaacGym API call
        self._frozen_object_root_states = None      # (N, 13) root states (pos, quat, linvel, angvel)
        self.support_handles = []
        # Where to dump viewer recordings (png sequence + mp4).
        # Backward compatible: if user doesn't set save_rendering_dir, default to a local folder.
        save_dir = getattr(config, "save_rendering_dir", None)
        self.save_rendering_dir = Path(save_dir) if save_dir is not None else Path("renderings")

    def set_frozen_object(self, actor_ids: torch.Tensor, root_states: torch.Tensor):
        """Freeze object actors by forcing their root states every physics step.
        Args:
            actor_ids: (N,) actor indices in flattened root state tensor.
            root_states: (N, 13) desired root state per actor.
        """
        if actor_ids is None or actor_ids.numel() == 0:
            self.clear_frozen_object()
            return
        if root_states is None or root_states.numel() == 0:
            self.clear_frozen_object()
            return
        if root_states.shape[-1] != 13:
            raise ValueError(f"root_states must have last dim 13, got {tuple(root_states.shape)}")
        if actor_ids.shape[0] != root_states.shape[0]:
            raise ValueError(f"actor_ids and root_states must have same length, got {actor_ids.shape[0]} vs {root_states.shape[0]}")

        actor_ids_long = actor_ids.to(device=self.device, dtype=torch.long)
        self._frozen_object_actor_ids_long = actor_ids_long
        self._frozen_object_actor_ids_int32 = actor_ids_long.to(dtype=torch.int32)
        self._frozen_object_root_states = root_states.to(device=self.device, dtype=self.all_root_states.dtype)

    def clear_frozen_object(self):
        self._frozen_object_actor_ids_long = None
        self._frozen_object_actor_ids_int32 = None
        self._frozen_object_root_states = None


    def load_objects(self):
        def _read_urdf_mass(urdf_path: str) -> float:
            """Read the first <mass value="..."/> from a URDF. Minimal and robust for our object URDFs."""
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            mass_elem = root.find(".//inertial/mass")
            if mass_elem is None or "value" not in mass_elem.attrib:
                raise ValueError(f"URDF has no inertial mass value: {urdf_path}")
            return float(mass_elem.attrib["value"])

        def _read_urdf_reference_size(urdf_path: str) -> Optional[float]:
            """Return the largest base geometric dimension in meters.

            Isaac Gym applies one uniform actor scale, so the largest dimension
            is used as the characteristic size for matching motion metadata.
            """
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            geometry = root.find(".//visual/geometry")
            if geometry is None or len(geometry) == 0:
                return None
            shape = next(iter(geometry))
            tag = shape.tag.lower()
            if tag == "box" and "size" in shape.attrib:
                dims = [float(v) for v in shape.attrib["size"].split()]
                return max(dims) if len(dims) == 3 else None
            if tag == "sphere" and "radius" in shape.attrib:
                return 2.0 * float(shape.attrib["radius"])
            if tag == "cylinder" and "radius" in shape.attrib and "length" in shape.attrib:
                return max(2.0 * float(shape.attrib["radius"]), float(shape.attrib["length"]))
            return None

        def _motion_object_size(motion_file) -> Optional[float]:
            """Read a representative object_size from a converted motion PKL."""
            if motion_file is None:
                return None
            path = Path(os.path.expanduser(str(motion_file)))
            if not path.is_absolute():
                path = Path.cwd() / path
            if not path.is_file() or path.suffix.lower() != ".pkl":
                return None
            try:
                loaded = joblib.load(path)
            except Exception as exc:
                logger.warning(f"Could not inspect object_size from motion file {path}: {exc}")
                return None
            motions = loaded.values() if isinstance(loaded, dict) else []
            sizes = []
            for motion in motions:
                if not isinstance(motion, dict) or "object_size" not in motion:
                    continue
                value = np.asarray(motion["object_size"], dtype=np.float32).reshape(-1)
                if value.size not in (1, 3) or not np.all(np.isfinite(value)) or not np.all(value > 0):
                    logger.warning(f"Ignoring invalid object_size in {path}: {value}")
                    continue
                sizes.append(float(np.max(value)))
            if not sizes:
                return None
            if max(sizes) - min(sizes) > 1e-5:
                logger.warning(
                    f"Motion file {path} contains multiple object_size values {sizes}; "
                    f"using the first value {sizes[0]:.6g} for the shared object asset."
                )
            return sizes[0]

        #setup object info
        self.num_of_objects = 2
        self.ball_texture_path = None
        self._ball_texture_handle = None
        self.support_box_size = (0.2, 0.2, 0.02)

        def _resolve_ball_texture_path(asset_root: str, urdf_file: str, texture_file):
            # Explicit config wins. Relative path is resolved from asset_root.
            if isinstance(texture_file, str):
                texture_file = texture_file.strip()
                if texture_file:
                    tex_path = texture_file if os.path.isabs(texture_file) else os.path.join(asset_root, texture_file)
                    tex_path = os.path.abspath(tex_path)
                    return tex_path if os.path.isfile(tex_path) else None

            # Auto mode: pick the first image in the URDF sibling directory.
            urdf_dir = os.path.dirname(os.path.abspath(os.path.join(asset_root, urdf_file)))
            if not os.path.isdir(urdf_dir):
                return None
            img_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            for name in sorted(os.listdir(urdf_dir)):
                if os.path.splitext(name)[1].lower() in img_exts:
                    candidate = os.path.join(urdf_dir, name)
                    if os.path.isfile(candidate):
                        return candidate
            return None
        
        # Read object scales from config
        if hasattr(self.simulator_config, 'object_scales'):
            configured_ball_scale = self.simulator_config.object_scales.get('ball_scale', None)
            self.proj_scale = self.simulator_config.object_scales.get('proj_scale', 1.5)
        else:
            configured_ball_scale = None
            self.proj_scale = 1.5
        
        # Load an explicit object asset, or infer it from the motion filename.
        if hasattr(self.simulator_config, 'objects') and hasattr(self.simulator_config.objects, 'ball'):
            ball_cfg = self.simulator_config.objects.ball
            asset_root = ball_cfg.get('asset_root', "humanoidverse/asset/objects")
            configured_urdf_file = ball_cfg.get('urdf_file', None)
            auto_detect_from_motion = bool(ball_cfg.get('auto_detect_from_motion', True))
            if configured_urdf_file:
                urdf_file = str(configured_urdf_file)
            elif auto_detect_from_motion:
                urdf_file = self._infer_object_urdf_from_motion_file(
                    getattr(getattr(self.robot_config, "motion", None), "motion_file", None)
                )
                if urdf_file is None:
                    urdf_file = "basketball/ball.urdf"
                    logger.warning(
                        f"Could not infer an object asset from motion_file="
                        f"{getattr(getattr(self.robot_config, 'motion', None), 'motion_file', None)!r}; "
                        "falling back to basketball/ball.urdf. "
                        "Set simulator.config.objects.ball.urdf_file explicitly "
                        "for custom object names."
                    )
            else:
                urdf_file = "basketball/ball.urdf"
            texture_file = ball_cfg.get("texture_file", None) if "texture_file" in ball_cfg else None
            # Asset options from config
            asset_options = gymapi.AssetOptions()
            # Backward compatible: default is dynamic ball (fix_base_link=False).
            # If True, the ball becomes effectively static/kinematic and will not move during simulation.
            asset_options.fix_base_link = ball_cfg.get('fix_base_link', False)
            asset_options.disable_gravity = ball_cfg.get('disable_gravity', False)
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.angular_damping = ball_cfg.get('angular_damping', 0.01)
            asset_options.linear_damping = ball_cfg.get('linear_damping', 0.01)
            asset_options.max_angular_velocity = ball_cfg.get('max_angular_velocity', 100.0)
            asset_options.density = ball_cfg.get('density', 1000.0)
            friction = ball_cfg.get('friction', 1.0)
            restitution = ball_cfg.get('restitution', 0.81)  # Default restitution for backward compatibility
            # mass policy (resolved after asset is loaded):
            # 1) if config provides a number -> use it as final mass
            # 2) if config is null OR missing -> use asset/URDF default mass (with scale^3)
            ball_mass_cfg = ball_cfg.get("mass", None) if "mass" in ball_cfg else None
        else:
            # Default values for backward compatibility
            asset_root = "humanoidverse/asset/objects"
            urdf_file = "basketball/ball.urdf"
            texture_file = None
            asset_options = gymapi.AssetOptions()
            asset_options.disable_gravity = False
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01
            asset_options.max_angular_velocity = 100.0
            asset_options.density = 1000.0
            friction = 1.0
            restitution = 0.81  # Default restitution for backward compatibility
            ball_mass_cfg = None

        self.ball_asset = self.gym.load_asset(self.sim, asset_root, urdf_file, asset_options)
        self.ball_asset_idx = 1
        self.ball_texture_path = _resolve_ball_texture_path(asset_root, urdf_file, texture_file)
        self.support_asset_idx = 2

        # Resolve default mass from URDF (IsaacGym API may not expose asset rigid-body props reliably).
        urdf_path = os.path.join(asset_root, urdf_file)
        if configured_ball_scale is None:
            source_size = _motion_object_size(
                getattr(getattr(self.robot_config, "motion", None), "motion_file", None)
            )
            asset_size = _read_urdf_reference_size(urdf_path)
            if source_size is not None and asset_size is not None:
                self.ball_scale = source_size / asset_size
                logger.info(
                    f"Auto-scaled object asset from motion object_size={source_size:.6g} m "
                    f"and URDF reference size={asset_size:.6g} m: ball_scale={self.ball_scale:.6g}"
                )
            else:
                self.ball_scale = 1.0
                if source_size is not None:
                    logger.warning(
                        f"Motion contains object_size={source_size:.6g} m but URDF geometry size "
                        f"could not be inferred from {urdf_path}; using ball_scale=1.0."
                    )
        else:
            self.ball_scale = float(configured_ball_scale)
            logger.info(f"Using configured object ball_scale={self.ball_scale:.6g}")
        self.ball_mass_urdf = _read_urdf_mass(urdf_path)
        self.ball_mass_asset_scaled = self.ball_mass_urdf * float(self.ball_scale) ** 3
        self.ball_mass_cfg = None if ball_mass_cfg is None else float(ball_mass_cfg)
        # Final mass used in _build_objects (no branching there)
        # self.ball_mass_final = self.ball_mass_asset_scaled if self.ball_mass_cfg is None else self.ball_mass_cfg
        self.ball_mass_final = self.ball_mass_urdf if self.ball_mass_cfg is None else self.ball_mass_cfg
        
        # Save friction and restitution for use in _build_objects
        self.ball_friction = friction
        self.ball_restitution = restitution
        
        #set rigid shape properties
        ball_asset_props = self.gym.get_asset_rigid_shape_properties(self.ball_asset)
        ball_asset_props[0].friction = friction
        self.gym.set_asset_rigid_shape_properties(self.ball_asset, ball_asset_props)

        support_cfg = None
        if hasattr(self.simulator_config, "objects") and hasattr(self.simulator_config.objects, "support"):
            support_cfg = self.simulator_config.objects.support
        if support_cfg is not None and "size" in support_cfg:
            support_size = tuple(float(v) for v in support_cfg.get("size"))
        else:
            support_size = self.support_box_size
        self.support_box_size = support_size
        self.support_box_thickness = float(self.support_box_size[2])

        support_asset_options = gymapi.AssetOptions()
        support_asset_options.fix_base_link = True
        support_asset_options.disable_gravity = True
        support_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self.support_asset = self.gym.create_box(
            self.sim,
            float(self.support_box_size[0]),
            float(self.support_box_size[1]),
            float(self.support_box_size[2]),
            support_asset_options,
        )
        support_friction = float(support_cfg.get("friction", friction)) if support_cfg is not None else float(friction)
        support_restitution = float(support_cfg.get("restitution", 0.0)) if support_cfg is not None else 0.0
        self.support_friction = support_friction
        self.support_restitution = support_restitution
        support_asset_props = self.gym.get_asset_rigid_shape_properties(self.support_asset)
        for p in support_asset_props:
            p.friction = support_friction
            p.restitution = support_restitution
        self.gym.set_asset_rigid_shape_properties(self.support_asset, support_asset_props)
        
        #set force sensor
        # sensor_pose = gymapi.Transform()
        # sensor_options = gymapi.ForceSensorProperties()
        # sensor_options.enable_forward_dynamics_forces = False # for example gravity
        # sensor_options.enable_constraint_solver_forces = True # for example contacts
        # sensor_options.use_world_frame = True # report forces in world frame (easier to get vertical components)
        # index = self.gym.find_asset_rigid_body_index(self.ball_asset, "ball")
        # self.gym.create_asset_force_sensor(self.ball_asset, index, sensor_pose, sensor_options)
        # self.force_sensor_step += 1

    @classmethod
    def _infer_object_urdf_from_motion_file(cls, motion_file):
        """Infer an object URDF from the motion filename's leading token."""
        if motion_file is None:
            return None

        motion_file = os.path.normpath(str(motion_file))
        filename = os.path.basename(motion_file)
        stem = os.path.splitext(filename)[0].lower()
        prefix = next(
            (
                candidate
                for candidate in sorted(cls._OBJECT_URDF_BY_PREFIX, key=len, reverse=True)
                if stem == candidate
                or re.match(rf"^{re.escape(candidate)}(?:[-_\s]|$)", stem)
            ),
            None,
        )
        urdf_file = cls._OBJECT_URDF_BY_PREFIX.get(prefix)
        if urdf_file is not None:
            logger.info(
                f"Auto-selected object asset {urdf_file} from motion filename "
                f"{filename} (prefix={prefix})"
            )
        return urdf_file

    def load_assets(self):
        # Create robot asset
        asset_root = self.robot_config.asset.asset_root
        asset_file = self.robot_config.asset.urdf_file
        self.robot_asset = self._setup_robot_asset_when_env_created(asset_root, asset_file, self.robot_config.asset)
        self.num_dof, self.num_bodies, self.dof_names, self.body_names = self._setup_robot_props_when_env_created()

        # Create table and cube assets
        self.load_objects()
        if self.robot_config.proj:
            self._load_proj_asset()
            self.num_of_objects += 1
        
        #asset info
        self.num_of_asset = 1 + self.num_of_objects#num of robot + num of objects
        
        # assert if  aligns with config
        assert self.num_dof == len(self.robot_config.dof_names), "Number of DOFs must be equal to number of actions"
        assert self.num_bodies == len(self.robot_config.body_names), "Number of bodies must be equal to number of body names"
        assert self.dof_names == self.robot_config.dof_names, "DOF names must match the config"
        assert self.body_names == self.robot_config.body_names, "Body names must match the config"


    def _load_proj_asset(self):
        asset_root = "humanoidverse/asset/objects" 
        small_asset_file = "basketball/ball.urdf"  
        
        small_asset_options = gymapi.AssetOptions()
        small_asset_options.angular_damping = 0.01
        small_asset_options.linear_damping = 0.01
        small_asset_options.max_angular_velocity = 100.0
        small_asset_options.density = 200.0
        # small_asset_options.fix_base_link = True
        small_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._small_proj_asset = self.gym.load_asset(self.sim, asset_root, small_asset_file, small_asset_options)

        # Optional projectile friction (backward compatible: if not set, do nothing)
        self.proj_friction = None
        if hasattr(self.simulator_config, "objects") and hasattr(self.simulator_config.objects, "proj"):
            proj_cfg = self.simulator_config.objects.proj
            # allow explicit null / missing to mean "leave Isaac Gym defaults unchanged"
            self.proj_friction = proj_cfg.get("friction", None)

        if self.proj_friction is not None:
            proj_asset_props = self.gym.get_asset_rigid_shape_properties(self._small_proj_asset)
            for p in proj_asset_props:
                p.friction = float(self.proj_friction)
            self.gym.set_asset_rigid_shape_properties(self._small_proj_asset, proj_asset_props)

        return


    def _setup_robot_asset_when_env_created(self, asset_root, asset_file, asset_cfg):
        asset_path = os.path.join(asset_root, asset_file)
        gym_asset_root = os.path.dirname(asset_path)
        gym_asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()

        # wyh
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.max_convex_hulls = 32

        def set_value_if_not_none(prev_value, new_value):
            return new_value if new_value is not None else prev_value

        asset_config_options = [
            "default_dof_drive_mode",
            "collapse_fixed_joints",
            "replace_cylinder_with_capsule",
            "flip_visual_attachments",
            "fix_base_link",
            "density",
            "angular_damping",
            "linear_damping",
            "max_angular_velocity",
            "max_linear_velocity",
            "armature",
            "thickness",
            "disable_gravity",
        ]
        for option in asset_config_options:
            option_value = set_value_if_not_none(
                getattr(asset_options, option), getattr(asset_cfg, option)
            )
            setattr(asset_options, option, option_value)

        self.robot_asset = self.gym.load_asset(self.sim, gym_asset_root, gym_asset_file, asset_options)
        
        #set force sensor
        # self.force_sensor_step = 0
        # hoi_bodies_name = self.robot_config.hoi_contact_bodies
        # for body_name in hoi_bodies_name:
        #     sensor_pose = gymapi.Transform()
        #     sensor_options = gymapi.ForceSensorProperties()
        #     sensor_options.enable_forward_dynamics_forces = False # for example gravity
        #     sensor_options.enable_constraint_solver_forces = True # for example contacts
        #     sensor_options.use_world_frame = True # report forces in world frame (easier to get vertical components)
        #     index = self.gym.find_asset_rigid_body_index(self.robot_asset, body_name)
        #     self.gym.create_asset_force_sensor(self.robot_asset, index, sensor_pose, sensor_options)
        #     self.force_sensor_step += 1
            
        return self.robot_asset
    
    def _setup_robot_props_when_env_created(self):
        props = super()._setup_robot_props_when_env_created()
        self.robot_asset_idx = 0  # assume robot loaded first
        return props

    def create_envs(self, num_envs, env_origins, base_init_state):
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.num_envs = num_envs
        self.env_config = self.config
        self.env_origins = env_origins
        self.base_init_state = base_init_state
        self.envs = []
        self.robot_handles = []
        self.ball_handles = []
        if self.robot_config.proj:
            self._proj_handles = []



        # # Define start pose for table
        # self.tableA_start_pose = gymapi.Transform()
        # self.tableA_start_pose.p = gymapi.Vec3(*self.init_tableA_states[:3])
        # self.tableA_start_pose.r = gymapi.Quat(*self.init_tableA_states[3:7])
        # self._table_surface_pos = np.array(self.init_tableA_states[:3]) + np.array([0, 0, self.tableA_thickness / 2])
        # # self.reward_settings["table_height"] = self._tableA_surface_pos[2]
        # self.tableB_start_pose = gymapi.Transform()
        # self.tableB_start_pose.p = gymapi.Vec3(*self.init_tableB_states[:3])
        # self.tableB_start_pose.r = gymapi.Quat(*self.init_tableB_states[3:7])
        # self._table_surface_pos = np.array(self.init_tableB_states[:3]) + np.array([0, 0, self.tableB_thickness / 2])
        # # self.reward_settings["table_height"] = self._tableA_surface_pos[2]
        # # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        # self.cubeA_start_pose = gymapi.Transform()
        # self.cubeA_start_pose.p = gymapi.Vec3(*self.init_cubeA_states[:3])
        # self.cubeA_start_pose.r = gymapi.Quat(*self.init_cubeA_states[3:7])
        # self.cubeB_start_pose = gymapi.Transform()
        # self.cubeB_start_pose.p = gymapi.Vec3(*self.init_cubeB_states[:3])
        # self.cubeB_start_pose.r = gymapi.Quat(*self.init_cubeB_states[3:7])

        with Progress() as progress:
            task = progress.add_task(
                f"Creating {self.num_envs} environments...", total=self.num_envs
            )
            # compute aggregate size
            num_robot_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
            num_robot_shapes = self.gym.get_asset_rigid_shape_count(self.robot_asset)
            # max_agg_bodies = num_robot_bodies + 4     # 1 for tableA/B, cubeA/B
            # max_agg_shapes = num_robot_shapes + 4     # 1 for tableA/B, cubeA/B
            max_agg_bodies = num_robot_bodies + 3
            max_agg_shapes = num_robot_shapes + 3
            for i in range(self.num_envs):
                # create env instance
                env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
                # self._build_each_env(i, env_handle)
                
                self.gym.begin_aggregate(env_handle, max_agg_bodies, max_agg_shapes, True)
                self._build_each_env(i, env_handle)
                self.gym.end_aggregate(env_handle)
                progress.update(task, advance=1)

        return self.envs, self.robot_handles

    def _build_objects(self, env_id, env_ptr):
        # # Create table
        # self._tableA_id = self.gym.create_actor(env_ptr, self.tableA_asset, self.tableA_start_pose, "tableA", env_id, 0, 0)
        # self._tableB_id = self.gym.create_actor(env_ptr, self.tableB_asset, self.tableB_start_pose, "tableB", env_id, 0, 0)
        # # Create cubes
        # self._cubeA_id = self.gym.create_actor(env_ptr, self.cubeA_asset, self.cubeA_start_pose, "cubeA", env_id, 0, 0)
        # self._cubeB_id = self.gym.create_actor(env_ptr, self.cubeB_asset, self.cubeB_start_pose, "cubeB", env_id, 0, 0)
        # # Set colors
        # self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, self.cubeA_color)
        # self.gym.set_rigid_body_color(env_ptr, self._cubeB_id, 0, gymapi.MESH_VISUAL, self.cubeB_color)
        # # Set mass
        # cubeA_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, self._cubeA_id)
        # cubeA_body_props[0].mass = 0.2
        # self.gym.set_actor_rigid_body_properties(env_ptr, self._cubeA_id, cubeA_body_props)
        # # Load basketball
        self.ball_pose = gymapi.Transform()
        self.ball_pose.p = gymapi.Vec3(0, 0, 0.8)  
        pos = self.env_origins[env_id].clone()
        pos[:2] += torch_rand_float(-1., 1., (2, 1), device=str(self.device)).squeeze(1)
        self.ball_pose.p = gymapi.Vec3(*pos)
        self.ball_pose.r = gymapi.Quat(0, 0, 0, 1) 
        
        ball_handle = self.gym.create_actor(env_ptr, 
                                              self.ball_asset, 
                                              self.ball_pose, 
                                              "ball", 
                                              env_id, 0)
        
        #set rigid shape properties
        ball_props =  self.gym.get_actor_rigid_shape_properties(env_ptr, ball_handle)
        # Modify the properties
        for b in ball_props:
            b.restitution = self.ball_restitution  # Use saved restitution value (with backward compatibility)
            b.friction = self.ball_friction  # Set friction from config
        self.gym.set_actor_rigid_shape_properties(env_ptr, ball_handle, ball_props)  

        # #set rigid body properties
        # ball_asset_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, ball_handle)
        # for b in ball_asset_body_props:
        #     b.mass = 0.3 #1.00 #0.66 #1.6
        # self.gym.set_actor_rigid_body_properties(env_ptr, ball_handle, ball_asset_body_props, recomputeInertia=True)
        # # Use ball scale from config
        # ball_size = self.ball_scale
        # self.gym.set_actor_scale(env_ptr, ball_handle, ball_size)

        # === Object DR (per-env) ===
        # Match existing per-env DR style: sample once per env at creation time.
        dr_cfg = getattr(self.env_config, "domain_rand", None)
        rand_obj_scale = bool(getattr(dr_cfg, "randomize_object_scale", False)) if dr_cfg is not None else False
        rand_obj_mass = bool(getattr(dr_cfg, "randomize_object_mass", False)) if dr_cfg is not None else False
        rand_obj_com = bool(getattr(dr_cfg, "randomize_object_com", False)) if dr_cfg is not None else False
        rand_obj_inertia = bool(getattr(dr_cfg, "randomize_object_inertia_scale", False)) if dr_cfg is not None else False

        base_scale = float(self.ball_scale)
        ball_size = base_scale
        if rand_obj_scale:
            rng = getattr(dr_cfg, "object_scale_range", [1.0, 1.0])
            scale_factor = float(np.random.uniform(rng[0], rng[1]))
            ball_size = base_scale * scale_factor

        # IMPORTANT: set scale BEFORE setting mass.
        # Isaac Gym's set_actor_scale may update mass properties (e.g., if density is set / volume changes).
        self.gym.set_actor_scale(env_ptr, ball_handle, ball_size)

        # Mass DR: do NOT couple mass to size/scale.
        # Start from base mass (as configured / resolved in load_objects) and optionally apply a factor.
        base_mass = float(self.ball_mass_final)
        mass = base_mass
        if rand_obj_mass:
            rng = getattr(dr_cfg, "object_mass_range", [1.0, 1.0])
            mass_factor = float(np.random.uniform(rng[0], rng[1]))
            mass = base_mass * mass_factor

        # Always set mass AFTER scaling.
        ball_asset_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, ball_handle)
        for b in ball_asset_body_props:
            b.mass = mass
        self.gym.set_actor_rigid_body_properties(env_ptr, ball_handle, ball_asset_body_props, recomputeInertia=True)

        # Independent COM & inertia randomization (OmniRetarget: COM ±0.08m, inertia 50–150%).
        # IMPORTANT: this must be applied AFTER recomputeInertia=True above, otherwise it may get overwritten.
        if rand_obj_com or rand_obj_inertia:
            # NOTE(Isaac Gym / GPU pipeline):
            # Under PhysX GPU pipeline, runtime COM updates via
            # set_actor_rigid_body_properties are often unsupported/ineffective.
            # We still attempt to write b.com here, but more reliable options are:
            # CPU pipeline, or baking COM into the asset inertial.
            if rand_obj_com and bool(getattr(self.sim_params, "use_gpu_pipeline", False)) and not getattr(self, "_warned_gpu_pipeline_com", False):
                logger.warning(
                    "IsaacGym GPU pipeline typically does not support runtime rigid-body "
                    "COM (center of mass) updates. randomize_object_com may be ineffective; "
                    "mass/inertia scaling is still available."
                )
                self._warned_gpu_pipeline_com = True
            ball_asset_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, ball_handle)
            # Assume single rigid body for ball assets used here.
            if len(ball_asset_body_props) > 0:
                b = ball_asset_body_props[0]

                if rand_obj_com:
                    # Object COM offset in meters (object local frame).
                    # Backward compatible: if range is missing, do nothing.
                    rng = getattr(dr_cfg, "object_com_range", None)
                    if rng is not None and len(rng) == 3:
                        dx = float(np.random.uniform(rng[0][0], rng[0][1]))
                        dy = float(np.random.uniform(rng[1][0], rng[1][1]))
                        dz = float(np.random.uniform(rng[2][0], rng[2][1]))
                        b.com.x += dx
                        b.com.y += dy
                        b.com.z += dz

                if rand_obj_inertia:
                    rng = getattr(dr_cfg, "object_inertia_scale_range", [1.0, 1.0])
                    s = float(np.random.uniform(rng[0], rng[1]))
                    # Gym rigid-body inertia is exposed as principal moments (Vec3): x,y,z
                    b.inertia.x *= s
                    b.inertia.y *= s
                    b.inertia.z *= s

                # Do NOT recompute inertia here, otherwise we lose the randomization above.
                self.gym.set_actor_rigid_body_properties(
                    env_ptr, ball_handle, ball_asset_body_props, recomputeInertia=False
                )

        if env_id < 3:
            try:
                _props_after = self.gym.get_actor_rigid_body_properties(env_ptr, ball_handle)
                logger.info(
                    f"[ball] env_id={env_id} scale={ball_size} "
                    f"mass_urdf(urdf)={self.ball_mass_urdf} "
                    f"mass_asset_scaled={self.ball_mass_asset_scaled} "
                    f"mass_cfg={self.ball_mass_cfg} "
                    f"mass_final={self.ball_mass_final} "
                    f"dr_scale={rand_obj_scale} dr_mass={rand_obj_mass} "
                    f"mass(base)={base_mass} mass(set)={mass} "
                    f"mass(after)={_props_after[0].mass}"
                )
            except Exception:
                pass
        

        #append handler
        self.ball_handles.append(ball_handle)
        
        #viewer setting
        if not self.headless:
            self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL,
                                        gymapi.Vec3(1.5, 1.5, 1.5))
                                        # gymapi.Vec3(0., 1.0, 1.5))
            if self.ball_texture_path:
                if self._ball_texture_handle is None:
                    self._ball_texture_handle = self.gym.create_texture_from_file(self.sim, self.ball_texture_path)
                self.gym.set_rigid_body_texture(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL, self._ball_texture_handle)

        support_pose = gymapi.Transform()
        support_pose.p = gymapi.Vec3(float(pos[0]), float(pos[1]), float(pos[2] - 100.0))
        support_pose.r = gymapi.Quat(0, 0, 0, 1)
        support_handle = self.gym.create_actor(
            env_ptr,
            self.support_asset,
            support_pose,
            "support",
            env_id,
            0,
        )
        support_props = self.gym.get_actor_rigid_shape_properties(env_ptr, support_handle)
        for p in support_props:
            p.friction = self.support_friction
            p.restitution = self.support_restitution
        self.gym.set_actor_rigid_shape_properties(env_ptr, support_handle, support_props)
        self.support_handles.append(support_handle)
        if not self.headless:
            self.gym.set_rigid_body_color(
                env_ptr,
                support_handle,
                0,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(0.2, 0.2, 0.2),
            )


    def _build_proj(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        for i, obj in enumerate(PERTURB_PROJECTORS):
            default_pose = gymapi.Transform()
            default_pose.p.x = 1 + i
            default_pose.p.z = 2.5
            obj_type = obj[0]
            if (obj_type == "small"):
                proj_asset = self._small_proj_asset
            elif (obj_type == "large"):
                proj_asset = self._large_proj_asset

            proj_handle = self.gym.create_actor(env_ptr, proj_asset, default_pose, "proj{:d}".format(i), col_group, col_filter, segmentation_id)
            self._proj_handles.append(proj_handle)
            # Use projectile scale from config
            self.gym.set_actor_scale(env_ptr, proj_handle, self.proj_scale)

        return


    def _build_each_env(self, env_id, env_ptr):
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        pos = self.env_origins[env_id].clone()
        pos[:2] += torch_rand_float(-1., 1., (2, 1), device=str(self.device)).squeeze(1)
        start_pose.p = gymapi.Vec3(*pos)

        robot_handle = self.gym.create_actor(env_ptr, 
                                             self.robot_asset, 
                                             start_pose, 
                                             self.env_config.robot.asset.robot_type, 
                                             env_id, 
                                             self.env_config.robot.asset.self_collisions, 0)
        self._body_list = self.gym.get_actor_rigid_body_names(env_ptr, robot_handle)

        # NOTE: set per-env friction on the ACTOR (instance) rather than mutating the shared asset defaults.
        rigid_shape_props_actor = self.gym.get_actor_rigid_shape_properties(env_ptr, robot_handle)
        rigid_shape_props_actor = self._process_rigid_shape_props(rigid_shape_props_actor, env_id)
        self.gym.set_actor_rigid_shape_properties(env_ptr, robot_handle, rigid_shape_props_actor)

        # Keep the DOF workflow consistent with other per-env properties: read actor defaults, then override.
        dof_props_actor = self.gym.get_actor_dof_properties(env_ptr, robot_handle)
        dof_props = self._process_dof_props(dof_props_actor, env_id)
        self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_props)
        body_props = self.gym.get_actor_rigid_body_properties(env_ptr, robot_handle)
        body_props = self._process_rigid_body_props(body_props, env_id)
        self.gym.set_actor_rigid_body_properties(env_ptr, robot_handle, body_props, recomputeInertia=True)

        # Build objects
        self._build_objects(env_id, env_ptr)

        #build proj
        if self.robot_config.proj:
            self._build_proj(env_id, env_ptr)

        self.envs.append(env_ptr)
        self.robot_handles.append(robot_handle)


    def _build_proj_tensors(self):
        self._proj_dist_min = 4
        self._proj_dist_max = 5
        self._proj_h_min = 0.25
        self._proj_h_max = 2
        self._proj_steps = 150
        self._proj_warmup_steps = 1
        self._proj_speed_min = 30
        self._proj_speed_max = 40

        num_actors = self._get_num_actors_per_env()
        num_objs = len(PERTURB_PROJECTORS)
        self._proj_states = self.all_root_states.view(self.num_envs, num_actors, self.all_root_states.shape[-1])[..., (num_actors - num_objs):, :]
        
        self._proj_actor_ids = num_actors * np.arange(self.num_envs)
        self._proj_actor_ids = np.expand_dims(self._proj_actor_ids, axis=-1)
        self._proj_actor_ids = self._proj_actor_ids + np.reshape(np.array(self._proj_handles), [self.num_envs, num_objs])
        self._proj_actor_ids = self._proj_actor_ids.flatten()
        self._proj_actor_ids = to_torch(self._proj_actor_ids, device=self.device, dtype=torch.int32)
        
        # bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        # contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        # contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        # self._proj_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., (num_actors - num_objs):, :]

        self._calc_perturb_times()
        
        return

    def _calc_perturb_times(self):
        self._perturb_timesteps = []
        total_steps = 0
        for i, obj in enumerate(PERTURB_PROJECTORS):
            curr_time = obj[1]
            total_steps += curr_time
            self._perturb_timesteps.append(total_steps)

        self._perturb_timesteps = np.array(self._perturb_timesteps)

        return




    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each rigid shape (from asset or actor)
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties (same list, mutated in-place)
        """
        if self.env_config.domain_rand.randomize_friction:
            self._ground_friction_values = torch.zeros(self.num_envs, self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)
            if env_id==0:
                # prepare friction randomization
                friction_range = self.env_config.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            if len(props) != self.num_bodies and env_id<3:
                logger.warning("Number of rigid shapes does not match number of bodies")
                logger.warning(f"len(RigidShapeProperties): {len(props)}")
                logger.warning(f"self.num_bodies: {self.num_bodies}")
                logger.warning("Only randomizing friction of number of bodies")

            num_available_friction_shapes = min(len(props), self.num_bodies)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

            for s in range(num_available_friction_shapes):
                props[s].friction = self.friction_coeffs[env_id]
                self._ground_friction_values[env_id, s] += self.friction_coeffs[env_id].squeeze()
                if env_id<3:
                    logger.debug(f"Friction of shape {s}: {props[s].friction} (after randomization)")
                    
        elif not self.env_config.domain_rand.randomize_friction:
            # set default friction
            for s in range(len(props)): # wyh
                props[s].friction = 1.0

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF (from asset or actor)
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        # Actor DOF properties are applied per-environment, so config-driven overrides
        # must be written back for every env, not only the first one.
        for i in range(len(props)):
            config_effort = float(self.robot_config.dof_effort_limit_list[i])
            urdf_effort = props["effort"][i].item()
            if env_id == 0 and abs(urdf_effort - config_effort) > 1e-5:
                logger.warning(
                    f"Overriding DOF effort limit from URDF with robot config: "
                    f"{self.dof_names[i]} urdf={urdf_effort} config={config_effort}"
                )
            props["effort"][i] = config_effort

        if env_id==0:
            self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.hard_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.hard_dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit

        #set armature
        self.armatures = np.zeros(self.num_dof)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            self.armatures[i] = self.robot_config.dof_armature_prop[name] 
        
        props["armature"] = self.armatures

        return props

    def _process_rigid_body_props(self, props, env_id):
        if env_id<3:
            sum = 0
            for i, p in enumerate(props):
                sum += p.mass
                logger.debug(f"Mass of body {i}: {p.mass} (before randomization)")
            logger.debug(f"Total mass {sum} (before randomization)")

        # randomize base com
        if self.env_config.domain_rand.randomize_base_com:
            # NOTE(Isaac Gym / GPU pipeline):
            # Under PhysX GPU pipeline, runtime rigid-body COM updates are usually
            # unsupported/ineffective (this limitation is documented by the API).
            # We still write props[*].com according to config, but with
            # use_gpu_pipeline=True this randomization may not affect final dynamics.
            if bool(getattr(self.sim_params, "use_gpu_pipeline", False)) and not getattr(self, "_warned_gpu_pipeline_com", False):
                logger.warning(
                    "IsaacGym GPU pipeline typically does not support runtime rigid-body "
                    "COM (center of mass) updates. randomize_base_com/randomize_object_com "
                    "may be ineffective. For COM domain randomization, use CPU pipeline or "
                    "write COM into URDF/asset inertial."
                )
                self._warned_gpu_pipeline_com = True
            self._base_com_bias = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
            if env_id<3:
                logger.debug("randomizing base com")
            try:
                torso_index = self._body_list.index("torso_link")
            except:
                torso_index = self._body_list.index("pelvis") # for fixed upper URDF we only have pelvis link
            assert torso_index != -1

            com_x_bias = np.random.uniform(self.env_config.domain_rand.base_com_range.x[0], self.env_config.domain_rand.base_com_range.x[1])
            com_y_bias = np.random.uniform(self.env_config.domain_rand.base_com_range.y[0], self.env_config.domain_rand.base_com_range.y[1])
            com_z_bias = np.random.uniform(self.env_config.domain_rand.base_com_range.z[0], self.env_config.domain_rand.base_com_range.z[1])

            self._base_com_bias[env_id, 0] += com_x_bias
            self._base_com_bias[env_id, 1] += com_y_bias
            self._base_com_bias[env_id, 2] += com_z_bias

            props[torso_index].com.x += com_x_bias
            props[torso_index].com.y += com_y_bias
            props[torso_index].com.z += com_z_bias

        # randomize link mass
        if self.env_config.domain_rand.randomize_link_mass:
            self._link_mass_scale = torch.ones(self.num_envs, len(self.env_config.robot.randomize_link_body_names), dtype=torch.float, device=self.device, requires_grad=False)
            if env_id<3:
                logger.debug("randomizing link mass")
            for i, body_name in enumerate(self.env_config.robot.randomize_link_body_names):
                body_index = self._body_list.index(body_name)
                assert body_index != -1

                mass_scale = np.random.uniform(self.env_config.domain_rand.link_mass_range[0], self.env_config.domain_rand.link_mass_range[1])
                props[body_index].mass *= mass_scale

                self._link_mass_scale[env_id, i] *= mass_scale

        # randomize base mass
        if self.env_config.domain_rand.randomize_base_mass:
            raise Exception("index 0 is for world, 13 is for torso!")
            raise NotImplementedError
            rng = self.env_config.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])

        if env_id<3:
            sum_mass = 0
            for i in range(len(props)):
                logger.debug(f"Mass of body {i}: {props[i].mass} (after randomization)")
                sum_mass += props[i].mass
            logger.debug(f"Total mass {sum_mass} (afters randomization)")
        return props

    def get_dof_limits_properties(self):
        # assert the isaacgym dof limits are the same as the config
        for i in range(self.num_dof):
            # import pdb; pdb.set_trace()
            assert abs(self.hard_dof_pos_limits[i, 0].item() - self.robot_config.dof_pos_lower_limit_list[i]) < 1e-5, f"DOF {i} lower limit does not match"
            assert abs(self.hard_dof_pos_limits[i, 1].item() - self.robot_config.dof_pos_upper_limit_list[i]) < 1e-5, f"DOF {i} upper limit does not match"
            assert abs(self.dof_vel_limits[i].item() - self.robot_config.dof_vel_limit_list[i]) < 1e-5, f"DOF {i} velocity limit does not match"
            assert abs(self.torque_limits[i].item() - self.robot_config.dof_effort_limit_list[i]) < 1e-5, f"DOF {i} effort limit does not match"
            # assert self.dof_pos_hard_dof_pos_limitslimits[i, 1].item() == self.robot_config.dof_pos_upper_limit_list[i], f"DOF {i} upper limit does not match"
            # assert self.dof_vel_limits[i].item() == self.robot_config.dof_vel_limit_list[i], f"DOF {i} velocity limit does not match"
            # assert self.torque_limits[i].item() == self.robot_config.dof_effort_limit_list[i], f"DOF {i} effort limit does not match"

        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    def find_rigid_body_indice(self, body_name):
        return self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handles[0], body_name)
    
    def prepare_sim(self):
        super().prepare_sim()
        num_actors = self._get_num_actors_per_env()
        self.ball_rigid_body_idx, self.support_rigid_body_idx = get_hoi_object_rigid_body_indices(self.num_bodies)
        self.object_root_state = self.all_root_states.view(
            self.num_envs, num_actors, self.all_root_states.shape[-1]
        )[..., 1, :]
        self.object_pos = self.object_root_state[..., 0:3]
        self.object_rot = self.object_root_state[..., 3:7]
        self.object_vel = self.object_root_state[..., 7:10]
        self.object_ang_vel = self.object_root_state[..., 10:13]
        self.support_root_state = self.all_root_states.view(
            self.num_envs, num_actors, self.all_root_states.shape[-1]
        )[..., self.support_asset_idx, :]
        self.support_pos = self.support_root_state[..., 0:3]

        if self.robot_config.proj:
            self._build_proj_tensors()
        
    def _get_num_actors_per_env(self):
        return self.all_root_states.shape[0] // self.num_envs
        # num_actors = (
        #     self.root_states.shape[0] - self.total_num_objects
        # ) // self.num_envs
        # return num_actors

    def apply_torques_at_dof(self, torques):
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))

    def simulate_at_each_physics_step(self):
        super().simulate_at_each_physics_step()
        # Actor-level freeze (per-env): overwrite root state right after simulation step.
        # Env will refresh actor root state tensors later in its post-step.
        if self._frozen_object_actor_ids_int32 is not None and self._frozen_object_actor_ids_int32.numel() > 0:
            ids_long = self._frozen_object_actor_ids_long
            ids_i32 = self._frozen_object_actor_ids_int32
            # Update local tensor (for consistency) then push to simulator for those actors.
            self.all_root_states[ids_long, :] = self._frozen_object_root_states
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.all_root_states),
                gymtorch.unwrap_tensor(ids_i32),
                len(ids_i32),
            )

    def setup_viewer(self):
        self.enable_viewer_sync = True
        self.visualize_viewer = True
        self.viewer = self.gym.create_viewer(
            self.sim, gymapi.CameraProperties())
        # subscribe to keyboard shortcuts
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_W, "forward_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_S, "backward_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_A, "left_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_D, "right_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_Q, "heading_left_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_E, "heading_right_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_Z, "zero_command"
        )
        # EE commands key
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_F, "ee_forward_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_H, "ee_backward_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_O, "ee_open_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_C, "ee_close_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_T, "ee_height_up_command"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_G, "ee_height_down_command"
        )

        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_P, "push_robots"
        )

        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_N, "next_task"
        )
        self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_R, "toggle_video_record"
            )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_SEMICOLON, "cancel_video_record"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_U, "height_up"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_L, "height_down"
        )        

        sim_params = self.sim_params
        if sim_params.up_axis == gymapi.UP_AXIS_Z:
            cam_pos = gymapi.Vec3(5.0, 5.0, 3.0)
            cam_target = gymapi.Vec3(0.0, 0.0, 3.0)
        else:
            cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
            cam_target = gymapi.Vec3(10.0, 0.0, 15.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # video recording
        self.user_is_recording, self.user_recording_state_change = False, False
        self.save_rendering_dir.mkdir(parents=True, exist_ok=True)
        self.user_recording_video_path = str(self.save_rendering_dir / f"{self.config.experiment_name}-%s")

        if self.robot_config.proj:
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "space_shoot") #ZC0
            self.gym.subscribe_viewer_mouse_event(self.viewer, gymapi.MOUSE_LEFT_BUTTON, "mouse_shoot")

    def render(self, sync_frame_time=True):
        # check for window closed
        if self.gym.query_viewer_has_closed(self.viewer):
            sys.exit()
        delete_user_viewer_recordings = False
        # check for keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self.enable_viewer_sync = not self.enable_viewer_sync
            elif evt.action == "forward_command" and evt.value > 0:
                self.commands[:, 0] += 0.1
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "backward_command" and evt.value > 0:
                self.commands[:, 0] -= 0.1
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "left_command" and evt.value > 0:
                self.commands[:, 1] -= 0.1
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "right_command" and evt.value > 0:
                self.commands[:, 1] += 0.1
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "heading_left_command" and evt.value > 0:
                self.commands[:, 3] -= 0.1
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "heading_right_command" and evt.value > 0:
                self.commands[:, 3] += 0.1
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "zero_command" and evt.value > 0:
                self.commands[:, :4] = 0
                logger.info(f"Current Command: {self.commands[:, ]}")
            # EE commands
            elif evt.action == "ee_forward_command" and evt.value > 0:
                self.commands[:, 5] += 0.05
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "ee_backward_command" and evt.value > 0:
                self.commands[:, 5] -= 0.05
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "ee_open_command" and evt.value > 0:
                self.commands[:, 6] += 0.05
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "ee_close_command" and evt.value > 0:
                self.commands[:, 6] -= 0.05
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "ee_height_up_command" and evt.value > 0:
                self.commands[:, 7] += 0.05
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "ee_height_down_command" and evt.value > 0:
                self.commands[:, 7] -= 0.05
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "push_robots" and evt.value > 0:
                logger.info("Push Robots")
                self._push_robots(torch.arange(self.num_envs, device=self.device))
            elif evt.action == "next_task" and evt.value > 0:
                self.next_task()
            elif evt.action == "toggle_video_record" and evt.value > 0:
                # https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/envs/base_interface/isaacgym.py#L179
                self.user_is_recording = not self.user_is_recording
                self.user_recording_state_change = True
            elif evt.action == "cancel_video_record" and evt.value > 0:
                # https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/envs/base_interface/isaacgym.py#L182
                self.user_is_recording = False
                self.user_recording_state_change = False
                delete_user_viewer_recordings = True
            elif evt.action == "height_up" and evt.value > 0:
                self.commands[:, 4] += 0.1
                logger.info(f"Current Command: {self.commands[:, ]}")
            elif evt.action == "height_down" and evt.value > 0:
                self.commands[:, 4] -= 0.1
                logger.info(f"Current Command: {self.commands[:, ]}")

        # fetch results
        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)

        # step graphics
        if self.enable_viewer_sync:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            if sync_frame_time:
                self.gym.sync_frame_time(self.sim)
        else:
            self.gym.poll_viewer_events(self.viewer)

        if self.visualize_viewer:
        # https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/envs/base_interface/isaacgym.py#L198
            if self.user_recording_state_change:
                if self.user_is_recording:
                    curr_date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                    self.curr_user_recording_name = (
                        self.user_recording_video_path % curr_date_time
                    )
                    self.user_recording_frame = 0
                    if not os.path.exists(self.curr_user_recording_name):
                        os.makedirs(self.curr_user_recording_name)

                    logger.info(
                        f"Started to record data into folder {self.curr_user_recording_name}"
                    )
                if not self.user_is_recording:
                    images = [
                        img
                        for img in os.listdir(self.curr_user_recording_name)
                        if img.endswith(".png")
                    ]
                    images.sort()
                    if len(images) == 0:
                        logger.warning(f"No frames found in {self.curr_user_recording_name}; skip video writing.")
                        self.user_recording_state_change = False
                        return
                    sample_frame = cv2.imread(
                        os.path.join(self.curr_user_recording_name, images[0])
                    )
                    height, width = sample_frame.shape[:2]

                    fps = self._get_recording_fps()
                    out_mp4 = str(self.curr_user_recording_name) + ".mp4"

                    # Prefer ffmpeg (H.264) for maximum player compatibility; fallback to OpenCV if unavailable.
                    wrote = self._encode_video_ffmpeg(self.curr_user_recording_name, out_mp4, fps)
                    if not wrote:
                        # OpenCV/FFMPEG: "MP4V" often triggers a warning for mp4 container; "mp4v" is the usual tag.
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video = cv2.VideoWriter(out_mp4, fourcc, fps, (width, height))
                        for image in images:
                            frame = cv2.imread(os.path.join(self.curr_user_recording_name, image))
                            if frame is None:
                                continue
                            video.write(frame)
                        cv2.destroyAllWindows()
                        video.release()
                        wrote = os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 0

                    # Only delete frames if we successfully produced a video file.
                    delete_user_viewer_recordings = bool(wrote)
                    if not wrote:
                        logger.warning(
                            f"Failed to write video {out_mp4}. Keeping png frames under {self.curr_user_recording_name} for debugging."
                        )

                    logger.info(
                        f"============ Video finished writing {self.curr_user_recording_name}.mp4 ============"
                    )
                else:
                    logger.info("============ Writing video ============")
                self.user_recording_state_change = False

            if self.user_is_recording:
                self.gym.write_viewer_image_to_file(
                    self.viewer,
                    self.curr_user_recording_name
                    + "/%04d.png" % self.user_recording_frame,
                )
                self.user_recording_frame += 1

            if delete_user_viewer_recordings:
                images = [
                    img
                    for img in os.listdir(self.curr_user_recording_name)
                    if img.endswith(".png")
                ]
                # delete all images
                for image in images:
                    os.remove(os.path.join(self.curr_user_recording_name, image))
                os.removedirs(self.curr_user_recording_name)

    def next_task(self):
        pass

    # debug visualization
    def clear_lines(self):
        self.gym.clear_lines(self.viewer)
