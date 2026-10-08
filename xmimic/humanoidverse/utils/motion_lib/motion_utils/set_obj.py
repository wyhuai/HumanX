import argparse
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from humanoidverse.utils.motion_lib.motion_utils.set_fps import (
    _format_override_fps_suffix,
    _parse_override_fps_factor,
    apply_override_fps_to_motion,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROBOT_CONFIG = REPO_ROOT / "humanoidverse/config/robot/g1/g1_29dof_model16.yaml"
DEFAULT_LEFT_PALM_OFFSET = np.asarray([0.1115, 0.0030, 0.0], dtype=np.float32)
DEFAULT_RIGHT_PALM_OFFSET = np.asarray([0.1115, -0.0030, 0.0], dtype=np.float32)
DEFAULT_PALM_CENTER_Z_OFFSET = 0 #-0.03
DEFAULT_PALM_CENTER_MIN_Z = 0.10
DEFAULT_BALL_RADIUS = 0.10
DEFAULT_BALL_GRAVITY = 9.81

_CURRENT_RULE_CONTEXT = None


def _default_output_pkl_path(input_pkl):
    input_pkl = Path(input_pkl)
    if input_pkl.stem.endswith("_w_ball"):
        return input_pkl.with_name(f"{input_pkl.stem}_w_contact.pkl")
    return input_pkl.with_name(f"{input_pkl.stem}_w_ball_w_contact.pkl")


def _default_output_pkl_path_with_fps_override(input_pkl, override_fps_factor):
    suffix = _format_override_fps_suffix(override_fps_factor)
    input_pkl = Path(input_pkl)
    if input_pkl.stem.endswith("_w_ball"):
        stem = f"{input_pkl.stem}_{suffix}_w_contact"
    else:
        stem = f"{input_pkl.stem}_{suffix}_w_ball_w_contact"
    return input_pkl.with_name(f"{stem}{input_pkl.suffix}")


def _load_motion_data(input_pkl):
    data = joblib.load(input_pkl)
    if not isinstance(data, dict):
        raise ValueError(f"{input_pkl} must contain a dict of motion entries")
    return data


def _get_motion_key(data, motion_key=None):
    if motion_key is None:
        keys = list(data.keys())
        if len(keys) == 1:
            return keys[0]
        raise ValueError(
            "input pkl contains multiple motion entries; please specify --motion-key explicitly"
        )
    if motion_key not in data:
        raise ValueError(f"motion-key '{motion_key}' not found in input pkl")
    return motion_key


def _resolve_fk_device(device):
    if device is None:
        return "cpu"
    return str(device)


def _clamp_palm_center_min_z(palm_center, min_z=DEFAULT_PALM_CENTER_MIN_Z):
    palm_center = np.asarray(palm_center, dtype=np.float32).copy()
    palm_center[2] = max(float(palm_center[2]), float(min_z))
    return palm_center


def _ball_vertex_colors(contact_active):
    if contact_active:
        return np.asarray([255, 0, 0], dtype=np.uint8)
    return np.asarray([255, 140, 0], dtype=np.uint8)


def _palm_center_radius():
    return DEFAULT_BALL_RADIUS


def _palm_center_line_radius_ui_points():
    return 0.5


def _palm_center_vertex_colors(contact_active):
    del contact_active
    return np.asarray([0, 220, 220, 40], dtype=np.uint8)


def _palm_center_wireframe(center, radius=None, latitude_segments=20, longitude_segments=20):
    center = np.asarray(center, dtype=np.float32).reshape(3)
    radius = _palm_center_radius() if radius is None else float(radius)
    latitude_segments = int(latitude_segments)
    longitude_segments = int(longitude_segments)

    theta = np.linspace(0.0, 2.0 * np.pi, longitude_segments + 1, dtype=np.float32)
    phi_lat = np.linspace(
        -0.5 * np.pi,
        0.5 * np.pi,
        latitude_segments + 1,
        dtype=np.float32,
    )[1:-1]
    phi_lon = np.linspace(
        -0.5 * np.pi,
        0.5 * np.pi,
        longitude_segments + 1,
        dtype=np.float32,
    )

    strips = []

    # Latitude circles.
    for phi in phi_lat:
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        strip = np.stack(
            [
                center[0] + radius * cos_phi * np.cos(theta),
                center[1] + radius * cos_phi * np.sin(theta),
                center[2] + radius * sin_phi * np.ones_like(theta),
            ],
            axis=-1,
        )
        strips.append(strip.astype(np.float32))

    # Longitude circles.
    lon_angles = np.linspace(0.0, 2.0 * np.pi, longitude_segments, endpoint=False, dtype=np.float32)
    for angle in lon_angles:
        strip = np.stack(
            [
                center[0] + radius * np.cos(phi_lon) * np.cos(angle),
                center[1] + radius * np.cos(phi_lon) * np.sin(angle),
                center[2] + radius * np.sin(phi_lon),
            ],
            axis=-1,
        )
        strips.append(strip.astype(np.float32))

    return np.asarray(strips, dtype=np.float32)


def _quat_rotate(quat_xyzw, vec_xyz):
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    vec_xyz = np.asarray(vec_xyz, dtype=np.float32)
    quat_vec = quat_xyzw[..., :3]
    quat_w = quat_xyzw[..., 3:4]
    uv = np.cross(quat_vec, vec_xyz)
    uuv = np.cross(quat_vec, uv)
    return vec_xyz + 2.0 * (quat_w * uv + uuv)


def _calc_heading_from_quat(quat_xyzw):
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    forward = np.zeros(quat_xyzw.shape[:-1] + (3,), dtype=np.float32)
    forward[..., 0] = 1.0
    rotated_forward = _quat_rotate(quat_xyzw, forward)
    return np.arctan2(rotated_forward[..., 1], rotated_forward[..., 0]).astype(np.float32)


def _solve_ground_hit_time(z0, vz0, dt, gravity, ground_z):
    if dt == 0.0:
        return None

    coeff = np.asarray(
        [-0.5 * float(gravity), float(vz0), float(z0) - float(ground_z)],
        dtype=np.float64,
    )
    roots = np.roots(coeff)
    if roots.size == 0:
        return None

    real_roots = roots[np.isreal(roots)].real
    if real_roots.size == 0:
        return None

    eps = 1e-9
    if dt > 0.0:
        candidates = [root for root in real_roots if eps < root <= dt + eps]
        if not candidates:
            return None
        return float(min(candidates))

    candidates = [root for root in real_roots if dt - eps <= root < -eps]
    if not candidates:
        return None
    return float(max(candidates))


def _integrate_ballistic_step(pos, vel, dt, gravity=DEFAULT_BALL_GRAVITY, ground_z=DEFAULT_BALL_RADIUS):
    pos = np.asarray(pos, dtype=np.float64).copy()
    vel = np.asarray(vel, dtype=np.float64).copy()
    acc = np.asarray([0.0, 0.0, -float(gravity)], dtype=np.float64)

    remaining = float(dt)
    eps = 1e-9
    max_bounces = 32

    for _ in range(max_bounces):
        if abs(remaining) <= eps:
            break

        hit_time = _solve_ground_hit_time(pos[2], vel[2], remaining, gravity, ground_z)
        if hit_time is None:
            pos = pos + vel * remaining + 0.5 * acc * remaining * remaining
            vel = vel + acc * remaining
            if pos[2] < ground_z:
                pos[2] = ground_z
            break

        pos = pos + vel * hit_time + 0.5 * acc * hit_time * hit_time
        pos[2] = ground_z
        vel = vel + acc * hit_time
        vel[2] = -vel[2]
        remaining -= hit_time
    else:
        raise RuntimeError("too many ballistic bounces within one frame step")

    return pos.astype(np.float32), vel.astype(np.float32)


def build_ballistic_trajectory(
    *,
    root_rot,
    event_frame,
    event_pos,
    speed,
    relative_heading_deg,
    elevation_deg,
    fps,
    gravity=DEFAULT_BALL_GRAVITY,
    ball_radius=DEFAULT_BALL_RADIUS,
    num_frames=None,
):
    root_rot = np.asarray(root_rot, dtype=np.float32).reshape(-1, 4)
    total_frames = int(root_rot.shape[0] if num_frames is None else num_frames)
    if total_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {total_frames}")
    if root_rot.shape[0] < total_frames:
        raise ValueError(f"root_rot has {root_rot.shape[0]} frames but {total_frames} are required")

    event_frame = int(event_frame)
    if event_frame < 0 or event_frame >= total_frames:
        raise ValueError(f"event_frame {event_frame} is out of range for {total_frames} frames")

    fps = float(fps)
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")

    speed = float(speed)
    if speed == 0.0:
        raise ValueError("speed must be non-zero")

    elevation_deg = float(elevation_deg)
    if not (-90.0 < elevation_deg < 90.0):
        raise ValueError(f"elevation_deg must be within (-90, 90), got {elevation_deg}")

    event_pos = np.asarray(event_pos, dtype=np.float32).reshape(3)
    event_heading = float(_calc_heading_from_quat(root_rot[event_frame]))
    yaw = event_heading + np.deg2rad(float(relative_heading_deg))
    elevation = np.deg2rad(elevation_deg)
    speed_mag = abs(speed)
    direction_sign = 1.0 if speed > 0.0 else -1.0

    velocity = np.asarray(
        [
            np.cos(elevation) * np.cos(yaw),
            np.cos(elevation) * np.sin(yaw),
            np.sin(elevation),
        ],
        dtype=np.float32,
    )
    velocity *= np.float32(direction_sign * speed_mag)

    dt = np.float32(1.0 / fps)
    trajectory = np.zeros((total_frames, 3), dtype=np.float32)
    trajectory[event_frame] = event_pos

    pos = event_pos.astype(np.float32).copy()
    vel = velocity.astype(np.float32).copy()
    for frame_idx in range(event_frame + 1, total_frames):
        pos, vel = _integrate_ballistic_step(
            pos,
            vel,
            float(dt),
            gravity=gravity,
            ground_z=ball_radius,
        )
        trajectory[frame_idx] = pos

    pos = event_pos.astype(np.float32).copy()
    vel = velocity.astype(np.float32).copy()
    for frame_idx in range(event_frame - 1, -1, -1):
        pos, vel = _integrate_ballistic_step(
            pos,
            vel,
            -float(dt),
            gravity=gravity,
            ground_z=ball_radius,
        )
        trajectory[frame_idx] = pos

    return trajectory.astype(np.float32)


def get_current_rule_context():
    if _CURRENT_RULE_CONTEXT is None:
        raise RuntimeError("rule context is only available while build_updated_frame_state is running")
    return _CURRENT_RULE_CONTEXT


def scale_ballistic_speed(configured_speed, fps_override_factor=None):
    if fps_override_factor is None:
        fps_override_factor = 1.0 if _CURRENT_RULE_CONTEXT is None else _CURRENT_RULE_CONTEXT.get("fps_override_factor", 1.0)
    return float(configured_speed) * float(fps_override_factor)


def custom_frame_state(frame_idx, motion_time, palm_center, fps):
    """Edit this function directly.

    Return a dict with:
    - obj_pos: np.ndarray shape (3,)
    - contact: 0.0 or 1.0
    """
    del motion_time

    palm_center = _clamp_palm_center_min_z(palm_center)

    # Example:
    # if frame_idx < 246:
    #     return {"obj_pos": np.asarray([2.666, 1.337, 0.20], dtype=np.float32), "contact": 1.0}
    # return {"obj_pos": palm_center, "contact": 0.0}
    #
    # For full-length ballistic helpers:
    ctx = _CURRENT_RULE_CONTEXT
    if ctx is not None:
        traj = ctx["cache"].get("ballistic_traj")
        if traj is None:
            traj = build_ballistic_trajectory(
                root_rot=ctx["motion"]["root_rot"],
                event_frame=56,
                event_pos=np.asarray([ 0.2801,  0.6380,  0.9557], dtype=np.float32),
                speed=scale_ballistic_speed(+5.0),
                relative_heading_deg=0.0,
                elevation_deg=-30.0,
                fps=fps,
            )
            ctx["cache"]["ballistic_traj"] = traj

    if frame_idx < 74:
        return {
            "obj_pos": [ 1.6234, -1.9879,  0.8367], #traj[frame_idx],
            "contact": 0.0,
        }
    elif frame_idx == 74:
        return {
            "obj_pos": [ 1.6234, -1.9879,  0.8367], #traj[frame_idx],
            "contact": 1.0,
        }
    else:
        return {
            "obj_pos": palm_center,
            "contact": 1.0,
        }

    # return {
    #     "obj_pos": palm_center,
    #     "contact": 1.0,
    # }

    # return {
    #     "obj_pos": [ 0.,  0.,  -1.], #palm_center,
    #     "contact": 0.0,
    # }

    return {}


def build_updated_frame_state(
    *,
    motion,
    palm_centers,
    rule_fn=None,
    fps_override_factor=1.0,
):
    global _CURRENT_RULE_CONTEXT

    fps = float(motion.get("fps", 0))
    if fps <= 0.0:
        raise ValueError(f"motion data is missing valid 'fps': {fps}")

    palm_centers = np.asarray(palm_centers, dtype=np.float32).reshape(-1, 3)
    original_obj_pos = motion.get("obj_pos", palm_centers)
    original_obj_pos = np.asarray(original_obj_pos, dtype=np.float32).reshape(-1, 3)
    original_contact = motion.get("contact", np.zeros((palm_centers.shape[0], 1), dtype=np.float32))
    original_contact = np.asarray(original_contact, dtype=np.float32).reshape(-1, 1)
    rule_fn = custom_frame_state if rule_fn is None else rule_fn

    updated_obj_pos = np.zeros_like(palm_centers, dtype=np.float32)
    updated_contact = np.zeros((palm_centers.shape[0], 1), dtype=np.float32)
    previous_rule_context = _CURRENT_RULE_CONTEXT
    _CURRENT_RULE_CONTEXT = {
        "motion": motion,
        "palm_centers": palm_centers,
        "original_obj_pos": original_obj_pos,
        "original_contact": original_contact,
        "fps_override_factor": float(fps_override_factor),
        "cache": {},
    }
    try:
        for frame_idx, palm_center in enumerate(palm_centers):
            motion_time = float(frame_idx) / fps
            frame_state = rule_fn(frame_idx, motion_time, palm_center, fps)
            obj_pos = frame_state.get("obj_pos", original_obj_pos[frame_idx])
            contact = frame_state.get("contact", original_contact[frame_idx, 0])
            updated_obj_pos[frame_idx] = np.asarray(obj_pos, dtype=np.float32).reshape(3)
            updated_contact[frame_idx, 0] = float(contact)
    finally:
        _CURRENT_RULE_CONTEXT = previous_rule_context
    return {
        "obj_pos": updated_obj_pos,
        "contact": updated_contact,
    }


def write_motion_with_updated_obj_pos(
    *,
    input_pkl,
    updated_obj_pos,
    updated_contact,
    motion_key=None,
    output_pkl=None,
    motion_overrides=None,
):
    data = _load_motion_data(input_pkl)
    motion_key = _get_motion_key(data, motion_key)
    motion = dict(data[motion_key])
    updated_obj_pos = np.asarray(updated_obj_pos, dtype=np.float32).reshape(-1, 3)
    updated_contact = np.asarray(updated_contact, dtype=np.float32).reshape(-1, 1)

    if "obj_pos" in motion:
        original_frames = np.asarray(motion["obj_pos"], dtype=np.float32).reshape(-1, 3).shape[0]
    else:
        original_frames = np.asarray(motion["root_trans_offset"], dtype=np.float32).reshape(-1, 3).shape[0]
    if updated_obj_pos.shape[0] != original_frames:
        raise ValueError(
            f"updated obj_pos row count mismatch: expected {original_frames}, got {updated_obj_pos.shape[0]}"
        )
    if updated_contact.shape[0] != original_frames:
        raise ValueError(
            f"updated contact row count mismatch: expected {original_frames}, got {updated_contact.shape[0]}"
        )

    motion["obj_pos"] = updated_obj_pos
    motion["contact"] = updated_contact
    if motion_overrides:
        for key, value in motion_overrides.items():
            motion[key] = value
    data = dict(data)
    data[motion_key] = motion

    output_path = Path(output_pkl) if output_pkl is not None else _default_output_pkl_path(input_pkl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, output_path)
    return output_path


def _resolve_robot_motion_cfg(robot_config):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(robot_config).robot.motion
    asset_root = Path(cfg.asset.assetRoot)
    if not asset_root.is_absolute():
        asset_root = REPO_ROOT / asset_root
    cfg.asset.assetRoot = str(asset_root)
    return cfg


def _apply_quat_to_vector(quats_xyzw, vector_xyz):
    from scipy.spatial.transform import Rotation as R

    vector_xyz = np.asarray(vector_xyz, dtype=np.float32).reshape(1, 3)
    vector_xyz = np.repeat(vector_xyz, quats_xyzw.shape[0], axis=0)
    return R.from_quat(quats_xyzw).apply(vector_xyz).astype(np.float32)


def compute_palm_centers(
    motion,
    *,
    robot_config=DEFAULT_ROBOT_CONFIG,
    device=None,
    left_offset=DEFAULT_LEFT_PALM_OFFSET,
    right_offset=DEFAULT_RIGHT_PALM_OFFSET,
    palm_center_z_offset=DEFAULT_PALM_CENTER_Z_OFFSET,
):
    import torch

    from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

    cfg = _resolve_robot_motion_cfg(robot_config)
    device = _resolve_fk_device(device)
    torch_device = torch.device(device)
    humanoid_batch = Humanoid_Batch(cfg, device=torch_device)

    pose_aa = np.asarray(motion["pose_aa"], dtype=np.float32)
    trans = np.asarray(motion["root_trans_offset"], dtype=np.float32)
    fps = float(motion.get("fps", 0))
    if fps <= 0.0:
        raise ValueError(f"motion data is missing valid 'fps': {fps}")

    fk_res = humanoid_batch.fk_batch(
        torch.from_numpy(pose_aa).unsqueeze(0).to(device=torch_device),
        torch.from_numpy(trans).unsqueeze(0).to(device=torch_device),
        return_full=False,
        dt=1.0 / fps,
    )

    body_names = humanoid_batch.body_names
    left_idx = body_names.index("left_wrist_yaw_link")
    right_idx = body_names.index("right_wrist_yaw_link")

    global_translation = fk_res.global_translation.squeeze(0).detach().cpu().numpy().astype(np.float32)
    global_rotation = fk_res.global_rotation.squeeze(0).detach().cpu().numpy().astype(np.float32)

    left_wrist_pos = global_translation[:, left_idx, :]
    right_wrist_pos = global_translation[:, right_idx, :]
    left_wrist_rot = global_rotation[:, left_idx, :]
    right_wrist_rot = global_rotation[:, right_idx, :]

    left_palm_pos = left_wrist_pos + _apply_quat_to_vector(left_wrist_rot, left_offset)
    right_palm_pos = right_wrist_pos + _apply_quat_to_vector(right_wrist_rot, right_offset)
    palm_centers = 0.5 * (left_palm_pos + right_palm_pos)
    palm_centers[:, 2] += float(palm_center_z_offset)
    return palm_centers.astype(np.float32)


def print_frame_debug(palm_centers, obj_positions, contact, fps):
    for frame_idx, (palm_center, obj_pos, contact_value) in enumerate(zip(palm_centers, obj_positions, contact)):
        motion_time = float(frame_idx) / float(fps)
        print(
            f"frame={frame_idx:04d} time={motion_time:8.4f}s "
            f"palm_center=[{palm_center[0]: .4f}, {palm_center[1]: .4f}, {palm_center[2]: .4f}] "
            f"obj_pos=[{obj_pos[0]: .4f}, {obj_pos[1]: .4f}, {obj_pos[2]: .4f}] "
            f"contact={float(contact_value[0]):.1f}"
        )


def preview_motion(
    *,
    motion,
    palm_centers,
    object_positions,
    contact,
    frame_mode="world",
    stream_rate="realtime",
    prefetch_frames=50,
):
    import rerun as rr

    from humanoidverse.utils.motion_lib.vis_rr import (
        RerunURDF,
        build_motion_data_for_vis,
        build_object_positions_for_vis,
        format_startup_summary,
        frame_time_seconds,
        prefetch_count,
        stream_sleep_seconds,
    )

    preview_motion = dict(motion)
    preview_motion["obj_pos"] = np.asarray(object_positions, dtype=np.float32)
    contact = np.asarray(contact, dtype=np.float32).reshape(-1, 1)

    fps = float(preview_motion.get("fps", 0))
    if fps <= 0.0:
        raise ValueError(f"motion data is missing valid 'fps': {fps}")

    rr.init("set_obj_preview", spawn=True)
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    motion_data = build_motion_data_for_vis(preview_motion, frame_mode=frame_mode)
    object_positions_for_vis = build_object_positions_for_vis(preview_motion, frame_mode=frame_mode)
    palm_positions_for_vis = np.asarray(palm_centers, dtype=np.float32)
    if frame_mode == "root-centered":
        root_pos = np.asarray(preview_motion["root_trans_offset"], dtype=np.float32)
        root_rot = np.asarray(preview_motion["root_rot"], dtype=np.float32)
        palm_positions_for_vis = palm_positions_for_vis - root_pos
        from humanoidverse.utils.motion_lib.vis_rr import quat_to_euler
        import math

        root_yaw0 = quat_to_euler(*root_rot[0])[2]
        c = math.cos(-root_yaw0)
        s = math.sin(-root_yaw0)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        palm_positions_for_vis = palm_positions_for_vis @ rot.T

    print(format_startup_summary(motion_data.shape[0], fps))

    rerun_urdf = RerunURDF("g1")
    total_frames = motion_data.shape[0]
    initial_frames = prefetch_count(total_frames, prefetch_frames)

    for frame_nr in range(initial_frames):
        rr.set_time_sequence("frame_nr", frame_nr)
        rr.set_time_seconds("time", frame_time_seconds(frame_nr, fps))
        rerun_urdf.update(motion_data[frame_nr, :])
        contact_active = contact[frame_nr, 0] >= 1.0
        rr.log(
            "object/palm_center",
            rr.LineStrips3D(
                _palm_center_wireframe(palm_positions_for_vis[frame_nr]),
                radii=rr.Radius.ui_points(_palm_center_line_radius_ui_points()),
                colors=_palm_center_vertex_colors(contact_active),
            ),
        )
        rr.log(
            "object/ball",
            rr.Points3D(object_positions_for_vis[frame_nr], radii=0.12, colors=_ball_vertex_colors(contact_active)),
        )

    if initial_frames < total_frames:
        sleep_s = stream_sleep_seconds(stream_rate, fps)
        for frame_nr in range(initial_frames, total_frames):
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            rr.set_time_sequence("frame_nr", frame_nr)
            rr.set_time_seconds("time", frame_time_seconds(frame_nr, fps))
            rerun_urdf.update(motion_data[frame_nr, :])
            contact_active = contact[frame_nr, 0] >= 1.0
            rr.log(
                "object/palm_center",
                rr.LineStrips3D(
                    _palm_center_wireframe(palm_positions_for_vis[frame_nr]),
                    radii=rr.Radius.ui_points(_palm_center_line_radius_ui_points()),
                    colors=_palm_center_vertex_colors(contact_active),
                ),
            )
            rr.log(
                "object/ball",
                rr.Points3D(object_positions_for_vis[frame_nr], radii=0.12, colors=_ball_vertex_colors(contact_active)),
            )


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Preview and optionally save updated obj_pos trajectories for a motion pkl."
    )
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--motion-key")
    parser.add_argument(
        "--robot-config",
        default=str(DEFAULT_ROBOT_CONFIG),
        help="Robot config yaml used for offline FK palm-center computation.",
    )
    parser.add_argument("--device", help="Torch device for FK, e.g. cpu or cuda:0.")
    parser.add_argument("--frame", choices=["world", "root-centered"], default="world")
    parser.add_argument("--stream-rate", choices=["realtime", "max"], default="realtime")
    parser.add_argument("--prefetch-frames", type=int, default=50)
    parser.add_argument("--save-pkl", action="store_true", help="Write updated obj_pos to a new pkl.")
    parser.add_argument("--output-pkl", help="Override output pkl path. Default adds _w_ball suffix.")
    parser.add_argument("--override-fps", help="FPS multiplier, e.g. 0p7 or 1p3. Metadata only, no resampling.")
    preview_group = parser.add_mutually_exclusive_group()
    preview_group.add_argument("--preview", dest="preview", action="store_true", default=True)
    preview_group.add_argument("--no-preview", dest="preview", action="store_false")
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    data = _load_motion_data(args.input_pkl)
    resolved_motion_key = _get_motion_key(data, args.motion_key)
    motion = data[resolved_motion_key]
    override_fps_factor = None
    if args.override_fps is not None:
        override_fps_factor = _parse_override_fps_factor(args.override_fps)
        motion = apply_override_fps_to_motion(motion, override_fps_factor)
    fps = float(motion.get("fps", 0))
    if fps <= 0.0:
        raise ValueError(f"motion data is missing valid 'fps': {fps}")

    if "pose_aa" not in motion:
        raise ValueError("motion data is missing required field 'pose_aa'")
    if "obj_pos" not in motion:
        warnings.warn("motion data is missing 'obj_pos'; a new obj_pos track will be written", UserWarning)
    if "contact" not in motion:
        warnings.warn("motion data is missing 'contact'; a new contact track will be written", UserWarning)

    palm_centers = compute_palm_centers(
        motion,
        robot_config=args.robot_config,
        device=args.device,
    )
    updated_frame_state = build_updated_frame_state(
        motion=motion,
        palm_centers=palm_centers,
        fps_override_factor=override_fps_factor if override_fps_factor is not None else 1.0,
    )
    updated_obj_pos = np.asarray(
        updated_frame_state.get("obj_pos", motion.get("obj_pos", palm_centers)),
        dtype=np.float32,
    ).reshape(-1, 3)
    updated_contact = np.asarray(
        updated_frame_state.get("contact", motion.get("contact", np.zeros((len(updated_obj_pos), 1), dtype=np.float32))),
        dtype=np.float32,
    ).reshape(-1, 1)

    print(f"[set_obj] motion_key={resolved_motion_key} total_frames={len(updated_obj_pos)} fps={fps}")
    print_frame_debug(palm_centers, updated_obj_pos, updated_contact, fps)

    if args.preview:
        preview_motion(
            motion=motion,
            palm_centers=palm_centers,
            object_positions=updated_obj_pos,
            contact=updated_contact,
            frame_mode=args.frame,
            stream_rate=args.stream_rate,
            prefetch_frames=args.prefetch_frames,
        )

    if args.save_pkl:
        output_pkl = args.output_pkl
        if output_pkl is None and override_fps_factor is not None:
            output_pkl = _default_output_pkl_path_with_fps_override(args.input_pkl, override_fps_factor)
        output_path = write_motion_with_updated_obj_pos(
            input_pkl=args.input_pkl,
            updated_obj_pos=updated_obj_pos,
            updated_contact=updated_contact,
            motion_key=resolved_motion_key,
            output_pkl=output_pkl,
            motion_overrides={"fps": fps} if override_fps_factor is not None else None,
        )
        print(f"[set_obj] saved updated obj_pos to {output_path}")


if __name__ == "__main__":
    main()
