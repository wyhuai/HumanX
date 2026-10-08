import argparse
import time
import numpy as np
import joblib
import math
from pathlib import Path

import rerun as rr
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[3]
G1_NUM_BODIES = 30
G1_NUM_DOF = 29


def resolve_robot_urdf_paths(robot_type):
    if robot_type != "g1":
        raise ValueError(f"Unsupported robot_type: {robot_type}")

    asset_root = REPO_ROOT / "humanoidverse/asset/robots/g1"
    urdf_path = asset_root / "g1_29dof_mode16.urdf"
    mesh_root = asset_root
    return urdf_path, mesh_root


def _joint_angles_from_pose_aa(pose_aa):
    pose_aa = np.asarray(pose_aa, dtype=np.float32)
    if pose_aa.shape[1] > G1_NUM_BODIES:
        pose_aa = pose_aa[:, :G1_NUM_BODIES, :]
    if pose_aa.shape[1] < G1_NUM_BODIES:
        raise ValueError(f"pose_aa must have at least {G1_NUM_BODIES} joints, got {pose_aa.shape[1]}")
    joint_pose = pose_aa[:, 1:G1_NUM_BODIES, :]
    return np.sum(joint_pose, axis=-1).astype(np.float32)


def _joint_angles_from_motion(motion):
    if "pose_aa" in motion:
        return _joint_angles_from_pose_aa(motion["pose_aa"])

    if "dof" in motion:
        return np.asarray(motion["dof"], dtype=np.float32).reshape(len(motion["root_trans_offset"]), -1)

    raise ValueError("motion data must contain either 'pose_aa' or 'dof'")


def build_motion_data_for_vis(motion, frame_mode="world"):
    dof = _joint_angles_from_motion(motion)
    motion_data = np.concatenate(
        (
            np.asarray(motion["root_trans_offset"], dtype=np.float32),
            np.asarray(motion["root_rot"], dtype=np.float32),
            dof,
        ),
        axis=1,
    )

    if frame_mode == "root-centered":
        motion_data[:, :3] *= 0
        motion_data[:, 3:7] = rebase_yaw(motion_data[:, 3:7])
    elif frame_mode != "world":
        raise ValueError(f"Unsupported frame_mode: {frame_mode}")

    return motion_data


def build_object_positions_for_vis(motion, frame_mode="world"):
    if "obj_pos" not in motion:
        return None

    obj_pos = np.asarray(motion["obj_pos"], dtype=np.float32)
    if frame_mode == "world":
        return obj_pos
    if frame_mode != "root-centered":
        raise ValueError(f"Unsupported frame_mode: {frame_mode}")

    root_pos = np.asarray(motion["root_trans_offset"], dtype=np.float32)
    rel = obj_pos - root_pos
    yaw0 = quat_to_euler(*np.asarray(motion["root_rot"], dtype=np.float32)[0])[2]
    c = math.cos(-yaw0)
    s = math.sin(-yaw0)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rel @ rot.T


def build_object_contact_for_vis(motion):
    if "contact" not in motion:
        return None
    return np.asarray(motion["contact"], dtype=np.float32).reshape(-1, 1)


def _ball_colors_from_contact(contact_active):
    if contact_active:
        return np.asarray([255, 0, 0], dtype=np.uint8)
    return np.asarray([255, 140, 0], dtype=np.uint8)


def frame_time_seconds(frame_idx, fps):
    fps = float(fps)
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    return float(frame_idx) / fps


def format_startup_summary(total_frames, fps):
    return f"[vis_rr] total frames: {int(total_frames)} | fps: {fps}"


def prefetch_count(num_frames, requested_prefetch):
    return min(int(num_frames), max(int(requested_prefetch), 0))


def stream_sleep_seconds(stream_rate, fps):
    if stream_rate == "realtime":
        return 1.0 / float(fps)
    if stream_rate == "max":
        return 0.0
    raise ValueError(f"Unsupported stream_rate: {stream_rate}")


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, help="File path", required=True)
    parser.add_argument('--frame', choices=['world', 'root-centered'], default='world')
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument('--stream', dest='stream', action='store_true', default=True)
    stream_group.add_argument('--no-stream', dest='stream', action='store_false')
    parser.add_argument('--stream-rate', choices=['realtime', 'max'], default='realtime')
    parser.add_argument('--prefetch-frames', type=int, default=50)
    return parser

class RerunURDF():
    def __init__(self, robot_type):
        import pinocchio as pin

        self.name = robot_type
        self._pin = pin
        urdf_path, mesh_root = resolve_robot_urdf_paths(robot_type)
        self.robot = pin.RobotWrapper.BuildFromURDF(
            str(urdf_path),
            str(mesh_root),
            pin.JointModelFreeFlyer(),
        )
        self.Tpose = np.array([0,0,0.785,0,0,0,1,
                                -0.15,0,0,0.3,-0.15,0,
                                -0.15,0,0,0.3,-0.15,0,
                                0,0,0,
                                0, 1.57,0,1.57,0,0,0,
                                0,-1.57,0,1.57,0,0,0]).astype(np.float32)
        
        # print all joints names
        # for i in range(self.robot.model.njoints):
        #     print(self.robot.model.names[i])
        
        self.link2mesh = self.get_link2mesh()
        self.load_visual_mesh()
        self.update()
    
    def get_link2mesh(self):
        link2mesh = {}
        for visual in self.robot.visual_model.geometryObjects:
            mesh = trimesh.load_mesh(visual.meshPath)
            name = visual.name[:-2]
            mesh.visual = trimesh.visual.ColorVisuals()
            mesh.visual.vertex_colors = visual.meshColor
            link2mesh[name] = mesh
        return link2mesh
   
    def load_visual_mesh(self):       
        self.robot.framesForwardKinematics(self._pin.neutral(self.robot.model))
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            mesh = self.link2mesh[frame_name]
            
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parentJoint
            parent_joint_name = self.robot.model.names[parent_joint_id]
            frame_tf = self.robot.data.oMf[frame_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]
            rr.log(f'urdf_{self.name}/{parent_joint_name}',
                   rr.Transform3D(translation=joint_tf.translation,
                                  mat3x3=joint_tf.rotation,
                                  axis_length=0.01))
            
            relative_tf = joint_tf.inverse() * frame_tf
            mesh.apply_transform(relative_tf.homogeneous)
            rr.log(f'urdf_{self.name}/{parent_joint_name}/{frame_name}',
                   rr.Mesh3D(
                       vertex_positions=mesh.vertices,
                       triangle_indices=mesh.faces,
                       vertex_normals=mesh.vertex_normals,
                       vertex_colors=mesh.visual.vertex_colors,
                       albedo_texture=None,
                       vertex_texcoords=None,
                   ),
                   static=True)
    
    def update(self, configuration = None):
        self.robot.framesForwardKinematics(self.Tpose if configuration is None else configuration)
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parentJoint
            parent_joint_name = self.robot.model.names[parent_joint_id]
            # print(parent_joint_name)
            joint_tf = self.robot.data.oMi[parent_joint_id]
            rr.log(f'urdf_{self.name}/{parent_joint_name}',
                   rr.Transform3D(translation=joint_tf.translation,
                                  mat3x3=joint_tf.rotation,
                                  axis_length=0.01))


def quat_to_euler(x, y, z, w):
    """Convert quaternion (xyzw) to Euler angles (roll, pitch, yaw) in radians"""
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        print("Warning: Pitch out of range, using 90 degrees")
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quat(roll, pitch, yaw):
    """Convert Euler angles (radians) to quaternion (xyzw)"""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy

    return np.array([x, y, z, w])

def remove_yaw(quaternions):
    """Remove yaw component by converting to Euler angles and zeroing yaw"""
    eulers = np.array([quat_to_euler(*q) for q in quaternions])
    eulers[:, 2] = 0  # Zero yaw
    new_quats = np.array([euler_to_quat(r, p, y) for r, p, y in eulers])
    return new_quats

def rebase_yaw(quaternions):
    """Remove yaw component by converting to Euler angles and zeroing yaw"""
    eulers = np.array([quat_to_euler(*q) for q in quaternions])
    # eulers[:, 2] = 0  # Zero yaw
    eulers[:, 2] -= eulers[0, 2] 
    new_quats = np.array([euler_to_quat(r, p, y) for r, p, y in eulers])
    return new_quats



if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    rr.init('Reviz', spawn=True)
    # rr.init(args.filepath, spawn=True)
    rr.log('', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    filepath = args.filepath
    robot_type = 'g1'
    data = joblib.load(filepath)
    keyname = list(data.keys())[0]
    data = data[keyname]
    fps = data.get("fps", None)
    if fps is None:
        raise ValueError("motion data is missing required field 'fps'")
    motion_data = build_motion_data_for_vis(
        data,
        frame_mode=args.frame,
    )
    object_positions = build_object_positions_for_vis(
        data,
        frame_mode=args.frame,
    )
    object_contact = build_object_contact_for_vis(data)
    print(format_startup_summary(motion_data.shape[0], fps))
    

    rerun_urdf = RerunURDF(robot_type)
    total_frames = motion_data.shape[0]
    initial_frames = prefetch_count(total_frames, args.prefetch_frames) if args.stream else total_frames
    for frame_nr in range(initial_frames):
        rr.set_time_sequence('frame_nr', frame_nr)
        rr.set_time_seconds('time', frame_time_seconds(frame_nr, fps))
        configuration = motion_data[frame_nr, :]
        rerun_urdf.update(configuration)
        if object_positions is not None:
            contact_active = object_contact is not None and object_contact[frame_nr, 0] >= 1.0
            rr.log(
                "object/ball",
                rr.Points3D(
                    object_positions[frame_nr],
                    radii=0.12,
                    colors=_ball_colors_from_contact(contact_active),
                ),
            )

    if args.stream and initial_frames < total_frames:
        sleep_s = stream_sleep_seconds(args.stream_rate, fps)
        for frame_nr in range(initial_frames, total_frames):
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            rr.set_time_sequence('frame_nr', frame_nr)
            rr.set_time_seconds('time', frame_time_seconds(frame_nr, fps))
            configuration = motion_data[frame_nr, :]
            rerun_urdf.update(configuration)
            if object_positions is not None:
                contact_active = object_contact is not None and object_contact[frame_nr, 0] >= 1.0
                rr.log(
                    "object/ball",
                    rr.Points3D(
                        object_positions[frame_nr],
                        radii=0.12,
                        colors=_ball_colors_from_contact(contact_active),
                    ),
                )
