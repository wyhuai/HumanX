import torch

from isaac_utils.rotations import xyzw_to_wxyz


def apply_motion_frame_to_state_tensors(
    *,
    env_ids: torch.Tensor,
    simulator_name: str,
    robot_root_states: torch.Tensor,
    object_root_state: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    motion_res: dict,
) -> None:
    env_ids = env_ids.to(dtype=torch.long)

    robot_root_rot = motion_res["root_rot"][env_ids]
    if simulator_name == "isaacsim":
        robot_root_rot = xyzw_to_wxyz(robot_root_rot)

    robot_root_states[env_ids, :3] = motion_res["root_pos"][env_ids]
    robot_root_states[env_ids, 3:7] = robot_root_rot
    robot_root_states[env_ids, 7:10] = motion_res["root_vel"][env_ids]
    robot_root_states[env_ids, 10:13] = motion_res["root_ang_vel"][env_ids]

    object_root_state[env_ids, :3] = motion_res["curr_obj_pos"][env_ids]
    object_root_state[env_ids, 3:7] = motion_res["curr_obj_rot"][env_ids]
    object_root_state[env_ids, 7:10] = motion_res["curr_obj_vel"][env_ids]
    object_root_state[env_ids, 10:13] = motion_res["curr_obj_ang_vel"][env_ids]

    dof_pos[env_ids] = motion_res["dof_pos"][env_ids]
    dof_vel[env_ids] = motion_res["dof_vel"][env_ids]
