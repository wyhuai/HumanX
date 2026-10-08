#!/usr/bin/env python3
"""Convert a joblib motion payload into the HOI editor NPZ schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mujoco
import numpy as np

from server import JOINT_NAMES, MUJOCO_JOINT_NAMES, MuJoCoModel, _quat_ang_vel


HERE = Path(__file__).resolve().parent
MODEL_XML = HERE.parent / "decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml"


def _wxyz_from_xyzw(quat: np.ndarray) -> np.ndarray:
    values = np.asarray(quat, dtype=np.float32)
    return values[..., [3, 0, 1, 2]]


def _gradient(values: np.ndarray, fps: float) -> np.ndarray:
    return np.gradient(values, 1.0 / max(float(fps), 1e-6), axis=0).astype(np.float32)


def _convert_joint_order(dof_mujoco: np.ndarray) -> np.ndarray:
    """Convert pkl/MJCF depth order into the editor's HOI articulation order."""
    mujoco_indices = {name: index for index, name in enumerate(MUJOCO_JOINT_NAMES)}
    return np.stack(
        [dof_mujoco[:, mujoco_indices[name]] for name in JOINT_NAMES],
        axis=1,
    ).astype(np.float32, copy=False)


def _load_payload(path: Path) -> dict[str, object]:
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError("pickle payload must be a dictionary")
    if len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
        return next(iter(payload.values()))
    return payload


def convert(source: Path, destination: Path, robot_only: bool = False) -> Path:
    source_payload = _load_payload(source)
    required = ("root_trans_offset", "root_rot", "dof")
    if not robot_only:
        required += ("obj_pos", "obj_rot_quat")
    missing = [key for key in required if key not in source_payload]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")

    base_pos = np.asarray(source_payload["root_trans_offset"], dtype=np.float32)
    base_quat = _wxyz_from_xyzw(source_payload["root_rot"])
    dof_mujoco = np.asarray(source_payload["dof"], dtype=np.float32).reshape(base_pos.shape[0], -1)
    joint_pos = _convert_joint_order(dof_mujoco)
    object_pos = None
    object_quat = None
    if not robot_only:
        object_pos = np.asarray(source_payload["obj_pos"], dtype=np.float32)
        object_quat = _wxyz_from_xyzw(source_payload["obj_rot_quat"])
    if base_pos.ndim != 2 or base_pos.shape[1] != 3:
        raise ValueError("root_trans_offset must have shape (T, 3)")
    if base_quat.shape != (base_pos.shape[0], 4):
        raise ValueError("root_rot must have shape (T, 4)")
    if dof_mujoco.shape != (base_pos.shape[0], 29):
        raise ValueError("dof must have shape (T, 29)")
    if not robot_only and (
        object_pos.shape != (base_pos.shape[0], 3)
        or object_quat.shape != (base_pos.shape[0], 4)
    ):
        raise ValueError("object pose arrays must match the root frame count")

    fps = float(np.asarray(source_payload.get("fps", 50.0)).reshape(-1)[0])
    model = MuJoCoModel(MODEL_XML)
    body_pose = np.stack([
        model.body_pose(base_pos[frame], base_quat[frame], joint_pos[frame])
        for frame in range(base_pos.shape[0])
    ])
    body_pos = body_pose[..., :3]
    body_quat = body_pose[..., 3:]
    body_lin_vel = _gradient(body_pos, fps)
    body_ang_vel = np.stack([
        _quat_ang_vel(body_quat[:, body], fps)
        for body in range(body_quat.shape[1])
    ], axis=1)

    # Resolve named EE bodies directly from MuJoCo so this remains independent
    # of body id ordering in the XML.
    def body_id(name: str) -> int:
        result = int(mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_BODY, name))
        if result < 0:
            raise ValueError(f"MuJoCo model is missing body {name}")
        return result

    ee_ids = [
        body_id("left_wrist_yaw_link"),
        body_id("right_wrist_yaw_link"),
        body_id("left_ankle_roll_link"),
        body_id("right_ankle_roll_link"),
        # This older G1 XML has no head body; torso is the stable fallback
        # expected by the editor's five-point EE display.
        body_id("torso_link"),
    ]
    ee_pos = body_pos[:, ee_ids]
    ee_quat = body_quat[:, ee_ids]

    arrays = {
        "fps": np.array([fps], dtype=np.float32),
        "base_pos_w": base_pos,
        "base_quat_w": base_quat,
        "joint_pos": joint_pos,
        "joint_vel": _gradient(joint_pos, fps),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": body_lin_vel,
        "body_ang_vel_w": body_ang_vel.astype(np.float32),
        "ee_pos_w": ee_pos.astype(np.float32),
        "ee_quat_w": ee_quat.astype(np.float32),
    }
    if not robot_only:
        source_contact = np.asarray(source_payload.get(
            "contact", np.zeros((base_pos.shape[0], 1), dtype=np.float32)
        ), dtype=np.float32).reshape(base_pos.shape[0], -1).max(axis=1)
        left_foot = np.asarray(source_payload.get(
            "lfoot_contact", np.zeros_like(source_contact)
        ), dtype=np.float32).reshape(base_pos.shape[0], -1).max(axis=1)
        right_foot = np.asarray(source_payload.get(
            "rfoot_contact", np.zeros_like(source_contact)
        ), dtype=np.float32).reshape(base_pos.shape[0], -1).max(axis=1)
        contact_info = np.zeros((base_pos.shape[0], 4, 1), dtype=np.float32)
        contact_info[:, 0, 0] = (left_foot > 0.5).astype(np.float32)
        contact_info[:, 1, 0] = (right_foot > 0.5).astype(np.float32)
        contact_info[:, 2, 0] = (source_contact > 0.5).astype(np.float32)
        contact_info[:, 3, 0] = (source_contact > 0.5).astype(np.float32)

        hand_span = np.linalg.norm(ee_pos[:, 0] - ee_pos[:, 1], axis=1)
        contact_mask = source_contact > 0.5
        object_size = (
            float(np.median(hand_span[contact_mask]))
            if np.any(contact_mask)
            else float(np.median(hand_span))
        )
        object_size = max(object_size, 0.02)
        zero_tables = np.zeros((base_pos.shape[0], 3), dtype=np.float32)
        arrays.update({
            "object_pos_w": object_pos,
            "object_quat_w": object_quat,
            "object_lin_vel_w": _gradient(object_pos, fps),
            "object_ang_vel_w": _quat_ang_vel(object_quat, fps),
            "object_size": np.array([object_size], dtype=np.float32),
            "object_type": np.array(["box"]),
            "object_yaw_only": np.array([False], dtype=np.bool_),
            "support_enabled": np.array([False], dtype=np.bool_),
            "table1_pos_w": zero_tables,
            "table2_pos_w": zero_tables.copy(),
            "contact_info": contact_info,
        })
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--robot-only",
        action="store_true",
        help="omit object and contact fields from the output",
    )
    args = parser.parse_args()
    result = convert(args.source, args.destination, robot_only=args.robot_only)
    print(result)


if __name__ == "__main__":
    main()
