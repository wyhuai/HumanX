#!/usr/bin/env python3
"""Convert XGen HOI npz batches to HumanX motion pkl format."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R


G1_29DOF_AXES = np.asarray(
    [
        [0, 1, 0],  # left_hip_pitch
        [1, 0, 0],  # left_hip_roll
        [0, 0, 1],  # left_hip_yaw
        [0, 1, 0],  # left_knee
        [0, 1, 0],  # left_ankle_pitch
        [1, 0, 0],  # left_ankle_roll
        [0, 1, 0],  # right_hip_pitch
        [1, 0, 0],  # right_hip_roll
        [0, 0, 1],  # right_hip_yaw
        [0, 1, 0],  # right_knee
        [0, 1, 0],  # right_ankle_pitch
        [1, 0, 0],  # right_ankle_roll
        [0, 0, 1],  # waist_yaw
        [1, 0, 0],  # waist_roll
        [0, 1, 0],  # waist_pitch
        [0, 1, 0],  # left_shoulder_pitch
        [1, 0, 0],  # left_shoulder_roll
        [0, 0, 1],  # left_shoulder_yaw
        [0, 1, 0],  # left_elbow
        [1, 0, 0],  # left_wrist_roll
        [0, 1, 0],  # left_wrist_pitch
        [0, 0, 1],  # left_wrist_yaw
        [0, 1, 0],  # right_shoulder_pitch
        [1, 0, 0],  # right_shoulder_roll
        [0, 0, 1],  # right_shoulder_yaw
        [0, 1, 0],  # right_elbow
        [1, 0, 0],  # right_wrist_roll
        [0, 1, 0],  # right_wrist_pitch
        [0, 0, 1],  # right_wrist_yaw
    ],
    dtype=np.float32,
)

HUMANX_DOF_NAMES = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)

XGEN_DOF_NAMES = (
    "left_hip_pitch",
    "right_hip_pitch",
    "waist_yaw",
    "left_hip_roll",
    "right_hip_roll",
    "waist_roll",
    "left_hip_yaw",
    "right_hip_yaw",
    "waist_pitch",
    "left_knee",
    "right_knee",
    "left_shoulder_pitch",
    "right_shoulder_pitch",
    "left_ankle_pitch",
    "right_ankle_pitch",
    "left_shoulder_roll",
    "right_shoulder_roll",
    "left_ankle_roll",
    "right_ankle_roll",
    "left_shoulder_yaw",
    "right_shoulder_yaw",
    "left_elbow",
    "right_elbow",
    "left_wrist_roll",
    "right_wrist_roll",
    "left_wrist_pitch",
    "right_wrist_pitch",
    "left_wrist_yaw",
    "right_wrist_yaw",
)

XGEN_TO_HUMANX_DOF_INDICES = np.asarray(
    [XGEN_DOF_NAMES.index(name) for name in HUMANX_DOF_NAMES],
    dtype=np.int64,
)


def _natural_path_key(path: Path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def _numeric_label_from_path(path: Path) -> int | None:
    match = re.match(r"^(\d+)_", path.stem)
    if match is None:
        return None
    return int(match.group(1))


def _as_float32(array: np.ndarray, *, shape_tail: tuple[int, ...], key: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.shape[-len(shape_tail) :] != shape_tail:
        raise ValueError(f"{key} has shape {value.shape}, expected trailing shape {shape_tail}")
    return value


def _wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape[-1] != 4:
        raise ValueError(f"quaternion has shape {quat.shape}, expected trailing dim 4")
    out = quat[..., [1, 2, 3, 0]].copy()
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError("quaternion contains zero-norm entries")
    return (out / norm).astype(np.float32)


def _quat_xyzw_to_rotvec(quat_xyzw: np.ndarray) -> np.ndarray:
    return R.from_quat(quat_xyzw).as_rotvec().astype(np.float32)


def _read_object_size(z: np.lib.npyio.NpzFile) -> np.ndarray | None:
    """Return the source object's size metadata when available.

    XGen currently stores this as either a scalar characteristic size or a
    three-element box extent. Keep the original shape so XMimic can choose an
    appropriate uniform asset scale at runtime.
    """
    if "object_size" not in z:
        return None
    value = np.asarray(z["object_size"], dtype=np.float32).squeeze()
    if value.ndim == 0:
        value = value.reshape(1)
    if value.ndim == 1 and value.size in (1, 3) and np.all(np.isfinite(value)) and np.all(value > 0):
        return value.astype(np.float32)
    raise ValueError(f"object_size has unsupported shape/value: shape={value.shape}, value={value}")


def _infer_foot_contact(z: np.lib.npyio.NpzFile, frames: int) -> tuple[np.ndarray, np.ndarray]:
    if "ee_pos_w" not in z:
        unknown = np.full((frames, 1), 0.5, dtype=np.float32)
        return unknown.copy(), unknown.copy()

    ee_pos = np.asarray(z["ee_pos_w"], dtype=np.float32)
    if ee_pos.ndim != 3 or ee_pos.shape[0] != frames or ee_pos.shape[1] < 4 or ee_pos.shape[2] != 3:
        unknown = np.full((frames, 1), 0.5, dtype=np.float32)
        return unknown.copy(), unknown.copy()

    ground_z = float(np.asarray(z.get("grounding_ground_z", [0.0]), dtype=np.float32).reshape(-1)[0])
    threshold = ground_z + 0.065
    left = (ee_pos[:, 2, 2] <= threshold).astype(np.float32).reshape(frames, 1)
    right = (ee_pos[:, 3, 2] <= threshold).astype(np.float32).reshape(frames, 1)
    return left, right


def convert_one(
    path: Path,
    *,
    freeze_until_release: bool,
    fixed_object: bool,
    ignore_support: bool,
    label_dim: int | None = None,
) -> dict:
    with np.load(path, allow_pickle=True) as z:
        root_trans = _as_float32(z["base_pos_w"], shape_tail=(3,), key="base_pos_w")
        frames = root_trans.shape[0]

        joint_pos_xgen = _as_float32(z["joint_pos"], shape_tail=(29,), key="joint_pos")
        if joint_pos_xgen.shape[0] != frames:
            raise ValueError(f"{path}: joint_pos length does not match base_pos_w")
        joint_pos = joint_pos_xgen[:, XGEN_TO_HUMANX_DOF_INDICES]

        root_rot = _wxyz_to_xyzw(z["base_quat_w"])
        obj_rot = _wxyz_to_xyzw(z["object_quat_w"])
        pose_aa = np.zeros((frames, 30, 3), dtype=np.float32)
        pose_aa[:, 0, :] = _quat_xyzw_to_rotvec(root_rot)
        pose_aa[:, 1:, :] = joint_pos[:, :, None] * G1_29DOF_AXES[None, :, :]

        contact_info = np.asarray(z.get("contact_info", np.zeros((frames, 1), dtype=np.float32)), dtype=np.float32)
        contact = (contact_info.reshape(frames, -1).max(axis=1, keepdims=True) > 0.5).astype(np.float32)
        lfoot_contact, rfoot_contact = _infer_foot_contact(z, frames)

        fps_array = np.asarray(z["fps"], dtype=np.float32).reshape(-1)
        fps = int(round(float(fps_array[0]))) if fps_array.size else 30

        obj_pos = _as_float32(z["object_pos_w"], shape_tail=(3,), key="object_pos_w")
        obj_vel = _as_float32(z["object_lin_vel_w"], shape_tail=(3,), key="object_lin_vel_w")
        obj_ang_vel = _as_float32(z["object_ang_vel_w"], shape_tail=(3,), key="object_ang_vel_w")
        object_size = _read_object_size(z)

        release_frame = frames - 1
        if "object_release_frame" in z:
            release_frame = int(np.asarray(z["object_release_frame"]).reshape(-1)[0])
            release_frame = max(0, min(release_frame, frames - 1))

        obj_freeze = np.zeros((frames, 1), dtype=np.int64)
        if fixed_object:
            obj_pos = np.repeat(obj_pos[:1], frames, axis=0).astype(np.float32)
            obj_rot = np.repeat(obj_rot[:1], frames, axis=0).astype(np.float32)
            obj_vel = np.zeros((frames, 3), dtype=np.float32)
            obj_ang_vel = np.zeros((frames, 3), dtype=np.float32)
            obj_freeze[:, 0] = 1
            release_frame = frames - 1
        elif freeze_until_release:
            obj_freeze[: release_frame + 1, 0] = 1

        numeric_label = _numeric_label_from_path(path)
        if label_dim is None:
            skill_label = np.zeros((frames, 32), dtype=np.float32)
        else:
            skill_label = np.zeros((frames, label_dim), dtype=np.float32)
            if numeric_label is None:
                raise ValueError(f"{path}: expected filename prefix like '1_...' for one-hot label")
            label_index = numeric_label - 1
            if label_index < 0 or label_index >= label_dim:
                raise ValueError(f"{path}: label {numeric_label} is outside label_dim={label_dim}")
            skill_label[:, label_index] = 1.0

        motion = {
            "fps": fps,
            "root_trans_offset": root_trans,
            "root_rot": root_rot,
            "dof": joint_pos[:, :, None].astype(np.float32),
            "pose_aa": pose_aa,
            "obj_pos": obj_pos,
            "obj_rot_quat": obj_rot,
            "obj_vel": obj_vel,
            "obj_ang_vel": obj_ang_vel,
            "contact": contact,
            "lfoot_contact": lfoot_contact,
            "rfoot_contact": rfoot_contact,
            "skill_label": skill_label,
            "stand_frame": release_frame,
            "obj_freeze": obj_freeze,
            **({"object_size": object_size} if object_size is not None else {}),
            "source_path": str(path),
        }

        if ignore_support:
            motion["need_support"] = False
            motion["support_top_pos"] = np.zeros(3, dtype=np.float32)

        return motion


def convert_batch(input_dir: Path, output: Path, *, freeze_until_release: bool, fixed_object: bool, ignore_support: bool) -> dict:
    paths = sorted(input_dir.glob("*.npz"), key=_natural_path_key)
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")
    numeric_labels = [_numeric_label_from_path(path) for path in paths]
    label_dim = max(label for label in numeric_labels if label is not None) if any(
        label is not None for label in numeric_labels
    ) else None

    converted = {}
    for path in paths:
        key = path.stem
        converted[key] = convert_one(
            path,
            freeze_until_release=freeze_until_release,
            fixed_object=fixed_object,
            ignore_support=ignore_support,
            label_dim=label_dim,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(converted, output)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/xgen/approach"))
    parser.add_argument("--output", type=Path, default=Path("data/motions/xgen_approach/approach.pkl"))
    parser.add_argument(
        "--no-freeze-until-release",
        action="store_true",
        help="Do not populate obj_freeze for the static pre-contact object segment.",
    )
    parser.add_argument(
        "--fixed-object",
        action="store_true",
        help="Keep the reference object fixed at its first-frame pose for the whole motion.",
    )
    parser.add_argument(
        "--use-support",
        action="store_true",
        help="Preserve support handling instead of forcing need_support=False.",
    )
    args = parser.parse_args()

    converted = convert_batch(
        args.input_dir,
        args.output,
        freeze_until_release=not args.no_freeze_until_release,
        fixed_object=args.fixed_object,
        ignore_support=not args.use_support,
    )
    total_frames = sum(m["root_trans_offset"].shape[0] for m in converted.values())
    print(f"Wrote {len(converted)} motions, {total_frames} frames -> {args.output}")


if __name__ == "__main__":
    main()
