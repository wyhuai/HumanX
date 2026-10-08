#!/usr/bin/env python3
"""Small local HTTP API for browsing and editing HOI trajectory NPZ files."""

from __future__ import annotations

import argparse
import copy
import json
import math
import mimetypes
import re
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import mujoco


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = HERE.parents[1] / "data"
MODEL_XML = (
    HERE.parent
    / "decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml"
)
MODEL_MESH_ROOT = MODEL_XML.parent / "meshes"
SUPPORT_THICKNESS = 0.02
OBJECT_BOX = "box"
OBJECT_SPHERE = "sphere"
OBJECT_TYPES = {OBJECT_BOX, OBJECT_SPHERE}
ANCHOR_PALM_CENTER = "palm_center"
ANCHOR_RIGHT_PALM_NORMAL = "right_palm_normal"
ANCHOR_MODES = {ANCHOR_PALM_CENTER, ANCHOR_RIGHT_PALM_NORMAL}
RIGHT_PALM_NORMAL_OFFSET = 0.05
RIGHT_PALM_NORMAL_LOCAL = np.array([1.0, 0.0, 0.0], dtype=np.float64)
CONTACT_TARGET_GAP = 0.0015
CONTACT_PALM_TARGET_GAP = 0.003
CONTACT_PALM_CENTER_SCALE = 0.045
CONTACT_PALM_CENTER_WEIGHT = 12.0
CONTACT_PALM_PROJECTION_SCALE = 0.018
CONTACT_PALM_PROJECTION_WEIGHT = 18.0
CONTACT_HAND_GAP_WEIGHT = 24000.0
CONTACT_HAND_CLEARANCE_WEIGHT = 5.0
CONTACT_JOINT_REGULARIZATION_WEIGHT = 1.5
CONTACT_OPTIMIZATION_ITERATIONS = 12
CONTACT_OPTIMIZATION_TRANSITION_FRAMES = 8
CONTACT_CORRECTION_SMOOTHING_PASSES = 8
CONTACT_CORRECTION_ENDPOINT_BLEND = 0.25
PRE_CONTACT_HOLD = "hold"
PRE_CONTACT_PARABOLIC = "parabolic"
PRE_CONTACT_MODES = {PRE_CONTACT_HOLD, PRE_CONTACT_PARABOLIC}
POST_CONTACT_HOLD = "hold"
POST_CONTACT_PARABOLIC = "parabolic"
POST_CONTACT_MODES = {POST_CONTACT_HOLD, POST_CONTACT_PARABOLIC}
OBJECT_GRAVITY_W = np.array([0.0, 0.0, -9.80665], dtype=np.float64)
FOOT_CONTACT_POINTS_LOCAL = np.array([
    [-0.05, 0.025, -0.03],
    [-0.05, -0.025, -0.03],
    [0.12, 0.03, -0.03],
    [0.12, -0.03, -0.03],
], dtype=np.float64)
GROUNDING_GROUND_PERCENTILE = 10.0
GROUNDING_SUPPORT_SPEED = 0.55
GROUNDING_SUPPORT_VERTICAL_SPEED = 0.22
GROUNDING_FLIGHT_SPEED = 0.75
GROUNDING_FLIGHT_VERTICAL_SPEED = 0.12
GROUNDING_FOOT_SEPARATION = 0.06
GROUNDING_LANDING_WINDOW = 3
GROUNDING_LANDING_TOLERANCE = 0.008
GROUNDING_CORRECTION_STEP = 0.025
GROUNDING_CORRECTION_ALPHA = 0.65
GROUNDING_PENETRATION_TOLERANCE = 0.004

MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# The HOI npz files store joint_pos in the Isaac/PhysX articulation order,
# which interleaves the left/right limbs and waist by tree depth. MuJoCo qpos
# uses the XML depth-first order above, so FK writes values by joint name.
JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# Limits from the G1 29-dof MJCF, listed in MuJoCo order then projected into
# the HOI data order.
MUJOCO_JOINT_RANGES = [
    (-2.5307, 2.8798),
    (-0.5236, 2.9671),
    (-2.7576, 2.7576),
    (-0.087267, 2.8798),
    (-0.87267, 0.5236),
    (-0.2618, 0.2618),
    (-2.5307, 2.8798),
    (-2.9671, 0.5236),
    (-2.7576, 2.7576),
    (-0.087267, 2.8798),
    (-0.87267, 0.5236),
    (-0.2618, 0.2618),
    (-2.618, 2.618),
    (-0.52, 0.52),
    (-0.52, 0.52),
    (-3.0892, 2.6704),
    (-1.5882, 2.2515),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.97222, 1.97222),
    (-1.61443, 1.61443),
    (-1.61443, 1.61443),
    (-3.0892, 2.6704),
    (-2.2515, 1.5882),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.97222, 1.97222),
    (-1.61443, 1.61443),
    (-1.61443, 1.61443),
]
JOINT_RANGES_BY_NAME = dict(zip(MUJOCO_JOINT_NAMES, MUJOCO_JOINT_RANGES))
JOINT_RANGES = [JOINT_RANGES_BY_NAME[name] for name in JOINT_NAMES]

FRIENDLY_JOINT_NAMES = [
    name.removesuffix("_joint").replace("_", " ").title() for name in JOINT_NAMES
]


def _finite_float(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    values = np.asarray(value, dtype=np.float32)
    if values.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must be finite")
    return values


def _json_value(value: np.ndarray | object) -> object:
    array = np.asarray(value)
    if array.dtype.kind in "fc":
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return array.tolist()


def _object_type(arrays: dict[str, np.ndarray]) -> str:
    value = np.asarray(arrays.get("object_type", [OBJECT_BOX])).reshape(-1)
    if value.size == 0:
        return OBJECT_BOX
    item = value[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8", errors="ignore")
    normalized = str(item).strip().lower()
    return normalized if normalized in OBJECT_TYPES else OBJECT_BOX


def _object_anchor_mode(arrays: dict[str, np.ndarray]) -> str:
    value = np.asarray(arrays.get("object_anchor_mode", [ANCHOR_PALM_CENTER])).reshape(-1)
    if value.size == 0:
        return ANCHOR_PALM_CENTER
    item = value[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8", errors="ignore")
    normalized = str(item).strip().lower()
    return normalized if normalized in ANCHOR_MODES else ANCHOR_PALM_CENTER


def _grounding_applied(arrays: dict[str, np.ndarray]) -> bool:
    value = np.asarray(arrays.get("grounding_correction_applied", [False])).reshape(-1)
    if value.size == 0:
        return False
    item = value[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8", errors="ignore")
    return str(item).strip().lower() in {"1", "true", "yes"}


def _normalize_object_type(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    normalized = str(value or OBJECT_BOX).strip().lower()
    if normalized not in OBJECT_TYPES:
        raise ValueError(f"object_type must be one of: {', '.join(sorted(OBJECT_TYPES))}")
    return normalized


def _normalize_anchor_mode(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    normalized = str(value or ANCHOR_PALM_CENTER).strip().lower()
    if normalized not in ANCHOR_MODES:
        raise ValueError(f"anchor_mode must be one of: {', '.join(sorted(ANCHOR_MODES))}")
    return normalized


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    value = np.asarray(quat, dtype=np.float64)[:4]
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return value / norm


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    q = _normalize_quat(quat)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = _normalize_quat(left)
    b = _normalize_quat(right)
    return _normalize_quat(_quat_mul_raw(a, b))


def _quat_mul_raw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)[:4]
    b = np.asarray(right, dtype=np.float64)[:4]
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def _quat_rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = _normalize_quat(quat)
    v = np.asarray(vector, dtype=np.float64)[:3]
    pure = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    return _quat_mul_raw(_quat_mul_raw(q, pure), _quat_conjugate(q))[1:]


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = _normalize_quat(quat)
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _average_quat(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = _normalize_quat(left)
    b = _normalize_quat(right)
    if float(np.dot(a, b)) < 0.0:
        b = -b
    return _normalize_quat(a + b)


def _yaw_from_quat(quat: np.ndarray) -> float:
    w, x, y, z = _normalize_quat(quat)
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _yaw_only_quat(quat: np.ndarray) -> np.ndarray:
    yaw = _yaw_from_quat(quat)
    return np.array([math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)], dtype=np.float64)


def _yaw_quat_from_angle(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)], dtype=np.float64)


def _wrist_center_pose(arrays: dict[str, np.ndarray], frame: int) -> tuple[np.ndarray, np.ndarray]:
    ee_pos = np.asarray(arrays["ee_pos_w"], dtype=np.float64)
    left_pos = ee_pos[frame, 0, :3]
    right_pos = ee_pos[frame, 1, :3]
    center_pos = (left_pos + right_pos) * 0.5
    ee_quat = np.asarray(arrays.get("ee_quat_w", np.zeros((ee_pos.shape[0], 0, 4))), dtype=np.float64)
    if ee_quat.ndim == 3 and ee_quat.shape[1] >= 2:
        center_quat = _average_quat(ee_quat[frame, 0, :4], ee_quat[frame, 1, :4])
    else:
        center_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return center_pos, center_quat


def _palm_anchor_pose(
    arrays: dict[str, np.ndarray],
    frame: int,
    yaw_only: bool,
    model: object | None = None,
    anchor_mode: str = ANCHOR_PALM_CENTER,
) -> tuple[np.ndarray, np.ndarray]:
    anchor_mode = _normalize_anchor_mode(anchor_mode)
    if model is not None and hasattr(model, "palm_pose"):
        palm_pose = model.palm_pose(
            arrays["base_pos_w"][frame],
            arrays["base_quat_w"][frame],
            arrays["joint_pos"][frame],
        )
        palm_pos = np.asarray(palm_pose[:, :3], dtype=np.float64)
        palm_quat = np.asarray(palm_pose[:, 3:7], dtype=np.float64)
        if anchor_mode == ANCHOR_RIGHT_PALM_NORMAL and palm_pos.shape[0] >= 2:
            right_quat = _normalize_quat(palm_quat[1])
            palm_normal = _quat_to_matrix(right_quat) @ RIGHT_PALM_NORMAL_LOCAL
            center_pos = palm_pos[1] + palm_normal * RIGHT_PALM_NORMAL_OFFSET
            center_quat = right_quat
        else:
            center_pos = (palm_pos[0] + palm_pos[1]) * 0.5
            center_quat = _average_quat(palm_quat[0], palm_quat[1])
        if not yaw_only:
            return center_pos, center_quat
        if anchor_mode == ANCHOR_RIGHT_PALM_NORMAL:
            return center_pos, _yaw_only_quat(center_quat)
        delta = palm_pos[1, :2] - palm_pos[0, :2]
        if np.isfinite(delta).all() and float(np.linalg.norm(delta)) > 1e-5:
            return center_pos, _yaw_quat_from_angle(
                float(math.atan2(delta[1], delta[0]))
            )
        return center_pos, _yaw_only_quat(center_quat)

    if anchor_mode == ANCHOR_RIGHT_PALM_NORMAL:
        ee_pos = np.asarray(arrays["ee_pos_w"], dtype=np.float64)
        if ee_pos.ndim == 3 and ee_pos.shape[1] >= 2:
            ee_quat = np.asarray(
                arrays.get("ee_quat_w", np.zeros((ee_pos.shape[0], 0, 4))),
                dtype=np.float64,
            )
            if ee_quat.ndim == 3 and ee_quat.shape[1] >= 2:
                center_quat = _normalize_quat(ee_quat[frame, 1, :4])
                center_pos = (
                    ee_pos[frame, 1, :3]
                    + (
                        _quat_to_matrix(center_quat)
                        @ RIGHT_PALM_NORMAL_LOCAL
                    ) * RIGHT_PALM_NORMAL_OFFSET
                )
                return (
                    center_pos,
                    _yaw_only_quat(center_quat) if yaw_only else center_quat,
                )

    center_pos, center_quat = _wrist_center_pose(arrays, frame)
    if not yaw_only:
        return center_pos, center_quat
    ee_pos = np.asarray(arrays["ee_pos_w"], dtype=np.float64)
    if ee_pos.ndim == 3 and ee_pos.shape[1] >= 2:
        delta = ee_pos[frame, 1, :2] - ee_pos[frame, 0, :2]
        if np.isfinite(delta).all() and float(np.linalg.norm(delta)) > 1e-5:
            return center_pos, _yaw_quat_from_angle(float(math.atan2(delta[1], delta[0])))
    return center_pos, _yaw_only_quat(center_quat)


def _default_object_pose(
    arrays: dict[str, np.ndarray],
    frame: int,
    model: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if "object_pos_w" in arrays and "object_quat_w" in arrays:
        return (
            np.asarray(arrays["object_pos_w"][frame], dtype=np.float64)[:3],
            _normalize_quat(arrays["object_quat_w"][frame]),
        )
    return _palm_anchor_pose(arrays, frame, False, model)


def _object_size(arrays: dict[str, np.ndarray], frame: int) -> float:
    if "object_size" in arrays:
        value = float(np.asarray(arrays["object_size"]).reshape(-1)[0])
        if math.isfinite(value) and value > 0.0:
            return value
    if "ee_pos_w" in arrays:
        ee_pos = np.asarray(arrays["ee_pos_w"], dtype=np.float64)
        if ee_pos.ndim == 3 and ee_pos.shape[1] >= 2:
            frame = max(0, min(int(frame), ee_pos.shape[0] - 1))
            return float(np.linalg.norm(ee_pos[frame, 0, :3] - ee_pos[frame, 1, :3]))
    return 0.28


def _wrist_center_positions(arrays: dict[str, np.ndarray]) -> np.ndarray:
    ee_pos = np.asarray(arrays.get("ee_pos_w"), dtype=np.float64)
    if ee_pos.ndim != 3 or ee_pos.shape[1] < 2 or ee_pos.shape[2] < 3:
        return np.zeros((0, 3), dtype=np.float64)
    return (ee_pos[:, 0, :3] + ee_pos[:, 1, :3]) * 0.5


def _wrist_center_velocity(arrays: dict[str, np.ndarray], frame: int) -> np.ndarray:
    centers = _wrist_center_positions(arrays)
    if centers.shape[0] < 2:
        return np.zeros(3, dtype=np.float64)
    frame = max(0, min(int(frame), centers.shape[0] - 1))
    fps = float(np.asarray(arrays.get("fps", [50.0])).reshape(-1)[0])
    velocity = np.gradient(centers, 1.0 / max(fps, 1e-6), axis=0)
    return np.asarray(velocity[frame], dtype=np.float64)


def _pre_contact_start_wrist_velocity(
    arrays: dict[str, np.ndarray], contact_start: int, window: int = 5
) -> np.ndarray:
    """Average wrist-center velocity over the frames immediately before contact."""
    centers = _wrist_center_positions(arrays)
    if centers.shape[0] < 2:
        return np.zeros(3, dtype=np.float64)
    fps = float(np.asarray(arrays.get("fps", [50.0])).reshape(-1)[0])
    start = max(0, min(int(contact_start), centers.shape[0] - 1))
    begin = max(0, start - max(1, int(window)))
    # Each sample is the velocity over one interval before contact:
    # begin -> begin+1, ..., start-1 -> start.
    samples = np.diff(centers[begin:start + 1], axis=0)
    if samples.shape[0] == 0:
        return _wrist_center_velocity(arrays, start)
    dt = 1.0 / max(fps, 1e-6)
    return (np.mean(samples, axis=0) / dt).astype(np.float64, copy=False)


def _post_contact_end_wrist_velocity(
    arrays: dict[str, np.ndarray], contact_end: int, window: int = 5
) -> np.ndarray:
    """Average wrist-center velocity over the frames immediately after release."""
    centers = _wrist_center_positions(arrays)
    if centers.shape[0] < 2:
        return np.zeros(3, dtype=np.float64)
    fps = float(np.asarray(arrays.get("fps", [50.0])).reshape(-1)[0])
    end = max(0, min(int(contact_end), centers.shape[0] - 1))
    count = min(max(1, int(window)), centers.shape[0] - 1 - end)
    # Each sample is the velocity over one interval after contact end:
    # end -> end+1, ..., end+count-1 -> end+count.
    samples = np.diff(centers[end:end + count + 1], axis=0)
    if samples.shape[0] == 0:
        return _wrist_center_velocity(arrays, end)
    dt = 1.0 / max(fps, 1e-6)
    return (np.mean(samples, axis=0) / dt).astype(np.float64, copy=False)


def _post_contact_mode(arrays: dict[str, np.ndarray]) -> str:
    value = arrays.get("object_post_contact_mode")
    if value is None:
        return POST_CONTACT_HOLD
    raw = np.asarray(value).reshape(-1)
    if raw.size == 0:
        return POST_CONTACT_HOLD
    item = raw[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8", errors="ignore")
    mode = str(item).strip().lower()
    return mode if mode in POST_CONTACT_MODES else POST_CONTACT_HOLD


def _pre_contact_mode(arrays: dict[str, np.ndarray]) -> str:
    value = arrays.get("object_pre_contact_mode")
    if value is None:
        return PRE_CONTACT_HOLD
    raw = np.asarray(value).reshape(-1)
    if raw.size == 0:
        return PRE_CONTACT_HOLD
    item = raw[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8", errors="ignore")
    mode = str(item).strip().lower()
    return mode if mode in PRE_CONTACT_MODES else PRE_CONTACT_HOLD


def _scalar_int(arrays: dict[str, np.ndarray], key: str, default: int = -1) -> int:
    value = arrays.get(key)
    if value is None:
        return default
    raw = np.asarray(value).reshape(-1)
    if raw.size == 0:
        return default
    try:
        return int(raw[0])
    except (TypeError, ValueError):
        return default


def _stored_vector(
    arrays: dict[str, np.ndarray], key: str, default: np.ndarray
) -> np.ndarray:
    value = arrays.get(key)
    if value is None:
        return np.asarray(default, dtype=np.float64).copy()
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size < 3 or not np.isfinite(result[:3]).all():
        return np.asarray(default, dtype=np.float64).copy()
    return result[:3].copy()


def _support_size(arrays: dict[str, np.ndarray], object_size: float) -> np.ndarray:
    for key in ("support_size", "table_size"):
        if key in arrays:
            values = np.asarray(arrays[key], dtype=np.float64).reshape(-1)
            if values.size >= 3 and np.isfinite(values[:3]).all() and float(np.min(values[:3])) > 0.0:
                return values[:3]
    if "support_enabled" in arrays or "support1_pos_w" in arrays or "support2_pos_w" in arrays:
        return np.array([object_size, object_size, SUPPORT_THICKNESS], dtype=np.float64)
    return np.array([0.72, 0.58, 0.12], dtype=np.float64)


def _support_enabled(arrays: dict[str, np.ndarray]) -> bool:
    if "support_enabled" in arrays:
        return bool(np.asarray(arrays["support_enabled"]).reshape(-1)[0])
    return (
        "support1_pos_w" in arrays
        or "support2_pos_w" in arrays
        or ("table1_pos_w" in arrays and "table2_pos_w" in arrays)
    )


def _object_extent_along_normal(
    object_type: str,
    object_size: float,
    object_quat: np.ndarray,
    normal: np.ndarray,
) -> float:
    half_size = float(object_size) * 0.5
    if object_type == OBJECT_SPHERE:
        return half_size
    rotation = _quat_to_matrix(object_quat)
    local_normal = rotation.T @ np.asarray(normal, dtype=np.float64)
    return half_size * float(np.abs(local_normal).sum())


def _quat_ang_vel(quats: np.ndarray, fps: float) -> np.ndarray:
    q = np.asarray(quats, dtype=np.float64).copy()
    for index in range(1, q.shape[0]):
        if float(np.dot(q[index - 1], q[index])) < 0.0:
            q[index] = -q[index]
    dt = 1.0 / max(float(fps), 1e-6)
    dq = np.gradient(q, dt, axis=0)
    ang_vel = np.zeros((q.shape[0], 3), dtype=np.float64)
    for index in range(q.shape[0]):
        omega_quat = _quat_mul_raw(dq[index], _quat_conjugate(q[index]))
        ang_vel[index] = 2.0 * omega_quat[1:]
    return ang_vel.astype(np.float32)


def _safe_relative(root: Path, requested: str) -> Path:
    relative = Path(unquote(requested)).expanduser()
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise ValueError("path must stay inside the data directory")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("path must stay inside the data directory")
    return path


def _safe_folder_name(requested: str) -> str:
    name = str(requested or "").strip()
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or Path(name).name != name
    ):
        raise ValueError("folder name must be a single safe path component")
    return name


def _friendly_file_name(path: Path) -> str:
    return path.as_posix()


class TrajectorySession:
    def __init__(self, path: Path, model: "MuJoCoModel") -> None:
        self.path = path
        self.model = model
        self.lock = threading.RLock()
        self.source_mtime_ns = 0
        self.reload()

    def reload(self) -> None:
        with np.load(self.path, allow_pickle=True) as source:
            self.original = {key: np.array(source[key], copy=True) for key in source.files}
        self.arrays = copy.deepcopy(self.original)
        self.dirty_frames: set[int] = set()
        self._grounding_profile_cache: dict[str, object] | None = None
        self.source_mtime_ns = self.path.stat().st_mtime_ns

    @property
    def frame_count(self) -> int:
        return int(self.arrays["joint_pos"].shape[0])

    @property
    def fps(self) -> float:
        return float(np.asarray(self.arrays.get("fps", [50.0])).reshape(-1)[0])

    def frame(self, index: int) -> dict[str, object]:
        index = max(0, min(int(index), self.frame_count - 1))
        with self.lock:
            arrays = self.arrays
            has_object = "object_pos_w" in arrays and "object_quat_w" in arrays
            has_contact = "contact_info" in arrays
            object_pos, object_quat = _default_object_pose(
                arrays,
                index,
                self.model,
            )
            object_size = _object_size(arrays, index)
            object_type = _object_type(arrays)
            object_anchor_mode = _object_anchor_mode(arrays)
            pre_contact_mode = _pre_contact_mode(arrays)
            post_contact_mode = _post_contact_mode(arrays)
            contact_start_frame = _scalar_int(arrays, "object_contact_start_frame")
            release_frame = _scalar_int(arrays, "object_release_frame")
            pre_contact_velocity_default = _pre_contact_start_wrist_velocity(
                arrays,
                contact_start_frame if contact_start_frame >= 0 else index,
            )
            release_velocity_default = _post_contact_end_wrist_velocity(
                arrays,
                release_frame if release_frame >= 0 else index,
            )
            pre_contact_velocity = _stored_vector(
                arrays,
                "object_pre_contact_velocity_w",
                pre_contact_velocity_default,
            )
            release_velocity = _stored_vector(
                arrays,
                "object_release_velocity_w",
                release_velocity_default,
            )
            contact_start_pos = _stored_vector(
                arrays,
                "object_contact_start_pos_w",
                object_pos,
            )
            contact_start_quat = _normalize_quat(
                np.asarray(
                    arrays.get("object_contact_start_quat_w", object_quat),
                    dtype=np.float64,
                ).reshape(-1)[:4]
            )
            release_pos = _stored_vector(arrays, "object_release_pos_w", object_pos)
            gravity = _stored_vector(arrays, "object_gravity_w", OBJECT_GRAVITY_W)
            contact = np.asarray(arrays.get("contact_info", np.zeros((self.frame_count, 4, 1))))
            contact_flat = contact[index].reshape(-1)
            body_pose = self.model.body_pose(
                arrays["base_pos_w"][index],
                arrays["base_quat_w"][index],
                arrays["joint_pos"][index],
            )
            palm_pose = self.model.palm_pose(
                arrays["base_pos_w"][index],
                arrays["base_quat_w"][index],
                arrays["joint_pos"][index],
            )
            return {
                "file": self.path.name,
                "frame": index,
                "time": round(index / max(self.fps, 1e-6), 4),
                "fps": self.fps,
                "total_frames": self.frame_count,
                "dirty": index in self.dirty_frames,
                "has_object": has_object,
                "has_contact": has_contact,
                "grounding_correction_applied": _grounding_applied(arrays),
                "joint_names": JOINT_NAMES,
                "friendly_joint_names": FRIENDLY_JOINT_NAMES,
                "joint_ranges": JOINT_RANGES,
                "base_pos_w": _json_value(arrays["base_pos_w"][index]),
                "base_quat_w": _json_value(arrays["base_quat_w"][index]),
                "joint_pos": _json_value(arrays["joint_pos"][index]),
                "joint_vel": _json_value(arrays.get("joint_vel", np.zeros_like(arrays["joint_pos"]))[index]),
                "body_pos_w": _json_value(arrays.get("body_pos_w", np.zeros((self.frame_count, 0, 3)))[index]),
                "body_quat_w": _json_value(arrays.get("body_quat_w", np.zeros((self.frame_count, 0, 4)))[index]),
                "body_pose_w": _json_value(body_pose),
                "palm_pos_w": _json_value(palm_pose[:, :3]),
                "palm_quat_w": _json_value(palm_pose[:, 3:]),
                "ee_pos_w": _json_value(arrays.get("ee_pos_w", np.zeros((self.frame_count, 0, 3)))[index]),
                "ee_quat_w": _json_value(arrays.get("ee_quat_w", np.zeros((self.frame_count, 0, 4)))[index]),
                "object_pos_w": _json_value(object_pos),
                "object_quat_w": _json_value(object_quat),
                "object_size": object_size,
                "object_type": object_type,
                "object_anchor_mode": object_anchor_mode,
                "object_yaw_only": bool(np.asarray(arrays.get("object_yaw_only", [False])).reshape(-1)[0]),
                "object_pre_contact_mode": pre_contact_mode,
                "object_contact_start_frame": contact_start_frame,
                "object_contact_start_pos_w": _json_value(contact_start_pos),
                "object_contact_start_quat_w": _json_value(contact_start_quat),
                "object_pre_contact_velocity_w": _json_value(pre_contact_velocity),
                "object_post_contact_mode": post_contact_mode,
                "object_release_frame": release_frame,
                "object_release_pos_w": _json_value(release_pos),
                "object_release_velocity_w": _json_value(release_velocity),
                "object_gravity_w": _json_value(gravity),
                "object_contact_optimized": bool(np.asarray(
                    arrays.get("object_contact_optimized", [False])
                ).reshape(-1)[0]),
                "object_contact_optimization_target_gap_w": float(np.asarray(
                    arrays.get(
                        "object_contact_optimization_target_gap_w",
                        [CONTACT_TARGET_GAP],
                    )
                ).reshape(-1)[0]),
                "object_contact_optimization_palm_target_gap_w": float(np.asarray(
                    arrays.get(
                        "object_contact_optimization_palm_target_gap_w",
                        [CONTACT_PALM_TARGET_GAP],
                    )
                ).reshape(-1)[0]),
                "support_enabled": _support_enabled(arrays),
                "support_size": _json_value(_support_size(arrays, object_size)),
                "table1_pos_w": _json_value(arrays.get("table1_pos_w", np.zeros((self.frame_count, 3)))[index]),
                "table2_pos_w": _json_value(arrays.get("table2_pos_w", np.zeros((self.frame_count, 3)))[index]),
                "contact_info": _json_value(contact[index]),
                "contact_active": [bool(float(value) > 0.5) for value in contact_flat[:4]],
                "keys": sorted(arrays.keys()),
            }

    def trajectory(self, limit: int = 900) -> dict[str, object]:
        with self.lock:
            count = self.frame_count
            stride = max(1, int(math.ceil(count / max(1, min(limit, count)))))
            indices = np.arange(0, count, stride, dtype=np.int64)
            if indices[-1] != count - 1:
                indices = np.append(indices, count - 1)
            arrays = self.arrays
            contact = np.asarray(arrays.get("contact_info", np.zeros((count, 4, 1))))
            if "object_pos_w" in arrays:
                object_xyz = arrays["object_pos_w"][indices]
            else:
                object_xyz = np.stack([
                    _palm_anchor_pose(
                        arrays,
                        int(index),
                        False,
                        self.model,
                    )[0]
                    for index in indices
                ], axis=0)
            return {
                "file": self.path.name,
                "fps": self.fps,
                "total_frames": count,
                "indices": indices.tolist(),
                "base_xy": _json_value(arrays["base_pos_w"][indices, :2]),
                "object_xyz": _json_value(object_xyz),
                "table1_xyz": _json_value(arrays.get("table1_pos_w", np.zeros((count, 3)))[indices]),
                "table2_xyz": _json_value(arrays.get("table2_pos_w", np.zeros((count, 3)))[indices]),
                "contact": _json_value(contact[indices].reshape(len(indices), -1).max(axis=1)),
            }

    def grounding_profile(self) -> dict[str, object]:
        """Estimate a fixed floor and a smoothed vertical correction preview."""
        with self.lock:
            if self._grounding_profile_cache is not None:
                return copy.deepcopy(self._grounding_profile_cache)

            count = self.frame_count
            fps = max(self.fps, 1e-6)
            foot_heights = np.zeros((count, 2), dtype=np.float64)
            foot_centers = np.zeros((count, 2, 3), dtype=np.float64)
            foot_body_ids = self.model.ee_body_ids[2:4]

            for frame in range(count):
                poses = self.model.body_pose(
                    self.arrays["base_pos_w"][frame],
                    self.arrays["base_quat_w"][frame],
                    self.arrays["joint_pos"][frame],
                )
                for foot_index, body_id in enumerate(foot_body_ids):
                    foot_pose = poses[body_id]
                    rotation = _quat_to_matrix(foot_pose[3:7])
                    points = (
                        FOOT_CONTACT_POINTS_LOCAL @ rotation.T
                        + foot_pose[:3]
                    )
                    foot_heights[frame, foot_index] = float(np.min(points[:, 2]))
                    foot_centers[frame, foot_index] = np.mean(points, axis=0)

            if count > 1:
                foot_velocity = np.gradient(
                    foot_centers,
                    1.0 / fps,
                    axis=0,
                )
            else:
                foot_velocity = np.zeros_like(foot_centers)
            foot_speed = np.linalg.norm(foot_velocity, axis=2)
            foot_vertical_speed = foot_velocity[:, :, 2]

            flat_heights = foot_heights.reshape(-1)
            low_cut = float(np.percentile(flat_heights, GROUNDING_GROUND_PERCENTILE))
            low_samples = flat_heights[flat_heights <= low_cut + 1e-9]
            ground_z = float(np.median(low_samples)) if low_samples.size else low_cut
            if _grounding_applied(self.arrays):
                correction = np.zeros(count, dtype=np.float64)
                corrected_heights = foot_heights - ground_z
                profile = {
                    "file": self.path.name,
                    "fps": self.fps,
                    "total_frames": count,
                    "ground_z": ground_z,
                    "grounding_method": "grounding already baked into trajectory",
                    "foot_height_w": _json_value(foot_heights.astype(np.float32)),
                    "foot_speed_mps": _json_value(foot_speed.astype(np.float32)),
                    "correction_z": _json_value(correction.astype(np.float32)),
                    "corrected_foot_height_w": _json_value(corrected_heights.astype(np.float32)),
                    "support_flags": np.zeros(count, dtype=bool).tolist(),
                    "airborne_flags": np.zeros(count, dtype=bool).tolist(),
                }
                self._grounding_profile_cache = profile
                return copy.deepcopy(profile)

            correction = np.zeros(count, dtype=np.float64)
            support_flags = np.zeros(count, dtype=bool)
            airborne_flags = np.zeros(count, dtype=bool)
            current_correction = 0.0
            previous_vertical_speed = foot_vertical_speed[0].copy()
            phase = "grounded"

            for frame in range(count):
                heights = foot_heights[frame]
                speeds = foot_speed[frame]
                vertical_speed = foot_vertical_speed[frame]
                quiet = (
                    (speeds <= GROUNDING_SUPPORT_SPEED)
                    & (np.abs(vertical_speed) <= GROUNDING_SUPPORT_VERTICAL_SPEED)
                )
                rising_together = (
                    np.all(vertical_speed >= GROUNDING_FLIGHT_VERTICAL_SPEED)
                    or (
                        np.all(speeds >= GROUNDING_FLIGHT_SPEED)
                        and np.all(vertical_speed >= 0.03)
                    )
                )
                if phase == "grounded" and frame >= 3 and rising_together:
                    phase = "flight"

                if phase == "flight":
                    landed = False
                    for foot in range(2):
                        start = max(0, frame - GROUNDING_LANDING_WINDOW)
                        stop = min(count, frame + GROUNDING_LANDING_WINDOW + 1)
                        local_min = float(np.min(foot_heights[start:stop, foot]))
                        landed = landed or (
                            quiet[foot]
                            and speeds[foot] <= 0.7
                            and heights[foot] <= local_min + GROUNDING_LANDING_TOLERANCE
                            and (
                                previous_vertical_speed[foot] <= -0.03
                                or (
                                    frame > 0
                                    and heights[foot] <= foot_heights[frame - 1, foot] + 0.005
                                )
                            )
                        )
                    if landed:
                        phase = "grounded"

                if phase == "grounded":
                    support_ids = np.flatnonzero(quiet)
                    if support_ids.size == 1:
                        other = 1 - int(support_ids[0])
                        if heights[int(support_ids[0])] > (
                            heights[other] + GROUNDING_FOOT_SEPARATION
                        ):
                            support_ids = np.empty(0, dtype=np.int64)
                    if support_ids.size:
                        support_flags[frame] = True
                        target = ground_z - float(np.min(heights[support_ids]))
                        delta = np.clip(
                            target - current_correction,
                            -GROUNDING_CORRECTION_STEP,
                            GROUNDING_CORRECTION_STEP,
                        )
                        current_correction += GROUNDING_CORRECTION_ALPHA * float(delta)

                corrected_min = float(np.min(heights) + current_correction - ground_z)
                if corrected_min < -GROUNDING_PENETRATION_TOLERANCE:
                    current_correction += min(
                        GROUNDING_CORRECTION_STEP,
                        -corrected_min,
                    ) * GROUNDING_CORRECTION_ALPHA

                correction[frame] = current_correction
                airborne_flags[frame] = phase == "flight"
                previous_vertical_speed = vertical_speed

            corrected_heights = foot_heights + correction[:, None] - ground_z
            profile = {
                "file": self.path.name,
                "fps": self.fps,
                "total_frames": count,
                "ground_z": ground_z,
                "grounding_method": "lowest foot contact site with support-phase tracking",
                "foot_height_w": _json_value(foot_heights.astype(np.float32)),
                "foot_speed_mps": _json_value(foot_speed.astype(np.float32)),
                "correction_z": _json_value(correction.astype(np.float32)),
                "corrected_foot_height_w": _json_value(corrected_heights.astype(np.float32)),
                "support_flags": support_flags.tolist(),
                "airborne_flags": airborne_flags.tolist(),
            }
            self._grounding_profile_cache = profile
            return copy.deepcopy(profile)

    def edit(self, frame: int, changes: dict[str, object]) -> dict[str, object]:
        frame = max(0, min(int(frame), self.frame_count - 1))
        with self.lock:
            if "object_type" in changes:
                self.arrays["object_type"] = np.array([
                    _normalize_object_type(changes["object_type"])
                ])

            if "object_anchor_mode" in changes:
                self.arrays["object_anchor_mode"] = np.array([
                    _normalize_anchor_mode(changes["object_anchor_mode"])
                ])

            if "joint_index" in changes:
                joint_index = int(changes["joint_index"])
                if joint_index < 0 or joint_index >= len(JOINT_NAMES):
                    raise ValueError("joint_index out of range")
                value = _finite_float(changes.get("joint_value"), "joint_value")
                low, high = JOINT_RANGES[joint_index]
                if not low <= value <= high:
                    raise ValueError(f"joint_value must be within [{low}, {high}]")
                self.arrays["joint_pos"][frame, joint_index] = value
                if "joint_vel" in self.arrays:
                    self.arrays["joint_vel"][:, joint_index] = np.gradient(
                        self.arrays["joint_pos"][:, joint_index], 1.0 / max(self.fps, 1e-6)
                    ).astype(self.arrays["joint_vel"].dtype, copy=False)

            for key, shape in (
                ("base_pos_w", (3,)),
                ("base_quat_w", (4,)),
                ("object_pos_w", (3,)),
                ("object_quat_w", (4,)),
            ):
                if key in changes:
                    value = _array(changes[key], shape, key)
                    if key.endswith("quat_w"):
                        norm = float(np.linalg.norm(value))
                        if norm < 1e-6:
                            raise ValueError(f"{key} quaternion cannot be zero")
                        value = value / norm
                    self.arrays[key][frame] = value

            if "object_size" in changes:
                value = _finite_float(changes["object_size"], "object_size")
                if value <= 0.0:
                    raise ValueError("object_size must be positive")
                self.arrays["object_size"] = np.array([value], dtype=np.float32)

            if "object_pos_w" in changes and "object_lin_vel_w" in self.arrays:
                self.arrays["object_lin_vel_w"][:] = np.gradient(
                    self.arrays["object_pos_w"], 1.0 / max(self.fps, 1e-6), axis=0
                ).astype(self.arrays["object_lin_vel_w"].dtype, copy=False)

            self.dirty_frames.add(frame)
            self._grounding_profile_cache = None
            return self.frame(frame)

    def _apply_grounding_correction(
        self,
        output: dict[str, np.ndarray],
        source_frame_indices: np.ndarray,
    ) -> dict[str, object]:
        """Bake the preview's vertical correction into an exported trajectory."""
        indices = np.asarray(source_frame_indices, dtype=np.int64).reshape(-1)
        count = int(indices.size)
        if count == 0:
            raise ValueError("grounding correction requires at least one frame")
        profile = self.grounding_profile()
        profile_correction = np.asarray(
            profile["correction_z"],
            dtype=np.float64,
        ).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= profile_correction.size):
            raise ValueError("grounding correction frame index out of range")
        correction = profile_correction[indices]
        fps = max(self.fps, 1e-6)
        correction_velocity = (
            np.gradient(correction, 1.0 / fps)
            if count > 1
            else np.zeros(1, dtype=np.float64)
        )

        base_pos = output.get("base_pos_w")
        if (
            base_pos is not None
            and base_pos.ndim == 2
            and base_pos.shape[0] == count
            and base_pos.shape[1] >= 3
        ):
            base_pos[:, 2] += correction.astype(base_pos.dtype, copy=False)

        if "joint_pos" in output and "base_pos_w" in output:
            _recompute_kinematic_fields(
                self.model,
                output,
                range(count),
                self.fps,
            )

        for key in ("root_lin_vel_world", "root_lin_vel_w", "base_lin_vel_w"):
            values = output.get(key)
            if (
                values is not None
                and values.ndim == 2
                and values.shape == (count, 3)
            ):
                values[:, 2] += correction_velocity.astype(values.dtype, copy=False)

        ground_z = float(profile["ground_z"])
        output["grounding_correction_applied"] = np.array([True], dtype=np.bool_)
        output["grounding_correction_z"] = correction.astype(np.float32)
        output["grounding_ground_z"] = np.array([ground_z], dtype=np.float32)
        return {
            "grounding_correction_applied": True,
            "grounding_ground_z": ground_z,
            "grounding_correction_max_abs_m": float(np.max(np.abs(correction))),
        }

    def save_copy(
        self,
        requested_name: str | None,
        apply_grounding_correction: bool = False,
        output_dir: Path | None = None,
    ) -> Path:
        with self.lock:
            edited_dir = output_dir if output_dir is not None else self.path.parent / "edited"
            edited_dir.mkdir(parents=True, exist_ok=True)
            if requested_name:
                clean_name = Path(requested_name).name
                if not clean_name.endswith(".npz"):
                    clean_name += ".npz"
            else:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                clean_name = f"{self.path.stem}_edited_{stamp}.npz"
            destination = (edited_dir / clean_name).resolve()
            if edited_dir.resolve() not in destination.parents:
                raise ValueError("save path must stay inside the selected folder")
            output = {key: np.array(value, copy=True) for key, value in self.arrays.items()}
            if "object_pos_w" in output and "object_quat_w" in output:
                output.setdefault("object_type", np.array([_object_type(self.arrays)]))
                output.setdefault(
                    "object_anchor_mode",
                    np.array([_object_anchor_mode(self.arrays)]),
                )
            if apply_grounding_correction:
                self._apply_grounding_correction(
                    output,
                    np.arange(self.frame_count, dtype=np.int64),
                )
            np.savez_compressed(destination, **output)
            return destination

    def crop_copy(
        self,
        start_frame: int,
        end_frame: int,
        requested_name: str | None,
        apply_grounding_correction: bool = False,
        output_dir: Path | None = None,
    ) -> tuple[Path, dict[str, int]]:
        with self.lock:
            count = self.frame_count
            start = max(0, min(int(start_frame), count - 1))
            end = max(0, min(int(end_frame), count - 1))
            if end < start:
                start, end = end, start
            new_count = end - start + 1

            output: dict[str, np.ndarray] = {}
            for key, value in self.arrays.items():
                array = np.asarray(value)
                if array.ndim >= 1 and array.shape[0] == count:
                    output[key] = np.array(array[start:end + 1], copy=True)
                else:
                    output[key] = np.array(array, copy=True)
            if "object_pos_w" in output and "object_quat_w" in output:
                output.setdefault("object_type", np.array([_object_type(self.arrays)]))
                output.setdefault(
                    "object_anchor_mode",
                    np.array([_object_anchor_mode(self.arrays)]),
                )

            for key in ("object_contact_start_frame", "object_release_frame"):
                if key not in output or output[key].size == 0:
                    continue
                original = int(np.asarray(output[key]).reshape(-1)[0])
                if original >= 0:
                    output[key].flat[0] = (
                        original - start
                        if start <= original <= end
                        else -1
                    )

            if apply_grounding_correction:
                self._apply_grounding_correction(
                    output,
                    np.arange(start, end + 1, dtype=np.int64),
                )
            crop_dir = output_dir if output_dir is not None else (
                self.path.parent.parent / "cropped"
                if self.path.parent.name == "robot_only"
                else self.path.parent / "cropped"
            )
            crop_dir.mkdir(parents=True, exist_ok=True)
            if requested_name:
                clean_name = Path(requested_name).name
                if not clean_name.endswith(".npz"):
                    clean_name += ".npz"
            else:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                clean_name = f"{self.path.stem}_crop_{start}_{end}_{stamp}.npz"
            destination = (crop_dir / clean_name).resolve()
            if crop_dir.resolve() not in destination.parents:
                raise ValueError("crop path must stay inside the selected folder")
            np.savez_compressed(destination, **output)
            return destination, {
                "crop_start": start,
                "crop_end": end,
                "frames": new_count,
            }

    def synthesize_copy(
        self,
        contact_start: int,
        contact_end: int,
        object_pos_w: object,
        object_quat_w: object,
        object_size: object | None,
        object_type: object,
        requested_name: str | None,
        yaw_only: bool = False,
        support_surfaces: bool = True,
        post_contact_mode: str = POST_CONTACT_HOLD,
        release_velocity_w: object | None = None,
        pre_contact_mode: str = PRE_CONTACT_HOLD,
        pre_contact_velocity_w: object | None = None,
        optimize_contact_stage: bool = False,
        preview_only: bool = False,
        apply_grounding_correction: bool = False,
        output_dir: Path | None = None,
        anchor_mode: object = ANCHOR_PALM_CENTER,
    ) -> tuple[Path | dict[str, np.ndarray], dict[str, object]]:
        with self.lock:
            pre_contact_mode = str(pre_contact_mode or PRE_CONTACT_HOLD).strip().lower()
            if pre_contact_mode not in PRE_CONTACT_MODES:
                raise ValueError(
                    f"pre_contact_mode must be one of: {', '.join(sorted(PRE_CONTACT_MODES))}"
                )
            post_contact_mode = str(post_contact_mode or POST_CONTACT_HOLD).strip().lower()
            if post_contact_mode not in POST_CONTACT_MODES:
                raise ValueError(
                    f"post_contact_mode must be one of: {', '.join(sorted(POST_CONTACT_MODES))}"
                )
            object_type = _normalize_object_type(object_type)
            anchor_mode = _normalize_anchor_mode(anchor_mode)
            count = self.frame_count
            start = max(0, min(int(contact_start), count - 1))
            end = max(0, min(int(contact_end), count - 1))
            if end < start:
                start, end = end, start

            output = {
                key: np.array(value, copy=True)
                for key, value in self.arrays.items()
            }
            joint_pos = np.asarray(output["joint_pos"], dtype=np.float32).copy()
            output["joint_pos"] = joint_pos
            grounding_meta = {
                "grounding_correction_applied": False,
            }
            if apply_grounding_correction:
                grounding_meta = self._apply_grounding_correction(
                    output,
                    np.arange(count, dtype=np.int64),
                )
                joint_pos = np.asarray(output["joint_pos"], dtype=np.float32).copy()
                output["joint_pos"] = joint_pos
            arrays = output

            seed_pos = _array(object_pos_w, (3,), "object_pos_w").astype(np.float64)
            seed_quat = _normalize_quat(_array(object_quat_w, (4,), "object_quat_w"))
            if yaw_only:
                seed_quat = _yaw_only_quat(seed_quat)
            start_center_pos, start_anchor_quat = _palm_anchor_pose(
                arrays,
                start,
                yaw_only,
                self.model,
                anchor_mode,
            )
            center_inv = _quat_conjugate(start_anchor_quat)
            relative_pos = _quat_rotate(center_inv, seed_pos - start_center_pos)
            relative_quat = _quat_mul(center_inv, seed_quat)
            contact_start_pos = start_center_pos + _quat_rotate(start_anchor_quat, relative_pos)
            contact_start_quat = _normalize_quat(_quat_mul(start_anchor_quat, relative_quat))
            if yaw_only:
                contact_start_quat = _yaw_only_quat(contact_start_quat)

            object_pos = np.zeros((count, 3), dtype=np.float32)
            object_quat = np.zeros((count, 4), dtype=np.float32)
            release_pos = contact_start_pos.copy()
            release_quat = contact_start_quat.copy()
            pre_contact_velocity = (
                _pre_contact_start_wrist_velocity(arrays, start)
                if pre_contact_velocity_w is None
                else _array(
                    pre_contact_velocity_w,
                    (3,),
                    "pre_contact_velocity_w",
                ).astype(np.float64)
            )
            release_velocity = (
                _post_contact_end_wrist_velocity(arrays, end)
                if release_velocity_w is None
                else _array(release_velocity_w, (3,), "release_velocity_w").astype(np.float64)
            )
            for frame in range(count):
                if frame < start:
                    if pre_contact_mode == PRE_CONTACT_PARABOLIC:
                        elapsed = (start - frame) / max(self.fps, 1e-6)
                        pos = (
                            contact_start_pos
                            - pre_contact_velocity * elapsed
                            + OBJECT_GRAVITY_W * (0.5 * elapsed * elapsed)
                        )
                        quat = contact_start_quat
                    else:
                        pos = seed_pos
                        quat = seed_quat
                elif frame <= end:
                    center_pos, anchor_quat = _palm_anchor_pose(
                        arrays,
                        frame,
                        yaw_only,
                        self.model,
                        anchor_mode,
                    )
                    pos = center_pos + _quat_rotate(anchor_quat, relative_pos)
                    quat = _quat_mul(anchor_quat, relative_quat)
                    if yaw_only:
                        quat = _yaw_only_quat(quat)
                    release_pos = pos
                    release_quat = quat
                else:
                    if post_contact_mode == POST_CONTACT_PARABOLIC:
                        elapsed = (frame - end) / max(self.fps, 1e-6)
                        pos = (
                            release_pos
                            + release_velocity * elapsed
                            + OBJECT_GRAVITY_W * (0.5 * elapsed * elapsed)
                        )
                    else:
                        pos = release_pos
                    quat = release_quat
                object_pos[frame] = pos.astype(np.float32)
                object_quat[frame] = _normalize_quat(quat).astype(np.float32)

            if object_size is None:
                wrist_span = float(np.linalg.norm(
                    np.asarray(arrays["ee_pos_w"][start, 0, :3])
                    - np.asarray(arrays["ee_pos_w"][start, 1, :3])
                ))
            else:
                wrist_span = _finite_float(object_size, "object_size")
                if wrist_span <= 0.0:
                    raise ValueError("object_size must be positive")
            dt = 1.0 / max(self.fps, 1e-6)
            object_lin_vel = np.gradient(object_pos, dt, axis=0).astype(np.float32)
            object_ang_vel = _quat_ang_vel(object_quat, self.fps)
            optimized = bool(optimize_contact_stage)
            optimization_max_object_shift = 0.0
            optimization_max_joint_shift = 0.0
            optimization_mean_loss = 0.0
            if optimized and end >= start:
                reference_joint_pos = np.asarray(
                    arrays["joint_pos"], dtype=np.float64
                ).copy()
                reference_object_pos = np.asarray(
                    object_pos, dtype=np.float64
                ).copy()
                contact_count = end - start + 1
                arm_count = len(CONTACT_ARM_JOINT_INDICES)
                joint_deltas = np.zeros(
                    (contact_count, arm_count),
                    dtype=np.float64,
                )
                losses = []
                previous_delta = None
                for local_index, frame in enumerate(range(start, end + 1)):
                    optimized_joints, loss = _optimize_contact_frame(
                        self.model,
                        arrays["base_pos_w"][frame],
                        arrays["base_quat_w"][frame],
                        reference_joint_pos[frame],
                        reference_object_pos[frame],
                        object_quat[frame],
                        wrist_span,
                        object_type,
                        previous_delta,
                    )
                    joint_delta = (
                        optimized_joints[CONTACT_ARM_JOINT_INDICES]
                        - reference_joint_pos[frame, CONTACT_ARM_JOINT_INDICES]
                    )
                    joint_deltas[local_index] = joint_delta
                    previous_delta = joint_delta.copy()
                    losses.append(loss)

                joint_deltas = _smooth_contact_corrections(joint_deltas)
                transition = CONTACT_OPTIMIZATION_TRANSITION_FRAMES
                first_delta = joint_deltas[0]
                last_delta = joint_deltas[-1]
                for offset in range(transition, 0, -1):
                    frame = start - offset
                    if frame >= 0:
                        progress = (
                            transition + 1 - offset
                        ) / (transition + 1)
                        weight = _smoothstep(progress)
                        joint_pos[frame, CONTACT_ARM_JOINT_INDICES] = (
                            reference_joint_pos[
                                frame, CONTACT_ARM_JOINT_INDICES
                            ]
                            + weight * first_delta
                        )
                for local_index, frame in enumerate(range(start, end + 1)):
                    joint_pos[frame, CONTACT_ARM_JOINT_INDICES] = (
                        reference_joint_pos[frame, CONTACT_ARM_JOINT_INDICES]
                        + joint_deltas[local_index]
                    )
                for offset in range(1, transition + 1):
                    frame = end + offset
                    if frame < count:
                        progress = offset / (transition + 1)
                        weight = 1.0 - _smoothstep(progress)
                        joint_pos[frame, CONTACT_ARM_JOINT_INDICES] = (
                            reference_joint_pos[
                                frame, CONTACT_ARM_JOINT_INDICES
                            ]
                            + weight * last_delta
                        )
                object_lin_vel = np.gradient(
                    object_pos,
                    dt,
                    axis=0,
                ).astype(np.float32)
                object_ang_vel = _quat_ang_vel(object_quat, self.fps)
                optimization_max_joint_shift = float(
                    np.abs(joint_deltas).max(initial=0.0)
                )
                optimization_mean_loss = float(np.mean(losses)) if losses else 0.0
            contact = np.zeros((count, 4, 1), dtype=np.float32)
            contact[start:end + 1, 2:4, 0] = 1.0

            output["joint_pos"] = joint_pos
            output["object_pos_w"] = object_pos
            output["object_quat_w"] = object_quat
            output["object_lin_vel_w"] = object_lin_vel
            output["object_ang_vel_w"] = object_ang_vel
            output["object_size"] = np.array([wrist_span], dtype=np.float32)
            output["object_type"] = np.array([object_type])
            output["object_anchor_mode"] = np.array([anchor_mode])
            output["object_yaw_only"] = np.array([bool(yaw_only)], dtype=np.bool_)
            output["object_pre_contact_mode"] = np.array([pre_contact_mode])
            output["object_contact_start_frame"] = np.array([start], dtype=np.int32)
            output["object_contact_start_pos_w"] = contact_start_pos.astype(np.float32)
            output["object_contact_start_quat_w"] = _normalize_quat(contact_start_quat).astype(np.float32)
            output["object_pre_contact_velocity_w"] = pre_contact_velocity.astype(np.float32)
            output["object_post_contact_mode"] = np.array([post_contact_mode])
            output["object_release_frame"] = np.array([end], dtype=np.int32)
            output["object_release_pos_w"] = release_pos.astype(np.float32)
            output["object_release_quat_w"] = _normalize_quat(release_quat).astype(np.float32)
            output["object_release_velocity_w"] = release_velocity.astype(np.float32)
            output["object_gravity_w"] = OBJECT_GRAVITY_W.astype(np.float32)
            output["contact_info"] = contact
            output["support_enabled"] = np.array([bool(support_surfaces)], dtype=np.bool_)
            output["object_contact_optimized"] = np.array([optimized], dtype=np.bool_)
            output["object_contact_optimization_iterations"] = np.array(
                [CONTACT_OPTIMIZATION_ITERATIONS],
                dtype=np.int32,
            )
            output["object_contact_optimization_target_gap_w"] = np.array(
                [CONTACT_TARGET_GAP],
                dtype=np.float32,
            )
            output["object_contact_optimization_palm_target_gap_w"] = np.array(
                [CONTACT_PALM_TARGET_GAP],
                dtype=np.float32,
            )

            if optimized:
                if "body_pos_w" in output or "ee_pos_w" in output:
                    transition_start = max(
                        0,
                        start - CONTACT_OPTIMIZATION_TRANSITION_FRAMES,
                    )
                    transition_end = min(
                        count - 1,
                        end + CONTACT_OPTIMIZATION_TRANSITION_FRAMES,
                    )
                    _recompute_kinematic_fields(
                        self.model,
                        output,
                        range(transition_start, transition_end + 1),
                        self.fps,
                    )

            if support_surfaces:
                support_size = np.array([wrist_span, wrist_span, SUPPORT_THICKNESS], dtype=np.float32)
                support1_bottom = np.array([
                    seed_pos[0],
                    seed_pos[1],
                    seed_pos[2] - wrist_span * 0.5 - SUPPORT_THICKNESS,
                ], dtype=np.float32)
                support2_bottom = np.array([
                    release_pos[0],
                    release_pos[1],
                    release_pos[2] - wrist_span * 0.5 - SUPPORT_THICKNESS,
                ], dtype=np.float32)
                support1 = np.tile(support1_bottom, (count, 1))
                support2 = np.tile(support2_bottom, (count, 1))
                output["support_size"] = support_size
                output["support1_pos_w"] = support1
                output["support2_pos_w"] = support2
                output["support1_quat_w"] = np.tile(
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    (count, 1),
                )
                output["support2_quat_w"] = np.tile(
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    (count, 1),
                )
                output["table1_pos_w"] = support1
                output["table2_pos_w"] = support2

            meta = {
                "contact_start": start,
                "contact_end": end,
                "object_size": wrist_span,
                "object_type": object_type,
                "object_anchor_mode": anchor_mode,
                "yaw_only": bool(yaw_only),
                "support_surfaces": bool(support_surfaces),
                "pre_contact_mode": pre_contact_mode,
                "pre_contact_velocity_w": _json_value(pre_contact_velocity.astype(np.float32)),
                "post_contact_mode": post_contact_mode,
                "release_velocity_w": _json_value(release_velocity.astype(np.float32)),
                "relative_pos": _json_value(relative_pos.astype(np.float32)),
                "relative_quat": _json_value(relative_quat.astype(np.float32)),
                "contact_optimized": optimized,
                "contact_optimization_max_object_shift_m": optimization_max_object_shift,
                "contact_optimization_max_joint_shift_rad": optimization_max_joint_shift,
                "contact_optimization_mean_loss": optimization_mean_loss,
                "contact_optimization_target_gap_w": CONTACT_TARGET_GAP,
                "contact_optimization_palm_target_gap_w": CONTACT_PALM_TARGET_GAP,
                **grounding_meta,
            }
            if preview_only:
                return output, meta

            synth_dir = output_dir if output_dir is not None else (
                self.path.parent.parent / "synthesized"
                if self.path.parent.name == "robot_only"
                else self.path.parent / "synthesized"
            )
            synth_dir.mkdir(parents=True, exist_ok=True)
            if requested_name:
                clean_name = Path(requested_name).name
                if not clean_name.endswith(".npz"):
                    clean_name += ".npz"
            else:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                clean_name = f"{self.path.stem}_hoi_{stamp}.npz"
            destination = (synth_dir / clean_name).resolve()
            if synth_dir.resolve() not in destination.parents:
                raise ValueError("synthesis path must stay inside the selected folder")
            np.savez_compressed(destination, **output)
            return destination, meta


class App:
    def __init__(self, data_root: Path, model: "MuJoCoModel") -> None:
        self.data_root = data_root.resolve()
        self.model = model
        self.sessions: dict[Path, TrajectorySession] = {}
        self.lock = threading.RLock()

    def _is_editor_npz(self, path: Path) -> bool:
        try:
            with np.load(path, allow_pickle=True) as data:
                return all(
                    key in data
                    for key in ("base_pos_w", "base_quat_w", "joint_pos")
                )
        except Exception:
            return False

    def folders(self) -> list[dict[str, object]]:
        result = [{
            "folder": "",
            "name": "All folders",
            "files": sum(
                1
                for path in self.data_root.rglob("*.npz")
                if self._is_editor_npz(path)
            ),
        }]
        for path in sorted(self.data_root.rglob("*")):
            if not path.is_dir():
                continue
            relative_parts = path.relative_to(self.data_root).parts
            if any(part.startswith(".") for part in relative_parts):
                continue
            relative = path.relative_to(self.data_root).as_posix()
            result.append({
                "folder": relative,
                "name": path.name,
                "files": sum(
                    1
                    for child in path.glob("*.npz")
                    if self._is_editor_npz(child)
                ),
            })
        return result

    def folder(self, requested: str | None) -> Path:
        path = _safe_relative(self.data_root, str(requested or ""))
        if not path.is_dir():
            raise ValueError("unknown data folder")
        return path

    def files(self, folder: str | None = None) -> list[dict[str, object]]:
        folder_path = self.folder(folder)
        paths = (
            sorted(self.data_root.rglob("*.npz"))
            if not folder
            else sorted(folder_path.glob("*.npz"))
        )
        result = []
        for path in paths:
            try:
                with np.load(path, allow_pickle=True) as data:
                    if any(
                        key not in data
                        for key in ("base_pos_w", "base_quat_w", "joint_pos")
                    ):
                        continue
                    frames = int(data["joint_pos"].shape[0])
                    fps = float(np.asarray(data.get("fps", [50.0])).reshape(-1)[0])
                    keys = list(data.files)
            except Exception:
                continue
            relative = path.relative_to(self.data_root).as_posix()
            session = self.sessions.get(path.resolve())
            result.append({
                "file": relative,
                "name": path.name,
                "frames": frames,
                "fps": fps,
                "duration": round(frames / max(fps, 1e-6), 3),
                "keys": keys,
                "dirty": bool(session and session.dirty_frames),
            })
        return result

    def session(self, requested: str) -> TrajectorySession:
        path = _safe_relative(self.data_root, requested)
        if path.suffix != ".npz" or not path.is_file():
            raise ValueError("unknown NPZ file")
        with self.lock:
            session = self.sessions.get(path)
            if session is None:
                session = TrajectorySession(path, self.model)
                self.sessions[path] = session
            elif not session.dirty_frames and path.stat().st_mtime_ns != session.source_mtime_ns:
                session.reload()
            return session


class MuJoCoModel:
    """Compiled G1 model used as the authoritative FK and mesh manifest."""

    def __init__(self, xml_path: Path) -> None:
        self.xml_path = xml_path.resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.lock = threading.RLock()
        self.ee_body_ids = [
            self._body_id("left_wrist_yaw_link"),
            self._body_id("right_wrist_yaw_link"),
            self._body_id("left_ankle_roll_link"),
            self._body_id("right_ankle_roll_link"),
            self._body_id("torso_link"),
        ]
        self.joint_qpos_adrs = []
        self.joints = []
        for index, name in enumerate(JOINT_NAMES):
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise ValueError(f"G1 model is missing joint {name}")
            self.joint_qpos_adrs.append(int(self.model.jnt_qposadr[joint_id]))
            self.joints.append({
                "name": name,
                "index": index,
                "body_id": int(self.model.jnt_bodyid[joint_id]),
                "axis": _json_value(self.model.jnt_axis[joint_id]),
                "range": JOINT_RANGES[index],
            })
        self._contact_model = None
        self._contact_data = None
        self._contact_object_qpos_adr = None
        self._contact_box_geom_id = None
        self._contact_sphere_geom_id = None
        self._contact_object_geom_id = None
        self._contact_hand_geom_ids = None
        self._contact_palm_geom_ids = None
        self._contact_hand_body_ids = None
        self._contact_hand_centers_local = None
        self._contact_palm_centers_local = None

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, name
        )
        if body_id < 0:
            raise ValueError(f"G1 model is missing body {name}")
        return int(body_id)

    def body_pose(
        self, base_pos: np.ndarray, base_quat: np.ndarray, joint_pos: np.ndarray
    ) -> np.ndarray:
        with self.lock:
            self.data.qpos[:] = 0.0
            self.data.qpos[0:3] = np.asarray(base_pos, dtype=np.float64)[:3]
            self.data.qpos[3:7] = np.asarray(base_quat, dtype=np.float64)[:4]
            for index, address in enumerate(self.joint_qpos_adrs):
                self.data.qpos[address] = float(joint_pos[index])
            mujoco.mj_forward(self.model, self.data)
            poses = np.zeros((self.model.nbody, 7), dtype=np.float32)
            for body_id in range(self.model.nbody):
                quat = np.empty(4, dtype=np.float64)
                mujoco.mju_mat2Quat(quat, self.data.xmat[body_id])
                poses[body_id, :3] = self.data.xpos[body_id]
                poses[body_id, 3:] = quat
            return poses

    def palm_pose(
        self, base_pos: np.ndarray, base_quat: np.ndarray, joint_pos: np.ndarray
    ) -> np.ndarray:
        """Return world poses for the centers and frames of both palm meshes."""
        with self.lock:
            self._ensure_contact_model()
            model = self._contact_model
            data = self._contact_data
            data.qpos[:] = 0.0
            data.qpos[0:3] = np.asarray(base_pos, dtype=np.float64)[:3]
            data.qpos[3:7] = _normalize_quat(base_quat)
            for index, address in enumerate(self._contact_joint_qpos_adrs):
                data.qpos[address] = float(joint_pos[index])
            mujoco.mj_forward(model, data)
            poses = np.zeros((2, 7), dtype=np.float32)
            for index, (body_id, local_center, geom_id) in enumerate(zip(
                self._contact_hand_body_ids,
                self._contact_palm_centers_local,
                self._contact_palm_geom_ids,
            )):
                rotation = data.xmat[body_id].reshape(3, 3)
                poses[index, :3] = data.xpos[body_id] + rotation @ local_center
                quat = np.empty(4, dtype=np.float64)
                mujoco.mju_mat2Quat(quat, data.geom_xmat[geom_id])
                poses[index, 3:] = _normalize_quat(quat)
            return poses

    def contact_metrics(
        self,
        base_pos: np.ndarray,
        base_quat: np.ndarray,
        joint_pos: np.ndarray,
        object_pos: np.ndarray,
        object_quat: np.ndarray,
        object_size: float,
        object_type: str,
    ) -> dict[str, np.ndarray]:
        with self.lock:
            self._ensure_contact_model()
            model = self._contact_model
            data = self._contact_data
            data.qpos[:] = 0.0
            data.qpos[:3] = np.asarray(base_pos, dtype=np.float64)[:3]
            data.qpos[3:7] = _normalize_quat(base_quat)
            for index, address in enumerate(self._contact_joint_qpos_adrs):
                data.qpos[address] = float(joint_pos[index])
            data.qpos[self._contact_object_qpos_adr:self._contact_object_qpos_adr + 3] = (
                np.asarray(object_pos, dtype=np.float64)[:3]
            )
            data.qpos[self._contact_object_qpos_adr + 3:self._contact_object_qpos_adr + 7] = (
                _normalize_quat(object_quat)
            )
            half_size = max(float(object_size) * 0.5, 1e-4)
            model.geom_size[self._contact_box_geom_id][:3] = half_size
            model.geom_size[self._contact_sphere_geom_id][0] = half_size
            self._contact_object_geom_id = (
                self._contact_sphere_geom_id
                if object_type == OBJECT_SPHERE
                else self._contact_box_geom_id
            )
            mujoco.mj_forward(model, data)

            distances = []
            for geom_id in self._contact_hand_geom_ids:
                fromto = np.zeros(6, dtype=np.float64)
                distances.append(
                    mujoco.mj_geomDistance(
                        model,
                        data,
                        geom_id,
                        self._contact_object_geom_id,
                        1.0,
                        fromto,
                    )
                )
            palm_distances = []
            for geom_id in self._contact_palm_geom_ids:
                fromto = np.zeros(6, dtype=np.float64)
                palm_distances.append(
                    mujoco.mj_geomDistance(
                        model,
                        data,
                        geom_id,
                        self._contact_object_geom_id,
                        1.0,
                        fromto,
                    )
                )
            hand_centers = []
            inward_normals = []
            for body_id, local_center, side in zip(
                self._contact_hand_body_ids,
                self._contact_hand_centers_local,
                (-1.0, 1.0),
            ):
                rotation = data.xmat[body_id].reshape(3, 3)
                hand_centers.append(data.xpos[body_id] + rotation @ local_center)
                inward_normals.append(rotation[:, 1] * side)
            palm_centers = []
            for body_id, local_center in zip(
                self._contact_hand_body_ids,
                self._contact_palm_centers_local,
            ):
                rotation = data.xmat[body_id].reshape(3, 3)
                palm_centers.append(data.xpos[body_id] + rotation @ local_center)
            return {
                "distance": np.asarray(distances, dtype=np.float64),
                "palm_distance": np.asarray(palm_distances, dtype=np.float64),
                "hand_center": np.asarray(hand_centers, dtype=np.float64),
                "palm_center": np.asarray(palm_centers, dtype=np.float64),
                "inward_normal": np.asarray(inward_normals, dtype=np.float64),
            }

    def _ensure_contact_model(self) -> None:
        if self._contact_model is not None:
            return
        xml = self.xml_path.read_text(encoding="utf-8")
        xml = xml.replace(
            'meshdir="meshes"',
            f'meshdir="{MODEL_MESH_ROOT.as_posix()}"',
        )
        xml = xml.replace(
            '<mesh name="left_rubber_hand" file="left_rubber_hand.STL" />',
            '<mesh name="left_rubber_hand" file="left_rubber_hand.STL" />'
            '<mesh name="left_hand_palm_link" file="left_hand_palm_link.STL" />',
        )
        xml = xml.replace(
            '<mesh name="right_rubber_hand" file="right_rubber_hand.STL" />',
            '<mesh name="right_rubber_hand" file="right_rubber_hand.STL" />'
            '<mesh name="right_hand_palm_link" file="right_hand_palm_link.STL" />',
        )
        object_body = (
            '<body name="contact_object">'
            '<freejoint name="contact_object_free"/>'
            '<geom name="contact_object_box" type="box" '
            'size="0.5 0.5 0.5" contype="0" conaffinity="0"/>'
            '<geom name="contact_object_sphere" type="sphere" '
            'size="0.5" contype="0" conaffinity="0"/>'
            '</body>'
        )
        # Use the standalone G1 palm meshes for the primary contact target.
        # The unified rubber-hand meshes remain active below for penetration
        # and clearance checks around the fingers and hand edge.
        left_palm_geom = (
            '<geom name="left_palm_proxy" pos="0.0415 0.003 0" '
            'type="mesh" mesh="left_hand_palm_link" '
            'contype="0" conaffinity="0"/>'
        )
        right_palm_geom = (
            '<geom name="right_palm_proxy" pos="0.0415 -0.003 0" '
            'type="mesh" mesh="right_hand_palm_link" '
            'contype="0" conaffinity="0"/>'
        )
        xml = xml.replace(
            '<body name="left_rubber_hand" >',
            f'<body name="left_rubber_hand" >{left_palm_geom}',
        )
        xml = xml.replace(
            '<body name="right_rubber_hand" >',
            f'<body name="right_rubber_hand" >{right_palm_geom}',
        )
        xml = xml.replace("</worldbody>", f"{object_body}</worldbody>")
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        self._contact_model = model
        self._contact_data = data
        self._contact_joint_qpos_adrs = [
            int(model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ])
            for name in JOINT_NAMES
        ]
        self._contact_object_qpos_adr = int(model.jnt_qposadr[
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, "contact_object_free"
            )
        ])
        self._contact_box_geom_id = int(mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "contact_object_box"
        ))
        self._contact_sphere_geom_id = int(mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "contact_object_sphere"
        ))
        self._contact_object_geom_id = self._contact_box_geom_id
        hand_body_ids = [
            int(mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "left_rubber_hand"
            )),
            int(mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "right_rubber_hand"
            )),
        ]
        hand_geom_ids = []
        hand_centers_local = []
        for body_name, body_id in zip(
            ("left_rubber_hand", "right_rubber_hand"), hand_body_ids
        ):
            mesh_id = int(mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_MESH, body_name
            ))
            geom_id = next(
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) == body_id
                and int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
                and int(model.geom_dataid[geom_id]) == mesh_id
            )
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            mesh_center = model.mesh_vert[start:start + count].mean(axis=0)
            local_center = model.geom_pos[geom_id] + _quat_rotate(
                model.geom_quat[geom_id], mesh_center
            )
            hand_geom_ids.append(geom_id)
            hand_centers_local.append(local_center)
        self._contact_hand_body_ids = hand_body_ids
        self._contact_hand_geom_ids = hand_geom_ids
        self._contact_palm_geom_ids = [
            int(mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "left_palm_proxy",
            )),
            int(mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "right_palm_proxy",
            )),
        ]
        self._contact_hand_centers_local = hand_centers_local
        palm_centers_local = []
        for geom_id in self._contact_palm_geom_ids:
            mesh_id = int(model.geom_dataid[geom_id])
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            mesh_center = model.mesh_vert[start:start + count].mean(axis=0)
            palm_centers_local.append(
                model.geom_pos[geom_id] + _quat_rotate(
                    model.geom_quat[geom_id],
                    mesh_center,
                )
            )
        self._contact_palm_centers_local = palm_centers_local

    def manifest(self) -> dict[str, object]:
        geoms = []
        seen_meshes: set[tuple[int, int]] = set()
        # Prefer group 1 visual geoms. The XML also has group 0 collision
        # duplicates, except for the rubber hand meshes which only have group 0.
        mesh_geom_ids = [
            index
            for index in range(self.model.ngeom)
            if self.model.geom_type[index] == mujoco.mjtGeom.mjGEOM_MESH
        ]
        mesh_geom_ids.sort(key=lambda index: int(self.model.geom_group[index]) != 1)
        for geom_id in mesh_geom_ids:
            body_id = int(self.model.geom_bodyid[geom_id])
            mesh_id = int(self.model.geom_dataid[geom_id])
            key = (body_id, mesh_id)
            if key in seen_meshes:
                continue
            seen_meshes.add(key)
            mesh_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_MESH, mesh_id
            )
            if not mesh_name:
                continue
            geoms.append({
                "body_id": body_id,
                "body_name": mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                ),
                "mesh": mesh_name,
                "file": f"{mesh_name}.STL",
                "pos": _json_value(self.model.geom_pos[geom_id]),
                "quat": _json_value(self.model.geom_quat[geom_id]),
                # MuJoCo compiles raw STL vertices into body-local mesh
                # coordinates using this transform before applying geom_pos/
                # geom_quat. The browser reproduces that step for exact mesh
                # placement.
                "mesh_pos": _json_value(self.model.mesh_pos[mesh_id]),
                "mesh_quat": _json_value(self.model.mesh_quat[mesh_id]),
                "mesh_scale": _json_value(self.model.mesh_scale[mesh_id]),
                "rgba": _json_value(self.model.geom_rgba[geom_id]),
            })
        bodies = [
            {
                "id": body_id,
                "name": mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                ),
                "parent_id": int(self.model.body_parentid[body_id]),
            }
            for body_id in range(self.model.nbody)
        ]
        return {
            "xml": str(self.xml_path.relative_to(HERE.parents[1])),
            "bodies": bodies,
            "joints": self.joints,
            "geoms": geoms,
        }


CONTACT_ARM_JOINT_NAMES = {
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
}
CONTACT_ARM_JOINT_INDICES = np.asarray([
    index for index, name in enumerate(JOINT_NAMES)
    if name in CONTACT_ARM_JOINT_NAMES
], dtype=np.int64)


def _contact_loss(
    model: MuJoCoModel,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    joint_pos: np.ndarray,
    object_pos: np.ndarray,
    object_quat: np.ndarray,
    object_size: float,
    object_type: str,
    reference_joint_pos: np.ndarray,
    previous_delta: np.ndarray | None,
    reference_palm_center: np.ndarray | None = None,
) -> float:
    metrics = model.contact_metrics(
        base_pos,
        base_quat,
        joint_pos,
        object_pos,
        object_quat,
        object_size,
        object_type,
    )
    distances = metrics["distance"]
    palm_distances = metrics["palm_distance"]
    loss = 0.0
    for distance in distances:
        distance_value = float(distance)
        penetration_error = max(0.0, CONTACT_TARGET_GAP - distance_value)
        excess_clearance = max(0.0, distance_value - (CONTACT_TARGET_GAP + 0.008))
        loss += CONTACT_HAND_GAP_WEIGHT * penetration_error ** 2
        loss += CONTACT_HAND_CLEARANCE_WEIGHT * excess_clearance ** 2
    for distance in palm_distances:
        distance_value = float(distance)
        palm_error = distance_value - CONTACT_PALM_TARGET_GAP
        loss += 24000.0 * palm_error ** 2
        if distance_value < 0.0:
            loss += 24000.0 * distance_value ** 2

    palm_centers = np.asarray(metrics["palm_center"], dtype=np.float64)
    if reference_palm_center is None:
        palm_center_target = (
            palm_centers.mean(axis=0)
            if palm_centers.size
            else np.zeros(3, dtype=np.float64)
        )
    else:
        palm_center_target = np.asarray(
            reference_palm_center,
            dtype=np.float64,
        )
    for palm_center, inward_normal in zip(
        palm_centers,
        metrics["inward_normal"],
    ):
        inward_normal = np.asarray(inward_normal, dtype=np.float64)
        normal_norm = float(np.linalg.norm(inward_normal))
        if normal_norm < 1e-8:
            continue
        inward_normal /= normal_norm
        extent = _object_extent_along_normal(
            object_type,
            object_size,
            object_quat,
            inward_normal,
        )
        palm_projection = float(np.dot(
            np.asarray(object_pos, dtype=np.float64) - palm_center,
            inward_normal,
        ))
        projection_error = palm_projection - (
            extent + CONTACT_PALM_TARGET_GAP
        )
        loss += CONTACT_PALM_PROJECTION_WEIGHT * float(
            (projection_error / CONTACT_PALM_PROJECTION_SCALE) ** 2
        )
        if palm_projection < 0.0:
            loss += 40.0 * palm_projection ** 2
        # A nearest-point objective alone can be satisfied by the palm edge
        # or a finger. Keep the object centered in the common palm plane so
        # the broad palm patch, rather than a side edge, is the contact area.
        tangent_error = (
            np.asarray(object_pos, dtype=np.float64)
            - palm_center_target
        )
        tangent_error -= float(np.dot(tangent_error, inward_normal)) * inward_normal
        loss += CONTACT_PALM_CENTER_WEIGHT * float(
            np.sum((tangent_error / CONTACT_PALM_CENTER_SCALE) ** 2)
        )

    if distances.size >= 2:
        loss += 16.0 * float(distances[0] - distances[1]) ** 2
    if palm_distances.size >= 2:
        loss += 24.0 * float(palm_distances[0] - palm_distances[1]) ** 2

    joint_delta = np.asarray(joint_pos, dtype=np.float64) - reference_joint_pos
    loss += CONTACT_JOINT_REGULARIZATION_WEIGHT * float(
        np.sum((joint_delta / 0.18) ** 2)
    )
    if previous_delta is not None:
        current_delta = joint_delta[CONTACT_ARM_JOINT_INDICES]
        loss += 0.35 * float(
            np.sum(((current_delta - previous_delta) / 0.07) ** 2)
        )
    return float(loss)


def _optimize_contact_frame(
    model: MuJoCoModel,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    reference_joint_pos: np.ndarray,
    reference_object_pos: np.ndarray,
    object_quat: np.ndarray,
    object_size: float,
    object_type: str,
    previous_delta: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    arm_indices = CONTACT_ARM_JOINT_INDICES
    reference_metrics = model.contact_metrics(
        base_pos,
        base_quat,
        reference_joint_pos,
        reference_object_pos,
        object_quat,
        object_size,
        object_type,
    )
    reference_palm_centers = np.asarray(
        reference_metrics["palm_center"],
        dtype=np.float64,
    )
    reference_palm_center = (
        reference_palm_centers.mean(axis=0)
        if reference_palm_centers.size
        else None
    )
    # The object pose is a fixed target. Only the arm joints are optimized.
    # This keeps the user-authored object trajectory bit-for-bit unchanged.
    object_pos = np.asarray(reference_object_pos, dtype=np.float64)
    initial = np.asarray(reference_joint_pos[arm_indices], dtype=np.float64)
    lower = np.asarray(
        [JOINT_RANGES[index][0] for index in arm_indices],
        dtype=np.float64,
    )
    upper = np.asarray(
        [JOINT_RANGES[index][1] for index in arm_indices],
        dtype=np.float64,
    )
    values = np.clip(initial, lower, upper)
    steps = np.full(len(arm_indices), 0.035, dtype=np.float64)

    def evaluate(candidate: np.ndarray) -> float:
        joints = np.asarray(reference_joint_pos, dtype=np.float64).copy()
        joints[arm_indices] = candidate
        return _contact_loss(
            model,
            base_pos,
            base_quat,
            joints,
            object_pos,
            object_quat,
            object_size,
            object_type,
            np.asarray(reference_joint_pos, dtype=np.float64),
            previous_delta,
            reference_palm_center,
        )

    best_loss = evaluate(values)
    for _ in range(CONTACT_OPTIMIZATION_ITERATIONS):
        improved = False
        for index in range(values.size):
            best_candidate = values
            for direction in (-1.0, 1.0):
                candidate = values.copy()
                candidate[index] = np.clip(
                    candidate[index] + direction * steps[index],
                    lower[index],
                    upper[index],
                )
                candidate_loss = evaluate(candidate)
                if candidate_loss + 1e-9 < best_loss:
                    best_candidate = candidate
                    best_loss = candidate_loss
                    improved = True
            values = best_candidate
        steps *= 0.68
        if not improved and float(np.max(steps)) < 0.001:
            break

    optimized_joints = np.asarray(reference_joint_pos, dtype=np.float64).copy()
    optimized_joints[arm_indices] = values
    return optimized_joints, best_loss


def _smooth_contact_corrections(joint_deltas: np.ndarray) -> np.ndarray:
    smoothed_joints = np.asarray(joint_deltas, dtype=np.float64).copy()
    if smoothed_joints.shape[0] < 3:
        return smoothed_joints
    for _ in range(CONTACT_CORRECTION_SMOOTHING_PASSES):
        previous = smoothed_joints.copy()
        smoothed_joints[1:-1] = (
            0.25 * previous[:-2]
            + 0.50 * previous[1:-1]
            + 0.25 * previous[2:]
        )
        smoothed_joints[0] = (
            (1.0 - CONTACT_CORRECTION_ENDPOINT_BLEND) * previous[0]
            + CONTACT_CORRECTION_ENDPOINT_BLEND * previous[1]
        )
        smoothed_joints[-1] = (
            (1.0 - CONTACT_CORRECTION_ENDPOINT_BLEND) * previous[-1]
            + CONTACT_CORRECTION_ENDPOINT_BLEND * previous[-2]
        )
    return smoothed_joints


def _smoothstep(progress: float) -> float:
    """Cubic blend with zero slope at both ends of a stage transition."""
    value = min(1.0, max(0.0, float(progress)))
    return value * value * (3.0 - 2.0 * value)


def _recompute_kinematic_fields(
    model: MuJoCoModel,
    arrays: dict[str, np.ndarray],
    frame_indices: range,
    fps: float,
) -> None:
    body_pos = arrays.get("body_pos_w")
    body_quat = arrays.get("body_quat_w")
    ee_pos = arrays.get("ee_pos_w")
    ee_quat = arrays.get("ee_quat_w")
    for frame in frame_indices:
        poses = model.body_pose(
            arrays["base_pos_w"][frame],
            arrays["base_quat_w"][frame],
            arrays["joint_pos"][frame],
        )
        if body_pos is not None and body_pos.shape[1:] == poses[:, :3].shape:
            body_pos[frame] = poses[:, :3]
        if body_quat is not None and body_quat.shape[1:] == poses[:, 3:].shape:
            body_quat[frame] = poses[:, 3:]
        if ee_pos is not None and ee_pos.shape[1:] == (len(model.ee_body_ids), 3):
            ee_pos[frame] = poses[model.ee_body_ids, :3]
        if ee_quat is not None and ee_quat.shape[1:] == (len(model.ee_body_ids), 4):
            ee_quat[frame] = poses[model.ee_body_ids, 3:]

    dt = 1.0 / max(float(fps), 1e-6)
    if "joint_vel" in arrays:
        arrays["joint_vel"][:] = np.gradient(
            arrays["joint_pos"], dt, axis=0
        ).astype(arrays["joint_vel"].dtype, copy=False)
    if body_pos is not None and "body_lin_vel_w" in arrays:
        arrays["body_lin_vel_w"][:] = np.gradient(
            body_pos, dt, axis=0
        ).astype(arrays["body_lin_vel_w"].dtype, copy=False)
    if body_quat is not None and "body_ang_vel_w" in arrays:
        body_ang_vel = np.stack([
            _quat_ang_vel(body_quat[:, body_id], fps)
            for body_id in range(body_quat.shape[1])
        ], axis=1)
        arrays["body_ang_vel_w"][:] = body_ang_vel.astype(
            arrays["body_ang_vel_w"].dtype,
            copy=False,
        )
class Handler(BaseHTTPRequestHandler):
    app: App

    def log_message(self, format: str, *args: object) -> None:
        print(f"[hoi-editor] {self.address_string()} - {format % args}", flush=True)

    def send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def error(self, message: str, status: int = 400) -> None:
        self.send_json({"error": message}, status=status)

    def read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 4_000_000:
            raise ValueError("request body is too large")
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8") or "{}")
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/folders":
                self.send_json({"folders": self.app.folders()})
                return
            if parsed.path == "/api/files":
                params = parse_qs(parsed.query)
                folder = params.get("folder", [None])[0]
                self.send_json({"files": self.app.files(folder)})
                return
            if parsed.path == "/api/model":
                self.send_json(self.app.model.manifest())
                return
            if parsed.path == "/api/frame":
                params = parse_qs(parsed.query)
                session = self.app.session(params.get("file", [""])[0])
                frame = int(params.get("frame", ["0"])[0])
                self.send_json(session.frame(frame))
                return
            if parsed.path == "/api/trajectory":
                params = parse_qs(parsed.query)
                session = self.app.session(params.get("file", [""])[0])
                limit = int(params.get("limit", ["900"])[0])
                self.send_json(session.trajectory(limit))
                return
            if parsed.path == "/api/grounding":
                params = parse_qs(parsed.query)
                session = self.app.session(params.get("file", [""])[0])
                self.send_json(session.grounding_profile())
                return
            self.serve_static(parsed.path)
        except (ValueError, KeyError, IndexError, OSError) as exc:
            self.error(str(exc), 400)

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        try:
            self.serve_static(parsed.path, head_only=True)
        except (ValueError, KeyError, IndexError, OSError) as exc:
            self.error(str(exc), 400)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self.read_json()
            if parsed.path == "/api/folders":
                parent = self.app.folder(payload.get("parent", ""))
                name = _safe_folder_name(str(payload.get("name", "")))
                target = (parent / name).resolve()
                if self.app.data_root not in target.parents:
                    raise ValueError("new folder must stay inside the data directory")
                if target.exists():
                    raise ValueError("folder already exists")
                target.mkdir(parents=False, exist_ok=False)
                self.send_json({
                    "created": True,
                    "folder": target.relative_to(self.app.data_root).as_posix(),
                    "name": target.name,
                })
                return
            if parsed.path == "/api/edit":
                session = self.app.session(str(payload["file"]))
                frame = session.edit(int(payload["frame"]), payload.get("changes", {}))
                self.send_json(frame)
                return
            if parsed.path == "/api/reset":
                session = self.app.session(str(payload["file"]))
                session.reload()
                self.send_json(session.frame(int(payload.get("frame", 0))))
                return
            if parsed.path == "/api/save":
                session = self.app.session(str(payload["file"]))
                grounding_applied = bool(payload.get("apply_grounding_correction", False))
                output_folder = payload.get("output_folder")
                output_dir = (
                    self.app.folder(str(output_folder))
                    if output_folder is not None
                    else None
                )
                destination = session.save_copy(
                    payload.get("filename"),
                    grounding_applied,
                    output_dir=output_dir,
                )
                self.send_json({
                    "saved": True,
                    "file": destination.relative_to(self.app.data_root).as_posix(),
                    "path": str(destination),
                    "output_folder": (
                        output_dir.relative_to(self.app.data_root).as_posix()
                        if output_dir is not None
                        else destination.parent.relative_to(self.app.data_root).as_posix()
                    ),
                    "grounding_correction_applied": grounding_applied,
                })
                return
            if parsed.path == "/api/crop":
                session = self.app.session(str(payload["file"]))
                grounding_applied = bool(payload.get("apply_grounding_correction", False))
                output_folder = payload.get("output_folder")
                output_dir = (
                    self.app.folder(str(output_folder))
                    if output_folder is not None
                    else None
                )
                destination, meta = session.crop_copy(
                    int(payload["start_frame"]),
                    int(payload["end_frame"]),
                    payload.get("filename"),
                    grounding_applied,
                    output_dir=output_dir,
                )
                self.send_json({
                    "saved": True,
                    "file": destination.relative_to(self.app.data_root).as_posix(),
                    "path": str(destination),
                    "output_folder": (
                        output_dir.relative_to(self.app.data_root).as_posix()
                        if output_dir is not None
                        else destination.parent.relative_to(self.app.data_root).as_posix()
                    ),
                    "grounding_correction_applied": grounding_applied,
                    **meta,
                })
                return
            if parsed.path == "/api/optimize_contact":
                session = self.app.session(str(payload["file"]))
                preview, meta = session.synthesize_copy(
                    int(payload["contact_start"]),
                    int(payload["contact_end"]),
                    payload["object_pos_w"],
                    payload["object_quat_w"],
                    payload.get("object_size"),
                    payload.get("object_type", OBJECT_BOX),
                    None,
                    bool(payload.get("yaw_only", False)),
                    bool(payload.get("support_surfaces", True)),
                    str(payload.get("post_contact_mode", POST_CONTACT_HOLD)),
                    payload.get("release_velocity_w"),
                    str(payload.get("pre_contact_mode", PRE_CONTACT_HOLD)),
                    payload.get("pre_contact_velocity_w"),
                    True,
                    True,
                    bool(payload.get("apply_grounding_correction", False)),
                    anchor_mode=payload.get("anchor_mode", ANCHOR_PALM_CENTER),
                )
                if not isinstance(preview, dict):
                    raise ValueError("contact optimization preview failed")
                start = int(meta["contact_start"])
                end = int(meta["contact_end"])
                transition = CONTACT_OPTIMIZATION_TRANSITION_FRAMES
                preview_start = max(0, start - transition)
                preview_end = min(
                    session.frame_count - 1,
                    end + transition,
                )
                frames = np.arange(
                    preview_start,
                    preview_end + 1,
                    dtype=np.int64,
                )
                # Keep the preview in the same coordinate convention as
                # /api/frame: grounding is displayed by the frontend's
                # contentRoot transform. The synthesis output may already
                # contain a baked grounding offset, so using its base_pos_w
                # here would apply that offset twice in the viewer.
                body_poses = np.stack([
                    session.model.body_pose(
                        session.arrays["base_pos_w"][frame],
                        session.arrays["base_quat_w"][frame],
                        preview["joint_pos"][frame],
                    )
                    for frame in frames
                ], axis=0)
                palm_poses = np.stack([
                    session.model.palm_pose(
                        session.arrays["base_pos_w"][frame],
                        session.arrays["base_quat_w"][frame],
                        preview["joint_pos"][frame],
                    )
                    for frame in frames
                ], axis=0)
                ee_ids = session.model.ee_body_ids
                self.send_json({
                    **meta,
                    "frames": frames.tolist(),
                    "joint_pos": _json_value(preview["joint_pos"][frames]),
                    "body_pose_w": _json_value(body_poses),
                    "palm_pos_w": _json_value(palm_poses[:, :, :3]),
                    "palm_quat_w": _json_value(palm_poses[:, :, 3:]),
                    "ee_pos_w": _json_value(body_poses[:, ee_ids, :3]),
                    "ee_quat_w": _json_value(body_poses[:, ee_ids, 3:]),
                    "object_pos_w": _json_value(preview["object_pos_w"][frames]),
                    "object_quat_w": _json_value(preview["object_quat_w"][frames]),
                })
                return
            if parsed.path == "/api/synthesize":
                session = self.app.session(str(payload["file"]))
                destination, meta = session.synthesize_copy(
                    int(payload["contact_start"]),
                    int(payload["contact_end"]),
                    payload["object_pos_w"],
                    payload["object_quat_w"],
                    payload.get("object_size"),
                    payload.get("object_type", OBJECT_BOX),
                    payload.get("filename"),
                    bool(payload.get("yaw_only", False)),
                    bool(payload.get("support_surfaces", True)),
                    str(payload.get("post_contact_mode", POST_CONTACT_HOLD)),
                    payload.get("release_velocity_w"),
                    str(payload.get("pre_contact_mode", PRE_CONTACT_HOLD)),
                    payload.get("pre_contact_velocity_w"),
                    bool(payload.get("optimize_contact_stage", False)),
                    apply_grounding_correction=bool(
                        payload.get("apply_grounding_correction", False)
                    ),
                    output_dir=(
                        self.app.folder(str(payload["output_folder"]))
                        if "output_folder" in payload
                        else None
                    ),
                    anchor_mode=payload.get("anchor_mode", ANCHOR_PALM_CENTER),
                )
                self.send_json({
                    "saved": True,
                    "file": destination.relative_to(self.app.data_root).as_posix(),
                    "path": str(destination),
                    **meta,
                })
                return
            self.error("unknown API endpoint", 404)
        except (ValueError, KeyError, IndexError, OSError, json.JSONDecodeError) as exc:
            self.error(str(exc), 400)

    def serve_static(self, requested_path: str, head_only: bool = False) -> None:
        path = unquote(requested_path)
        if path.startswith("/g1-meshes/"):
            requested_mesh = Path(path.removeprefix("/g1-meshes/"))
            target = (MODEL_MESH_ROOT / requested_mesh).resolve()
            if (
                target.suffix.lower() != ".stl"
                or MODEL_MESH_ROOT.resolve() not in target.parents
                or not target.is_file()
            ):
                self.error("mesh not found", 404)
                return
            content = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "public, max-age=86400, immutable")
            self.end_headers()
            if not head_only:
                self.wfile.write(content)
            return
        if path in ("", "/"):
            path = "/index.html"
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", path) or ".." in Path(path).parts:
            self.error("invalid static path", 404)
            return
        target = (HERE / path.lstrip("/")).resolve()
        if HERE not in target.parents or not target.is_file():
            self.error("not found", 404)
            return
        content = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        if not head_only:
            self.wfile.write(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local HOI trajectory editor.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"data directory does not exist: {data_root}")
    if not MODEL_XML.is_file():
        raise SystemExit(f"G1 MuJoCo model does not exist: {MODEL_XML}")
    model = MuJoCoModel(MODEL_XML)
    app = App(data_root, model)
    handler = type("HOIEditorHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"[hoi-editor] data root: {data_root}", flush=True)
    print(f"[hoi-editor] open http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[hoi-editor] stopped", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
