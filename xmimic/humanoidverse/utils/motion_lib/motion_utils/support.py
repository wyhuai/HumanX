from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def _is_torch_tensor(value) -> bool:
    return torch.is_tensor(value)


def _to_numpy_float32(value) -> np.ndarray:
    if _is_torch_tensor(value):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _window_is_static(window: np.ndarray, static_threshold: float) -> bool:
    if window.shape[0] == 0:
        return False
    return bool(np.max(np.ptp(window, axis=0)) <= float(static_threshold))


def _candidate_static_windows(
    obj_pos: np.ndarray,
    contact: np.ndarray,
    window_size: int,
    static_threshold: float,
) -> Iterable[np.ndarray]:
    transitions = np.nonzero(contact[1:] != contact[:-1])[0] + 1
    for transition in transitions.tolist():
        prev_window = obj_pos[max(0, transition - window_size):transition]
        next_window = obj_pos[transition:min(obj_pos.shape[0], transition + window_size)]
        if prev_window.shape[0] == window_size and _window_is_static(prev_window, static_threshold):
            yield prev_window
        if next_window.shape[0] == window_size and _window_is_static(next_window, static_threshold):
            yield next_window


def derive_support_top_pose_from_motion(
    obj_pos,
    contact,
    window_size: int = 5,
    static_threshold: float = 5e-3,
    z_offset: float = 0.1,
) -> np.ndarray:
    obj_pos_np = _to_numpy_float32(obj_pos).reshape(-1, 3)
    contact_np = _to_numpy_float32(contact).reshape(-1)
    if obj_pos_np.shape[0] != contact_np.shape[0]:
        raise ValueError(
            f"obj_pos/contact length mismatch: {obj_pos_np.shape[0]} vs {contact_np.shape[0]}"
        )
    if int(window_size) <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    static_windows = list(
        _candidate_static_windows(
            obj_pos=obj_pos_np,
            contact=contact_np,
            window_size=int(window_size),
            static_threshold=float(static_threshold),
        )
    )
    if not static_windows:
        raise ValueError("need_support=True but no static obj_pos window was found around a contact transition")

    stacked = np.concatenate(static_windows, axis=0)
    support_top = stacked.mean(axis=0).astype(np.float32, copy=False)
    support_top[2] -= float(z_offset)
    return support_top


def support_top_to_center(support_top_pos, thickness: float):
    if _is_torch_tensor(support_top_pos):
        center = support_top_pos.clone()
        center[..., 2] = center[..., 2] - float(thickness) * 0.5
        return center

    center = np.asarray(support_top_pos, dtype=np.float32).copy()
    center[..., 2] = center[..., 2] - float(thickness) * 0.5
    return center


def parked_support_center(env_origins, dz: float = 100.0):
    if _is_torch_tensor(env_origins):
        parked = env_origins.clone()
        parked[..., 2] = parked[..., 2] - float(dz)
        return parked

    parked = np.asarray(env_origins, dtype=np.float32).copy()
    parked[..., 2] = parked[..., 2] - float(dz)
    return parked
