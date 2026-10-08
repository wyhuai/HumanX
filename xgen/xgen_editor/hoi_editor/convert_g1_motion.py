#!/usr/bin/env python3
"""Convert g1_motion NPZ sequences into the XGen robot-only editor schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from server import JOINT_NAMES, MuJoCoModel, _quat_ang_vel


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = HERE.parents[1] / "data" / "g1_motion"
DEFAULT_OUTPUT_DIR = HERE.parents[1] / "data" / "robot_only"
MODEL_XML = (
    HERE.parent
    / "decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml"
)


def _as_array(
    payload: np.lib.npyio.NpzFile,
    key: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if key not in payload:
        raise ValueError(f"missing required field: {key}")
    value = np.asarray(payload[key])
    if value.shape != shape:
        raise ValueError(f"{key} must have shape {shape}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{key} contains non-finite values")
    return value


def _normalize_quaternions(quats: np.ndarray) -> np.ndarray:
    values = np.asarray(quats, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms < 1e-9):
        raise ValueError("root_quat_wxyz contains a zero quaternion")
    return (values / norms[:, None]).astype(np.float32)


def _reorder_by_joint_names(
    values: np.ndarray,
    source_names: list[str],
) -> np.ndarray:
    if len(source_names) != len(set(source_names)):
        raise ValueError("joint_names contains duplicates")
    editor_names = set(JOINT_NAMES)
    source_indices = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in JOINT_NAMES if name not in source_indices]
    extra = [name for name in source_names if name not in editor_names]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(
            "joint_names do not match the G1 editor model: "
            + ", ".join(details)
        )
    return np.stack(
        [values[:, source_indices[name]] for name in JOINT_NAMES],
        axis=1,
    ).astype(np.float32, copy=False)


def _gradient(values: np.ndarray, fps: float) -> np.ndarray:
    return np.gradient(
        np.asarray(values, dtype=np.float32),
        1.0 / max(float(fps), 1e-6),
        axis=0,
    ).astype(np.float32)


def convert(source: Path, destination: Path, model: MuJoCoModel) -> Path:
    with np.load(source, allow_pickle=True) as payload:
        if "joint_pos" not in payload:
            raise ValueError("missing required field: joint_pos")
        raw_joint_pos = np.asarray(payload["joint_pos"], dtype=np.float32)
        if raw_joint_pos.ndim != 2 or raw_joint_pos.shape[1] != len(JOINT_NAMES):
            raise ValueError(
                f"joint_pos must have shape (T, {len(JOINT_NAMES)}), "
                f"got {raw_joint_pos.shape}"
            )
        frame_count = raw_joint_pos.shape[0]
        base_pos = _as_array(
            payload,
            "root_pos",
            (frame_count, 3),
        ).astype(np.float32)
        base_quat = _normalize_quaternions(_as_array(
            payload,
            "root_quat_wxyz",
            (frame_count, 4),
        ))
        source_names = [
            str(name)
            for name in np.asarray(payload["joint_names"]).reshape(-1).tolist()
        ]
        if len(source_names) != len(JOINT_NAMES):
            raise ValueError(
                f"joint_names must contain {len(JOINT_NAMES)} entries, "
                f"got {len(source_names)}"
            )
        joint_pos = _reorder_by_joint_names(raw_joint_pos, source_names)

        fps = float(np.asarray(payload.get("fps", 50.0)).reshape(-1)[0])
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"fps must be positive and finite, got {fps}")
        if "joint_vel" in payload:
            raw_joint_vel = np.asarray(payload["joint_vel"], dtype=np.float32)
            if raw_joint_vel.shape != raw_joint_pos.shape:
                raise ValueError(
                    f"joint_vel must have shape {raw_joint_pos.shape}, "
                    f"got {raw_joint_vel.shape}"
                )
            joint_vel = _reorder_by_joint_names(raw_joint_vel, source_names)
        else:
            joint_vel = _gradient(joint_pos, fps)

    body_pose = np.stack([
        model.body_pose(base_pos[frame], base_quat[frame], joint_pos[frame])
        for frame in range(frame_count)
    ], axis=0)
    body_pos = body_pose[:, :, :3].astype(np.float32, copy=False)
    body_quat = body_pose[:, :, 3:].astype(np.float32, copy=False)
    body_lin_vel = _gradient(body_pos, fps)
    body_ang_vel = np.stack([
        _quat_ang_vel(body_quat[:, body_id], fps)
        for body_id in range(body_quat.shape[1])
    ], axis=1).astype(np.float32, copy=False)
    ee_pos = body_pos[:, model.ee_body_ids]
    ee_quat = body_quat[:, model.ee_body_ids]

    arrays = {
        "fps": np.array([fps], dtype=np.float32),
        "base_pos_w": base_pos,
        "base_quat_w": base_quat,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin_vel,
        "body_ang_vel_w": body_ang_vel,
        "ee_pos_w": ee_pos,
        "ee_quat_w": ee_quat,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=DEFAULT_SOURCE_DIR,
        help="one source NPZ or a directory of source NPZ files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for XGen robot-only NPZ files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing converted files",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if source.is_dir():
        sources = sorted(source.glob("*.npz"))
    elif source.suffix == ".npz":
        sources = [source]
    else:
        raise SystemExit(f"source must be an NPZ file or directory: {source}")
    if not sources:
        raise SystemExit(f"no NPZ files found under {source}")

    model = MuJoCoModel(MODEL_XML)
    output_dir = args.output_dir.resolve()
    converted = 0
    skipped = 0
    for source_path in sources:
        destination = output_dir / f"{source_path.stem}_robot_only.npz"
        if destination.exists() and not args.overwrite:
            print(f"skip {destination}")
            skipped += 1
            continue
        result = convert(source_path, destination, model)
        print(f"{source_path} -> {result}")
        converted += 1
    print(f"converted={converted} skipped={skipped} total={len(sources)}")


if __name__ == "__main__":
    main()
