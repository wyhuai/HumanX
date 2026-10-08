#!/usr/bin/env python3
"""Serve a local 3D viewer for human_smpl_joints_local motion files."""

from __future__ import annotations

import argparse
import json
import mimetypes
import posixpath
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import joblib
import numpy as np


DEFAULT_MOTION_ROOT = Path("data/motions/intertracker_1")
STATIC_DIR = Path(__file__).resolve().parent / "static"

JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_thumb",
    "right_thumb",
]


def discover_motions(root: Path) -> List[Dict[str, str]]:
    """Return all pickle motion files below root as stable POSIX relative paths."""
    root = root.resolve()
    motions = []
    for path in sorted(root.rglob("*.pkl")):
        rel = path.relative_to(root).as_posix()
        motions.append({"path": rel, "name": rel})
    return motions


def _resolve_motion_path(root: Path, rel_path: str) -> Path:
    root = root.resolve()
    path = (root / rel_path).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"Motion path escapes root: {rel_path}")
    if path.suffix != ".pkl":
        raise ValueError(f"Motion path must be a .pkl file: {rel_path}")
    return path


def _select_motion(data: Any, preferred_key: str | None = None) -> Tuple[str | None, Dict[str, Any]]:
    if not isinstance(data, dict):
        raise ValueError("Motion pickle must contain a dictionary")
    if "human_smpl_joints_local" in data or "smpl_joints_local" in data:
        return None, data
    if preferred_key and preferred_key in data and isinstance(data[preferred_key], dict):
        return preferred_key, data[preferred_key]
    for key, value in data.items():
        if isinstance(value, dict) and (
            "human_smpl_joints_local" in value or "smpl_joints_local" in value
        ):
            return str(key), value
    raise ValueError("No human_smpl_joints_local or smpl_joints_local entry found")


def _as_float_list(array: Any) -> List[Any]:
    return np.asarray(array, dtype=np.float32).tolist()


def load_motion_payload(root: Path, rel_path: str, motion_key: str | None = None) -> Dict[str, Any]:
    """Load one motion pickle and shape it for the browser viewer."""
    path = _resolve_motion_path(root, rel_path)
    data = joblib.load(path)
    selected_key, motion = _select_motion(data, motion_key)
    joints_key = "human_smpl_joints_local" if "human_smpl_joints_local" in motion else "smpl_joints_local"
    joints = np.asarray(motion[joints_key], dtype=np.float32)
    if joints.ndim != 3 or joints.shape[1:] != (24, 3):
        raise ValueError(f"{joints_key} must have shape (frames, 24, 3), got {joints.shape}")

    fps = float(motion.get("fps", 50))
    payload: Dict[str, Any] = {
        "path": rel_path,
        "motion_key": selected_key,
        "joints_key": joints_key,
        "joint_names": JOINT_NAMES,
        "fps": fps,
        "num_frames": int(joints.shape[0]),
        "num_joints": int(joints.shape[1]),
        "joints": _as_float_list(joints),
    }
    if "obj_pos" in motion:
        obj_pos = np.asarray(motion["obj_pos"], dtype=np.float32)
        if obj_pos.ndim == 2 and obj_pos.shape[1] == 3 and obj_pos.shape[0] == joints.shape[0]:
            payload["obj_pos"] = _as_float_list(obj_pos)
    if "human_global_orient_quat" in motion:
        global_orient = np.asarray(motion["human_global_orient_quat"], dtype=np.float32)
        if global_orient.ndim == 2 and global_orient.shape[1] == 4 and global_orient.shape[0] == joints.shape[0]:
            payload["human_global_orient_quat"] = _as_float_list(global_orient)
            payload["human_global_orient_quat_order"] = "wxyz"
            payload["human_global_orient_quat_xyzw"] = _as_float_list(global_orient[:, [1, 2, 3, 0]])
    return payload


class ViewerHandler(BaseHTTPRequestHandler):
    motion_root: Path = DEFAULT_MOTION_ROOT

    def log_message(self, fmt: str, *args: Any) -> None:
        print("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/motions":
                self._send_json({"root": self.motion_root.as_posix(), "motions": discover_motions(self.motion_root)})
                return
            if parsed.path == "/api/motion":
                query = parse_qs(parsed.query)
                rel_path = query.get("path", [""])[0]
                motion_key = query.get("key", [None])[0]
                if not rel_path:
                    self._send_error(HTTPStatus.BAD_REQUEST, "Missing path query parameter")
                    return
                self._send_json(load_motion_payload(self.motion_root, rel_path, motion_key=motion_key))
                return
            self._serve_static(parsed.path)
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _send_json(self, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, request_path: str) -> None:
        clean_path = posixpath.normpath(unquote(request_path)).lstrip("/")
        if clean_path in ("", "."):
            clean_path = "index.html"
        static_root = STATIC_DIR.resolve()
        file_path = (static_root / clean_path).resolve()
        if static_root != file_path and static_root not in file_path.parents:
            self._send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not file_path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        body = file_path.read_bytes()
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_server(host: str, port: int, motion_root: Path) -> ThreadingHTTPServer:
    handler = type("ConfiguredViewerHandler", (ViewerHandler,), {"motion_root": motion_root.resolve()})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_MOTION_ROOT, help="Motion directory to scan")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    args = parser.parse_args()

    server = build_server(args.host, args.port, args.root)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving human SMPL viewer at {url}")
    print(f"Motion root: {args.root.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
