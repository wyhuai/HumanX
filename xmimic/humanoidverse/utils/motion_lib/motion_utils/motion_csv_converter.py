import argparse
import warnings
from pathlib import Path
import xml.etree.ElementTree as ETree

import joblib
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from lxml.etree import XMLParser, parse
from io import BytesIO


def _load_robot_motion_cfg(robot_config):
    return OmegaConf.load(robot_config).robot.motion


REPO_ROOT = Path(__file__).resolve().parents[4]


def _build_robot_metadata(robot_config):
    cfg = _load_robot_motion_cfg(robot_config)
    asset_root = Path(cfg.asset.assetRoot)
    if not asset_root.is_absolute():
        asset_root = REPO_ROOT / asset_root
    mjcf_file = asset_root / cfg.asset.assetFileName

    parser = XMLParser(remove_blank_text=True)
    with open(mjcf_file, "rb") as f:
        tree = parse(BytesIO(f.read()), parser=parser)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Invalid MJCF file: missing worldbody in {mjcf_file}")

    root_body = worldbody.find("body")
    if root_body is None:
        raise ValueError(f"Invalid MJCF file: missing root body in {mjcf_file}")

    joints = sorted([j.attrib["name"] for j in root.find("worldbody").findall(".//joint")])
    motors = sorted([m.attrib["name"] for m in root.find("actuator").getchildren()])
    num_dof = len(motors)

    node_names = []

    def _add_xml_node(xml_node):
        node_names.append(xml_node.attrib.get("name"))
        for next_node in xml_node.findall("body"):
            _add_xml_node(next_node)

    _add_xml_node(root_body)

    dof_axis = []
    joint_nodes = root.find("worldbody").findall(".//joint")
    if "type" in joint_nodes[0].attrib and joint_nodes[0].attrib["type"] == "free":
        selected_joint_nodes = joint_nodes[1:]
    elif "type" not in joint_nodes[0].attrib:
        selected_joint_nodes = joint_nodes
    else:
        selected_joint_nodes = joint_nodes[6:]
    for joint in selected_joint_nodes:
        dof_axis.append([int(i) for i in joint.attrib["axis"].split(" ")])

    dof_axis = np.asarray(dof_axis, dtype=np.float32)
    if dof_axis.shape[0] != num_dof:
        raise ValueError(
            f"Joint axis count {dof_axis.shape[0]} does not match motor count {num_dof} in {mjcf_file}"
        )

    return {
        "num_dof": num_dof,
        "num_bodies": len(node_names),
        "dof_axis": dof_axis,
    }


def _default_export_csv_path(input_pkl, motion_key):
    input_pkl = Path(input_pkl)
    if input_pkl.stem == motion_key:
        return input_pkl.with_suffix(".csv")
    return input_pkl.with_name(f"{input_pkl.stem}__{motion_key}.csv")


def _default_import_pkl_path(input_pkl):
    input_pkl = Path(input_pkl)
    return input_pkl.with_name(f"{input_pkl.stem}_modified.pkl")


def _load_motion_data(input_pkl):
    data = joblib.load(input_pkl)
    if not isinstance(data, dict):
        raise ValueError(f"{input_pkl} must contain a dict of motion entries")
    return data


def _get_motion_entry(data, motion_key=None):
    if motion_key is None:
        keys = list(data.keys())
        if len(keys) == 1:
            return data[keys[0]]
        raise ValueError(
            "input pkl contains multiple motion entries; please specify --motion-key explicitly"
        )
    if motion_key not in data:
        warnings.warn(f"motion-key '{motion_key}' not found; skipping", UserWarning, stacklevel=2)
        return None
    return data[motion_key]


def _normalize_quaternions_xyzw(quats):
    quats = np.asarray(quats, dtype=np.float32).reshape(-1, 4)
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return quats / norms


def _joint_angles_from_motion(motion, robot_meta):
    if "dof" in motion:
        return np.asarray(motion["dof"], dtype=np.float32).reshape(len(motion["root_trans_offset"]), -1)

    pose_aa = np.asarray(motion["pose_aa"], dtype=np.float32)
    if pose_aa.shape[1] > robot_meta["num_bodies"]:
        pose_aa = pose_aa[:, : robot_meta["num_bodies"], :]
    joint_pose = pose_aa[:, 1 : 1 + robot_meta["num_dof"], :]
    dof_axis = robot_meta["dof_axis"]
    return np.sum(joint_pose * dof_axis[None, :, :], axis=-1).astype(np.float32)


def export_motion_to_csv(*, input_pkl, motion_key=None, output_csv=None, robot_config):
    data = _load_motion_data(input_pkl)
    motion = _get_motion_entry(data, motion_key)
    if motion is None:
        return False

    robot_meta = _build_robot_metadata(robot_config)
    base_xyz = np.asarray(motion["root_trans_offset"], dtype=np.float32).reshape(-1, 3)
    root_quat = np.asarray(motion["root_rot"], dtype=np.float32).reshape(-1, 4)
    joint_angles = _joint_angles_from_motion(motion, robot_meta)

    csv = np.concatenate([base_xyz, root_quat, joint_angles], axis=1)
    resolved_motion_key = motion_key if motion_key is not None else next(iter(data.keys()))
    output_csv = Path(output_csv) if output_csv is not None else _default_export_csv_path(input_pkl, resolved_motion_key)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_csv, csv, delimiter=",", fmt="%.9g")
    return True


def _pose_aa_from_csv(csv, robot_meta):
    csv = np.asarray(csv, dtype=np.float32)
    quats = _normalize_quaternions_xyzw(csv[:, 3:7])
    joints = csv[:, 7:]
    num_frames = csv.shape[0]

    pose_aa = np.zeros((num_frames, robot_meta["num_bodies"], 3), dtype=np.float32)
    pose_aa[:, 0, :] = R.from_quat(quats).as_rotvec().astype(np.float32)
    dof_axis = robot_meta["dof_axis"]
    pose_aa[:, 1 : 1 + robot_meta["num_dof"], :] = joints[:, :, None] * dof_axis[None, :, :]
    return pose_aa, quats


def import_csv_to_motion_file(
    *,
    input_pkl,
    motion_key=None,
    input_csv,
    robot_config,
    output_pkl=None,
    inplace=False,
):
    data = _load_motion_data(input_pkl)
    motion = _get_motion_entry(data, motion_key)
    if motion is None:
        return False

    csv = np.loadtxt(input_csv, delimiter=",", dtype=np.float32)
    if csv.ndim == 1:
        csv = csv.reshape(1, -1)

    robot_meta = _build_robot_metadata(robot_config)
    expected_cols = 7 + robot_meta["num_dof"]
    if csv.shape[1] != expected_cols:
        raise ValueError(f"CSV column count mismatch: expected {expected_cols}, got {csv.shape[1]}")

    original_rows = int(np.asarray(motion["root_trans_offset"]).shape[0])
    if csv.shape[0] != original_rows:
        raise ValueError(f"CSV row count mismatch: expected {original_rows}, got {csv.shape[0]}")

    pose_aa, quats = _pose_aa_from_csv(csv, robot_meta)
    updated_motion = dict(motion)
    updated_motion["root_trans_offset"] = csv[:, 0:3].astype(np.float32)
    updated_motion["root_rot"] = quats.astype(np.float32)
    updated_motion["dof"] = csv[:, 7:].astype(np.float32)[:, :, None]
    updated_motion["pose_aa"] = pose_aa.astype(np.float32)
    resolved_motion_key = motion_key if motion_key is not None else next(iter(data.keys()))
    data[resolved_motion_key] = updated_motion

    if inplace:
        output_path = Path(input_pkl)
    else:
        output_path = Path(output_pkl) if output_pkl is not None else _default_import_pkl_path(input_pkl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, output_path)
    return True


def _build_parser():
    parser = argparse.ArgumentParser(description="Convert robot motion data to and from trajectory CSV format.")
    parser.add_argument(
        "--robot-config",
        default="humanoidverse/config/robot/g1/g1_29dof_model16.yaml",
        help="Robot config yaml used to resolve DOF ordering and axes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-csv", help="Export one motion entry to trajectory CSV.")
    export_parser.add_argument("--input-pkl", required=True)
    export_parser.add_argument("--motion-key")
    export_parser.add_argument("--output-csv")

    import_parser = subparsers.add_parser("import-csv", help="Import trajectory CSV and overwrite robot fields.")
    import_parser.add_argument("--input-pkl", required=True)
    import_parser.add_argument("--motion-key")
    import_parser.add_argument("--input-csv", required=True)
    import_parser.add_argument("--output-pkl")
    import_parser.add_argument("--inplace", action="store_true")

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "export-csv":
        export_motion_to_csv(
            input_pkl=args.input_pkl,
            motion_key=args.motion_key,
            output_csv=args.output_csv,
            robot_config=args.robot_config,
        )
        return

    import_csv_to_motion_file(
        input_pkl=args.input_pkl,
        motion_key=args.motion_key,
        input_csv=args.input_csv,
        robot_config=args.robot_config,
        output_pkl=args.output_pkl,
        inplace=args.inplace,
    )


if __name__ == "__main__":
    main()
