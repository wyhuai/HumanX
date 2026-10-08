import argparse
from pathlib import Path

import joblib


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


def _parse_override_fps_factor(value):
    normalized = str(value).strip().lower().replace("p", ".")
    factor = float(normalized)
    if factor <= 0.0:
        raise ValueError(f"override-fps factor must be positive, got {value}")
    return factor


def _format_override_fps_suffix(factor):
    factor_str = format(float(factor), "g").replace(".", "p")
    return f"x{factor_str}"


def _default_output_pkl_path(input_pkl, factor):
    input_pkl = Path(input_pkl)
    suffix = _format_override_fps_suffix(factor)
    stem = input_pkl.stem
    if stem.endswith("_w_ball_w_contact"):
        stem = f"{stem[:-len('_w_ball_w_contact')]}_{suffix}_w_ball_w_contact"
    elif stem.endswith("_w_contact"):
        stem = f"{stem[:-len('_w_contact')]}_{suffix}_w_contact"
    else:
        stem = f"{stem}_{suffix}"
    return input_pkl.with_name(f"{stem}.pkl")


def apply_override_fps_to_motion(motion, factor):
    factor = float(factor)
    motion = dict(motion)
    original_fps = float(motion.get("fps", 0))
    if original_fps <= 0.0:
        raise ValueError(f"motion data is missing valid 'fps': {original_fps}")
    motion["fps"] = original_fps * factor
    return motion


def write_motion_with_overridden_fps(
    *,
    input_pkl,
    factor,
    motion_key=None,
    output_pkl=None,
):
    data = _load_motion_data(input_pkl)
    motion_key = _get_motion_key(data, motion_key)
    updated_motion = apply_override_fps_to_motion(data[motion_key], factor)
    data = dict(data)
    data[motion_key] = updated_motion

    output_path = Path(output_pkl) if output_pkl is not None else _default_output_pkl_path(input_pkl, factor)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, output_path)
    return output_path


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Override motion fps metadata without resampling frame data.")
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--motion-key")
    parser.add_argument("--override-fps", required=True, help="FPS multiplier, e.g. 0p7 or 1p3.")
    parser.add_argument("--output-pkl")
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    factor = _parse_override_fps_factor(args.override_fps)
    output_path = write_motion_with_overridden_fps(
        input_pkl=args.input_pkl,
        factor=factor,
        motion_key=args.motion_key,
        output_pkl=args.output_pkl,
    )
    print(f"[set_fps] saved overridden motion to {output_path}")


if __name__ == "__main__":
    main()
