# XMimic

XMimic is a research codebase for learning humanoid motion tracking and
human-object interaction (HOI) skills from motion data. The main supported
runtime is NVIDIA Isaac Gym with a G1 humanoid and configurable object assets.

This repository currently releases the simulation-side training and evaluation
code. It does not include a complete hardware deployment, ROS/Unitree control,
or sim-to-real calibration pipeline.

## What Is Included

- Motion tracking and HOI environments.
- Isaac Gym training and evaluation entry points.
- PPO and AMP-PPO training.
- DAgger, DAggerBC, and DAggerKL training configurations.
- Reference observations with future targets, contact labels, phase, history,
  and task-specific overrides.
- Object interaction assets and contact-conditioned termination logic.
- XGen NPZ to HumanX PKL conversion.
- TorchScript and ONNX policy export during evaluation.


## Requirements

- Linux and Python 3.8.
- NVIDIA GPU and a CUDA-compatible PyTorch installation.
- NVIDIA Isaac Gym Preview 4 for the Isaac Gym backend.
- A CUDA/PyTorch version compatible with the Isaac Gym Python bindings.

`setup.py` installs the Python-level project dependencies, but it does not
install PyTorch, Isaac Gym, Isaac Sim, or Genesis. Install those components
separately when they are needed.

## Installation

Run the following commands from the `xmimic` directory:

```bash
conda create -n hxmimic python=3.8 -y
conda activate hxmimic

# Install a PyTorch build compatible with the local CUDA driver first.

# Install Isaac Gym Preview 4 from its extracted directory.
pip install -e /path/to/isaacgym/python

pip install -e .
pip install -e isaac_utils
```

On systems where Isaac Gym cannot find `libpython3.8.so.1.0`, expose the
active conda environment libraries before importing Isaac Gym:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

The repository uses Hydra. Commands below are intended to be run from the
`xmimic` directory so that relative motion, asset, log, and checkpoint paths
resolve consistently.

Before running a training or evaluation command, activate the environment and
change into this directory:

```bash
cd /path/to/HumanX/xmimic
conda activate hxmimic
```

## Motion Data

XMimic consumes HumanX motion files in `.pkl` format. The repository includes
examples for **basketball catch-and-throw, basketball pump-fake shots, box
pick-and-place, and football kicking**. These example files are stored under
`data/motions/example/`; the football example contains multiple motion
variants.

For new skills, the recommended video-to-policy pipeline is:

1. **[GVHMR](https://github.com/zju3dv/GVHMR)** recovers world-grounded human
   motion from monocular video.
2. **[UMR](https://github.com/hanyang9/UMR)** retargets the recovered human
   motion to the G1 humanoid.
3. **[XGen](../xgen/README.md)** edits and synthesizes physically
   plausible humanoid-object interaction motion, and exports HOI `.npz` files.
4. XMimic converts the XGen HOI `.npz` files into HumanX `.pkl` files and
   trains the interaction policy. See [XGen to XMimic Data Pipeline](#xgen-to-xmimic-data-pipeline) for the
conversion command and input format.

## Training

### PPO HOI teacher

The following command uses a motion file that is present in the current
working tree. Replace it with another HumanX PKL file when needed.

```bash
python humanoidverse/train_agent.py \
  +simulator=isaacgym_hoi \
  +exp=humanx/hoi_teacher \
  +domain_rand=domain_rand_low \
  +rewards=humanx/hoi \
  +robot=g1/g1_29dof_model16 \
  +terrain=terrain_locomotion_plane \
  +obs=humanx/hoi_teacher_tracking \
  num_envs=4096 \
  project_name=HumanX \
  experiment_name=HOI_Teacher \
  robot.motion.motion_file=data/motions/example/basketball_catch_throw.pkl \
  rewards.reward_penalty_curriculum=true \
  rewards.reward_penalty_degree=0.00002 \
  env.config.packet_loss.enable=true \
  env.config.termination.contact_conditioned_ig_termination=true \
  env.config.init_noise_scale.obj_pos=0.1
```

For the AMP teacher configuration, replace the experiment and observation
overrides with:

```text
+exp=humanx/hoi_teacher_w_amp
+obs=humanx/hoi_teacher_w_amp
```

These are Hydra overrides, so the leading `+` is intentional.

### DAggerBC student

```bash
python humanoidverse/train_agent.py \
  +simulator=isaacgym_hoi \
  +exp=humanx/hoi_DAggerBC \
  +domain_rand=domain_rand_low \
  +rewards=humanx/hoi \
  +robot=g1/g1_29dof_model16 \
  +terrain=terrain_locomotion_plane \
  +obs=humanx/mocap/hoi_DAggerBC \
  num_envs=4096 \
  project_name=HumanX \
  experiment_name=HOI_Student \
  robot.motion.motion_file=data/motions/example/basketball_catch_throw.pkl \
  rewards.reward_penalty_curriculum=true \
  rewards.reward_penalty_degree=0.00002 \
  env.config.packet_loss.enable=true \
  env.config.termination.contact_conditioned_ig_termination=true \
  env.config.init_noise_scale.obj_pos=0.1 \
  algo.config.teacher_ckpt_path=logs/HumanX/<teacher_run>/model_<iter>.pt
```

This DAggerBC observation config is compatible with a teacher trained using
`+obs=humanx/hoi_teacher_tracking` with the same reference-command settings.
For the teacher checkpoint used in this example, the expected input sizes are
`teacher actor=572` and `teacher critic=1135`.

If the student critic does not match the teacher critic architecture, add:

```text
algo.config.load_teacher_critic=false
```

### Common overrides

The object asset is normally inferred from the motion filename, so training
and evaluation commands do not need an object-specific override. For example,
the following command automatically selects `football/ball.urdf` from the
`football_*.pkl` filename:

```bash
# Use the football asset and foot contact bodies.
robot.motion.motion_file=data/motions/example/football_front_kick_64augs.pkl \
robot.hoi_contact_bodies='["left_ankle_roll_link","right_ankle_roll_link"]' \
env.config.termination.contact_conditioned_ig_termination=true
```

For a custom filename or an asset that is not in the built-in mapping, select
the asset explicitly:

```text
simulator.config.objects.ball.urdf_file=football/ball.urdf
```

For a fixed object, use an asset configuration with
`simulator.config.objects.ball.fix_base_link=true`.

### Automatic object asset selection

When `simulator.config.objects.ball.urdf_file` is left as `null` and
`auto_detect_from_motion=true`, XMimic selects the object asset from the
leading category in the motion filename. The current built-in mapping is:

```text
bin_*          -> bin/bin.urdf
basketball_*   -> basketball/ball.urdf
football_*     -> football/ball.urdf
shuttlecock_*  -> shuttlecock/ball.urdf
box_*           -> box/box.urdf
cube_*          -> box/box.urdf  (legacy alias)
```

When a converted motion contains `object_size`, XMimic automatically computes
the uniform `ball_scale` from the motion size and the base geometry size in
the selected URDF. This is enabled by default when
`simulator.config.object_scales.ball_scale=null`. Passing an explicit
`simulator.config.object_scales.ball_scale=<value>` overrides automatic
scaling.

## Evaluation

The `checkpoint=` override is an existing field in `base_eval.yaml`; do not
prefix it with `+`.

```bash
python humanoidverse/eval_agent.py \
  checkpoint=logs/HumanX/<run_dir>/model_<iter>.pt \
  headless=false \
  num_envs=1 \
  export.policy=false \
  export.onnx=false
```

To visualize motion data without running policy control while reusing a
training configuration, try

```bash
python humanoidverse/eval_agent.py \
  +simulator=isaacgym_hoi \
  +exp=humanx/hoi_teacher \
  +domain_rand=domain_rand_low \
  +rewards=humanx/hoi \
  +robot=g1/g1_29dof_model16 \
  +terrain=terrain_locomotion_plane \
  +obs=humanx/hoi_teacher_tracking \
  robot.motion.motion_file=data/motions/example/basketball_catch_throw.pkl \
  env.config.visualize_motion_data_only=true \
  env.config.use_custom_motion_start=true \
  env.config.motion_start_frame=0 \
  headless=false \
  num_envs=1 \
  export.policy=false \
  export.onnx=false
```

## XGen to XMimic Data Pipeline

The standard XGen-to-XMimic converter is:

```text
tools/convert_xgen_hoi_npz.py
```

### Example: `speed_capped_5mps`

This is the standard conversion used for the XGen basketball data in this
repository:

```bash
python tools/convert_xgen_hoi_npz.py \
  --input-dir data/motions/speed_capped_5mps \
  --output data/motions/xgen_speed_capped_5mps/speed_capped_5mps_dynamic.pkl \
  --no-freeze-until-release
```

Because `--use-support` is not specified, supporting-surface information is
ignored. Because `--fixed-object` is not specified and
`--no-freeze-until-release` is enabled, the object is not fixed at any frame.

The converted file currently used in this repository is:

```text
data/motions/xgen_speed_capped_5mps/speed_capped_5mps_dynamic.pkl
```

For the current 42-file dataset, the converted result is 42 motion entries,
30 FPS, 4613 total frames, 42-dimensional one-hot labels, no object-freeze
frames, and `need_support=False`.

### Validate the Converted PKL

Run this check after conversion:

```bash
python - <<'PY'
import joblib
import numpy as np

path = "data/motions/xgen_speed_capped_5mps/speed_capped_5mps_dynamic.pkl"
motions = joblib.load(path)

assert motions, "No motions found"
assert all(int(m["fps"]) == 30 for m in motions.values())
assert all(not np.asarray(m["obj_freeze"]).astype(bool).any() for m in motions.values())
assert all(bool(m["need_support"]) is False for m in motions.values())
assert all(m["skill_label"].shape[-1] == 42 for m in motions.values())

total_frames = sum(len(m["root_trans_offset"]) for m in motions.values())
print(f"validated {len(motions)} motions and {total_frames} frames")
PY
```

### Visualize Before Training

XMimic does not have a separate `play_dataset.py` entry point. The equivalent
motion playback path is `eval_agent.py` with
`env.config.visualize_motion_data_only=true`. Use a checkpoint whose
environment and observation configuration are compatible with the motion:

```bash
python humanoidverse/eval_agent.py \
  checkpoint=logs/HumanX/<compatible_run>/model_<iter>.pt \
  robot.motion.motion_file=data/motions/xgen_speed_capped_5mps/speed_capped_5mps_dynamic.pkl \
  env.config.visualize_motion_data_only=true \
  headless=false \
  num_envs=1 \
  export.policy=false \
  export.onnx=false
```

The playback check should verify, at minimum:

- The G1 pose is not twisted at the hips, shoulders, wrists, or root.
- The basketball follows the intended XGen trajectory.
- The object has the expected size and asset type.
- Contact labels agree with the visible hand-object interaction.
- The object does not move because of an accidental supporting surface.
- The motion plays at the source FPS rather than being silently resampled.

### Use the Converted File for Training

Pass the generated PKL to any compatible HOI training command:

```bash
python humanoidverse/train_agent.py \
  +simulator=isaacgym_hoi \
  +exp=humanx/hoi_teacher \
  +domain_rand=domain_rand_low \
  +rewards=humanx/hoi \
  +robot=g1/g1_29dof_model16 \
  +terrain=terrain_locomotion_plane \
  +obs=humanx/hoi_teacher_tracking \
  num_envs=4096 \
  project_name=HumanX \
  experiment_name=SpeedCapped5mps \
  robot.motion.motion_file=data/motions/xgen_speed_capped_5mps/speed_capped_5mps_dynamic.pkl
```

The converter naturally sorts input files by their numeric prefixes, so
`1_...npz` through `42_...npz` become stable motion entries and 42-dimensional
one-hot labels for unified-policy experiments.
