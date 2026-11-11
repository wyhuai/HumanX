#!/usr/bin/env bash
# Run from xmimic with its documented Isaac Gym environment activated.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set a comma-separated list of GPU IDs}"
: "${HUMANX_TRAIN_OUTPUT:?Set an output directory for the final checkpoint}"
num_gpus=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
if (( num_gpus < 2 )); then
  echo 'Select at least two GPUs.' >&2
  exit 2
fi
total_envs=${HUMANX_TOTAL_ENVS:-16384}
iterations=${HUMANX_ITERATIONS:-200}
if (( total_envs <= 0 || total_envs % num_gpus != 0 || iterations <= 0 )); then
  echo 'The positive total environment count must be divisible by the GPU count.' >&2
  exit 2
fi
mkdir -p "$HUMANX_TRAIN_OUTPUT"
exec python -m torch.distributed.run --standalone --nproc_per_node="$num_gpus" \
  experimental/multigpu_hoi/train.py humanoidverse/train_agent.py \
  +simulator=isaacgym_hoi +exp=humanx/hoi_teacher +domain_rand=domain_rand_low \
  +rewards=humanx/hoi +robot=g1/g1_29dof_model16 +terrain=terrain_locomotion_plane \
  +obs=humanx/hoi_teacher_tracking num_envs="$((total_envs / num_gpus))" \
  project_name=HumanX_MultiGPU experiment_name=hoi_teacher_experimental \
  robot.motion.motion_file=data/motions/example/basketball_catch_throw.pkl \
  rewards.reward_penalty_curriculum=true rewards.reward_penalty_degree=0.00002 \
  env.config.packet_loss.enable=true env.config.termination.contact_conditioned_ig_termination=true \
  env.config.init_noise_scale.obj_pos=0.1 headless=true \
  algo.config.num_learning_iterations="$iterations" algo.config.save_interval=10000 "$@"
