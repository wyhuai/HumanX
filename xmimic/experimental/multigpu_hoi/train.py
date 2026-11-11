"""Experimental sharded HOI PPO; see README for constraints and validation."""
import json
import hashlib
from datetime import timedelta
import os
import random
import runpy
import statistics
import sys
import time

# Give each Isaac Gym process one GPU before CUDA initialization. torchrun
# assigns LOCAL_RANK relative to the caller's CUDA_VISIBLE_DEVICES list.
if "LOCAL_RANK" in os.environ:
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    local_rank = int(os.environ["LOCAL_RANK"])
    if not devices or not devices[0] or local_rank >= len(devices):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to one device per local rank")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank]

import isaacgym  # Isaac Gym must be imported before torch.
import numpy as np
import torch
import torch.distributed as dist
from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.agents.modules.data_utils import RolloutStorage

rank = int(os.environ.get("RANK", os.environ.get("HUMANX_SAMPLING_RANK", "0")))
world = int(os.environ.get("WORLD_SIZE", os.environ.get("HUMANX_SAMPLING_WORLD_SIZE", "1")))
include_ppo = True
sync_curriculum = world > 1
backend = os.environ.get("HUMANX_DISTRIBUTED_BACKEND", "nccl")
process_group_backend = "nccl" if backend == "nccl_p2p" else backend
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
if world > 1:
    dist.init_process_group(process_group_backend, rank=rank, world_size=world, timeout=timedelta(seconds=120))


def sync_models(algo):
    if world == 1:
        return
    for model in (algo.actor, algo.critic):
        for tensor in model.state_dict().values():
            if process_group_backend == "nccl":
                dist.broadcast(tensor, src=0)
            else:
                host = tensor.detach().cpu().contiguous()
                dist.broadcast(host, src=0)
                tensor.copy_(host.to(tensor.device))
    if os.environ.get("HUMANX_VERIFY_MODEL_SYNC") == "1":
        digest = hashlib.sha256()
        for model in (algo.actor, algo.critic):
            for tensor in model.state_dict().values():
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        device = algo.device if process_group_backend == "nccl" else "cpu"
        local_hash = torch.tensor(list(digest.digest()), dtype=torch.uint8, device=device)
        hashes = [torch.empty_like(local_hash) for _ in range(world)]
        dist.all_gather(hashes, local_hash)
        if any(not torch.equal(hashes[0], other) for other in hashes[1:]):
            raise AssertionError("policy/critic weights are not synchronized across ranks")
        if rank == 0:
            print("HUMANX_MODELS_SYNCHRONIZED", digest.hexdigest(), flush=True)


def gather_rollouts(algo):
    """Gather rollout shards on rank zero and normalize advantages globally."""
    local_storage = algo.storage
    global_storage = None
    if rank == 0:
        global_storage = RolloutStorage(local_storage.num_envs * world, local_storage.num_transitions_per_env,
                                        device=algo.device)
    for key in local_storage.stored_keys:
        local = local_storage.query_key(key).detach().contiguous()
        if key == "advantages":
            # Local advantages were normalized per rank; rebuild them globally below.
            if rank == 0:
                global_storage.register_key(key, shape=tuple(local.shape[2:]), dtype=local.dtype)
            continue
        if backend == "nccl_p2p":
            if rank == 0:
                received = [local]
                for peer in range(1, world):
                    remote = torch.empty_like(local)
                    dist.recv(remote, src=peer)
                    received.append(remote)
            else:
                dist.send(local, dst=0)
                received = None
        elif backend == "nccl":
            received = [torch.empty_like(local) for _ in range(world)]
            dist.all_gather(received, local)
        else:
            local = local.cpu()
            received = [torch.empty_like(local) for _ in range(world)] if rank == 0 else None
            dist.gather(local, gather_list=received, dst=0)
        if rank == 0:
            global_storage.register_key(key, shape=tuple(local.shape[2:]), dtype=local.dtype)
            global_storage.query_key(key).copy_(torch.cat(received, dim=1).to(algo.device))
    if rank == 0:
        advantages = global_storage.returns - global_storage.values
        global_storage.advantages.copy_((advantages - advantages.mean()) / (advantages.std() + 1e-8))
    return global_storage


def benchmark(self):
    if bool(self._config_get(self._config_get(self.config, "bc_warmstart", {}), "enabled", False)):
        raise ValueError("BC warmstart is not supported by the experimental trainer")
    sync_models(self)
    # Independent streams after rank 0's initial policy was shared.
    random.seed(42 + rank)
    np.random.seed(42 + rank)
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed_all(42 + rank)
    if self.init_at_random_ep_len:
        self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
    obs = self.env.reset_all()
    obs = {key: value.to(self.device) for key, value in obs.items()}
    self._train_mode()
    if sync_curriculum:
        import joblib
        snapshot = os.environ.get("HUMANX_CURRICULUM_SNAPSHOT")
        resume_state = joblib.load(snapshot) if snapshot else {}
        for key in ("terminate_when_motion_far_threshold", "terminate_when_ball_status_far_threshold",
                    "soft_dof_pos_curriculum_value", "soft_dof_vel_curriculum_value",
                    "soft_torque_curriculum_value", "reward_penalty_scale"):
            if key in resume_state and hasattr(self.env, key):
                setattr(self.env, key, resume_state[key])
        # Original checkpoints do not contain the EMA; callers may supply an approximation.
        if "HUMANX_CURRICULUM_AVERAGE" in os.environ:
            self.env.average_episode_length = torch.tensor(
                float(os.environ["HUMANX_CURRICULUM_AVERAGE"]), device=self.env.device)
        from humanx_distributed_curriculum import GlobalResetCurriculum
        curriculum = GlobalResetCurriculum(self.env)
    durations = []
    start_iteration = self.current_learning_iteration + (1 if any(
        arg.startswith("checkpoint=") and arg.split("=", 1)[1] not in ("", "null", "None")
        for arg in sys.argv) else 0)
    for iteration_offset in range(self.num_learning_iterations):
        if world > 1:
            dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.start_time = time.time()
        obs = self._rollout_step(obs)
        if sync_curriculum and os.environ.get("HUMANX_VERIFY_CURRICULUM") == "1":
            curriculum.verify()
            if rank == 0:
                print("HUMANX_CURRICULUM_SYNCHRONIZED", flush=True)
        local_storage = self.storage
        if include_ppo:
            if world > 1:
                self.storage = gather_rollouts(self)
            if rank == 0:
                self._training_step()
            self.storage = local_storage
            if world > 1:
                sync_models(self)
        torch.cuda.synchronize()
        if world > 1:
            dist.barrier()
        durations.append(time.perf_counter() - start)
        if rank == 0:
            metrics = {
                "iteration": start_iteration + iteration_offset,
                "seconds": durations[-1],
                "mean_episode_reward_rank0": statistics.mean(self.rewbuffer) if self.rewbuffer else None,
                "mean_episode_length_rank0": statistics.mean(self.lenbuffer) if self.lenbuffer else None,
                "curriculum_average_episode_length": float(self.env.average_episode_length),
            }
            print("HUMANX_TRAIN_METRICS " + json.dumps(metrics), flush=True)
        self.storage.clear()
        self.ep_infos.clear()
    if rank == 0:
        self.current_learning_iteration = start_iteration + self.num_learning_iterations - 1
        self.log_it = self.current_learning_iteration
        output = os.environ["HUMANX_TRAIN_OUTPUT"]
        self.save(os.path.join(output, f"model_{self.current_learning_iteration}.pt"))
        print("HUMANX_CHECKPOINT_SAVED", flush=True)
        steady = durations[2:] if len(durations) > 2 else durations
        print("HUMANX_PARALLEL_ROLLOUT " + json.dumps({
            "world_size": world,
            "local_envs": self.env.num_envs,
            "steps_per_env": self.num_steps_per_env,
            "durations_s": durations,
            "median_s_after_two_warmup": statistics.median(steady),
            "transitions_per_s": world * self.env.num_envs * self.num_steps_per_env / statistics.median(steady),
            "includes_optimizer_or_rollout_transfer": include_ppo,
            "global_advantages": include_ppo,
            "env_curriculum_synchronized": sync_curriculum,
            "rollout_transfer": backend if include_ppo and world > 1 else "none",
        }), flush=True)
    if world > 1:
        dist.destroy_process_group()


PPO.learn = benchmark
for index, arg in enumerate(sys.argv):
    if arg.startswith("project_name="):
        sys.argv[index] = arg + "_rank" + str(rank)
script = sys.argv.pop(1)
sys.argv[0] = script
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
