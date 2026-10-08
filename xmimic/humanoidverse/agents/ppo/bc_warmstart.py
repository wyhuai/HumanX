from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger


def _cfg_get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _cfg_bool(config, key, default=False):
    return bool(_cfg_get(config, key, default))


def _cfg_int(config, key, default):
    return int(_cfg_get(config, key, default))


def _cfg_float(config, key, default):
    return float(_cfg_get(config, key, default))


def _to_device_obs(obs_dict, device):
    return {key: value.to(device) for key, value in obs_dict.items()}


def bc_expert_actions(env, target_clip=None):
    """Convert env reference DOF targets into policy action coordinates."""
    actions = (env.expert_dof_pos - env.default_dof_pos) / env.action_scales
    if target_clip is not None:
        clip = float(target_clip)
        actions = torch.clamp(actions, -clip, clip)
    return actions


def _validate_supported_actor(actor):
    if getattr(actor, "add_ref_action", False):
        raise NotImplementedError("BC warmstart does not support add_ref_action=true yet.")
    if not hasattr(actor, "actor"):
        raise TypeError("BC warmstart requires an actor with an `.actor` module.")


def collect_bc_snapshots(env, config, device):
    """Collect short reset-snapshot BC data while letting the env build observations."""
    num_batches = _cfg_int(config, "num_batches", 512)
    history_warmup_steps = _cfg_int(config, "history_warmup_steps", 4)
    collect_horizon = _cfg_int(config, "collect_horizon", 1)
    target_clip = _cfg_get(config, "target_clip", None)

    if num_batches <= 0:
        raise ValueError(f"bc_warmstart.num_batches must be positive, got {num_batches}")
    if history_warmup_steps < 0:
        raise ValueError(
            f"bc_warmstart.history_warmup_steps must be non-negative, got {history_warmup_steps}"
        )
    if collect_horizon <= 0:
        raise ValueError(f"bc_warmstart.collect_horizon must be positive, got {collect_horizon}")

    obs_batches = []
    label_batches = []

    # Do not use inference_mode here: the env mutates internal action/history buffers
    # during reset/step, and inference tensors forbid later inplace updates.
    with torch.no_grad():
        for _ in range(num_batches):
            obs_dict = _to_device_obs(env.reset_all(), device)

            for _ in range(history_warmup_steps):
                actions = bc_expert_actions(env, target_clip=target_clip).to(device)
                obs_dict, _, _, _ = env.step({"actions": actions})
                obs_dict = _to_device_obs(obs_dict, device)

            for horizon_step in range(collect_horizon):
                if "actor_obs" not in obs_dict:
                    raise KeyError("BC warmstart requires obs_dict['actor_obs'].")
                labels = bc_expert_actions(env, target_clip=target_clip).to(device)
                obs_batches.append(obs_dict["actor_obs"].detach().clone())
                label_batches.append(labels.detach().clone())

                if horizon_step + 1 < collect_horizon:
                    obs_dict, _, _, _ = env.step({"actions": labels})
                    obs_dict = _to_device_obs(obs_dict, device)

    return torch.cat(obs_batches, dim=0), torch.cat(label_batches, dim=0)


def train_bc_actor(actor, actor_obs, labels, config, device):
    """Run supervised BC on actor.actor only; PPO optimizer state is left untouched."""
    if actor_obs.numel() == 0 or labels.numel() == 0:
        raise ValueError("BC warmstart received an empty dataset.")

    epochs = _cfg_int(config, "epochs", 3)
    minibatch_size = _cfg_int(config, "minibatch_size", 1024)
    lr = _cfg_float(config, "lr", 3.0e-4)
    max_grad_norm = _cfg_float(config, "max_grad_norm", 1.0)

    if epochs <= 0:
        raise ValueError(f"bc_warmstart.epochs must be positive, got {epochs}")
    if minibatch_size <= 0:
        raise ValueError(f"bc_warmstart.minibatch_size must be positive, got {minibatch_size}")

    actor_obs = actor_obs.to(device)
    labels = labels.to(device)
    actor_module = actor.actor
    parameters = [param for param in actor_module.parameters() if param.requires_grad]
    if len(parameters) == 0:
        raise ValueError("BC warmstart actor module has no trainable parameters.")

    optimizer = optim.Adam(parameters, lr=lr)
    loss_fn = nn.MSELoss()
    was_training = actor.training
    actor.train()

    with torch.inference_mode():
        initial_loss = loss_fn(actor_module(actor_obs), labels).item()

    final_loss = initial_loss
    num_samples = actor_obs.shape[0]
    for _ in range(epochs):
        permutation = torch.randperm(num_samples, device=actor_obs.device)
        for start in range(0, num_samples, minibatch_size):
            batch_ids = permutation[start : start + minibatch_size]
            pred = actor_module(actor_obs[batch_ids])
            loss = loss_fn(pred, labels[batch_ids])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            optimizer.step()
            final_loss = loss.item()

    with torch.inference_mode():
        final_loss = loss_fn(actor_module(actor_obs), labels).item()

    if not was_training:
        actor.eval()

    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "num_samples": int(num_samples),
    }


def run_bc_warmstart(actor, env, config, device, writer=None, log_dir=None):
    if not _cfg_bool(config, "enabled", False):
        return None

    _validate_supported_actor(actor)
    logger.info("Starting PPO BC warmstart")
    actor_obs, labels = collect_bc_snapshots(env=env, config=config, device=device)
    metrics = train_bc_actor(
        actor=actor,
        actor_obs=actor_obs,
        labels=labels,
        config=config,
        device=device,
    )

    logger.info(
        "BC warmstart complete: samples={} initial_loss={:.6f} final_loss={:.6f}",
        metrics["num_samples"],
        metrics["initial_loss"],
        metrics["final_loss"],
    )
    if writer is not None:
        writer.add_scalar("BCWarmstart/initial_loss", metrics["initial_loss"], 0)
        writer.add_scalar("BCWarmstart/final_loss", metrics["final_loss"], 0)
        writer.add_scalar("BCWarmstart/num_samples", metrics["num_samples"], 0)

    if _cfg_bool(config, "save_checkpoint", True) and log_dir is not None:
        path = Path(log_dir) / "bc_warmstart_actor.pt"
        torch.save(
            {
                "actor_model_state_dict": actor.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        logger.info(f"Saved BC warmstart actor checkpoint to {path}")

    return metrics
