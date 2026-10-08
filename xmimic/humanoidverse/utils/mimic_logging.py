from __future__ import annotations

from typing import Dict, Tuple

import torch


def apply_termination_mask(
    log_dict: Dict[str, torch.Tensor],
    reset_buf: torch.Tensor,
    time_out_buf: torch.Tensor,
    name: str,
    raw_mask: torch.Tensor,
    applied_mask: torch.Tensor | None = None,
    *,
    mark_timeout: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one termination cause and log raw/applied/first-hit fractions.

    `first` means the environments for which this cause is the first one in the
    current step to set `reset_buf=True`, so ordering in the caller defines the
    attribution precedence.
    """
    raw_mask = raw_mask.bool()
    if applied_mask is None:
        applied_mask = raw_mask
    else:
        applied_mask = applied_mask.bool()

    first_mask = applied_mask & (~reset_buf.bool())
    log_dict[f"term_raw_{name}_frac"] = raw_mask.float().mean()
    log_dict[f"term_applied_{name}_frac"] = applied_mask.float().mean()
    log_dict[f"term_first_{name}_frac"] = first_mask.float().mean()

    next_time_out_buf = time_out_buf | applied_mask if mark_timeout else time_out_buf
    next_reset_buf = reset_buf | applied_mask
    return next_reset_buf, next_time_out_buf, first_mask


def summarize_reward_groups(
    episode_sums: Dict[str, torch.Tensor],
    reward_scales: Dict[str, float],
    episode_lengths: torch.Tensor,
    max_episode_length_s: float,
    step_dt: float,
) -> Dict[str, torch.Tensor]:
    """Build positive/penalty/termination reward-group metrics.

    The returned metrics mirror the existing `Episode/rew_*` normalization and
    add per-step grouped summaries for easier cross-run comparison.
    """
    device = episode_lengths.device
    dtype = episode_lengths.dtype
    lengths = torch.clamp(episode_lengths.to(torch.float32), min=1.0)

    groups = {
        "positive": torch.zeros_like(lengths, dtype=torch.float32, device=device),
        "penalty": torch.zeros_like(lengths, dtype=torch.float32, device=device),
        "termination": torch.zeros_like(lengths, dtype=torch.float32, device=device),
    }

    for name, values in episode_sums.items():
        if name == "termination":
            groups["termination"] += values.to(torch.float32)
            continue
        scale = float(reward_scales.get(name, 0.0))
        if scale > 0.0:
            groups["positive"] += values.to(torch.float32)
        elif scale < 0.0:
            groups["penalty"] += values.to(torch.float32)

    metrics: Dict[str, torch.Tensor] = {}
    denom = float(max_episode_length_s)
    seconds = torch.clamp(lengths * float(step_dt), min=max(float(step_dt), 1e-8))
    for group_name, values in groups.items():
        metrics[f"rew_group_{group_name}"] = torch.mean(values) / denom
        metrics[f"rew_step_group_{group_name}"] = torch.mean(values / lengths)
        metrics[f"rew_second_group_{group_name}"] = torch.mean(values / seconds)

    return metrics


def summarize_contact_error_metrics(
    window_count: torch.Tensor,
    total_error_count: torch.Tensor,
    forbidden_contact_error_count: torch.Tensor,
    object_contact_mismatch_error_count: torch.Tensor,
    missed_object_contact_error_count: torch.Tensor,
    unexpected_object_contact_error_count: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Summarize per-episode contact error rates and split root causes."""
    denom = torch.clamp(window_count.to(torch.float32), min=1.0)
    total_per_env = total_error_count.to(torch.float32) / denom
    forbidden_per_env = forbidden_contact_error_count.to(torch.float32) / denom
    object_mismatch_per_env = object_contact_mismatch_error_count.to(torch.float32) / denom
    missed_object_per_env = missed_object_contact_error_count.to(torch.float32) / denom
    unexpected_object_per_env = unexpected_object_contact_error_count.to(torch.float32) / denom

    return {
        "contact_error_rate": torch.mean(total_per_env),
        "contact_error_rate_per_env": total_per_env,
        "forbidden_contact_error_rate": torch.mean(forbidden_per_env),
        "forbidden_contact_error_rate_per_env": forbidden_per_env,
        "object_contact_mismatch_error_rate": torch.mean(object_mismatch_per_env),
        "object_contact_mismatch_error_rate_per_env": object_mismatch_per_env,
        "missed_object_contact_error_rate": torch.mean(missed_object_per_env),
        "missed_object_contact_error_rate_per_env": missed_object_per_env,
        "unexpected_object_contact_error_rate": torch.mean(unexpected_object_per_env),
        "unexpected_object_contact_error_rate_per_env": unexpected_object_per_env,
    }
