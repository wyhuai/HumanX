import torch
import torch.nn as nn
from torch import autograd


import importlib
import os
import pathlib
from typing import Callable, Tuple

import git
# import numpy as np
import torch


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = (), device="cpu"):
        """
        Calculates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = torch.zeros(shape, dtype=torch.float32).to(device)
        self.var = torch.ones(shape, dtype=torch.float32).to(device)
        self.device = device
        self.count = epsilon

    def update(self, arr: torch.Tensor) -> None:
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class Normalizer(RunningMeanStd):
    def __init__(self, obs_dim_dict, module_config_dict, device="cpu", epsilon=1e-4, clip_obs=10.0):
        self._setup(obs_dim_dict, module_config_dict)
        super().__init__(shape=self.input_dim, device=device)
        self.epsilon = epsilon
        self.clip_obs = clip_obs
        
    def _setup(self, obs_dim_dict, module_config_dict):
        self.input_dim = 0
        for input_dim_name in module_config_dict['input_dim']:
            if input_dim_name in obs_dim_dict.keys():
                self.input_dim += obs_dim_dict[input_dim_name]

    def normalize(self, input):
        return torch.clip((input - self.mean) / torch.sqrt(self.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def normalize_torch(self, input, device):
        mean_torch = self.mean.to(torch.float32)
        std_torch = torch.sqrt((self.var + self.epsilon).to(torch.float32))
        return torch.clamp((input - mean_torch) / std_torch, -self.clip_obs, self.clip_obs)


class Discriminator(nn.Module):
    """
    Discriminator neural network for adversarial motion priors (AMP) reward prediction.

    Args:
        input_dim (int): Dimension of the input feature vector (concatenated state and next state).
        amp_reward_coef (float): Coefficient to scale the AMP reward.
        hidden_layer_sizes (list[int]): Sizes of hidden layers in the MLP trunk.
        device (torch.device): Device to run the model on (CPU or GPU).
        task_reward_lerp (float, optional): Interpolation factor between AMP reward and task reward.
            Defaults to 0.0 (only AMP reward).

    Attributes:
        trunk (nn.Sequential): MLP layers processing input features.
        amp_linear (nn.Linear): Final linear layer producing discriminator output.
        task_reward_lerp (float): Interpolation factor for combining rewards.
    """

    def __init__(self, obs_dim_dict, module_config_dict, device):
        
        super().__init__()
        
        self._setup(obs_dim_dict, module_config_dict)

        self.device = device

        self.amp_reward_coef = module_config_dict["amp_reward_coef"]
        amp_layers = []
        curr_in_dim = self.input_dim
        self.hidden_dims = module_config_dict["hidden_dims"]
        for hidden_dim in self.hidden_dims:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(self.hidden_dims[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = module_config_dict["task_reward_lerp"]
        
    def _setup(self, obs_dim_dict, module_config_dict):
        self.input_dim = 0
        for input_dim_name in module_config_dict['input_dim']:
            if input_dim_name in obs_dim_dict.keys():
                self.input_dim += obs_dim_dict[input_dim_name]
                

    def forward(self, x):
        """
        Forward pass through the discriminator network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Discriminator output logits with shape (batch_size, 1).
        """
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self, amp_expert_state, lambda_=10):
        """
        Compute gradient penalty for the expert data, used to regularize the discriminator.

        Args:
            expert_state (torch.Tensor): Batch of expert states.
            expert_next_state (torch.Tensor): Batch of expert next states.
            lambda_ (float, optional): Gradient penalty coefficient. Defaults to 10.

        Returns:
            torch.Tensor: Scalar gradient penalty loss.
        """
        expert_data = amp_expert_state
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(self, amp_obs, task_reward, normalizer=None):
        """
        Predict the AMP reward given current and next states, optionally interpolated with a task reward.

        Args:
            state (torch.Tensor): Current state tensor.
            next_state (torch.Tensor): Next state tensor.
            task_reward (torch.Tensor): Task-specific reward tensor.
            normalizer (optional): Normalizer object to normalize input states before prediction.

        Returns:
            tuple:
                - reward (torch.Tensor): Predicted AMP reward (optionally interpolated) with shape (batch_size,).
                - d (torch.Tensor): Raw discriminator output logits with shape (batch_size, 1).
        """
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                amp_obs = normalizer.normalize_torch(amp_obs, self.device)

            d = self.amp_linear(self.trunk(amp_obs))
            reward = self.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
            self.train()
        return reward.squeeze(), d

    def _lerp_reward(self, disc_r, task_r):
        """
        Linearly interpolate between discriminator reward and task reward.

        Args:
            disc_r (torch.Tensor): Discriminator reward.
            task_r (torch.Tensor): Task reward.

        Returns:
            torch.Tensor: Interpolated reward.
        """
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r