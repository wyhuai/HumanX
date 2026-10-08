import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
import os
import statistics
from collections import deque
import copy
from typing import Optional, Tuple, Dict
from isaacgym import gymapi
from isaacgym import gymtorch
from hydra.utils import instantiate


class BaseWorldModel():
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device

    def step(self, action):
        raise NotImplementedError
    
    def next(self, obs, action):
        raise NotImplementedError

    def reset(self, state=None, buffer=None):
        raise NotImplementedError


class SimWorldModel(BaseWorldModel):
    def __init__(self, 
                 config, 
                 env,
                 n_samples:Optional[int]=100, 
                 device='cpu'):
        super(SimWorldModel, self).__init__(config, device)
        self.env = env
        self.env.set_is_evaluating()
        self.num_samples = n_samples
        self.action_dim = self.env.config.robot.actions_dim
        self.obs_dim = self.env.config.robot.policy_obs_dim
        self.rollout_ids = torch.arange(self.num_samples, device=self.device) + 1
        self.reset()
        self.states = None # states for all sim environments

    def reset(self, state=None, buffer=None):
        # Remark: privileged_obs is for critic learning, unused in model-based RL algo
        if state is None:
            self.env.reset_envs_idx(self.rollout_ids)
        else:
            self.set_envs_to_state(state, self.rollout_ids)

        self.states = {"dof_states": copy.deepcopy(self.env.dof_state.view(1 + self.num_samples, self.env.num_dof, 2)[1:]),
                       "root_states": copy.deepcopy(self.env.robot_root_states[1:])}
        
    def set_envs_to_state(self, desired_state, ids=None):
        # TODO
        if ids is None:
            ids = list(range(self.env.num_envs))
        root_states = {}
        for key in ["dof_states", "root_states"]:
            root_states[key] = desired_state[key][1:]

        # Create a numpy array to hold the states for all environments
        self.env.reset_envs_idx(self.rollout_ids, target_states=root_states)
    
    def step(self, actions):
        """
        parameters:
        actions: (n_samples, action_dim)
        outputs:
        obs: (n_samples, obs_dim)
        rewards: (n_samples,)
        dones: (n_samples,)
        """
        actions = torch.cat([torch.zeros_like(actions[[0]], device=self.device), actions], dim=0) # add a zero action for the first env
        obs, rewards, dones, infos = self.env.step(actions) # returns all envs' info (self.num_samples + 1)
        self.states = {"dof_states": copy.deepcopy(self.env.dof_state.view(1 + self.num_samples, self.env.num_dof, 2)[1:]),
                       "root_states": copy.deepcopy(self.env.robot_root_states[1:])}
        self.obs_buf = obs["actor_obs"]
        self.privileged_obs_buf = obs["critic_obs"]
        return obs["actor_obs"][1:], rewards[1:], (1 - dones[1:]).bool()
    
    def next(self, states, actions):
        raise NotImplementedError
    

# TODO: Implement the WorldModel class
# class RSSM(BaseWorldModel):
#     def __init__(self, env, config, device='cpu'):
#         super(RSSM, self).__init__(env, config, device)
#         self.model = instantiate(config.model)
#         self.model = self.model.to(self.device)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
#         self.loss_fn = torch.nn.MSELoss()
#         self._dynamics = None
#         self._reward = None
#         self._pi = None
#         self._task_emb = None
#         self._critic = None
#         self.obs, self.privileged_obs = self.reset()

#     def next(self, obs, action):
#         raise NotImplementedError


class MultiSimWorldModel(BaseWorldModel):
    def __init__(self, 
                 config,
                 sim_config,
                 n_samples:Optional[int]=100, 
                 command:Optional=None,
                 device='cpu'):
        super(MultiSimWorldModel, self).__init__(config, device)
        self.env = instantiate(sim_config, device=device)
        self.env.set_is_evaluating(command)
        self.num_samples = n_samples
        self.action_dim = self.env.config.robot.actions_dim
        self.obs_dim = self.env.config.robot.policy_obs_dim
        self.rollout_ids = torch.arange(self.num_samples, device=self.device)
        self.reset()
        self.states = None

    def reset(self, state=None, buffer=None):
        # Remark: privileged_obs is for critic learning, unused in model-based RL algo
        if state is None:
            self.env.reset_envs_idx(self.rollout_ids)
        else:
            self.set_envs_to_state(state, buffer, self.rollout_ids)

        self.states = {"dof_states": copy.deepcopy(self.env.dof_state.view(self.num_samples, self.env.num_dof, 2)),
                       "root_states": copy.deepcopy(self.env.robot_root_states)}
        
    def set_envs_to_state(self, desired_state, desired_buf, ids=None):
        if ids is None:
            ids = list(range(self.env.num_envs))
        # Create a numpy array to hold the states for all environments
        self.env.reset_envs_idx(self.rollout_ids, target_states=desired_state, target_buf=desired_buf)

    def get_env_dim(self):
        # TODO
        # get sim setups for all envs
        return {
            "dof_shape": [self.num_samples, self.env.num_dof, 2],
            "root_states_shape": self.env.robot_root_states.shape,
            "obs_dim": self.obs_dim,
        }
    
    def step(self, actions):
        """
        parameters:
        actions: (n_samples, action_dim)
        outputs:
        obs: (n_samples, obs_dim)
        rewards: (n_samples,)
        dones: (n_samples,)
        """
        obs, rewards, dones, infos = self.env.step(actions) # returns all envs' info (self.num_samples + 1)
        self.states = {"dof_states": copy.deepcopy(self.env.dof_state.view(self.num_samples, self.env.num_dof, 2)),
                       "root_states": copy.deepcopy(self.env.robot_root_states)}
        #print("commands:", self.env.commands)
        self.obs_buf = obs["actor_obs"]
        self.privileged_obs_buf = obs["critic_obs"]
        return obs["actor_obs"], rewards, (1 - dones).bool()
    
    def next(self, states, actions):
        raise NotImplementedError

