from __future__ import annotations
from copy import deepcopy

from easydict import EasyDict
import torch
import torch.nn as nn
from torch.distributions import Normal

from .modules import BaseModule


def _ensure_module_config(module_config_dict):
    if hasattr(module_config_dict, "layer_config"):
        return module_config_dict
    return EasyDict(module_config_dict)

class DiscretePPOActor(nn.Module):
    def __init__(self,
                 obs_dim_dict,
                 module_config_dict,
                 num_discrete_actions,  # Number of discrete choices per action dimension
                 action_dim):           # Number of action dimensions
        super(DiscretePPOActor, self).__init__()

        # Set output dim to action_dim * num_discrete_actions.
        module_config_dict = self._process_module_config(module_config_dict, action_dim * num_discrete_actions)
        
        self.actor_module = BaseModule(obs_dim_dict, _ensure_module_config(module_config_dict))
        
        # Discrete-action related params.
        self.num_discrete_actions = num_discrete_actions
        self.action_dim = action_dim
        
        # Remove continuous-action noise params.
        self.distribution = None

    def _process_module_config(self, module_config_dict, output_dim):
        """Process module config and set the correct output dimension."""
        for idx, dim in enumerate(module_config_dict['output_dim']):
            if dim == 'robot_action_dim':
                module_config_dict['output_dim'][idx] = output_dim
        return module_config_dict

    @property
    def actor(self):
        return self.actor_module
    
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    def update_distribution(self, actor_obs):
        """Update action distribution (discrete version)."""
        # Get logits, shape: [batch_size, action_dim * num_discrete_actions].
        logits = self.actor(actor_obs)
        
        # Reshape to [batch_size, action_dim, num_discrete_actions].
        logits = logits.view(-1, self.action_dim, self.num_discrete_actions)
        
        # Create categorical distribution.
        self.distribution = torch.distributions.Categorical(logits=logits)

    def act(self, actor_obs, **kwargs):
        """Sample discrete actions."""
        self.update_distribution(actor_obs)
        actions = self.distribution.sample()  # shape: [batch_size, action_dim]
        
        # Get action logits for storage.
        action_logits = self.distribution.logits
        
        return actions, action_logits.detach()
    
    def get_actions_log_prob(self, actions):
        """Compute log-probabilities for discrete actions."""
        if self.distribution is None:
            raise ValueError("Distribution not initialized. Call update_distribution first.")
        
        # Compute per-dimension log-probabilities.
        log_probs = self.distribution.log_prob(actions)  # shape: [batch_size, action_dim]
        
        # For multi-dimensional actions, sum log-probabilities across dimensions.
        if len(log_probs.shape) > 1:
            log_probs = log_probs.sum(dim=-1)
        
        return log_probs

    def get_policy_actions_for_log_prob(self, actions):
        return actions

    def act_inference(self, actor_obs):
        """Inference mode: pick the most probable action."""
        self.update_distribution(actor_obs)
        # Select argmax action for each action dimension.
        actions = torch.argmax(self.distribution.probs, dim=-1)
        return actions
    
    @property
    def entropy(self):
        """Compute entropy (discrete version)."""
        if self.distribution is None:
            raise ValueError("Distribution not initialized.")
        return self.distribution.entropy().sum(dim=-1)  # sum entropy across dimensions
    
    def get_action_logits(self, obs_dict):
        """Get action logits."""
        actor_obs = obs_dict['actor_obs'] if isinstance(obs_dict, dict) else obs_dict
        logits = self.actor(actor_obs)
        logits = logits.view(-1, self.action_dim, self.num_discrete_actions)
        return logits
    
    def to_cpu(self):
        self.actor_module = deepcopy(self.actor_module).to('cpu')



class PPOActor(nn.Module):
    def __init__(self,
                obs_dim_dict,
                module_config_dict,
                num_actions,
                init_noise_std,
                add_ref_action: bool = False):
        super(PPOActor, self).__init__()

        module_config_dict = self._process_module_config(module_config_dict, num_actions)

        self.actor_module = BaseModule(obs_dim_dict, _ensure_module_config(module_config_dict))

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        # self.std = init_noise_std * torch.ones(num_actions).to('cuda')
        self.distribution = None
        # HDMI-compat: optionally add reference dof_pos (expert-default) to the network output.
        # When enabled, actor.act/act_inference can be called as (actor_obs, ref_dof_pos).
        self.add_ref_action = bool(add_ref_action)
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict['output_dim']):
            if output_dim == 'robot_action_dim':
                module_config_dict['output_dim'][idx] = num_actions
        return module_config_dict

    @property
    def actor(self):
        return self.actor_module
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_obs, ref_dof_pos=None, **kwargs):
        # Accept either `ref_dof_pos` or `ref_obs` (common naming across repos).
        if ref_dof_pos is None and "ref_obs" in kwargs:
            ref_dof_pos = kwargs["ref_obs"]
        if self.add_ref_action and ref_dof_pos is None:
            raise TypeError(
                "PPOActor.update_distribution() missing required reference input: "
                "`ref_dof_pos`/`ref_obs` is None while `add_ref_action=true`. "
                "Fix by ensuring the env provides `obs['ref_obs']` (e.g. define obs_dict.ref_obs "
                "or set env.config.obs.synthesize_ref_obs=true), and pass it into the actor."
            )
        mean = self.actor(actor_obs)
        if self.add_ref_action and ref_dof_pos is not None:
            mean = mean + ref_dof_pos
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, actor_obs, ref_dof_pos=None, **kwargs):
        self.update_distribution(actor_obs, ref_dof_pos=ref_dof_pos, **kwargs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_policy_actions_for_log_prob(self, actions):
        return actions

    def act_inference(self, actor_obs, ref_dof_pos=None, **kwargs):
        if ref_dof_pos is None and "ref_obs" in kwargs:
            ref_dof_pos = kwargs["ref_obs"]
        if self.add_ref_action and ref_dof_pos is None:
            raise TypeError(
                "PPOActor.act_inference() missing required reference input: "
                "`ref_dof_pos`/`ref_obs` is None while `add_ref_action=true`. "
                "Fix by ensuring the env provides `obs['ref_obs']` (e.g. define obs_dict.ref_obs "
                "or set env.config.obs.synthesize_ref_obs=true), and pass it into the actor."
            )
        actions_mean = self.actor(actor_obs)
        if self.add_ref_action and ref_dof_pos is not None:
            actions_mean = actions_mean + ref_dof_pos
        return actions_mean
    
    def to_cpu(self):
        self.actor = deepcopy(self.actor).to('cpu')
        self.std.to('cpu')


class PPOCritic(nn.Module):
    def __init__(self,
                obs_dim_dict,
                module_config_dict):
        super(PPOCritic, self).__init__()

        self.critic_module = BaseModule(obs_dim_dict, _ensure_module_config(module_config_dict))

    @property
    def critic(self):
        return self.critic_module
    
    def reset(self, dones=None):
        pass
    
    def evaluate(self, critic_obs, **kwargs):
        value = self.critic(critic_obs)
        return value

# Deprecated: TODO: Let Wenli Fix this
class PPOActorFixSigma(PPOActor):
    def __init__(self,                 
                 obs_dim_dict,
                network_dict,
                network_load_dict,
                num_actions,):
        super(PPOActorFixSigma, self).__init__(obs_dim_dict, network_dict, network_load_dict, num_actions, 0.0)
        
    def update_distribution(self, obs_dict):
        mean = self.actor(obs_dict)['head']
        self.distribution = mean

    @property
    def action_mean(self):
        return self.distribution
    
    def get_actions_log_prob(self, actions):
        raise NotImplementedError
    
    def act(self, obs_dict, **kwargs):
        self.update_distribution(obs_dict)
        return self.distribution
