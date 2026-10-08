import torch
import torch.nn as nn
import torch.optim as optim

from humanoidverse.agents.modules.ppo_modules import PPOActor, PPOCritic
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.agents.base_algo import BaseAlgo
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.utils.average_meters import TensorAverageMeterDict

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
import time
import os
import statistics
from collections import deque
from hydra.utils import instantiate
from loguru import logger
from rich.progress import track
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
console = Console()

##################################################################
######## Note: This agent is ONLY for EVALUATION purposes ########
##################################################################
class PPOLocoManip(BaseAlgo):
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):

        self.device= device
        self.env = env
        self.config = config
        self.log_dir = log_dir

        self._init_config()

        self.tot_timesteps = 0
        self.tot_time = 0

        # Book keeping
        self.ep_infos = []
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.eval_callbacks: list[RL_EvalCallback] = []
        self.episode_env_tensors = TensorAverageMeterDict()
        _ = self.env.reset_all()

        # Actor observations for stand and loco policies
        self.actor_obs = ['actor_stand_obs', 'actor_loco_obs']

    def _init_config(self):
        # Env related Config
        self.num_envs: int = self.env.config.num_envs
        self.algo_obs_dim_dict = self.env.config.robot.algo_obs_dim_dict
        self.num_act = self.env.config.robot.lower_body_actions_dim

    def setup(self):
        logger.info("Setting up PPO Loco Manip")
        self._setup_models_and_optimizer()

    def _setup_models_and_optimizer(self):
        self.actor_stand = PPOActor(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self.config.module_dict.actor_stand,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std
        ).to(self.device)
        self.actor_loco = PPOActor(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self.config.module_dict.actor_loco,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std
        ).to(self.device)
        self.actor = [self.actor_stand, self.actor_loco]

    def _eval_mode(self):
        self.actor[0].eval()
        self.actor[1].eval()

    def load(self, ckpt_stand_path, ckpt_loco_path):
        if ckpt_stand_path is not None:
            logger.info(f"Loading standing checkpoint from {ckpt_stand_path}")
            loaded_dict_stand = torch.load(ckpt_stand_path, map_location=self.device)
            self.actor[0].load_state_dict(loaded_dict_stand["actor_model_state_dict"])
        if ckpt_loco_path is not None:
            logger.info(f"Loading locomotion checkpoint from {ckpt_loco_path}")
            loaded_dict_loco = torch.load(ckpt_loco_path, map_location=self.device)
            self.actor[1].load_state_dict(loaded_dict_loco["actor_model_state_dict"])
        return loaded_dict_stand['infos'], loaded_dict_loco["infos"]

    def _process_env_step(self, rewards, dones, infos):
        self.actor[self.env.control_mode].reset(dones)

    def _actor_act_step(self, obs_dict):
        return self.actor[self.env.control_mode].act(obs_dict)
    
    @property
    def inference_model(self):
        return self.actor

    ##########################################################################################
    # Code for Evaluation
    ##########################################################################################

    def env_step(self, actor_state):
        actions_lower_body = actor_state["actions"]
        actions_upper_body = self.env.ref_upper_dof_pos
        actions = torch.cat([actions_lower_body, actions_upper_body], dim=1)
        obs_dict, rewards, dones, extras = self.env.step(actions)
        actor_state.update(
            {"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras}
        )
        return actor_state

    @torch.no_grad()
    def get_example_obs(self):
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            print(obs_key, sorted(self.env.config.obs.obs_dict[obs_key]))
        # move to cpu
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return obs_dict

    @torch.no_grad()
    def evaluate_loco_manip_policy(self):
        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state = self._create_actor_state()
        step = 0
        self.eval_policy = self._get_inference_policy()
        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        actor_state = self._pre_eval_env_step(actor_state)
        while True:
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)
            step += 1

    def _create_actor_state(self):
        return {"done_indices": [], "stop": False}

    def _create_eval_callbacks(self):
        if self.config.eval_callbacks is not None:
            for cb in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb], training_loop=self))

    def _pre_evaluate_policy(self, reset_env=True):
        self._eval_mode()
        self.env.set_is_evaluating()
        if reset_env:
            _ = self.env.reset_all()

        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def _post_evaluate_policy(self):
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _pre_eval_env_step(self, actor_state: dict):
        eval_policy = self.eval_policy[self.env.control_mode]
        actor_obs = actor_state["obs"][self.actor_obs[self.env.control_mode]]
        actions = eval_policy(actor_obs)
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state):
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state

    def _get_inference_policy(self, device=None):
        self._eval_mode()
        if device is not None:
            self.actor[0].to(device)
            self.actor[1].to(device)
        return [self.actor[0].act_inference,
                self.actor[1].act_inference]