import torch
from torch import Tensor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import json
from flask import send_file
from smpl_sim.smpllib.smpl_eval import compute_metrics_lite, compute_error_vel, compute_error_accel
from tqdm import tqdm

from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.agents.ppo.ppo import PPO

class AnalysisPlotMotionTracking(RL_EvalCallback):
    training_loop: PPO
    env: LeggedRobotBase

    def __init__(self, config, training_loop: PPO):
        super().__init__(config, training_loop)
        env: LeggedRobotBase = self.training_loop.env
        self.env = env
        self.policy = self.training_loop._get_inference_policy()
        self.num_envs = self.env.num_envs
        self.use_weblogger = self.config.use_weblogger if hasattr(self.config, 'use_weblogger') else True
        if self.use_weblogger:
            self.logger = WebLogger(self.config.sim_dt)
        self.reset_buffers()
        self.log_single_robot = self.config.log_single_robot
        self.compute_metrics = self.config.compute_metrics

        # if genesis, only print the results of the second iteration of evaluation
        if self.env.config.simulator.config.name == "genesis":
            self.only_print_second_iter = True
        else:
            self.only_print_second_iter = False

        self.iter_count = 0

    def reset_buffers(self):
        self.obs_buf = [[] for _ in range(self.num_envs)]
        self.critic_obs_buf = [[] for _ in range(self.num_envs)]
        self.act_buf = [[] for _ in range(self.num_envs)]

    def on_pre_evaluate_policy(self):
        # Doing this in two lines because of type annotation issues.
        self.robot_num_dofs = self.env.num_dofs
        self.log_dof_pos_limits = self.env.dof_pos_limits.cpu().numpy()
        self.log_dof_vel_limits = self.env.dof_vel_limits.cpu().numpy()
        self.log_dof_torque_limits = self.env.torque_limits.cpu().numpy()
        if self.use_weblogger:
            self.logger.set_robot_limits(self.log_dof_pos_limits, self.log_dof_vel_limits, self.log_dof_torque_limits)
            self.logger.set_robot_num_dofs(self.robot_num_dofs)
        
        if self.compute_metrics:
            self.pbar = tqdm(range(self.env._motion_lib._num_unique_motions // self.env.num_envs))
            self.success_rate = 0
            self.curr_stpes = 0
            self.env.start_compute_metrics()
            self.terminate_state = torch.zeros(self.env.num_envs, device=self.device)
            self.terminate_memory = []
            self.mpjpe, self.mpjpe_all = [], []
            self.gt_pos, self.gt_pos_all = [], []
            self.gt_rot, self.gt_rot_all = [], []
            self.pred_pos, self.pred_pos_all = [], []
            self.pred_rot, self.pred_rot_all = [], []
            
        
        

    def on_post_evaluate_policy(self):
        pass

    def on_pre_eval_env_step(self, actor_state):
        obs: Tensor = actor_state["obs"]["actor_obs"].cpu()
        critic_obs: Tensor = actor_state["obs"]["critic_obs"].cpu()
        actions: Tensor = actor_state["actions"].cpu()
        

        for i in range(self.num_envs):
            self.obs_buf[i].append(obs[i])
            self.critic_obs_buf[i].append(critic_obs[i])
            self.act_buf[i].append(actions[i])

        if not self.use_weblogger:
            return actor_state

        if self.log_single_robot:
            self.logger.log_states(
                {
                'dof_pos_target': actions[0].cpu().numpy(),
                'dof_pos': self.env.dof_pos[0].cpu().numpy(),
                'dof_vel': self.env.dof_vel[0].cpu().numpy(),
                'dof_torque': self.env.torques[0].cpu().numpy(),
                'base_vel_x': self.env.base_lin_vel[0, 0].item(),
                'base_vel_y': self.env.base_lin_vel[0, 1].item(),
                'base_vel_z': self.env.base_lin_vel[0, 2].item(),
                'base_vel_yaw': self.env.base_ang_vel[0, 2].item(),
                'contact_forces_z': self.env.contact_forces[0, self.env.feet_indices, 2].cpu().numpy()
                }
            )
        else:
            # log average of all robots
            self.logger.log_states(
                {
                    'dof_pos_target': actions.mean(dim=0).cpu().numpy(),
                    'dof_pos': self.env.simulator.dof_pos.mean(dim=0).cpu().numpy(),
                    'dof_vel': self.env.simulator.dof_vel.mean(dim=0).cpu().numpy(),
                    'dof_torque': self.env.torques.mean(dim=0).cpu().numpy(),
                    'base_vel_x': self.env.base_lin_vel[:, 0].mean().item(),
                    'base_vel_y': self.env.base_lin_vel[:, 1].mean().item(),
                    'base_vel_z': self.env.base_lin_vel[:, 2].mean().item(),
                    'base_vel_yaw': self.env.base_ang_vel[:, 2].mean().item(),
                    'contact_forces_z': self.env.simulator.contact_forces[:, self.env.feet_indices, 2].mean(dim=1).cpu().numpy()
                }
            )
        return actor_state

    def on_post_eval_env_step(self, actor_state):
        step = actor_state["step"]
        

        if self.compute_metrics:
            self.gt_pos.append(self.env.extras['ref_body_pos_extend'].cpu().numpy())
            self.gt_rot.append(self.env.extras['ref_body_rot_extend'].cpu().numpy())
            
            self.pred_pos.append(self.env._rigid_body_pos_extend.cpu().numpy())
            self.pred_rot.append(self.env._rigid_body_rot_extend.cpu().numpy())
            
            self.mpjpe.append(self.env.dif_global_body_pos.norm(dim=-1).cpu() * 1000)
            
            
            died = actor_state['dones']
            died[actor_state['extras']["time_outs"]] = False
            
            termination_state = torch.logical_and(self.curr_stpes <= self.env._motion_lib.get_motion_num_steps() - 1, died) # if terminate after the last frame, then it is not a termination. curr_step is one step behind simulation. 
            self.terminate_state = torch.logical_or(termination_state, self.terminate_state)


            if (~self.terminate_state).sum() > 0:
                max_possible_id = self.env._motion_lib._num_unique_motions - 1
                curr_ids = self.env._motion_lib._curr_motion_ids
                if (max_possible_id == curr_ids).sum() > 0: # When you are running out of motions. 
                    bound = (max_possible_id == curr_ids).nonzero()[0] + 1
                    if (~self.terminate_state[:bound]).sum() > 0:
                        curr_max = self.env._motion_lib.get_motion_num_steps()[:bound][~self.terminate_state[:bound]].max()
                    else:
                        curr_max = (self.curr_stpes - 1)  # the ones that should be counted have teimrated
                else:
                    curr_max = self.env._motion_lib.get_motion_num_steps()[~self.terminate_state].max()

                if self.curr_stpes >= curr_max: curr_max = self.curr_stpes + 1  # For matching up the current steps and max steps. 
            else:
                curr_max = self.env._motion_lib.get_motion_num_steps().max()
                
            self.curr_stpes += 1
            if self.curr_stpes >= curr_max or self.terminate_state.sum() == self.env.num_envs:
                
                self.terminate_memory.append(self.terminate_state.cpu().numpy())
                self.success_rate = (1 - np.concatenate(self.terminate_memory)[: self.env._motion_lib._num_unique_motions].mean())

                # MPJPE
                all_mpjpe = torch.stack(self.mpjpe)
                try:
                    assert(all_mpjpe.shape[0] == curr_max or self.terminate_state.sum() == self.env.num_envs) # Max should be the same as the number of frames in the motion.
                except:
                    import ipdb; ipdb.set_trace()
                    print('??')

                all_body_pos_pred = np.stack(self.pred_pos)
                all_body_pos_gt = np.stack(self.gt_pos)
                all_body_rot_pred = np.stack(self.pred_rot)
                all_body_rot_gt = np.stack(self.gt_rot)
                
                all_mpjpe = [all_mpjpe[: (i - 1), idx].mean() for idx, i in enumerate(self.env._motion_lib.get_motion_num_steps())] # -1 since we do not count the first frame. 
                all_body_pos_pred = [all_body_pos_pred[: (i - 1), idx] for idx, i in enumerate(self.env._motion_lib.get_motion_num_steps())]
                all_body_pos_gt = [all_body_pos_gt[: (i - 1), idx] for idx, i in enumerate(self.env._motion_lib.get_motion_num_steps())]
                all_body_rot_pred = [all_body_rot_pred[: (i - 1), idx] for idx, i in enumerate(self.env._motion_lib.get_motion_num_steps())]
                all_body_rot_gt = [all_body_rot_gt[: (i - 1), idx] for idx, i in enumerate(self.env._motion_lib.get_motion_num_steps())]


                self.mpjpe_all.append(all_mpjpe)
                self.pred_pos_all += all_body_pos_pred
                self.gt_pos_all += all_body_pos_gt
                self.pred_rot_all += all_body_rot_pred
                self.gt_rot_all += all_body_rot_gt

                if (self.env.start_idx + self.env.num_envs >= self.env._motion_lib._num_unique_motions):
                    terminate_hist = np.concatenate(self.terminate_memory)
                    succ_idxes = np.nonzero(~terminate_hist[: self.env._motion_lib._num_unique_motions])[0].tolist()

                    pred_pos_all_succ = [(self.pred_pos_all[:self.env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
                    gt_pos_all_succ = [(self.gt_pos_all[: self.env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
                    pred_rot_all_succ = [(self.pred_rot_all[: self.env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
                    gt_rot_all_succ = [(self.gt_rot_all[: self.env._motion_lib._num_unique_motions])[i] for i in succ_idxes]

                    pred_pos_all = self.pred_pos_all[:self.env._motion_lib._num_unique_motions]
                    gt_pos_all = self.gt_pos_all[: self.env._motion_lib._num_unique_motions]
                    pred_rot_all = self.pred_rot_all[: self.env._motion_lib._num_unique_motions]
                    gt_rot_all = self.gt_rot_all[: self.env._motion_lib._num_unique_motions]
                    
                    # np.sum([i.shape[0] for i in self.pred_pos_all[:self.env._motion_lib._num_unique_motions]])
                    # self.env._motion_lib.get_motion_num_steps().sum()

                    failed_keys = self.env._motion_lib._motion_data_keys[terminate_hist[: self.env._motion_lib._num_unique_motions]]
                    success_keys = self.env._motion_lib._motion_data_keys[~terminate_hist[: self.env._motion_lib._num_unique_motions]]
                    # print("failed", self.env._motion_lib._motion_data_keys[np.concatenate(self.terminate_memory)[:self.env._motion_lib._num_unique_motions]])
                    
                    # metrics = compute_metrics_lite(pred_pos_all, gt_pos_all, pred_rot_all, gt_rot_all)
                    # metrics_succ = compute_metrics_lite(pred_pos_all_succ, gt_pos_all_succ, pred_rot_all_succ, gt_rot_all_succ)
                    

                    
                    metrics = compute_metrics_lite(pred_pos_all, gt_pos_all)
                   

                    
                    


                    metrics_succ = compute_metrics_lite(pred_pos_all_succ, gt_pos_all_succ)

                    metrics_all_print = {m: np.mean(v) for m, v in metrics.items()}
                    metrics_print = {m: np.mean(v) for m, v in metrics_succ.items()}

                    if self.only_print_second_iter:
                        if self.iter_count == 1:
                            print("------------------------------------------")
                            print("------------------------------------------")

                            # ------------------------------- SAVE FOR PLOTTING -------------------------------
                            # npz_metrics = {}
                            # npz_metrics['mpjpe_g'] = np.mean(metrics['mpjpe_g'], axis=1, keepdims=True)
                            # npz_metrics['mpjpe_l'] = np.mean(metrics['mpjpe_l'], axis=1, keepdims=True)
                            # npz_metrics['mpjpe_pa'] = np.mean(metrics['mpjpe_pa'], axis=1, keepdims=True)
                            # npz_metrics['accel_dist'] = metrics['accel_dist']
                            # npz_metrics['vel_dist'] = metrics['vel_dist']
                            # import ipdb; ipdb.set_trace()
                            # np.savez(f'TairanTestbed_0109_eval_g129dof_anneal_23dof_checkpoints_20250112_1853/horizon_minimalnorm_ablation/isaacsim_4300_shoot_level1_horizon1.0_norm0.1.npz', **npz_metrics)
                            
                            # ------------------------------- SAVE FOR PLOTTING -------------------------------


                            print(f"Success Rate: {self.success_rate:.10f}")
                            print("All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]))
                            print("Succ: "," \t".join([f"{k}: {v:.3f}" for k, v in metrics_print.items()]))
                            
                            exit(0)
                    else:
                        # ------------------------------- SAVE FOR PLOTTING -------------------------------
                        # npz_metrics = {}
                        # npz_metrics['mpjpe_g'] = np.mean(metrics['mpjpe_g'], axis=1, keepdims=True)
                        # npz_metrics['mpjpe_l'] = np.mean(metrics['mpjpe_l'], axis=1, keepdims=True)
                        # npz_metrics['mpjpe_pa'] = np.mean(metrics['mpjpe_pa'], axis=1, keepdims=True)
                        # npz_metrics['accel_dist'] = metrics['accel_dist']
                        # npz_metrics['vel_dist'] = metrics['vel_dist']

                        # print("------------------------------------------")
                        # print("------------------------------------------")
                        print(f"Success Rate: {self.success_rate:.10f}")
                        print("All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]))
                        print("Succ: "," \t".join([f"{k}: {v:.3f}" for k, v in metrics_print.items()]))
                        import ipdb; ipdb.set_trace()
                        # np.savez(f'TairanTestbed_0109_eval_g129dof_anneal_23dof_checkpoints_20250112_1853/horizon_minimalnorm_ablation/isaacsim_4300_shoot_level1_horizon0.5_norm0.1.npz', **npz_metrics)
                        # ------------------------------- SAVE FOR PLOTTING -------------------------------

                        

                        
                        exit(0)
                    
                    self.iter_count += 1

                    
 
                # self.env.forward_motion_samples()
                # self.terminate_state = torch.zeros(
                #     self.lab_env.num_envs, device=self.device
                # )

                self.pbar = tqdm(range(self.env._motion_lib._num_unique_motions // self.env.num_envs))
                self.success_rate = 0
                self.curr_stpes = 0
                self.env.start_compute_metrics()
                self.terminate_state = torch.zeros(self.env.num_envs, device=self.device)
                self.terminate_memory = []
                self.mpjpe, self.mpjpe_all = [], []
                self.gt_pos, self.gt_pos_all = [], []
                self.gt_rot, self.gt_rot_all = [], []
                self.pred_pos, self.pred_pos_all = [], []
                self.pred_rot, self.pred_rot_all = [], []

                self.pbar.update(1)
                self.pbar.refresh()
                self.mpjpe, self.gt_pos, self.pred_pos,  = [], [], []
                self.curr_stpes = 0
                    
            update_str = f"Terminated: {self.terminate_state.sum().item()} | max frames: {curr_max} | steps {self.curr_stpes} | Start: {self.env.start_idx} | Succ rate: {self.success_rate:.3f} | Mpjpe: {np.mean(self.mpjpe_all) * 1000:.3f}"
            self.pbar.set_description(update_str)
                    
        
        if self.use_weblogger and (step + 1) % self.config.plot_update_interval == 0:
            self.logger.plot_states()
        return actor_state

class WebLogger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.thread = None

    def set_robot_limits(self, dof_pos_limits, dof_vel_limits, dof_torque_limits):
        self.log_dof_pos_limits = dof_pos_limits
        self.log_dof_vel_limits = dof_vel_limits
        self.log_dof_torque_limits = dof_torque_limits

    def set_robot_num_dofs(self, num_dofs):
        self.robot_num_dofs = num_dofs

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run_server)
            self.thread.start()
        self._update_plot()
    
    def _run_server(self):
        @self.app.route('/')
        def index():
            return send_file('analysis_plot_template.html')

        self.socketio.run(self.app, debug=False, use_reloader=False)
    
    def _update_plot(self):
        log = self.state_log.copy()
        
        total_time = len(next(iter(log.values()))) * self.dt
        time = np.linspace(0, total_time, len(next(iter(log.values()))))

        for key in log:
            if isinstance(log[key], list):
                log[key] = np.array(log[key])

        BLUE = '#005A9D'
        RED = '#DA2513'
        YELLOW = '#EEDE70'

        num_dofs = self.robot_num_dofs

        def get_subplot_titles():
            titles = [
                'Base velocity x', 'Base velocity y', 'Base velocity yaw', 'Base velocity z',
                'Vertical Contact forces', 'Vertical Contact forces', 'Vertical Contact forces', 'Vertical Contact forces'
            ]
            for i in range(num_dofs):
                titles.extend([f'DOF {i} Position', f'DOF {i} Velocity', f'DOF {i} Torque', f'DOF {i} Torque/Velocity'])
            return titles

        # Calculate number of rows needed
        num_rows = 2 + num_dofs

        fig = make_subplots(rows=num_rows, cols=4, subplot_titles=get_subplot_titles())

        def add_trace(x, y, color, row, col, name=None, show_legend=False):
            fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=color), name=name, showlegend=show_legend), row=row, col=col)

        # Base velocities and commands
        add_trace(time, log["base_vel_x"], BLUE, 1, 1, "Base vel x")
        add_trace(time, log["base_vel_y"], BLUE, 1, 2, "Base vel y")
        add_trace(time, log["base_vel_yaw"], BLUE, 1, 3, "Base vel yaw")
        add_trace(time, log["base_vel_z"], BLUE, 1, 4, "Base vel z")

        # Vertical Contact forces
        forces = log["contact_forces_z"]
        add_trace(time, forces[:,0], BLUE, 2, 1, "Force Left")

        # Command velocities

        def add_limit_lines(row, col, lower, upper, color=YELLOW):
            fig.add_shape(type="rect", x0=time[0], x1=time[-1], y0=lower, y1=upper,
                        fillcolor=color, line=dict(width=0), layer='below', row=row, col=col)

        # DOF Positions, Velocities, and Torques
        for i in range(num_dofs):
            row = i + 3  # Start from the third row

            # Position
            add_trace(time, [pos[i] for pos in log["dof_pos"]], BLUE, row, 1, f"DOF {i} pos")
            add_trace(time, [pos[i] for pos in log["dof_pos_target"]], RED, row, 1, f"DOF {i} pos target")
            add_limit_lines(row, 1, self.log_dof_pos_limits[i, 0], self.log_dof_pos_limits[i, 1])
            # Velocity
            add_trace(time, [vel[i] for vel in log["dof_vel"]], BLUE, row, 2, f"DOF {i} vel")
            add_limit_lines(row, 2, -self.log_dof_vel_limits[i], self.log_dof_vel_limits[i])
            # Torque
            add_trace(time, [torque[i] for torque in log["dof_torque"]], BLUE, row, 3, f"DOF {i} torque")
            add_limit_lines(row, 3, -self.log_dof_torque_limits[i], self.log_dof_torque_limits[i])
            
            # Torque/Velocity curve
            fig.add_trace(go.Scatter(
                x=[vel[i] for vel in log["dof_vel"]], 
                y=[torque[i] for torque in log["dof_torque"]], 
                mode='markers', 
                marker=dict(color=BLUE, size=2), 
                showlegend=False,
                name=f"DOF {i} Torque/Velocity"
            ), row=row, col=4)

            # Add velocity limits
            fig.add_shape(type="line", x0=-self.log_dof_vel_limits[i], y0=-self.log_dof_torque_limits[i], 
                        x1=-self.log_dof_vel_limits[i], y1=self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)
            fig.add_shape(type="line", x0=self.log_dof_vel_limits[i], y0=-self.log_dof_torque_limits[i], 
                        x1=self.log_dof_vel_limits[i], y1=self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)

            # Add torque limits
            fig.add_shape(type="line", x0=-self.log_dof_vel_limits[i], y0=-self.log_dof_torque_limits[i], 
                        x1=self.log_dof_vel_limits[i], y1=-self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)
            fig.add_shape(type="line", x0=-self.log_dof_vel_limits[i], y0=self.log_dof_torque_limits[i], 
                        x1=self.log_dof_vel_limits[i], y1=self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)

        fig.update_layout(height=300*num_rows, width=1500, title_text="Robot State Plots", showlegend=True)
        
        # Update x and y axis labels
        for i in range(num_rows):
            for j in range(3):
                fig.update_xaxes(title_text="time [s]", row=i+1, col=j+1)
            fig.update_xaxes(title_text="", row=i+1, col=4)
        
        fig.update_yaxes(title_text="base lin vel [m/s]", row=1, col=1)
        fig.update_yaxes(title_text="base lin vel [m/s]", row=1, col=2)
        fig.update_yaxes(title_text="base ang vel [rad/s]", row=1, col=3)
        fig.update_yaxes(title_text="base lin vel [m/s]", row=1, col=4)
        fig.update_yaxes(title_text="Forces z [N]", row=2, col=1)


        for i in range(3, num_rows + 1):
            fig.update_yaxes(title_text="Position [rad]", row=i, col=1)
            fig.update_yaxes(title_text="Velocity [rad/s]", row=i, col=2)
            fig.update_yaxes(title_text="Torque [Nm]", row=i, col=3)
            fig.update_yaxes(title_text="Torque/Velocity", row=i, col=4)

        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        self.socketio.emit('update_plots', plot_json)

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.thread:
            self.socketio.stop()
            self.thread.join()