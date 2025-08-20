# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
""" 
    Modified based on track_agileVer3.py
    Do not use isaac gym anymore
"""
import numpy as np
import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch.utils.data import DataLoader
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_euler_angles, euler_angles_to_matrix, matrix_to_quaternion, axis_angle_to_matrix


from isaacgym.torch_utils import *
from .track_agile_config import TrackAgileCfg
from Zagreus.utils.helpers import asset_class_to_AssetOptions
from Zagreus.utils.mymath import rand_circle_point
import torch.nn.functional as F
from Zagreus.config import ROOT_DIR

class TrackAgileVer4():

    def __init__(self, cfg: TrackAgileCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg

        self.max_episode_length = int(self.cfg.env.episode_length_s / self.cfg.sim.dt)
        self.max_len_sample = self.cfg.env.max_sample_length

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device_id = sim_device
        self.headless = headless

        self.num_envs = self.cfg.env.num_envs

        self.device = sim_device
        self.tar_acc_norm = 1
        self.tar_acc_intervel = 100 # How many time steps will acceleration change once
        self.tar_acc = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)

        self.count_step = torch.zeros((self.num_envs, ), dtype=torch.long, device=self.device)
        self.initial_vector = torch.tensor([[1.0, 0.0, 0.0]] * self.num_envs, device=self.device)

        self.tar_state = torch.zeros((self.num_envs, 12), device=self.device)
        self.quad_state = torch.zeros((self.num_envs, 12), device=self.device)

        self.reset_buf = torch.zeros((self.num_envs, ), dtype=torch.long, device=self.device)

        self.counter = 0

    def step(self, new_states):


        self.pre_physics_step(new_states)
        self.count_step += 1

        
        return self.get_quad_state(), self.get_tar_state()
    

    def reset(self, reset_buf=None):
        """ Reset all robots"""
        if reset_buf is None:
            reset_idx = torch.arange(self.num_envs, device=self.device)
        else:
            reset_idx = torch.nonzero(reset_buf).squeeze(-1)

        if len(reset_idx):
            self.set_reset_idx(reset_idx)

        return self.get_quad_state()

    def set_reset_idx(self, env_ids):
        num_resets = len(env_ids)
        self.count_step[env_ids] = 0

        # reset position
        self.tar_state[env_ids, 0] = 2
        self.tar_state[env_ids, 1] = 0
        self.tar_state[env_ids, 2] = 2

        # reset linevels
        self.tar_state[env_ids, 6:9] = 0
        # self.tar_state[env_ids, 7:10] = self.tar_traj[env_ids, self.count_step[env_ids], 6:9]
        # reset angvels
        self.tar_state[env_ids, 9:12] = 0
        # reset euler
        self.tar_state[env_ids, 3:6] = 0

        self.tar_acc_norm = torch.rand(num_resets, device=self.device) * 2
        # self.tar_acc_norm = 1
        self.tar_acc[env_ids] = rand_circle_point(num_resets, self.tar_acc_norm, self.device)
        # reset position
        self.quad_state[env_ids, 0:3] = 0
        self.quad_state[env_ids, 2] = 2
        self.quad_state[env_ids, :3] -= torch.tensor([0.21, 0, 0.05], device=self.device)
        # reset linevels
        self.quad_state[env_ids, 6:9] = 0
        # reset angvels
        self.quad_state[env_ids, 9:12] = 0
        # reset euler
        self.quad_state[env_ids, 3:6] = 0


        self.reset_buf[env_ids] = 0


    def pre_physics_step(self, tar_state):
        # resets
        if self.counter % 250 == 0:
            print("self.counter:", self.counter)
        self.counter += 1

        self.quad_state = tar_state.clone()


        inv_acc_idx = torch.nonzero((self.count_step % self.tar_acc_intervel) == 0).squeeze(-1)
        self.tar_acc[inv_acc_idx] *= -1
        inv_acc_idx = torch.nonzero((self.count_step %( self.tar_acc_intervel * 2)) == 0).squeeze(-1)
        self.tar_acc[inv_acc_idx] *= -1
        # change_acc_idx = torch.nonzero(((self.count_step % (self.tar_acc_intervel * 4)) == 0)).squeeze(-1)

        # set position
        self.tar_state[:, 2] = 2
        # set linearvels
        self.tar_state[:, 7:9] += self.tar_acc * self.cfg.sim.dt
        self.tar_state[:, 9] = 0
        # set angvels
        self.tar_state[:, 10:13] = 0
        # set quats
        self.tar_state[:, 3:7] = 0
        self.tar_state[:, 6] = 1



    def qua2euler(self, qua):
        rotation_matrices = quaternion_to_matrix(
            qua[:, [3, 0, 1, 2]])
        euler_angles = matrix_to_euler_angles(
            rotation_matrices, "ZYX")[:, [2, 1, 0]]
        return euler_angles

    def euler2qua(self, euler):
        rotation_matrices = euler_angles_to_matrix(euler, "ZYX")
        qua = matrix_to_quaternion(rotation_matrices)[:, [3, 2, 1, 0]]
        return qua
    
    def get_quad_state(self):
        return self.quad_state.detach()

    def get_tar_state(self):
        return self.tar_state.detach()

    
    def check_out_space(self):
        ones = torch.ones_like(self.reset_buf)
        out_space = torch.zeros_like(self.reset_buf)
        obs = self.quad_state.clone()
        
        out_space = torch.where(torch.logical_or(obs[:, 0] > 10, obs[:, 0] < -10), ones, out_space)
        out_space = torch.where(torch.logical_or(obs[:, 1] > 10, obs[:, 1] < -10), ones, out_space)
        out_space = torch.where(torch.logical_or(obs[:, 2] > 10, obs[:, 2] < 0), ones, out_space)
        return out_space

        
    def check_reset_out(self):

        ones = torch.ones_like(self.reset_buf)
        out_space = torch.zeros_like(self.reset_buf)
        obs = self.quad_state.clone()
        out_space = torch.where(torch.logical_or(obs[:, 0] > 15, obs[:, 0] < -15), ones, out_space)
        out_space = torch.where(torch.logical_or(obs[:, 1] > 15, obs[:, 1] < -15), ones, out_space)
        out_space = torch.where(torch.logical_or(obs[:, 2] > 15, obs[:, 2] < 0), ones, out_space)
        out_space = torch.where(torch.any(torch.isnan(obs[:, :3]), dim=1).bool(), ones, out_space)
        out_space_idx = torch.nonzero(out_space).squeeze(-1)
        reset_buf = out_space
        reset_idx = torch.nonzero(reset_buf).squeeze(-1)
        
        
        return reset_buf, reset_idx
