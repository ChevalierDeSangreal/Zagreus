import numpy as np
import torch
import os
import json
from pathlib import Path
import torch.nn as nn
from .learnable_controller import DroneBodyRateController

class MyDynamics(nn.Module):

    def __init__(self, modified_params={}, device='cpu'):
        """
        Initialize quadrotor dynamics
        Args:
            modified_params (dict, optional): dynamic mismatch. Defaults to {}.
        """
        super().__init__()
        with open(
            os.path.join(Path(__file__).parent.absolute(), "config_quad_iris.json"),
            "r"
        ) as infile:
            self.cfg = json.load(infile)

        # update with modified parameters
        self.cfg.update(modified_params)

        torch.cuda.set_device(device)
        self.device = device
        

        # NUMPY PARAMETERS
        self.mass = self.cfg["mass"]
        self.inertia_vector = np.array(self.cfg["inertia"])

        self.torch_gravity = torch.tensor(self.cfg["gravity"]).to(device)
        self.torch_inertia_vector = torch.from_numpy(self.inertia_vector).float().to(device)
        self.torch_inertia_J = torch.diag(self.torch_inertia_vector).to(device)
        self.torch_inertia_J_inv = torch.diag(1 / self.torch_inertia_vector).to(device)


        self.force_weight = torch.nn.Parameter(torch.ones(1, device=device))
        self.force_bias = torch.nn.Parameter(torch.zeros(1, device=device))
        self.rotate_weight = torch.nn.Parameter(torch.ones(3, device=device))

        self.torch_translational_drag = torch.nn.Parameter(
            torch.tensor(self.cfg["translational_drag"], dtype=torch.float32, device=device)
        )



    @staticmethod
    def world_to_body_matrix(attitude):
        """
        Creates a transformation matrix for directions from world frame
        to body frame for a body with attitude given by `euler` Euler angles.
        :param euler: The Euler angles of the body frame.
        :return: The transformation matrix.
        """

        # check if we have a cached result already available
        roll = attitude[:, 0]
        pitch = attitude[:, 1]
        yaw = attitude[:, 2]

        Cy = torch.cos(yaw)
        Sy = torch.sin(yaw)
        Cp = torch.cos(pitch)
        Sp = torch.sin(pitch)
        Cr = torch.cos(roll)
        Sr = torch.sin(roll)

        # create matrix
        m1 = torch.transpose(torch.vstack([Cy * Cp, Sy * Cp, -Sp]), 0, 1)
        m2 = torch.transpose(
            torch.vstack(
                [Cy * Sp * Sr - Cr * Sy, Cr * Cy + Sr * Sy * Sp, Cp * Sr]
            ), 0, 1
        )
        m3 = torch.transpose(
            torch.vstack(
                [Cy * Sp * Cr + Sr * Sy, Cr * Sy * Sp - Cy * Sr, Cr * Cp]
            ), 0, 1
        )
        matrix = torch.stack((m1, m2, m3), dim=1)

        return matrix

    @staticmethod
    def to_euler_matrix(attitude):
        # attitude is [roll, pitch, yaw]
        pitch = attitude[:, 1]
        roll = attitude[:, 0]
        Cp = torch.cos(pitch)
        Sp = torch.sin(pitch)
        Cr = torch.cos(roll)
        Sr = torch.sin(roll)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        zero_vec_bs = torch.zeros(Sp.size()).to(device)
        ones_vec_bs = torch.ones(Sp.size()).to(device)

        # create matrix
        m1 = torch.transpose(
            torch.vstack([ones_vec_bs, zero_vec_bs, -Sp]), 0, 1
        )
        m2 = torch.transpose(torch.vstack([zero_vec_bs, Cr, Cp * Sr]), 0, 1)
        m3 = torch.transpose(torch.vstack([zero_vec_bs, -Sr, Cp * Cr]), 0, 1)
        matrix = torch.stack((m1, m2, m3), dim=1)

        # matrix = torch.tensor([[1, 0, -Sp], [0, Cr, Cp * Sr], [0, -Sr, Cp * Cr]])
        return matrix

    @staticmethod
    def euler_rate(attitude, angular_velocity):
        euler_matrix = MyDynamics.to_euler_matrix(attitude)
        together = torch.matmul(
            euler_matrix, torch.unsqueeze(angular_velocity.float(), 2)
        )
        # print("output euler rate", together.size())
        return torch.squeeze(together)

class LearnableDynamics(MyDynamics):

    def __init__(self, num_env=1, dt=0.02, controller_dt=1e-3, device='cpu'):
        super().__init__(modified_params={}, device=device)
        self.device = device
        self.dt = controller_dt # Simulator will be num_control_per_step times, each simulate takes controller_dt s
        self.num_env = num_env
        self.num_control_per_step = dt / controller_dt
        self.controller = DroneBodyRateController(num_envs=num_env, dt=controller_dt, device=device).to(device)

    def linear_dynamics(self, force, attitude, velocity):
        """
        linear dynamics
        """

        world_to_body = self.world_to_body_matrix(attitude)
        body_to_world = torch.transpose(world_to_body, 1, 2)

        # print("force in ld ", force.size())
        thrust = 1 / self.mass * torch.matmul(
            body_to_world, torch.unsqueeze(force, 2)
        )
        drag = - self.torch_translational_drag * velocity
        thrust_min_grav = (
            thrust[:, :, 0] + self.torch_gravity + drag
        )
        return thrust_min_grav

    def run_flight_control(self, action, av, current_vel, current_attitude):
        """
        action: (num_env, 4) tensor containing [thrust, body_rates], in [-1, 1]
        av: (num_env, 3) current angular velocity
        """
        force, body_torque_des = self.controller(action, av, current_vel, current_attitude)
        # print(force.shape, body_torque_des.shape)
        thrust_and_torque = torch.unsqueeze(
            torch.cat((force, body_torque_des), dim=1), 2
        )
        return thrust_and_torque[:, :, 0]

    def __call__(self, action, state):
        for _ in range(int(self.num_control_per_step)):
            state, _ = self.simulate_quadrotor(action, state)
        return state

    def detach_controller(self):
        self.controller.detach_state()

    def reset_controller(self):
        self.controller.reset()

    def simulate_quadrotor(self, action, state):
        """
        Pytorch implementation of the dynamics in Flightmare simulator
        """
        # extract state
        position = state[:, :3]
        attitude = state[:, 3:6]
        velocity = state[:, 6:9]
        angular_velocity = state[:, 9:]

        action_scaled = action.clone()
        action_scaled[:, 0] = (action[:, 0] * 2 - 1) * self.force_weight + self.force_bias
        action_scaled[:, 1:] = (action[:, 1:] / 4) * self.rotate_weight
        # print("After delay", action)
        # ctl_dt ist simulation time,
        # remainer wird immer -sim_dt gemacht in jedem loop
        # precompute cross product (batch, 3, 1)
        inertia_av = torch.matmul(
            self.torch_inertia_J.to(self.device), torch.unsqueeze(angular_velocity, 2)
        )[:, :, 0]
        cross_prod = torch.cross(angular_velocity, inertia_av, dim=1)

        force_torques = self.run_flight_control(
            action_scaled, angular_velocity, velocity, attitude
        ).to(self.device)
        # 2) angular acceleration
        tau = force_torques[:, 3:]
        torch_inertia_J_inv = torch.inverse(self.torch_inertia_J.to(self.device))
        angular_acc = torch.matmul(
            torch_inertia_J_inv.to(self.device), torch.unsqueeze((tau - cross_prod).to(self.device), 2)
        )[:, :, 0]
        new_angular_velocity = angular_velocity + self.dt * angular_acc


        attitude = attitude * 0.93 + attitude.detach() * 0.07 + self.dt * self.euler_rate(attitude, new_angular_velocity)

        # 1) linear dynamics
        force = force_torques[:, :3]  # (num_envs, 3)

        # print("Force:", force[0])
        # print("Attitude:", attitude[0])
        # print("Velocity:", velocity[0])
        acceleration = self.linear_dynamics(force, attitude, velocity)

        position = (
            position * 0.93 + position.detach() * 0.07 + 0.5 * self.dt * self.dt * acceleration + self.dt * velocity
        )
        velocity = velocity * 0.93 + velocity.detach() * 0.07 + self.dt * acceleration



        # set final state
        state = torch.hstack(
            (position, attitude, velocity, new_angular_velocity)
        )
        return state.float(), acceleration
        # return state.float()
