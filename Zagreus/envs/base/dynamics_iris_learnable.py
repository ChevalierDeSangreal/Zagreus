import numpy as np
import torch
import os
import json
from pathlib import Path
import torch.nn as nn

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
        self.arm_length = 0.2
        self.inertia_vector = np.array(self.cfg["inertia"])

        self.torch_gravity = torch.tensor(self.cfg["gravity"]).to(device)
        self.torch_inertia_vector = torch.from_numpy(self.inertia_vector).float().to(device)
        self.torch_inertia_J = torch.diag(self.torch_inertia_vector).to(device)
        self.torch_inertia_J_inv = torch.diag(1 / self.torch_inertia_vector).to(device)

        self.prev_rotate = torch.zeros(3, device=device)
        self.prev_translate = torch.zeros(1, device=device)


        # learnable parameters
        self.tau_rotate = torch.nn.Parameter(torch.tensor(self.cfg["tau_rotate"], dtype=torch.float32, device=device))
        self.tau_translate = torch.nn.Parameter(torch.tensor(self.cfg["tau_translate"], dtype=torch.float32, device=device))

        self.kinv_ang_vel_tau = torch.nn.Parameter(
            torch.tensor(self.cfg["kinv_ang_vel_tau"], dtype=torch.float32, device=device)
        )
        self.torch_kinv_ang_vel_tau = torch.diag(self.kinv_ang_vel_tau)

        self.torch_rotational_drag = torch.nn.Parameter(
            torch.tensor(self.cfg["rotational_drag"], dtype=torch.float32, device=device)
        )

        self.torch_translational_drag = torch.nn.Parameter(
            torch.tensor(self.cfg["translational_drag"], dtype=torch.float32, device=device)
        )

        self.force_weight = torch.nn.Parameter(torch.ones(1, device=device))
        self.force_bias = torch.nn.Parameter(torch.zeros(1, device=device))


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

class IrisDynamics(MyDynamics):

    def __init__(self, modified_params={}, device='cpu'):
        super().__init__(modified_params=modified_params, device=device)
        self.device = device

    def apply_rotate_delay(self, rotate, dt):
        """
        Apply first-order lag to simulate actuator delay
        """
        alpha = dt / (self.tau_rotate + dt)
        self.prev_rotate = (1 - alpha) * self.prev_rotate + alpha * rotate
        return self.prev_rotate

    def apply_translate_delay(self, translate, dt):
        alpha = dt / (self.tau_translate + dt)
        self.prev_translate = (1 - alpha) * self.prev_translate + alpha * translate
        return self.prev_translate
    
    def reset_action(self):
        self.prev_rotate = torch.zeros(3, device=self.device)
        self.prev_translate = torch.zeros(1, device=self.device)

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

    def run_flight_control(self, thrust, av, body_rates, cross_prod):
        """
        thrust: command first signal (around 9.81)
        omega = av: current angular velocity
        command = body_rates: body rates in command
        """
        force = torch.unsqueeze(thrust, 1)

        # constants
        omega_change = torch.unsqueeze(body_rates - av, 2)
        kinv_times_change = torch.matmul(
            self.torch_kinv_ang_vel_tau.to(self.device), omega_change.to(self.device)
        )
        first_part = torch.matmul(self.torch_inertia_J.to(self.device), kinv_times_change.to(self.device))
        # print("first_part", first_part.size())
        rotational_drag_torque = - self.torch_rotational_drag * av
        body_torque_des = (
            first_part[:, :, 0] + cross_prod + rotational_drag_torque
        )
        # print(force.shape, body_torque_des.shape)
        thrust_and_torque = torch.unsqueeze(
            torch.cat((force, body_torque_des), dim=1), 2
        )
        return thrust_and_torque[:, :, 0]

    def _pretty_print(self, varname, torch_var):
        np.set_printoptions(suppress=1, precision=7)
        if len(torch_var) > 1:
            print("ERR: batch size larger 1", torch_var.size())
        print(varname, torch_var[0].detach().numpy())

    def __call__(self, action, state, dt):
        return self.simulate_quadrotor(action, state, dt)
        

    def simulate_quadrotor(self, action, state, dt):
        """
        Pytorch implementation of the dynamics in Flightmare simulator
        """
        # extract state
        position = state[:, :3]
        attitude = state[:, 3:6]
        velocity = state[:, 6:9]
        angular_velocity = state[:, 9:]

        # action is normalized between 0 and 1 --> rescale
        # print(action.shape)
        total_thrust = - action[:, 0] * self.mass * (self.torch_gravity[2]) * self.force_weight + self.force_bias
        # print(total_thrust.shape)
        # total_thrust = action[:, 0] * 7.5 + self.mass * (-self.torch_gravity[2])
        body_rates = action[:, 1:]

        total_thrust = self.apply_translate_delay(total_thrust, dt)
        body_rates = self.apply_rotate_delay(body_rates, dt)
        # print("After delay", action)
        # ctl_dt ist simulation time,
        # remainer wird immer -sim_dt gemacht in jedem loop
        # precompute cross product (batch, 3, 1)
        inertia_av = torch.matmul(
            self.torch_inertia_J.to(self.device), torch.unsqueeze(angular_velocity, 2)
        )[:, :, 0]
        cross_prod = torch.cross(angular_velocity, inertia_av, dim=1)

        force_torques = self.run_flight_control(
            total_thrust, angular_velocity, body_rates, cross_prod
        ).to(self.device)

        # 2) angular acceleration
        tau = force_torques[:, 1:]
        torch_inertia_J_inv = torch.inverse(self.torch_inertia_J.to(self.device))
        angular_acc = torch.matmul(
            torch_inertia_J_inv.to(self.device), torch.unsqueeze((tau - cross_prod).to(self.device), 2)
        )[:, :, 0]
        new_angular_velocity = angular_velocity + dt * angular_acc


        attitude = attitude * 0.93 + attitude.detach() * 0.07 + dt * self.euler_rate(attitude, new_angular_velocity)

        # 1) linear dynamics
        force_expanded = torch.unsqueeze(force_torques[:, 0], 1)
        f_s = force_expanded.size()
        force = torch.cat(
            (torch.zeros(f_s).to(self.device), torch.zeros(f_s).to(self.device), force_expanded), dim=1
        )

        acceleration = self.linear_dynamics(force, attitude, velocity)

        position = (
            position * 0.93 + position * 0.07 + 0.5 * dt * dt * acceleration + dt * velocity
        )
        velocity = velocity * 0.93 + velocity * 0.07 + dt * acceleration



        # set final state
        state = torch.hstack(
            (position, attitude, velocity, new_angular_velocity)
        )
        return state.float(), acceleration
        # return state.float()