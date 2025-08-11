import os
import random
import time

import isaacgym  # noqa
from isaacgym import gymutil
from isaacgym.torch_utils import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from pytorch3d.transforms import euler_angles_to_matrix


import pytz
from datetime import datetime
import sys

from Zagreus.config import ROOT_DIR
from Zagreus.utils import AgileLoss, agile_lossVer7
from Zagreus.models import TrackAgileModuleVer11
from Zagreus.envs import IrisDynamics, task_registry
# os.path.basename(__file__).rstrip(".py")

import asyncio

from mavsdk import System
from mavsdk.offboard import (AttitudeRate, OffboardError)
from mavsdk.offboard import PositionNedYaw

"""
To test train_trackagileVer1.py
"""

def get_args():
	custom_parameters = [
		{"name": "--task", "type": str, "default": "track_agileVer3", "help": "The name of the task."},
		{"name": "--experiment_name", "type": str, "default": "test_trackagileVer6_onboard", "help": "Name of the experiment to run or load."},
		{"name": "--headless", "action": "store_true", "help": "Force display off at all times"},
		{"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
		{"name": "--num_envs", "type": int, "default": 2, "help": "Number of environments to create. Batch size will be equal to this"},
		{"name": "--seed", "type": int, "default": 42, "help": "Random seed. Overrides config file if provided."},

		# train setting
		{"name": "--learning_rate", "type":float, "default": 1.6e-4,
			"help": "the learning rate of the optimizer"},
		{"name": "--batch_size", "type":int, "default": 2,
			"help": "batch size of training. Notice that batch_size should be equal to num_envs"},
		{"name": "--num_worker", "type":int, "default": 4,
			"help": "num worker of dataloader"},
		{"name": "--num_epoch", "type":int, "default": 500,
			"help": "num of epoch"},
		{"name": "--len_sample", "type":int, "default": 50,
			"help": "length of a sample"},
		{"name": "--tmp", "type": bool, "default": False, "help": "Set false to officially save the trainning log"},
		{"name": "--gamma", "type":int, "default": 0.8,
			"help": "how much will learning rate decrease"},
		{"name": "--slide_size", "type":int, "default": 10,
			"help": "size of GRU input window"},
		{"name": "--step_size", "type":int, "default": 100,
			"help": "learning rate will decrease every step_size steps"},

		# model setting
		{"name": "--param_save_name", "type":str, "default": 'track_agileVer6.pth',
			"help": "The path to model parameters"},
		{"name": "--param_load_path", "type":str, "default": 'track_agileVer6.pth',
			"help": "The path to model parameters"},
		
		]
	# parse arguments
	args = gymutil.parse_arguments(
		description="APG Policy",
		custom_parameters=custom_parameters)
	assert args.batch_size == args.num_envs, "batch_size should be equal to num_envs"

	# name allignment
	args.sim_device_id = args.compute_device_id
	args.sim_device = args.sim_device_type
	if args.sim_device=='cuda':
		args.sim_device += f":{args.sim_device_id}"


	return args

def get_time():

	timestamp = time.time()  # 替换为您的时间戳

	# 将时间戳转换为datetime对象
	dt_object_utc = datetime.utcfromtimestamp(timestamp)

	# 指定目标时区（例如"Asia/Shanghai"）
	target_timezone = pytz.timezone("Asia/Shanghai")
	dt_object_local = dt_object_utc.replace(tzinfo=pytz.utc).astimezone(target_timezone)

	# 将datetime对象格式化为字符串
	formatted_time_local = dt_object_local.strftime("%Y-%m-%d %H:%M:%S %Z")

	return formatted_time_local


async def run():
	
	# torch.autograd.set_detect_anomaly(True)
	args = get_args()
	run_name = f"{args.task}__{args.experiment_name}__{args.seed}__{get_time()}"
	if args.tmp:
		run_name = 'tmp_' + run_name
	# print(args.tmp)
	
	param_save_path = os.path.join(ROOT_DIR, 'param', args.param_save_name)
	param_load_path = os.path.join(ROOT_DIR, 'param', args.param_load_path)
	data_path = os.path.join(ROOT_DIR, 'data')
	log_path = os.path.join(ROOT_DIR, 'test_runs', run_name)

	writer = SummaryWriter(log_path)
	writer.add_text(
		"hyperparameters",
		"|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
	)

	device = args.sim_device
	print("using device:", device)
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	dynamic = IrisDynamics()

	model = TrackAgileModuleVer11(device=device).to(device)

	model.load_model(param_load_path)
	model.eval()
	
	tar_ori = torch.zeros((args.batch_size, 3)).to(device)
	init_vec = torch.tensor([[1.0, 0.0, 0.0]] * args.batch_size, device=device).unsqueeze(-1)


	drone = System()
	await drone.connect(system_address="udp://:14540")

	print("Waiting for drone to connect...")
	async for state in drone.core.connection_state():
		if state.is_connected:
			print(f"-- Connected to drone!")
			break

	print("-- Arming")
	await drone.action.arm()

	print("-- Taking off")
	await drone.action.takeoff()
	await asyncio.sleep(10)

	async for odometry in drone.telemetry.odometry():
		# print(odometry)
		print(f"Odometry -> Position: {odometry.position_body}, Velocity: {odometry.velocity_body}")
		# print(f"z-Body: {odometry.position_body.z_m}, x-Body: {odometry.position_body.x_m}, y-Body: {odometry.position_body.y_m}")
		desired_z = odometry.position_body.z_m  # Adjust desired z position
		break

	target_yaw_deg = 0.0
	await drone.offboard.set_position_ned(PositionNedYaw(north_m=0, east_m=0, down_m=desired_z, yaw_deg=target_yaw_deg))
	await drone.offboard.start()

	print("-- Initializing Position")
	for _ in range(40):
		await drone.offboard.set_position_ned(PositionNedYaw(north_m=0, east_m=0, down_m=desired_z, yaw_deg=target_yaw_deg))
		await asyncio.sleep(0.05)

	mass = 1.9
	dt = 0.02


	tar_acc_intervel = 100
	tar_acc_norm = 0
	theta = torch.rand(2, device=device) * 2 * torch.tensor(np.pi, device=device)
	x = tar_acc_norm * torch.cos(theta)
	y = tar_acc_norm * torch.sin(theta)
	tar_acc = torch.stack([x, y], dim=1).to(device)

	with torch.no_grad():
			
		old_loss = AgileLoss(args.batch_size, device=device)
		
		timer = torch.zeros((args.batch_size,), device=device)

		input_buffer = torch.zeros(args.slide_size, args.batch_size, 6+3).to(device)
		predict_buffer = torch.zeros(args.batch_size, 10, 9).to(device)
		last_action = None

		now_quad_state = torch.zeros((args.batch_size, 12), device=device)
		tar_state = torch.zeros((args.batch_size, 12), device=device)
		
		async for odometry in drone.telemetry.odometry():
			# print(odometry)
			now_quad_state[:, :3] = torch.tensor([odometry.position_body.x_m, odometry.position_body.y_m, odometry.position_body.z_m], device=device).unsqueeze(0)
			now_quad_state[:, 6:9] = torch.tensor([odometry.velocity_body.x_m_s, odometry.velocity_body.y_m_s, odometry.velocity_body.z_m_s], device=device).unsqueeze(0)
			# print(f"z-Body: {odometry.position_body.z_m}, x-Body: {odometry.position_body.x_m}, y-Body: {odometry.position_body.y_m}")
			now_quad_state[:, 9:] = torch.tensor([odometry.angular_velocity_body.roll_rad_s, odometry.angular_velocity_body.pitch_rad_s, odometry.angular_velocity_body.yaw_rad_s], device=device).unsqueeze(0)
			break
		async for attitude in drone.telemetry.attitude_euler():
			now_quad_state[:, 3:6] = torch.tensor([attitude.roll_deg, attitude.pitch_deg, attitude.yaw_deg], device=device).unsqueeze(0) / 180.0 * np.pi
			break
		tar_state = now_quad_state.clone().detach()
		tar_state[:, 0] += 2
		

		# train
		for step in range(args.len_sample):
			

			rel_dis = tar_state[:, :3] - now_quad_state[:, :3]
			world_to_body = dynamic.world_to_body_matrix(now_quad_state[:, 3:6].detach())
			body_to_world = torch.transpose(world_to_body, 1, 2)

			body_rel_dis = torch.matmul(world_to_body, torch.unsqueeze(rel_dis, 2)).squeeze(-1)
			body_vel = torch.matmul(world_to_body, torch.unsqueeze(now_quad_state[:, 6:9], 2)).squeeze(-1)
			
			tmp_input = torch.cat((body_vel, now_quad_state[:, 3:6], body_rel_dis), dim=1)
		
			tmp_input = tmp_input.unsqueeze(0)
			input_buffer = input_buffer[1:].clone()
			input_buffer = torch.cat((input_buffer, tmp_input), dim=0)
			
			action = model.decision_module(input_buffer.clone())
			
			input_action = torch.zeros((args.batch_size, 4), device=device)
			input_action[:, 0] = action[:, 0] * 1
			input_action[:, 1:] = (action[:, 1:] * 2 - 1) * 3 * 180 / np.pi
			await drone.offboard.set_attitude_rate(AttitudeRate(input_action[0, 1], input_action[0, 2], input_action[0, 3], input_action[0, 0]))
			await asyncio.sleep(dt)
			print(f"Step: {step}--------------------------------")
			print("Action:" , input_action[0, :])
			print("Quadrotor state:", now_quad_state[0, :])
			print("Target state:", tar_state[0, :])

			# for debug
			# now_quad_state, acceleration = dynamic(now_quad_state, action, dt)

			
			# update state of quadrotor
			async for odometry in drone.telemetry.odometry():
				# print(odometry)
				now_quad_state[:, :3] = torch.tensor([odometry.position_body.x_m, odometry.position_body.y_m, odometry.position_body.z_m], device=device).unsqueeze(0)
				now_quad_state[:, 6:9] = torch.tensor([odometry.velocity_body.x_m_s, odometry.velocity_body.y_m_s, odometry.velocity_body.z_m_s], device=device).unsqueeze(0)
				# print(f"z-Body: {odometry.position_body.z_m}, x-Body: {odometry.position_body.x_m}, y-Body: {odometry.position_body.y_m}")
				now_quad_state[:, 9:] = torch.tensor([odometry.angular_velocity_body.roll_rad_s, odometry.angular_velocity_body.pitch_rad_s, odometry.angular_velocity_body.yaw_rad_s], device=device).unsqueeze(0)
				break
			async for attitude in drone.telemetry.attitude_euler():
				now_quad_state[:, 3:6] = torch.tensor([attitude.roll_deg, attitude.pitch_deg, attitude.yaw_deg], device=device).unsqueeze(0) / 180.0 * np.pi
				break

			
			print(f"Step: {step}, Position: {now_quad_state[0, :3]}, Velocity: {now_quad_state[0, 6:9]}, Orientation: {now_quad_state[0, 3:6]}")
			
			# update state of target
			if (step % tar_acc_intervel) == 0:
				tar_acc *= -1
			if (step % (tar_acc_intervel * 2)) == 0:
				tar_acc *= -1
			tar_state[:, :2] = tar_state[:, :2] + tar_state[:, 6:8] * dt + tar_acc * 0.5 * dt * dt
			tar_state[:, 6:8] = tar_acc * dt + tar_state[:, 6:8]
			tar_pos = tar_state[:, :3].detach()
			
			

			loss_agile, new_loss = agile_lossVer7(old_loss, now_quad_state, tar_state, tar_state[:, 2].clone(), tar_ori, 2, timer, dt, init_vec, action, last_action)
			old_loss = new_loss

			rotation_matrices = euler_angles_to_matrix(now_quad_state[:, 3:6], convention='XYZ')
			direction_vector = rotation_matrices @ init_vec
			direction_vector = direction_vector.squeeze()

			cos_sim = F.cosine_similarity(direction_vector, rel_dis, dim=1)
			theta = torch.acos(cos_sim)
			theta_degrees = theta * 180.0 / np.pi

			cos_sim_hor = F.cosine_similarity(direction_vector[:, :2], rel_dis[:, :2], dim=1)
			theta_hor = torch.acos(cos_sim_hor)
			theta_degrees_hor = theta_hor * 180.0 / np.pi
			
			item_tested = 0
			horizon_dis = torch.norm(now_quad_state[item_tested, :2] - tar_pos[item_tested, :2], dim=0, p=4)
			speed = torch.norm(now_quad_state[item_tested, 6:9], dim=0, p=2)

			writer.add_scalar(f'Total Loss', loss_agile[item_tested], step)
			writer.add_scalar(f'Direction Loss/sum', old_loss.direction[item_tested], step)
			writer.add_scalar(f'Direction Loss/xy', old_loss.direction_hor[item_tested], step)
			writer.add_scalar(f'Direction Loss/z', old_loss.direction_ver[item_tested], step)
			writer.add_scalar(f'Distance Loss', old_loss.distance[item_tested], step)
			writer.add_scalar(f'Velocity Loss', old_loss.vel[item_tested], step)
			writer.add_scalar(f'Orientation Loss', old_loss.ori[item_tested], step)
			writer.add_scalar(f'Smooth Loss', old_loss.aux[item_tested], step)
			writer.add_scalar(f'Height Loss', old_loss.h[item_tested], step)

			writer.add_scalar(f'Orientation/X', direction_vector[item_tested, 0], step)
			writer.add_scalar(f'Orientation/Y', direction_vector[item_tested, 1], step)
			writer.add_scalar(f'Orientation/Z', direction_vector[item_tested, 2], step)
			writer.add_scalar(f'Orientation/Theta', theta_degrees[item_tested], step)
			writer.add_scalar(f'Orientation/ThetaXY', theta_degrees_hor[item_tested], step)
			writer.add_scalar(f'Horizon Distance', horizon_dis, step)
			writer.add_scalar(f'Position/X', now_quad_state[item_tested, 0], step)
			writer.add_scalar(f'Position/Y', now_quad_state[item_tested, 1], step)
			writer.add_scalar(f'Target Position/X', tar_pos[item_tested, 0], step)
			writer.add_scalar(f'Target Position/Y', tar_pos[item_tested, 1], step)
			writer.add_scalar(f'Velocity/X', now_quad_state[item_tested, 6], step)
			writer.add_scalar(f'Velocity/Y', now_quad_state[item_tested, 7], step)
			writer.add_scalar(f'Distance/X', tar_pos[item_tested, 0] - now_quad_state[item_tested, 0], step)
			writer.add_scalar(f'Distance/Y', tar_pos[item_tested, 1] - now_quad_state[item_tested, 1], step)
			writer.add_scalar(f'Speed/Z', now_quad_state[item_tested, 8], step)
			writer.add_scalar(f'Speed', speed, step)
			writer.add_scalar(f'Height', now_quad_state[item_tested, 2], step)

			
			timer = timer + 1
			last_action = action.clone().detach()


if __name__ == "__main__":
	asyncio.run(run())