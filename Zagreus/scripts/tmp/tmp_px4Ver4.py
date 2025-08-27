#!/usr/bin/env python3

# Warning: Only try this in simulation!
#          The direct attitude interface is a low level interface to be used
#          with caution. On real vehicles the thrust values are likely not
#          adjusted properly and you need to close the loop using altitude.

import asyncio

from mavsdk import System
from mavsdk.offboard import (Attitude, OffboardError)
from mavsdk.offboard import PositionNedYaw

import os
os.environ["MAVSDK_CLIENT_LOG_LEVEL"] = "debug"

async def run():
	""" Does Offboard control using attitude commands. """

	drone = System()
	print("Connecting...")
	await drone.connect(system_address="udp://:14540")

	print("Waiting for drone to connect...")
	async for state in drone.core.connection_state():
		if state.is_connected:
			print(f"-- Connected to drone!")
			break

	print("Waiting for drone to have a global position estimate...")
	async for health in drone.telemetry.health():
		if health.is_global_position_ok and health.is_home_position_ok:
			print("-- Global position estimate OK")
			break

	print("-- Arming")
	await drone.action.arm()

	await drone.action.takeoff()

	await asyncio.sleep(10)


	print("-- Go up at 50% thrust")
	await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.30))
	await asyncio.sleep(10)
	
	await drone.action.land()
	await asyncio.sleep(10)
	
	desired_z = -2.0  # Desired altitude in meters (negative for NED frame)
	
	target_yaw_deg = 0.0
	await drone.offboard.set_position_ned(PositionNedYaw(north_m=0, east_m=0, down_m=desired_z, yaw_deg=target_yaw_deg))
	await drone.offboard.start()

	print("-- Initializing Position")
	for _ in range(40):
		await drone.offboard.set_position_ned(PositionNedYaw(north_m=0, east_m=0, down_m=desired_z, yaw_deg=target_yaw_deg))
		await asyncio.sleep(0.05)

	print("-- First Position")
	for _ in range(40):
		await drone.offboard.set_position_ned(PositionNedYaw(north_m=1, east_m=1, down_m=desired_z, yaw_deg=target_yaw_deg))
		await asyncio.sleep(0.05)

	print("-- First Position")
	for _ in range(40):
		await drone.offboard.set_position_ned(PositionNedYaw(north_m=-1, east_m=-1, down_m=desired_z, yaw_deg=target_yaw_deg))
		await asyncio.sleep(0.05)

	print("-- Finish")
	await drone.action.disarm()


if __name__ == "__main__":
	# Run the asyncio loop
	asyncio.run(run())