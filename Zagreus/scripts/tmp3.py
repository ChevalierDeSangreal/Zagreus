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

	print("-- Getting current attitude and position")
	async for attitude in drone.telemetry.attitude_euler():
		print(f"Attitude -> Roll: {attitude.roll_deg}, Pitch: {attitude.pitch_deg}, Yaw: {attitude.yaw_deg}")
		break

	async for position in drone.telemetry.position():
		print(f"Position -> Lat: {position.latitude_deg}, Lon: {position.longitude_deg}, Alt: {position.relative_altitude_m}")
		break

	async for odometry in drone.telemetry.odometry():
		# print(odometry)
		print(f"Odometry -> Position: {odometry.position_body}, Velocity: {odometry.velocity_body}")
		# print(f"z-Body: {odometry.position_body.z_m}, x-Body: {odometry.position_body.x_m}, y-Body: {odometry.position_body.y_m}")
		desired_z = odometry.position_body.z_m  # Adjust desired z position
		break
	async for imu in drone.telemetry.imu():
		print(f"Accelerometer: {imu.acceleration_frd.down_m_s2}, {imu.acceleration_frd.forward_m_s2}, {imu.acceleration_frd.right_m_s2}")
		break


	
	# print("-- Setting initial setpoint")
	# await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.0))

	# print("-- Starting offboard")
	# try:
	#     await drone.offboard.start()
	# except OffboardError as error:
	#     print(f"Starting offboard mode failed with error code: \
	#           {error._result.result}")
	#     print("-- Disarming")
	#     await drone.action.disarm()
	#     return

	print("-- Go up at 50% thrust")
	await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.30))
	await asyncio.sleep(10)
	
	await drone.action.land()
	await asyncio.sleep(10)

	await drone.action.disarm()


if __name__ == "__main__":
	# Run the asyncio loop
	asyncio.run(run())