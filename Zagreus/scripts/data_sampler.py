#!/usr/bin/env python3

import asyncio
import random
import time
from mavsdk import System
from mavsdk.offboard import PositionNedYaw
import subprocess
import os
import signal

async def run_main(ts=60):
    """ Random position commands with random intervals until total duration ts seconds """
    print("Connecting to drone...")

    drone = System()
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

    desired_z = -2.0  # 固定高度
    target_yaw_deg = 0.0

    # 初始化 Offboard
    await drone.offboard.set_position_ned(
        PositionNedYaw(north_m=0, east_m=0, down_m=desired_z, yaw_deg=target_yaw_deg)
    )
    await drone.offboard.start()

    print("-- Initializing Position Hold")
    for _ in range(40):  # 稍作初始化
        await drone.offboard.set_position_ned(
            PositionNedYaw(north_m=0, east_m=0, down_m=desired_z, yaw_deg=target_yaw_deg)
        )
        await asyncio.sleep(0.05)

    print(f"-- Starting Random Trajectory Sampling for {ts} seconds")
    start_time = time.time()
    elapsed = 0
    command_id = 1

    while elapsed < ts - 1e-1:
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        yaw = random.uniform(-180, 180)
        dt = random.uniform(2, 4)

        remaining = ts - elapsed
        duration = min(dt, remaining)

        print(f"[{command_id}] Command -> x={x:.2f}, y={y:.2f}, z={desired_z}, yaw={yaw:.1f}, duration={duration:.1f}s")

        steps = int(duration / 0.05)
        for _ in range(steps):
            await drone.offboard.set_position_ned(
                PositionNedYaw(north_m=x, east_m=y, down_m=desired_z, yaw_deg=yaw)
            )
            await asyncio.sleep(0.05)

        elapsed = time.time() - start_time
        command_id += 1

    print("-- Trajectory Sampling Completed")


async def run():
    """ 启动 PX4 SITL 并执行轨迹采集 """
    px4_dir = os.path.expanduser("/home/core/wangzimo/PX4-Autopilot")
    print("Opening PX4 SITL...")
    env = os.environ.copy()
    env["HEADLESS"] = "1"
    proc = subprocess.Popen(
        ["make", "px4_sitl", "gazebo", "-j4"],
        cwd=px4_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )

    try:
        # 等待 PX4 启动
        await asyncio.sleep(5)
        # 执行轨迹采集
        await run_main(ts=60)

    except KeyboardInterrupt:
        print("\n-- Ctrl+C detected! Stopping...")

    finally:
        # 确保子进程被终止
        if proc.poll() is None:
            print("-- Terminating PX4 SITL subprocess")
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)  # 发送 SIGTERM 给整个进程组
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("-- Subprocess terminated")


if __name__ == "__main__":
    asyncio.run(run())
