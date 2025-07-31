# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .base.dynamics_isaac import IsaacGymDynamics, NRIsaacGymDynamics
from .base.dynamics_newton import NewtonDynamics
from .base.dynamics_simple import SimpleDynamics, NRSimpleDynamics
from .base.dynamics_isaac_origin import IsaacGymOriDynamics
from .base.dynamics_iris import IrisDynamics
from .base.track_agileVer2 import TrackAgileVer2
from .base.track_agile_config import TrackAgileCfg
from Zagreus.utils.task_registry import task_registry

# task_registry.register( "quad", AerialRobot, AerialRobotCfg())
# task_registry.register("quad_with_obstacles", AerialRobotWithObstacles, AerialRobotWithObstaclesCfg())
# print("Here I am!!!")
task_registry.register("track_agileVer2", TrackAgileVer2, TrackAgileCfg())