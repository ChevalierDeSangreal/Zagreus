
from dataclasses import dataclass
from typing import List


@dataclass
class MotorConfig:
    """Motor configuration parameters"""
    taus_up: List[float] = (0.0125, 0.0125, 0.0125, 0.0125)
    """Time constants for each motor speed up."""
    
    taus_down: List[float] = (0.025, 0.025, 0.025, 0.025)
    """Time constants for each motor speed down."""
    
    init: List[float] = (100.0, 100.0, 100.0, 100.0)
    """Initial angular velocities for each motor in rad/s."""
    
    max_rotor_acc: List[float] = (50000.0, 50000.0, 50000.0, 50000.0)
    """Maximum rate of change of angular velocities for each motor in rad/s^2."""
    
    min_rotor_acc: List[float] = (-50000.0, -50000.0, -50000.0, -50000.0)
    """Minimum rate of change of angular velocities for each motor in rad/s^2."""
    
    use_motor_model: bool = False
    """Flag to determine if motor delay is bypassed."""


@dataclass
class BodyRateControllerConfig:
    """Body rate controller configuration parameters"""
    kp: List[float] = (0.15, 0.15, 0.15)
    """Proportional gain for the body rate controller. (x, y, z) axes."""
    
    ki: List[float] = (0.2, 0.2, 0.2)
    """Integral gain for the body rate controller. (x, y, z) axes."""
    
    kd: List[float] = (0.003, 0.003, 0.003)
    """Derivative gain for the body rate controller. (x, y, z) axes."""
    
    k_ff: List[float] = (0.0, 0.0, 0.0)
    """Feedforward gain for the controller."""
    
    cutoff_hz: List[float] = (40, 30)
    """Butterworth Low-pass filter cutoff frequency [Hz] for the rate and rate derivative term."""
    
    max_body_rate: float = 15.0
    """Maximum body rate in rad/s."""


@dataclass
class ControlAllocatorConfig:
    """Control allocator configuration parameters"""

    
    rotor_pos_com: float = 0.056
    """Rotor position w.r.t CoM in meters."""
    
    ctl_thrust_coef: float = 1.0
    """Thrust coefficient for the rotor control, thrust = thrust_coeff * u^2."""
    
    ctl_moment_coef: float = 0.0088
    """Moment coefficient for the rotor control, moment = moment_coeff * thrust."""
    



@dataclass
class IrisDroneConfig:
    """Complete drone configuration"""
    arm_length: float = 0.25554 * 2
    """Length of the arms of the drone in meters. Diagonal distance between two opposite rotors."""
    
    drag_coef: float = 2.57532e-9
    """Drag torque coefficient."""
    
    thrust_coef: float = 2.9265e-7
    """Thrust coefficient."""
    
    input_scaling: float = 5400.0
    """Scaling for 0-1 actuator setpoint to rotor speed in rad/s."""
    
    zero_position_armed: float = 100.0
    """Zero position when the drone is armed."""

    motor: MotorConfig = MotorConfig()
    """Motor configuration"""
    
    body_rate_controller: BodyRateControllerConfig = BodyRateControllerConfig()
    """Body rate controller configuration"""
    
    control_allocator: ControlAllocatorConfig = ControlAllocatorConfig()
    """Control allocator configuration"""

    quad_x_typical_config: bool = True
    """Flag to determine if the quadrotor is configured in X configuration."""

# Quad X configuration reference:
"""
    4 (CW)       2 (CCW)
  \             /
   \           /
    \         /
     \       /
      \     /
       \   /
        \ /
         |
         ↑ Front
         |
        / \
       /   \
      /     \
     /       \
    /         \
   /           \
3 (CCW)         1 (CW)
"""