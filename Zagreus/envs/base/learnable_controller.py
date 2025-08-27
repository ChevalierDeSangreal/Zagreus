import torch
import math
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List
from .iris_config import IrisDroneConfig


def assert_no_nan(tensor, name):
    if tensor is None:
        return
    if torch.is_floating_point(tensor) and torch.isnan(tensor).any():
        raise RuntimeError(f"NaN detected in {name}, shape={tuple(tensor.shape)}")
    if torch.is_floating_point(tensor) and torch.isinf(tensor).any():
        raise RuntimeError(f"Inf detected in {name}, shape={tuple(tensor.shape)}")


class ButterworthFilter(nn.Module):
    def __init__(self, num_envs, dt, cutoff_hz, device="cpu", dtype=torch.float32):
        """
        Second-order Butterworth low-pass filter.
        Args:
            num_envs (int): Number of parallel environments.
            dt (float): Sampling period.
            cutoff_hz (float): Cutoff frequency (Hz).
            device (str): Torch device.
            dtype (torch.dtype): Torch data type.

        Modified:
            - Converted to nn.Module to support PyTorch autograd.
            - 'cutoff_hz' is now a learnable parameter, using softplus to ensure positivity.
            - Historical states (_x1, _x2, _y1, _y2) stored as module attributes (buffers or plain tensors) 
            to maintain state across calls without affecting gradients.
            - Coefficient computation fully implemented with torch operations to preserve gradient flow.
        """
        super().__init__()
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.dtype = dtype

        cutoff_tensor = torch.tensor(float(cutoff_hz), device=device, dtype=dtype)
        # self.cutoff_hz_raw = nn.Parameter(torch.log(torch.exp(cutoff_tensor) - 1))
        self.cutoff_hz_raw = torch.log(torch.exp(cutoff_tensor) - 1)
        self._x1 = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self._x2 = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self._y1 = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self._y2 = torch.zeros(num_envs, 3, device=device, dtype=dtype)

    def _compute_coeffs(self, cutoff_hz: torch.Tensor):
        omega_c = 2 * math.pi * cutoff_hz
        tan_wc = torch.tan(omega_c * self.dt / 2)
        sqrt2 = torch.sqrt(torch.tensor(2.0, device=self.device, dtype=self.dtype))
        assert_no_nan(tan_wc, "tan_wc")
        a0 = 1 + sqrt2 * tan_wc + tan_wc * tan_wc
        b0 = (tan_wc * tan_wc) / a0
        b1 = 2 * b0
        b2 = b0
        a1 = 2 * (tan_wc * tan_wc - 1) / a0
        a2 = (1 - sqrt2 * tan_wc + tan_wc * tan_wc) / a0
        return b0, b1, b2, a1, a2

    def reset(self, env_ids=None):
        """
        Reset the filter states.
        Args:
            env_ids (list or torch.Tensor or None): Indices of environments to reset. If None, reset all.
        """
        if env_ids is None:
            self._x1 = torch.zeros_like(self._x1)
            self._x2 = torch.zeros_like(self._x2)
            self._y1 = torch.zeros_like(self._y1)
            self._y2 = torch.zeros_like(self._y2)
        else:
            x1 = self._x1.clone()
            x2 = self._x2.clone()
            y1 = self._y1.clone()
            y2 = self._y2.clone()

            x1[env_ids] = 0
            x2[env_ids] = 0
            y1[env_ids] = 0
            y2[env_ids] = 0

            self._x1, self._x2, self._y1, self._y2 = x1, x2, y1, y2

    def forward(self, x: torch.Tensor):
        """
        Apply the second-order Butterworth low-pass filter to the input.
        Args:
            x (torch.Tensor): Input tensor of shape (num_envs, 3)
        Returns:
            y (torch.Tensor): Filtered output tensor of shape (num_envs, 3)
        """
        cutoff_hz = torch.nn.functional.softplus(self.cutoff_hz_raw)
        assert_no_nan(cutoff_hz, "cutoff_hz")
        b0, b1, b2, a1, a2 = self._compute_coeffs(cutoff_hz)
        y = b0 * x + b1 * self._x1 + b2 * self._x2 - a1 * self._y1 - a2 * self._y2

        self._x2 = self._x1.clone()
        self._x1 = x.clone()
        self._y2 = self._y1.clone()
        self._y1 = y.clone()

        return y
    
    def detach_state(self):
        """Detach historical states from the current computation graph."""
        self._x1 = self._x1.detach()
        self._x2 = self._x2.detach()
        self._y1 = self._y1.detach()
        self._y2 = self._y2.detach()

class Allocation(nn.Module):
    def __init__(self, num_envs, quad_x_typical_config, arm_length, thrust_coeff, drag_coeff, device="cpu", dtype=torch.float32):
        """
        Initializes the allocation matrix for a quadrotor for multiple environments.
        Control allocator computes the normalized PWM and thrust from the normalized wrench.
        Allocation module, instead, simulates the motor speed to the ACTUAL wrench.
        
        Parameters:
        - num_envs (int): Number of environments
        - quad_x_typical_config (bool): If True, uses the quad X configuration; otherwise, uses the quad + configuration
        - arm_length (float): Distance from the center to the rotor
        - thrust_coeff (float): Rotor thrust constant
        - drag_coeff (float): Rotor torque constant
        - device (str): 'cpu' or 'cuda'
        - dtype (torch.dtype): Desired tensor dtype
        """
        super().__init__()
        self.num_envs = num_envs
        self.quad_x_typical_config = quad_x_typical_config
        self.device = device
        self.dtype = dtype

        self.arm_length = nn.Parameter(torch.tensor(float(arm_length), device=device, dtype=dtype))
        self.thrust_coeff = nn.Parameter(torch.tensor(float(thrust_coeff), device=device, dtype=dtype))
        self.drag_coeff = nn.Parameter(torch.tensor(float(drag_coeff), device=device, dtype=dtype))


    def _compute_allocation_matrix(self):
        sqrt2_inv = 1.0 / torch.sqrt(torch.tensor(2.0, dtype=self.dtype, device=self.device))

        if not self.quad_x_typical_config:  # quad + config
            A = torch.stack([
                torch.ones(4, dtype=self.dtype, device=self.device),
                torch.tensor([1, -1, -1, 1], dtype=self.dtype, device=self.device) * self.arm_length * sqrt2_inv,
                torch.tensor([-1, -1, 1, 1], dtype=self.dtype, device=self.device) * self.arm_length * sqrt2_inv,
                torch.tensor([1, -1, 1, -1], dtype=self.dtype, device=self.device) * self.drag_coeff
            ])
        else:
            A = torch.stack([
                torch.ones(4, dtype=self.dtype, device=self.device),
                torch.tensor([1, -1, -1, 1], dtype=self.dtype, device=self.device) * self.arm_length * sqrt2_inv,
                torch.tensor([1, 1, -1, -1], dtype=self.dtype, device=self.device) * self.arm_length * sqrt2_inv,
                torch.tensor([1, -1, -1, 1], dtype=self.dtype, device=self.device) * self.drag_coeff
            ])

        return A.unsqueeze(0).repeat(self.num_envs, 1, 1)

    def forward(self, omega: torch.Tensor):
        """
        Compute total thrust and body torques from rotor angular velocities.

        Args:
            omega (torch.Tensor): Shape (num_envs, 4), rotor angular velocities.

        Returns:
            thrust_torque (torch.Tensor): Shape (num_envs, 4), [total thrust, tau_x, tau_y, tau_z]
        """
        allocation_matrix = self._compute_allocation_matrix()
        # print("Value of omega", omega)
        thrusts_ref = self.thrust_coeff * omega**2
        thrust_torque = torch.bmm(allocation_matrix, thrusts_ref.unsqueeze(-1)).squeeze(-1)
        return thrust_torque

class Motor(nn.Module):
    def __init__(self, num_envs, taus_up, taus_down, init, max_rotor_acc, min_rotor_acc, dt, use, device="cpu", dtype=torch.float32):
        """
        Initializes the motor model.

        Modified:
        - Converted to nn.Module for integration into PyTorch pipelines.
        - taus_up, taus_down, max_rotor_acc, min_rotor_acc, init are learnable nn.Parameter (autograd-friendly).
        - Vectorized update equations (no Python for-loops).
        - forward() replaces compute(), following PyTorch convention.

        Parameters:
        - num_envs: Number of envs.
        - taus_up: (4,) Tensor or list specifying time constants per motor.
        - taus_down: (4,) Tensor or list specifying time constants per motor.
        - init: (4,) Tensor or list specifying the initial omega per motor. (rad/s)
        - max_rotor_acc: (4,) Tensor or list specifying max rate of change of omega per motor. (rad/s^2)
        - min_rotor_acc: (4,) Tensor or list specifying min rate of change of omega per motor. (rad/s^2)
        - dt: Time step for integration.
        - use: Boolean indicating whether to use motor dynamics.
        - device: 'cpu' or 'cuda' for tensor operations.
        - dtype: Data type for tensors.
        """
        super().__init__()
        self.num_envs = num_envs
        self.num_motors = len(taus_up)
        self.dt = dt
        self.use = use
        self.device = device
        self.dtype = dtype

        # Learnable parameters
        self.taus_up = nn.Parameter(torch.tensor(taus_up, device=device, dtype=dtype))
        self.taus_down = nn.Parameter(torch.tensor(taus_down, device=device, dtype=dtype))
        self.max_rotor_acc = nn.Parameter(torch.tensor(max_rotor_acc, device=device, dtype=dtype))
        self.min_rotor_acc = nn.Parameter(torch.tensor(min_rotor_acc, device=device, dtype=dtype))
        self.init_omega = nn.Parameter(torch.tensor(init, device=device, dtype=dtype))

        self.omega = torch.zeros((num_envs, 4), dtype=dtype, device=device) + self.init_omega

    def forward(self, omega_ref: torch.Tensor):
        """
        Computes the new omega values based on reference omega and motor dynamics.

        Parameters:
        - omega_ref: (num_envs, num_motors) Tensor of reference omega values.

        Returns:
        - omega: (num_envs, num_motors) Tensor of updated omega values.
        """

        if not self.use:
            self.omega = omega_ref
            return self.omega

        # Determine tau depending on up/down phase
        tau = torch.where(
            omega_ref >= self.omega,
            self.taus_up.expand_as(self.omega),
            self.taus_down.expand_as(self.omega),
        )

        # Rate of change
        omega_rate = (1.0 / tau) * (omega_ref - self.omega)

        # Clamp acceleration
        omega_rate = omega_rate.clamp(self.min_rotor_acc, self.max_rotor_acc)

        # Integrate
        self.omega = self.omega + self.dt * omega_rate
        return self.omega

    def reset(self, env_ids):
        """
        Resets the motor model to initial conditions.
        """
        self.omega[env_ids] = self.omega = torch.zeros((len(env_ids), 4), dtype=self.dtype, device=self.device) + self.init_omega

    def detach_state(self):
        self.omega = self.omega.detach()

def get_control_allocator(allocation):
    assert_no_nan(allocation, "allocation")
    return torch.linalg.pinv(allocation)

@torch.jit.script
def compute_control_allocation(ctl_allocators, norm_thrust, norm_torque):
    """Compute the control allocation based on the normalized thrust and torque.
    Args:
        ctl_allocators (torch.Tensor): Control allocation matrix (num_envs, 4, 4).
        norm_thrust (torch.Tensor): Normalized thrust in [ 0, 1] (num_envs, 1).
        norm_torque (torch.Tensor): Normalized torque in [-1, 1] (num_envs, 3).
    Returns:
        torch.Tensor: Computed actuator setpoint u (num_envs, 4).
    """
    # print("norm_thrust:", norm_thrust[0])
    # print("norm_torque:", norm_torque[0])
    wrench_vec = torch.cat([norm_thrust, norm_torque], dim=1).unsqueeze(-1)  # (num_envs, 4, 1)
    u_square_vec = torch.bmm(ctl_allocators, wrench_vec).squeeze(-1) # (num_envs, 4)
    return torch.sqrt(torch.clamp(u_square_vec, min=1e-6, max=1.0)) # (num_envs, 4)

class ControlAllocator(nn.Module):
    def __init__(self, num_envs, rotor_pos_com, ctl_thrust_coef, ctl_moment_coef, device="cpu", dtype=torch.float32):
        """Control allocator for quadcopters. 
        Control allocator computes the normalized PWM and thrust from the normalized wrench.
        Allocation module, instead, simulates the motor speed to the ACTUAL wrench.
        
        1. Transfer the normalized torque and thrust to the normalized PWM. 
        2. Normalized PWM to the Normalized Thrust based on the thrust curve.


        Args:
            num_envs (int): Number of environments.
            rotor_pos_com(float): The position of the rotors in xy axes w.r.t the CoM.
                           This value may not be the true geometry value, due to the PX4 wrong configuration.
            ctl_thrust_coef (float): Thrust coefficient for the rotor control, thrust = thrust_coeff * u^2.
            ctl_moment_coef (float): Moment coefficient for the rotor control, moment = moment_coeff * thrust.
                           This value may not be the true geometry value, due to the PX4 wrong configuration.
            device (str): Device to run the computations on.
            dtype (torch.dtype): Data type for the computations.

        Modified:
            - Changed to inherit from nn.Module.
            - Made rotor_pos_com, ctl_thrust_coef, ctl_moment_coef learnable nn.Parameter.
            - Moved compute() to forward(), rebuilding effectiveness matrix with learnable parameters.
            
        """
        super().__init__()
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype
        

        self.rotor_pos_com = nn.Parameter(torch.tensor(float(rotor_pos_com), dtype=dtype, device=device))
        self.ctl_thrust_coef = nn.Parameter(torch.tensor(float(ctl_thrust_coef), dtype=dtype, device=device))
        self.ctl_moment_coef = nn.Parameter(torch.tensor(float(ctl_moment_coef), dtype=dtype, device=device))
        
    def forward(self, norm_thrust: torch.Tensor, norm_torque: torch.Tensor) -> torch.Tensor:

        pos_com = self.rotor_pos_com
        ct = self.ctl_thrust_coef
        km = self.ctl_moment_coef
        # print("pos_com, ct, km", pos_com.item(), ct.item(), km.item())
        # assert_no_nan(pos_com, "pos_com")
        assert_no_nan(ct, "ct")
        assert_no_nan(km, "km")
        eff_matrix = torch.stack([
            torch.stack([ ct,        ct,        ct,       ct       ]),
            torch.stack([ pos_com*ct, -pos_com*ct, -pos_com*ct, pos_com*ct ]),
            torch.stack([ pos_com*ct,  pos_com*ct, -pos_com*ct, -pos_com*ct ]),
            torch.stack([ km*ct,     -km*ct,     -km*ct,    km*ct ]),
        ], dim=0).to(self.device, self.dtype)  # (4,4)
        # cond_num = torch.linalg.cond(eff_matrix)
        # print("cond(eff_matrix) =", cond_num)
        assert_no_nan(eff_matrix, "eff_matrix")
        ctl_allocator = get_control_allocator(eff_matrix)
        # print("ctl_allocator:", ctl_allocator)
        ctl_allocators = ctl_allocator.unsqueeze(0).repeat(self.num_envs, 1, 1)  # (num_envs, 4, 4)

        return compute_control_allocation(ctl_allocators, norm_thrust, norm_torque)



class BodyRateController(nn.Module):
     
    def __init__(self, num_envs, dt, kp, ki, kd, k_ff, cutoff_hz, device="cpu", dtype=torch.float32):
        """ Body rate PID controller for quadcopters.
        A batch parallel implementation of the PX4 body rate controller.
        
        Args:
        - num_envs (int): Number of environments.
        - dt (float): Time step duration, should be 1 KHz (0.001 seconds) for the PX4 controller.
        - kp (torch.Tensor): Proportional gain for the controller.
        - ki (torch.Tensor): Integral gain for the controller.
        - kd (torch.Tensor): Derivative gain for the controller.
        - k_ff (torch.Tensor): Feedforward gain for the controller.
        - cutoff_hz list(float): Butterworth Low-pass filter cutoff frequency [Hz] \
                                for the rate and rate derivate term.
        """

        super().__init__()
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.dtype = dtype
        
        # 可学习参数
        self.kp = nn.Parameter(torch.tensor(kp, device=device, dtype=dtype))
        self.ki = nn.Parameter(torch.tensor(ki, device=device, dtype=dtype))
        self.kd = nn.Parameter(torch.tensor(kd, device=device, dtype=dtype))
        self.k_ff = nn.Parameter(torch.tensor(k_ff, device=device, dtype=dtype))

        self._integral = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self._prev_rate = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self._prev_error = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        
        # low-pass filter for the derivative term
        self._derivative_filtered = torch.zeros(self.num_envs, 3, device=device, dtype=dtype)
        
        # body rate butterworth filter 
        self.rate_filter = ButterworthFilter(
            num_envs = num_envs,
            dt = dt,
            cutoff_hz = cutoff_hz[0],  # Typical cutoff frequency for body rate control
            device = device,
            dtype = dtype
        )
        
        # body rate derivate butterworth filter
        self.rate_derivate_filter = ButterworthFilter(
            num_envs = num_envs,
            dt = dt,
            cutoff_hz = cutoff_hz[1],  # Typical cutoff frequency for body rate control
            device = device,
            dtype = dtype
        )
        
        """Initialize the integral and previous error terms."""
        
        
    def forward(self, rate_ref, rate, debug=False):
        """Compute the control action based on the reference and current angular velocities.
        
        Args:
        - rate_ref (torch.Tensor): Reference body rate (rad/s) (num_envs, 3).
        - rate (torch.Tensor): Current body rate (rad/s) (num_envs, 3).
        - debug (bool): If True, return additional debug information.
        
        Returns:
        - torque (torch.Tensor): torque to be applied (num_envs, 3).
        
        Remarks:
        - No anti-windup is applied to the control action.
        - The control output, named `torque`, follows the PX4 body rate controller convention.
            However, since the typical MC_ROLLRATE_P = 0.05, it means the so called `torque`
            is actually just a quanity within [-1,1] for the follwing mixer.
            
        - normalized wrench -> [mixer (unit scale)] -> normalized PWM
          normalized PWM -> [thrust curve] -> normalized ACTUAL thrust -> normalized ACTUAL wrench
            If the [thrust curve] is simplified linear (Y=X), 
            Then
                normalized wrench = normalized ACTUAL wrench
        """
        assert_no_nan(rate, "rate")
        # 1. Compute the rate derivative
        d_rate = (rate - self._prev_rate) / self.dt
        # print("d_rate in PID", d_rate[0])
        # print("rate in PID:", rate[0])
        # 2. Apply low-pass filter to the rate and the rate derivative
        rate_filtered = self.rate_filter(rate) 
        d_rate_filtered = self.rate_derivate_filter(d_rate)  
        # rate_filtered = rate.clone()
        # d_rate_filtered = d_rate.clone()
        
        assert_no_nan(d_rate, "d_rate")
        assert_no_nan(d_rate_filtered, "d_rate_filtered")
        assert_no_nan(rate_filtered, "rate_filtered")
        
        # 3. Compute the error, error integral, and rate derivate term based on
        #    the reference and filter body rate
        error = rate_ref - rate_filtered
        self._integral = self._integral + error * self.dt

        # print("rate_filtered in PID", rate_filtered[0])
        # print("error in PID:", error[0])
        # print("ki in PID:", self.ki)
        # print("integral in PID:", self._integral[0])
        # print("derivative in PID:", d_rate_filtered[0])
        # print("k_ff in PID:", self.k_ff)
        # print("kd in PID:", self.kd)
        # 4. PID with the feedforward term
        torque = (
            self.kp * error + 
            self.ki * self._integral + 
            self.kd * d_rate_filtered + 
            self.k_ff * rate_ref
        )
        # print("torque in PID:", torque[0])
        # 5. Update with original (unfiltered) rate
        self._prev_rate = rate.clone()


        if debug:
            debug_info = {
                "error": error.clone(),
                "integral": self._integral.clone(),
                "derivative": d_rate.clone(),
                "torque": torque.clone(),
            }
            return torque, debug_info
        else:
            return torque, {}
    
    def reset(self, env_ids):
        """Reset the controller state."""
        self._integral = self._integral.clone()
        self._integral[env_ids] = 0

        self._prev_error = self._prev_error.clone()
        self._prev_error[env_ids] = 0

        self._prev_rate = self._prev_rate.clone()
        self._prev_rate[env_ids] = 0

        self._derivative_filtered = self._derivative_filtered.clone()
        self._derivative_filtered[env_ids] = 0
        
        self.rate_filter.reset(env_ids)
        self.rate_derivate_filter.reset(env_ids)

    def detach_state(self):
        self._integral = self._integral.detach()
        self._prev_error = self._prev_error.detach()
        self._prev_rate = self._prev_rate.detach()
        self._derivative_filtered = self._derivative_filtered.detach()

        self.rate_filter.detach_state()
        self.rate_derivate_filter.detach_state()



class DroneBodyRateController(nn.Module):
    """
    Combined drone motor + body rate control + allocation module (pure PyTorch implementation).

    This class receives normalized thrust and body rate commands in [-1, 1] range,
    computes the motor speed response, and outputs the corresponding force/torque vector.

    Notes:
        - Removed IsaacSim/Articulation dependencies
        - Single forward() handles both processing actions and computing output
        - Input actions: (num_envs, 4), normalized thrust + body rate [-1, 1]
        - Output: thrust (num_envs, 1), torque (num_envs, 3)
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype

        self.config = IrisDroneConfig() 


        self.zero_position_armed = nn.Parameter(torch.tensor(float(self.config.zero_position_armed), device=device, dtype=dtype))
        self.input_scaling = nn.Parameter(torch.tensor(float(self.config.input_scaling), device=device, dtype=dtype))

        # Motor module
        self._motor = Motor(
            num_envs=num_envs,
            taus_up=self.config.motor.taus_up,
            taus_down=self.config.motor.taus_down,
            init=self.config.motor.init,
            max_rotor_acc=self.config.motor.max_rotor_acc,
            min_rotor_acc=self.config.motor.min_rotor_acc,
            dt=dt,
            use=self.config.motor.use_motor_model,
            device=device,
            dtype=dtype,
        )
        self._allocation = Allocation(
            num_envs=self.num_envs,
            quad_x_typical_config=self.config.quad_x_typical_config,
            arm_length=self.config.arm_length,
            thrust_coeff=self.config.thrust_coef,
            drag_coeff=self.config.drag_coef,
            device=self.device,
            dtype=dtype,
        )
        # Body rate PID controller
        self._body_rate_controller = BodyRateController(
            num_envs=num_envs,
            dt=dt,
            kp=self.config.body_rate_controller.kp,
            ki=self.config.body_rate_controller.ki,
            kd=self.config.body_rate_controller.kd,
            k_ff=self.config.body_rate_controller.k_ff,
            cutoff_hz=self.config.body_rate_controller.cutoff_hz,
            device=device,
            dtype=dtype,
        )

        # Control allocator
        self._ctl_allocator = ControlAllocator(
            num_envs=num_envs,
            rotor_pos_com=self.config.control_allocator.rotor_pos_com,
            ctl_thrust_coef=self.config.control_allocator.ctl_thrust_coef,
            ctl_moment_coef=self.config.control_allocator.ctl_moment_coef,
            device=device,
            dtype=dtype,
        )

    def forward(self, actions: torch.Tensor, current_rate: torch.Tensor):
        """
        Compute the force/torque vector from normalized thrust and body rate commands.

        Args:
            actions (torch.Tensor): (num_envs, 4), normalized thrust + body rate [-1, 1]
            current_rate (torch.Tensor): (num_envs, 3), current body rates

        Returns:
            thrust (torch.Tensor): (num_envs, 1)
            torque (torch.Tensor): (num_envs, 3)
        """
        # 1. Clamp and scale inputs
        thrust_clamped = actions[:, 0].unsqueeze(-1).clamp(-1.0, 1.0)
        body_rate_clamped = actions[:, 1:4].clamp(-1.0, 1.0)

        # map thrust to [0, 1]
        z_thrust = (thrust_clamped + 1.0) / 2.0
        # map body rate to actual scale (currently identity, can multiply by max_body_rate)
        body_rate_scaled = body_rate_clamped * 1.0

        assert_no_nan(z_thrust, "z_thrust")
        assert_no_nan(body_rate_scaled, "body_rate_scaled")
        assert_no_nan(current_rate, "current_rate")

        # 2. PID controller computes normalized torque
        norm_torque, _ = self._body_rate_controller.forward(
            rate_ref=body_rate_scaled,
            rate=current_rate,
            debug=False,
        )
        # print("norm_torque after body rate pid:", norm_torque[0])
        # 3. Control allocation: normalized actuator command
        actuator_sp = self._ctl_allocator(
            norm_thrust=z_thrust,
            norm_torque=norm_torque,
        )

        # 4. Map actuator_sp to rotor speed reference
        omega_ref = self.zero_position_armed + actuator_sp * self.input_scaling

        # 5. Motor computes actual rotor speeds
        omega_real = self._motor(omega_ref)

        # 6. Allocation computes final thrust/torque
        force_torque = self._allocation(omega_real)

        # Split into thrust and torque
        thrust = force_torque[:, 0:1]
        torque = force_torque[:, 1:4]

        return thrust, torque

    def reset(self):
        """Reset internal states"""
        self._motor.reset(torch.arange(self.num_envs, device=self.device))
        self._body_rate_controller.reset(torch.arange(self.num_envs, device=self.device))
        self.detach_state()

    def detach_state(self):
        self._motor.detach_state()
        self._body_rate_controller.detach_state()
