"""
Venturi Controller Implementation
==========================
PID controllers and Venturi flow management for the 3-Venturi system.
Named after Giovanni Battista Venturi (1746-1822).
"""

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class PID:
    """PID Controller for Venturi flow management."""
    kp: float  # Proportional gain
    ki: float  # Integral gain  
    kd: float  # Derivative gain
    integral: float = 0.0
    prev_error: Optional[float] = None

    def step(self, error: float, dt: float) -> float:
        """Compute PID control output for given error and time delta."""
        self.integral += error * dt
        
        # Derivative term (avoid division by zero)
        if self.prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / max(dt, 1e-6)
        
        self.prev_error = error
        
        # PID output
        output = (self.kp * error + 
                 self.ki * self.integral + 
                 self.kd * derivative)
        
        return output


@dataclass
class VenturiState:
    """State container for a Venturi controller."""
    setpoint: float    # Target value
    control: float     # Current control output
    pid: PID          # PID controller instance


class Venturi:
    """
    Venturi flow controller implementing fluid dynamics-inspired control.
    
    Models computational flow constraints using Venturi principles:
    - Flow acceleration through constriction
    - Pressure differential monitoring  
    - Flow recovery and stabilization
    """
    
    def __init__(self, name: str, setpoint: float, pid: PID, 
                 clamp: tuple[float, float]):
        """
        Initialize Venturi controller.
        
        Args:
            name: Venturi identifier (e.g., 'ingestion', 'processing', 'output')
            setpoint: Target value for control
            pid: PID controller instance
            clamp: (min, max) bounds for control output
        """
        self.name = name
        self.state = VenturiState(setpoint=setpoint, control=setpoint, pid=pid)
        self.clamp = clamp
        self._last_time = time.perf_counter()
        
    def update(self, measured: float) -> float:
        """
        Update Venturi control based on measured value.
        
        Args:
            measured: Current measured value (flow, latency, etc.)
            
        Returns:
            New control output value
        """
        current_time = time.perf_counter()
        dt = max(current_time - self._last_time, 1e-3)  # Avoid zero dt
        self._last_time = current_time
        
        # Calculate error (setpoint - measured)
        error = self.state.setpoint - measured
        
        # Get PID correction
        pid_output = self.state.pid.step(error, dt)
        
        # Update control value
        new_control = self.state.control + pid_output
        
        # Apply clamping
        min_val, max_val = self.clamp
        self.state.control = max(min_val, min(max_val, new_control))
        
        return self.state.control
        
    def reset(self):
        """Reset PID controller internal state."""
        self.state.pid.integral = 0.0
        self.state.pid.prev_error = None
        self._last_time = time.perf_counter()
        
    def set_setpoint(self, new_setpoint: float):
        """Update the target setpoint."""
        self.state.setpoint = new_setpoint
        
    def get_telemetry(self) -> dict:
        """Get current telemetry data."""
        return {
            "name": self.name,
            "setpoint": self.state.setpoint,
            "control": self.state.control,
            "pid_state": {
                "integral": self.state.pid.integral,
                "prev_error": self.state.pid.prev_error
            }
        }
