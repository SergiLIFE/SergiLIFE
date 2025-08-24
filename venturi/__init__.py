"""
3-Venturi Control System for L.I.F.E Theory
============================================

Three coordinated Venturi stages modeled as controllable flow constraints
to regulate computational throughput and stability like a fluid system.

Named after Giovanni Battista Venturi (1746-1822).
"""

from .controller import PID, Venturi, VenturiState
from .metrics import VenturiTelemetry, efficiency, pressure_drop

__version__ = "1.0.0"
__all__ = [
    "PID", 
    "Venturi", 
    "VenturiState",
    "VenturiTelemetry", 
    "efficiency", 
    "pressure_drop"
]
