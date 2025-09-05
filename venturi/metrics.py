"""
Venturi Metrics and Telemetry
======================
Telemetry collection and efficiency calculations for the 3-Venturi system.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class VenturiTelemetry:
    """Telemetry data for a Venturi stage."""
    flow_rps: float         # items/sec throughput
    latency_ms: float       # mean or p50 latency
    queue_depth: int        # current queue/buffer depth
    efficiency: float       # 0..1 (within budget)
    pressure_drop: float    # latency vs baseline


def efficiency(lat_ms: float, budget_ms: float) -> float:
    """
    Calculate efficiency as fraction of budget used.
    
    Args:
        lat_ms: Current latency in milliseconds
        budget_ms: Total budget in milliseconds (e.g., 100ms clinical limit)
        
    Returns:
        Efficiency ratio in [0, 1] where 1.0 = perfect efficiency
    """
    if lat_ms <= 0 or budget_ms <= 0:
        return 0.0
    
    # Efficiency = budget_remaining / budget_total
    # Higher efficiency when latency is lower
    return max(0.0, min(1.0, budget_ms / max(lat_ms, 1e-6)))


def pressure_drop(lat_ms: float, baseline_ms: float) -> float:
    """
    Calculate pressure drop as latency increase over baseline.
    
    Args:
        lat_ms: Current latency in milliseconds
        baseline_ms: Baseline latency in milliseconds
        
    Returns:
        Pressure drop in milliseconds (always >= 0)
    """
    return max(0.0, lat_ms - baseline_ms)


def flow_rate(items_processed: int, time_window_sec: float) -> float:
    """
    Calculate flow rate in items per second.
    
    Args:
        items_processed: Number of items processed
        time_window_sec: Time window in seconds
        
    Returns:
        Flow rate in items/second
    """
    if time_window_sec <= 0:
        return 0.0
    return items_processed / time_window_sec


def calculate_venturi_metrics(
    stage_name: str,
    latency_ms: float,
    throughput_rps: float,
    queue_depth: int,
    baseline_ms: float,
    budget_ms: float
) -> VenturiTelemetry:
    """
    Calculate comprehensive metrics for a Venturi stage.
    
    Args:
        stage_name: Name of the stage
        latency_ms: Current latency
        throughput_rps: Current throughput
        queue_depth: Current queue depth
        baseline_ms: Baseline latency
        budget_ms: Total latency budget
        
    Returns:
        VenturiTelemetry instance with calculated metrics
    """
    return VenturiTelemetry(
        flow_rps=throughput_rps,
        latency_ms=latency_ms,
        queue_depth=queue_depth,
        efficiency=efficiency(latency_ms, budget_ms),
        pressure_drop=pressure_drop(latency_ms, baseline_ms)
    )


def aggregate_telemetry(telemetry_list: list[VenturiTelemetry]) -> Dict[str, Any]:
    """
    Aggregate telemetry from multiple Venturi stages.
    
    Args:
        telemetry_list: List of VenturiTelemetry instances
        
    Returns:
        Aggregated metrics dictionary
    """
    if not telemetry_list:
        return {}
    
    total_flow = sum(t.flow_rps for t in telemetry_list)
    avg_latency = sum(t.latency_ms for t in telemetry_list) / len(telemetry_list)
    total_queue = sum(t.queue_depth for t in telemetry_list)
    avg_efficiency = sum(t.efficiency for t in telemetry_list) / len(telemetry_list)
    total_pressure_drop = sum(t.pressure_drop for t in telemetry_list)
    
    return {
        "total_flow_rps": total_flow,
        "average_latency_ms": avg_latency,
        "total_queue_depth": total_queue,
        "average_efficiency": avg_efficiency,
        "total_pressure_drop_ms": total_pressure_drop,
        "stage_count": len(telemetry_list)
    }
