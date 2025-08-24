# 3‑Venturi Control in L.I.F.E THEORY

## Overview

This module implements three coordinated "Venturi" controllers named after Giovanni Battista Venturi (1746-1822), the Italian physicist who discovered the Venturi effect in fluid dynamics.

The 3-Venturi system models computational flow constraints as controllable stages, regulating throughput and stability like a fluid system while maintaining clinical BCI requirements.

## The Three Venturi Stages

### 1. Ingestion Venturi
- **Purpose**: Controls input flow and queue depth
- **Function**: Smooths and rate-limits raw data intake
- **Controls**: Batch/window sizing, debounce, queue back-pressure
- **Metrics**: Input flow rate (trials/sec), queue depth, smoothing efficiency

### 2. Processing Venturi  
- **Purpose**: Controls feature/inference latency via fast_mode/cache and workload sizing
- **Function**: Regulates computational workload to meet clinical latency requirements
- **Controls**: Fast/normal mode switching, cache enablement, batch sizing
- **Metrics**: Processing latency (ms), cache hit rate, computational efficiency

### 3. Output Venturi
- **Purpose**: Controls emission cadence and downstream IO
- **Function**: Regulates output delivery to prevent downstream overload
- **Controls**: Write/emit cadence, coalescing, flush policy
- **Metrics**: Output flow rate (items/sec), emission latency, IO efficiency

## Physics Principles Applied

Each Venturi implements fluid dynamics concepts:

- **Bernoulli's Principle**: Flow acceleration through computational constrictions
- **Pressure Differential**: Monitoring latency increases as "pressure drops"
- **Flow Recovery**: Stabilization after processing constraints
- **Conservation of Energy**: Maintaining total system efficiency within clinical budgets

## PID Control System

Each Venturi exposes a PID (Proportional-Integral-Derivative) controller:

- **Setpoint**: Target value (flow rate, latency, etc.)
- **Measured**: Actual current value
- **Error**: Setpoint - Measured
- **Control Output**: Adjusted parameter to minimize error

## Telemetry Mapping

Software metrics mapped to fluid dynamics:

| Fluid Concept | Software Metric | Units |
|---------------|-----------------|-------|
| Flow | items/sec or trials/sec | rps |
| Pressure Drop | latency increase or queue depth | ms |
| Efficiency | useful work / total budget | ratio |
| Constriction | computational bottleneck | % |
| Recovery | system stabilization | time |

## Clinical BCI Integration

The system maintains clinical-grade BCI requirements:

- **Latency Budget**: ≤100ms end-to-end for clinical compliance
- **Accuracy Floor**: ≥75% minimum performance
- **Real-time Processing**: Sub-20ms feature extraction
- **Stability**: Consistent performance under varying loads

## Usage

```python
from venturi.controller import PID, Venturi
from venturi.metrics import efficiency, pressure_drop

# Initialize Processing Venturi
processing_pid = PID(kp=0.7, ki=0.08, kd=0.04)
processing_venturi = Venturi(
    name="processing",
    setpoint=20.0,  # target latency in ms
    pid=processing_pid,
    clamp=(5.0, 60.0)  # min/max latency bounds
)

# Update with measured latency
measured_latency = 18.5  # ms
control_output = processing_venturi.update(measured_latency)

# Calculate efficiency
eff = efficiency(measured_latency, budget_ms=100.0)
```

## Files Structure

- `controller.py`: PID and Venturi controller implementations
- `metrics.py`: Telemetry calculation and efficiency functions  
- `config.yaml`: Configuration for all three Venturi stages
- `__init__.py`: Module initialization and exports

## Integration Points

The Venturi system integrates with:

1. **L.I.F.E Theory Pipeline**: Real-time EEG processing
2. **Autonomous Optimizer**: Performance optimization loops
3. **SOTA Benchmarking**: Contract-based validation
4. **Clinical Standards**: BCI compliance monitoring
5. **CI/CD Pipeline**: Automated testing and validation

## Audit and Compliance

All Venturi operations are logged for:

- **Clinical Validation**: Meeting BCI latency requirements
- **Performance Audits**: Optimization effectiveness tracking
- **Regulatory Compliance**: Evidence-based performance documentation
- **Quality Assurance**: Continuous monitoring and alerting

## Named After Giovanni Battista Venturi

Giovanni Battista Venturi (1746-1822) was an Italian physicist who:
- Discovered the Venturi effect in fluid dynamics
- Contributed to understanding of flow through constrictions
- Developed principles of pressure differential measurement
- Advanced the study of fluid mechanics and hydraulic engineering

The 3-Venturi system honors his contributions by applying fluid dynamics principles to computational flow control in brain-computer interface systems.
