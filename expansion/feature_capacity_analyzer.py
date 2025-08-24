#!/usr/bin/env python
"""Feature expansion capacity analyzer.

Evaluates current headroom for introducing new features(e.g., additional
intervention types, ML models, telemetry enrichment) based on resource
utilization & scaling recommendations.

Usage:
  python expansion/feature_capacity_analyzer.py --report
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from importlib import import_module


@dataclass
class CapacitySnapshot:
    cpu_avg: float
    cpu_peak: float
    mem_avg: float
    mem_peak: float
    p95_latency_ms: float
    rps_peak: float
    replicas_max: int


def load_scale_recommendation() -> Dict:
    """Attempt to reuse scaling analyzer logic.

    If the scaling analyzer module is unavailable, generate synthetic values.
    """
    try:
        scale_mod = import_module('scaling.scale_analyzer')
        # Create synthetic windows then recommend
        synth_windows = getattr(scale_mod, 'synthetic_windows')()
        rec = getattr(scale_mod, 'recommend')(synth_windows, 250.0)
        return rec
    except Exception:  # pragma: no cover
        return {
            'traffic': {'peak_rps': 20},
            'cpu': {'avg_pct': 60, 'peak_pct': 82},
            'latency': {'avg_p95_ms': 270},
            'replicas': {'recommended_max': 4},
        }


def derive_capacity(snapshot: CapacitySnapshot) -> Dict:
    cpu_headroom_pct = max(0.0, 85 - snapshot.cpu_peak) / 85 * 100
    mem_headroom_pct = max(0.0, 80 - snapshot.mem_peak) / 80 * 100
    latency_margin_ms = max(0.0, snapshot.p95_latency_ms - 250)

    # Rough estimate: each new feature variant adds 3% CPU + 1% memory + 5ms p95.
    feat_cpu_cost = 3.0
    feat_mem_cost = 1.0
    feat_latency_cost = 5.0

    max_features_cpu = math.floor(cpu_headroom_pct / feat_cpu_cost) if feat_cpu_cost else 0
    max_features_mem = math.floor(mem_headroom_pct / feat_mem_cost) if feat_mem_cost else 0
    max_features_latency = math.floor((50 - latency_margin_ms) / feat_latency_cost) if latency_margin_ms < 50 else 0
    expandable_features = max(0, min(max_features_cpu, max_features_mem, max_features_latency))

    risk = 'high' if latency_margin_ms > 100 or snapshot.cpu_peak > 85 else (
        'medium' if latency_margin_ms > 50 or snapshot.cpu_peak > 75 else 'low'
    )

    return {
        'cpu_headroom_pct': round(cpu_headroom_pct, 2),
        'memory_headroom_pct': round(mem_headroom_pct, 2),
        'latency_margin_ms': round(latency_margin_ms, 2),
        'estimated_additional_features_supported': expandable_features,
        'risk_level': risk,
        'assumptions': {
            'feature_cpu_cost_pct': feat_cpu_cost,
            'feature_mem_cost_pct': feat_mem_cost,
            'feature_p95_cost_ms': feat_latency_cost,
            'latency_budget_extra_ms': 50,
            'cpu_peak_threshold_pct': 85,
        }
    }


def main():
    ap = argparse.ArgumentParser(description='Feature expansion capacity analyzer')
    ap.add_argument('--report', action='store_true', help='Write JSON + Markdown report to ./reports')
    args = ap.parse_args()

    scale_rec = load_scale_recommendation()
    snap = CapacitySnapshot(
        cpu_avg=scale_rec['cpu']['avg_pct'],
        cpu_peak=scale_rec['cpu']['peak_pct'],
        mem_avg=45.0,  # placeholder until memory metrics integrated
        mem_peak=60.0,  # placeholder
        p95_latency_ms=scale_rec['latency']['avg_p95_ms'],
        rps_peak=scale_rec['traffic']['peak_rps'],
        replicas_max=scale_rec['replicas']['recommended_max'],
    )
    capacity = derive_capacity(snap)
    result = {
        'snapshot': snap.__dict__,
        'capacity': capacity,
    }
    print(json.dumps(result, indent=2))

    if args.report:
        report_dir = Path('reports')
        report_dir.mkdir(exist_ok=True)
        json_path = report_dir / 'feature_capacity_report.json'
        md_path = report_dir / 'feature_capacity_report.md'
        json_path.write_text(json.dumps(result, indent=2))
        md = [
            '# Feature Capacity Report', '',
            f"CPU Headroom: {capacity['cpu_headroom_pct']}%", f"Memory Headroom: {capacity['memory_headroom_pct']}%", f"Latency Margin: {capacity['latency_margin_ms']} ms over target",
            f"Estimated Additional Features Supported: {capacity['estimated_additional_features_supported']}", f"Risk Level: {capacity['risk_level']}", '', '## Assumptions'
        ]
        for k, v in capacity['assumptions'].items():
            md.append(f"- {k}: {v}")
        md_path.write_text('\n'.join(md))
        print(f"Reports written to {report_dir}")


if __name__ == '__main__':  # pragma: no cover
    main()
