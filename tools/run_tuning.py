#!/usr/bin/env python3
"""
Optimizer tuning script for LIFE Theory feature extraction.
Performs bounded grid search over optimizer configuration flags.
"""
import os
import sys
import argparse
import itertools
import json
import time
from pathlib import Path
# Add current directory to path for imports
sys.path.insert(0, '.')
from tools.sota_benchmark import main as benchmark_main
def test_configuration(flags: dict, run_id: int, total_configs: int) -> bool:
    """Test a specific optimizer configuration."""
    print(f"Testing configuration {run_id}/{total_configs}: {flags}")
    # Set environment flags
    for key, value in flags.items():
        os.environ[key] = str(value)
    # Run benchmark with clean sys.argv
    original_argv = sys.argv[:]
    try:
        sys.argv = ['sota_benchmark.py', '--runs', '3']  # Use fewer runs for tuning speed
        result = benchmark_main()
        if result == 0:
            print(f"✅ Configuration {run_id} passed!")
            return True
        else:
            print(f"❌ Configuration {run_id} failed with exit code {result}")
            return False
    except SystemExit as e:
        if e.code == 0:
            print(f"✅ Configuration {run_id} passed!")
            return True
        else:
            print(f"❌ Configuration {run_id} failed with exit code {e.code}")
            return False
    except Exception as e:
        print(f"❌ Configuration {run_id} failed with exception: {e}")
        return False
    finally:
        sys.argv = original_argv
def main():
    parser = argparse.ArgumentParser(description="LIFE Theory optimizer tuning")
    parser.add_argument("--budget", type=int, default=30, 
                       help="Time budget in minutes (used as max configurations)")
    parser.add_argument("--output", type=str, default="artifacts/tuning_results.json",
                       help="Output file for tuning results")
    parser.add_argument("--respect-production", action="store_true", default=True,
                       help="Respect locked production flags (default: True)")
    args = parser.parse_args()
    print(f"🚀 Running optimizer tuning with budget: {args.budget} configurations")
    # Simplified configuration for production stability
    flag_combinations = [
        # Production baseline 
        {"LIFE_FAST_MODE": "1", "LIFE_OPT_CACHE": "1"},
        # Production + core optimizations
        {"LIFE_FAST_MODE": "1", "LIFE_OPT_CACHE": "1", "LIFE_OPT_PREALLOC": "1"},
        {"LIFE_FAST_MODE": "1", "LIFE_OPT_CACHE": "1", "LIFE_OPT_RFFT": "1"}
    ]
    results = []
    passed_configs = 0
    for i, flags in enumerate(flag_combinations[:args.budget], 1):
        if test_configuration(flags, i, len(flag_combinations)):
            passed_configs += 1
            results.append({"config": flags, "status": "passed", "run_id": i})
        else:
            results.append({"config": flags, "status": "failed", "run_id": i})
    # Save results
    Path("artifacts").mkdir(exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_results = {
        "summary": {
            "total_configs": len(flag_combinations),
            "tested_configs": min(args.budget, len(flag_combinations)),
            "passed_configs": passed_configs,
            "success_rate": passed_configs / min(args.budget, len(flag_combinations))
        },
        "results": results
    }
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"📊 Results: {passed_configs}/{min(args.budget, len(flag_combinations))} configurations passed")
    print(f"💾 Results saved to: {output_path}")
    return 0
if __name__ == "__main__":
    exit(main())
