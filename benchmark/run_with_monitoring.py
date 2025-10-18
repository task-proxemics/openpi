#!/usr/bin/env python3
"""
Run OpenPI performance test with continuous system monitoring.

This script starts system monitoring in the background, then runs
the performance test, and finally analyzes the collected data.
"""

import argparse
from datetime import datetime
import json
import os
import subprocess
import time

import numpy as np


def start_monitoring(interval=0.5, output_file=None):
    """Start system monitoring in background."""
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/monitor_{timestamp}.json"

    print("Starting system monitoring...")
    print(f"Output file: {output_file}")

    # Start monitoring process
    cmd = [
        "python",
        "system_monitor.py",
        "--interval",
        str(interval),
        "--output",
        output_file,
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Give it more time to start and create the file
    time.sleep(3)

    # Check if the process is still running
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        print("Monitoring process failed to start:")
        print(f"STDOUT: {stdout.decode()}")
        print(f"STDERR: {stderr.decode()}")
        return None, None

    return process, output_file


def stop_monitoring(process):
    """Stop the monitoring process."""
    print("Stopping system monitoring...")
    process.terminate()
    process.wait(timeout=5)


def run_performance_test(iterations=100):
    """Run the performance test."""
    print(f"Running performance test with {iterations} iterations...")

    cmd = ["python", "simple_performance_test.py", "--iterations", str(iterations)]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    return result.returncode, result.stdout, result.stderr


def analyze_monitoring_data(output_file):
    """Analyze the collected monitoring data."""
    if not os.path.exists(output_file):
        print(f"Warning: Monitoring file {output_file} not found")
        # Try to find any monitoring files
        import glob

        monitor_files = glob.glob("results/monitor_*.json")
        if monitor_files:
            latest_file = max(monitor_files, key=os.path.getctime)
            print(f"Using latest monitoring file: {latest_file}")
            output_file = latest_file
        else:
            return

    print(f"\nAnalyzing monitoring data from {output_file}...")

    with open(output_file) as f:
        data = json.load(f)

    if not data:
        print("No monitoring data collected")
        return

    print(f"Collected {len(data)} data points")
    print(f"Monitoring duration: {data[-1]['elapsed_seconds'] - data[0]['elapsed_seconds']:.1f} seconds")

    # Extract stats
    gpu_utils = [dp["gpu"]["gpu_util"] for dp in data if dp["gpu"]]
    cpu_percents = [dp["system"]["cpu_percent"] for dp in data if dp["system"]]
    memory_percents = [dp["system"]["memory_percent"] for dp in data if dp["system"]]

    if gpu_utils:
        print("\nGPU Utilization:")
        print(f"  Average: {np.mean(gpu_utils):.1f}%")
        print(f"  Peak: {np.max(gpu_utils):.1f}%")
        print(f"  Min: {np.min(gpu_utils):.1f}%")

    if cpu_percents:
        print("\nCPU Utilization:")
        print(f"  Average: {np.mean(cpu_percents):.1f}%")
        print(f"  Peak: {np.max(cpu_percents):.1f}%")

    if memory_percents:
        print("\nMemory Utilization:")
        print(f"  Average: {np.mean(memory_percents):.1f}%")
        print(f"  Peak: {np.max(memory_percents):.1f}%")

    # Bottleneck analysis
    print("\nBOTTLENECK ANALYSIS:")
    avg_gpu_util = np.mean(gpu_utils) if gpu_utils else 0
    avg_cpu_util = np.mean(cpu_percents) if cpu_percents else 0
    avg_memory_util = np.mean(memory_percents) if memory_percents else 0

    if avg_gpu_util < 10:
        print(f"  ⚠️  GPU utilization is very low ({avg_gpu_util:.1f}%) - GPU is NOT the bottleneck")
    if avg_cpu_util > 80:
        print(f"  🔥 CPU utilization is high ({avg_cpu_util:.1f}%) - CPU may be the bottleneck")
    if avg_memory_util > 90:
        print(f"  🔥 Memory utilization is high ({avg_memory_util:.1f}%) - Memory may be the bottleneck")
    if avg_gpu_util < 10 and avg_cpu_util < 50 and avg_memory_util < 80:
        print("  🤔 Low resource utilization suggests the bottleneck may be:")
        print("     - Network latency (WebSocket communication)")
        print("     - Model loading/initialization overhead")
        print("     - Data serialization/deserialization")
        print("     - JAX compilation overhead")


def main():
    parser = argparse.ArgumentParser(description="Run OpenPI performance test with system monitoring")
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=100,
        help="Number of inference iterations (default: 100)",
    )
    parser.add_argument("--quick", action="store_true", help="Run quick test with 50 iterations")
    parser.add_argument(
        "--monitor-interval",
        "-i",
        type=float,
        default=0.5,
        help="Monitoring interval in seconds (default: 0.5)",
    )
    parser.add_argument("--output", "-o", type=str, help="Output file for monitoring data")

    args = parser.parse_args()

    if args.quick:
        iterations = 50
    else:
        iterations = args.iterations

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    try:
        # Start monitoring
        monitor_process, output_file = start_monitoring(interval=args.monitor_interval, output_file=args.output)

        # Run performance test
        return_code, stdout, stderr = run_performance_test(iterations)

        # Stop monitoring
        stop_monitoring(monitor_process)

        # Show performance results
        print("\n" + "=" * 60)
        print("PERFORMANCE TEST RESULTS")
        print("=" * 60)
        print(stdout)
        if stderr:
            print("STDERR:")
            print(stderr)

        # Analyze monitoring data
        analyze_monitoring_data(output_file)

        # Save performance results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        perf_log = f"logs/performance_{timestamp}.log"
        with open(perf_log, "w") as f:
            f.write(stdout)
            if stderr:
                f.write("\nSTDERR:\n")
                f.write(stderr)
        print(f"\nPerformance log saved to: {perf_log}")
        print(f"Monitoring data saved to: {output_file}")

        return return_code

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        if "monitor_process" in locals():
            stop_monitoring(monitor_process)
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if "monitor_process" in locals():
            stop_monitoring(monitor_process)
        return 1


if __name__ == "__main__":
    exit(main())
