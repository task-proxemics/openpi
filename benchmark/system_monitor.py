#!/usr/bin/env python3
"""
System Resource Monitor for OpenPI Performance Testing

This script runs separately from the performance test to monitor
GPU, CPU, memory, and other system resources without affecting
the performance measurements.
"""

import argparse
import json
import signal
import subprocess
import sys
import time

import psutil


def get_gpu_stats():
    """Get current GPU utilization and memory stats."""
    try:
        # Try the detailed query first
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            line = result.stdout.strip()
            if line and line != "[N/A], [N/A], [N/A], [N/A], [N/A]":
                parts = line.split(", ")
                return {
                    "gpu_util": float(parts[0]) if parts[0] != "[N/A]" else 0,
                    "memory_used": float(parts[1]) if parts[1] != "[N/A]" else 0,
                    "memory_total": float(parts[2]) if parts[2] != "[N/A]" else 0,
                    "temperature": float(parts[3]) if parts[3] != "[N/A]" else 0,
                    "power_draw": float(parts[4]) if parts[4] != "[N/A]" else 0,
                }

        # Fallback: try basic GPU utilization only
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            line = result.stdout.strip()
            if line and line != "[N/A]":
                gpu_util = float(line) if line != "[N/A]" else 0
                return {
                    "gpu_util": gpu_util,
                    "memory_used": 0,  # Not supported on Jetson
                    "memory_total": 0,  # Not supported on Jetson
                    "temperature": 0,  # Not available
                    "power_draw": 0,  # Not available
                }

    except Exception as e:
        print(f"Warning: Could not get GPU stats: {e}")

    return None


def get_system_stats():
    """Get CPU, memory, and other system stats."""
    try:
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "load_avg": psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else 0,
        }
    except Exception as e:
        print(f"Warning: Could not get system stats: {e}")
        return None


def monitor_resources(interval=1.0, duration=None, output_file=None):
    """Monitor system resources and optionally save to file."""
    print(f"Starting system monitoring (interval: {interval}s)")
    if duration:
        print(f"Will monitor for {duration} seconds")
    if output_file:
        print(f"Results will be saved to: {output_file}")
    else:
        # Default to results directory
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/monitor_{timestamp}.json"
        print(f"Results will be saved to: {output_file}")

    start_time = time.time()
    data_points = []

    # Signal handler to save data on termination
    def save_and_exit(signum, frame):
        print(f"\nReceived signal {signum}, saving data...")
        if output_file and data_points:
            with open(output_file, "w") as f:
                json.dump(data_points, f, indent=2)
            print(f"Saved {len(data_points)} data points to {output_file}")
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGTERM, save_and_exit)
    signal.signal(signal.SIGINT, save_and_exit)

    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time

            if duration and elapsed >= duration:
                break

            # Get stats
            gpu_stats = get_gpu_stats()
            system_stats = get_system_stats()

            # Create data point
            data_point = {
                "timestamp": current_time,
                "elapsed_seconds": elapsed,
                "gpu": gpu_stats,
                "system": system_stats,
            }
            data_points.append(data_point)

            # Print current status
            status_parts = [f"[{elapsed:6.1f}s]"]

            if gpu_stats:
                status_parts.append(f"GPU: {gpu_stats['gpu_util']:5.1f}%")

            if system_stats:
                status_parts.append(f"CPU: {system_stats['cpu_percent']:5.1f}%")
                status_parts.append(f"RAM: {system_stats['memory_percent']:5.1f}%")
                if system_stats["load_avg"] > 0:
                    status_parts.append(f"Load: {system_stats['load_avg']:4.2f}")

            print(" ".join(status_parts))

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

    # Save data if requested
    if output_file and data_points:
        with open(output_file, "w") as f:
            json.dump(data_points, f, indent=2)
        print(f"Saved {len(data_points)} data points to {output_file}")

    # Print summary statistics
    if data_points:
        print(f"\n{'=' * 60}")
        print("MONITORING SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total monitoring time: {elapsed:.1f} seconds")
        print(f"Data points collected: {len(data_points)}")

        # GPU summary
        gpu_utils = [dp["gpu"]["gpu_util"] for dp in data_points if dp["gpu"]]
        if gpu_utils:
            print("\nGPU Utilization:")
            print(f"  Average: {sum(gpu_utils) / len(gpu_utils):.1f}%")
            print(f"  Peak: {max(gpu_utils):.1f}%")
            print(f"  Min: {min(gpu_utils):.1f}%")

        # System summary
        cpu_percents = [dp["system"]["cpu_percent"] for dp in data_points if dp["system"]]
        memory_percents = [dp["system"]["memory_percent"] for dp in data_points if dp["system"]]

        if cpu_percents:
            print("\nCPU Utilization:")
            print(f"  Average: {sum(cpu_percents) / len(cpu_percents):.1f}%")
            print(f"  Peak: {max(cpu_percents):.1f}%")

        if memory_percents:
            print("\nMemory Utilization:")
            print(f"  Average: {sum(memory_percents) / len(memory_percents):.1f}%")
            print(f"  Peak: {max(memory_percents):.1f}%")

        # Bottleneck analysis
        print("\nBOTTLENECK ANALYSIS:")
        avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
        avg_cpu_util = sum(cpu_percents) / len(cpu_percents) if cpu_percents else 0
        avg_memory_util = sum(memory_percents) / len(memory_percents) if memory_percents else 0

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
    parser = argparse.ArgumentParser(description="Monitor system resources during OpenPI performance testing")
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=1.0,
        help="Monitoring interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        help="Duration to monitor in seconds (default: monitor until Ctrl+C)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file to save monitoring data (JSON format)",
    )

    args = parser.parse_args()

    monitor_resources(interval=args.interval, duration=args.duration, output_file=args.output)


if __name__ == "__main__":
    main()
