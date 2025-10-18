#!/usr/bin/env python3
"""
Direct inference benchmark - bypasses WebSocket server overhead.

This script loads the OpenPI policy directly and runs inference,
measuring pure model performance without network/serialization overhead.
"""

import os

# Disable type checking for performance
os.environ["JAXTYPING_DISABLE"] = "1"

import argparse
import logging
import time

import numpy as np

# OpenPI imports (same as serve_policy.py)
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def create_aloha_policy(checkpoint_dir="gs://openpi-assets/checkpoints/pi05_base"):
    """Create ALOHA policy directly (same as server does)."""
    logging.info("Loading ALOHA policy from %s", checkpoint_dir)
    policy = _policy_config.create_trained_policy(_config.get_config("pi05_aloha"), checkpoint_dir, default_prompt=None)
    logging.info("Policy loaded successfully")
    return policy


def create_dummy_observation():
    """Create a dummy observation matching ALOHA format.

    Format should match make_aloha_example() from aloha_policy.py:
    - state: (14,) float
    - images: dict of (3, 224, 224) uint8 arrays [channels, height, width]
    """
    return {
        "state": np.ones(14, dtype=np.float32),  # Using ones like the example
        "images": {
            "cam_high": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def warmup_policy(policy, num_warmup=5):
    """Warmup the policy with a few inference calls to ensure JIT compilation."""
    logging.info("Warming up policy with %d iterations...", num_warmup)
    obs = create_dummy_observation()

    for i in range(num_warmup):
        _ = policy.infer(obs)
        logging.info("  Warmup %d/%d complete", i + 1, num_warmup)

    logging.info("Warmup complete")


def benchmark_inference(policy, num_iterations=1000):
    """Run inference benchmark and collect timing statistics."""
    logging.info("Starting benchmark with %d iterations...", num_iterations)

    obs = create_dummy_observation()
    timings = []

    start_overall = time.time()

    for i in range(num_iterations):
        start = time.perf_counter()
        result = policy.infer(obs)
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000
        timings.append(elapsed_ms)

        if (i + 1) % 100 == 0:
            logging.info("  Completed %d/%d iterations", i + 1, num_iterations)

    end_overall = time.time()
    total_time = end_overall - start_overall

    # Calculate statistics
    timings_array = np.array(timings)

    print("\n" + "=" * 60)
    print("DIRECT INFERENCE BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total iterations: {num_iterations}")
    print(f"Total time: {total_time:.2f}s")
    print("\nInference Latency:")
    print(f"  Average: {np.mean(timings_array):.2f}ms")
    print(f"  Median: {np.median(timings_array):.2f}ms")
    print(f"  Min: {np.min(timings_array):.2f}ms")
    print(f"  Max: {np.max(timings_array):.2f}ms")
    print(f"  Std Dev: {np.std(timings_array):.2f}ms")
    print("\nPercentiles:")
    print(f"  50th: {np.percentile(timings_array, 50):.2f}ms")
    print(f"  90th: {np.percentile(timings_array, 90):.2f}ms")
    print(f"  95th: {np.percentile(timings_array, 95):.2f}ms")
    print(f"  99th: {np.percentile(timings_array, 99):.2f}ms")
    print("\nThroughput:")
    print(f"  {num_iterations / total_time:.2f} inferences/second")
    print(f"  {1000 / np.mean(timings_array):.2f} Hz")
    print("=" * 60)

    return timings_array


def main():
    parser = argparse.ArgumentParser(description="Direct inference benchmark (no WebSocket overhead)")
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=1000,
        help="Number of inference iterations (default: 1000)",
    )
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="gs://openpi-assets/checkpoints/pi05_base",
        help="Checkpoint directory",
    )

    args = parser.parse_args()

    # Load policy
    print("=" * 60)
    print("LOADING POLICY")
    print("=" * 60)
    policy = create_aloha_policy(args.checkpoint)

    # Warmup
    print("\n" + "=" * 60)
    print("WARMUP PHASE")
    print("=" * 60)
    warmup_policy(policy, args.warmup)

    # Benchmark
    print("\n" + "=" * 60)
    print("BENCHMARK PHASE")
    print("=" * 60)
    timings = benchmark_inference(policy, args.iterations)

    # Save results
    output_file = f"direct_inference_results_{args.iterations}iters.txt"
    with open(output_file, "w") as f:
        f.write("Direct Inference Benchmark Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Iterations: {args.iterations}\n")
        f.write(f"Average latency: {np.mean(timings):.2f}ms\n")
        f.write(f"Throughput: {args.iterations / np.sum(timings) * 1000:.2f} Hz\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
