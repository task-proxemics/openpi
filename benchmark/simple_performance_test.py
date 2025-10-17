#!/usr/bin/env python3
"""
Simple Performance Test for OpenPI Server

This script tests the inference performance of the OpenPI server
by measuring timing without triggering data type errors.
"""

import sys
import time

import numpy as np

# Add the src directory to the path
sys.path.append("/workspace/openpi_jax/src")


def test_server_performance(num_iterations=200):
    """Test the server performance by measuring inference times.

    Args:
        num_iterations: Number of inference iterations to run (default: 200)
    """

    try:
        import asyncio
        import json

        import websockets

        def zeros_obs():
            """Create observation exactly like the working test_server.py"""
            H = W = 224

            def img():
                return [[[0] * 3 for _ in range(W)] for __ in range(H)]

            return {
                "images": {
                    "base_0_rgb": img(),
                    "left_wrist_0_rgb": img(),
                    "right_wrist_0_rgb": img(),
                },
                "state": [0.0] * 14,
                "prompt": "pick up the red block",
            }

        async def test_inference():
            """Use the exact same logic as the working test_server.py but handle MessagePack"""
            async with websockets.connect("ws://localhost:8000", max_size=None) as ws:
                payload = {"obs": zeros_obs()}
                await ws.send(json.dumps(payload))
                raw = await ws.recv()
                try:
                    msg = json.loads(raw)
                    return msg
                except Exception:
                    # Server returns MessagePack, decode it
                    try:
                        import msgpack_numpy

                        msg = msgpack_numpy.unpackb(raw)
                        return msg
                    except Exception as e:
                        print(f"Failed to decode MessagePack: {e}")
                        print(
                            "Raw reply:",
                            ((raw[:200] + "...") if isinstance(raw, (str, bytes)) and len(raw) > 200 else raw),
                        )
                        return None

        print("Connecting to OpenPI server...")
        print("Testing server connection...")

        # Test single inference first
        try:
            print("Testing single inference...")
            action = asyncio.run(test_inference())
            print(f"SUCCESS! Got response with keys: {list(action.keys())}")
            if "actions" in action:
                actions = np.array(action["actions"])
                print(f"Actions shape: {actions.shape}")
                print(f"First action sample: {actions[0][:5] if len(actions) > 0 else 'No actions'}")
        except Exception as e:
            print(f"Single inference failed: {e}")
            return

        print("Running performance test...")
        timings = []

        # Run iterations for comprehensive benchmark
        print(f"Running {num_iterations} inferences for comprehensive benchmark...")

        for i in range(num_iterations):
            try:
                start = time.time()
                action = asyncio.run(test_inference())
                end = time.time()

                total_time = (end - start) * 1000  # Convert to milliseconds
                timings.append(total_time)

                if (i + 1) % 25 == 0:  # Progress every 25 iterations
                    print(f"Completed {i + 1}/{num_iterations} inferences...")

            except Exception as e:
                print(f"Error on inference {i + 1}: {str(e)[:100]}...")
                continue

        if timings:
            print(f"\n{'=' * 50}")
            print("PERFORMANCE RESULTS")
            print(f"{'=' * 50}")
            print(f"Successful inferences: {len(timings)}/{num_iterations}")
            print(f"Average inference time: {np.mean(timings):.1f}ms")
            print(f"Min inference time: {np.min(timings):.1f}ms")
            print(f"Max inference time: {np.max(timings):.1f}ms")
            print(f"Standard deviation: {np.std(timings):.1f}ms")
            print(f"Inference rate: {1000 / np.mean(timings):.2f} Hz")
            print(f"Throughput: {1000 / np.mean(timings):.2f} actions/second")

            # Calculate percentiles
            percentiles = [50, 90, 95, 99]
            print("\nPercentiles:")
            for p in percentiles:
                value = np.percentile(timings, p)
                print(f"  {p}th percentile: {value:.1f}ms")

            # Save results to file
            results_file = "/workspace/benchmark_results.txt"
            with open(results_file, "w") as f:
                f.write("OpenPI Performance Test Results\n")
                f.write("==============================\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Successful inferences: {len(timings)}/50\n")
                f.write(f"Average inference time: {np.mean(timings):.1f}ms\n")
                f.write(f"Min inference time: {np.min(timings):.1f}ms\n")
                f.write(f"Max inference time: {np.max(timings):.1f}ms\n")
                f.write(f"Standard deviation: {np.std(timings):.1f}ms\n")
                f.write(f"Inference rate: {1000 / np.mean(timings):.2f} Hz\n")
                f.write(f"Throughput: {1000 / np.mean(timings):.2f} actions/second\n")
                f.write("\nPercentiles:\n")
                for p in percentiles:
                    value = np.percentile(timings, p)
                    f.write(f"  {p}th percentile: {value:.1f}ms\n")

            print(f"\nResults saved to: {results_file}")

        else:
            print("No successful inferences - all failed due to data type issues")
            print("This indicates a problem with the data format or server configuration")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenPI Server Performance Test")
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=200,
        help="Number of inference iterations to run (default: 200)",
    )
    parser.add_argument("--quick", action="store_true", help="Run a quick test with 50 iterations")

    args = parser.parse_args()

    if args.quick:
        num_iterations = 50
    else:
        num_iterations = args.iterations

    print(f"Starting performance test with {num_iterations} iterations...")
    test_server_performance(num_iterations)
