#!/bin/bash
# Simple script to run performance test with monitoring
# Usage: ./run_test.sh [iterations]

set -e

# Default values
ITERATIONS=${1:-100}
MONITOR_INTERVAL=0.5

# Create directories
mkdir -p results logs

# Generate timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MONITOR_FILE="results/monitor_${TIMESTAMP}.json"
PERF_LOG="logs/performance_${TIMESTAMP}.log"

echo "============================================================"
echo "OpenPI Performance Test with Monitoring"
echo "============================================================"
echo "Iterations: ${ITERATIONS}"
echo "Monitor file: ${MONITOR_FILE}"
echo "Performance log: ${PERF_LOG}"
echo ""

# Start monitoring in background
echo "Starting system monitor in background..."
python system_monitor.py --interval ${MONITOR_INTERVAL} --output ${MONITOR_FILE} &
MONITOR_PID=$!
echo "Monitor PID: ${MONITOR_PID}"

# Give monitor time to start
sleep 3
echo ""

# Check if monitor is still running
if ! ps -p ${MONITOR_PID} > /dev/null; then
    echo "ERROR: Monitor process failed to start!"
    exit 1
fi

# Run performance test
echo "Starting performance test..."
python simple_performance_test.py --iterations ${ITERATIONS} | tee ${PERF_LOG}

# Stop monitoring
echo ""
echo "Stopping monitor..."
kill ${MONITOR_PID}
sleep 2  # Give monitor time to finish writing
wait ${MONITOR_PID} 2>/dev/null || true

# Analyze monitoring data
echo ""
echo "============================================================"
echo "MONITORING ANALYSIS"
echo "============================================================"

if [ -f "${MONITOR_FILE}" ]; then
    # Quick analysis with Python
    python -c "
import json
import sys

with open('${MONITOR_FILE}', 'r') as f:
    data = json.load(f)

if not data:
    print('No monitoring data collected')
    sys.exit(0)

print(f'Collected {len(data)} data points')
print(f'Duration: {data[-1][\"elapsed_seconds\"] - data[0][\"elapsed_seconds\"]:.1f}s')

# GPU stats
gpu_utils = [dp['gpu']['gpu_util'] for dp in data if dp['gpu']]
if gpu_utils:
    print(f'\nGPU Utilization:')
    print(f'  Average: {sum(gpu_utils)/len(gpu_utils):.1f}%')
    print(f'  Peak: {max(gpu_utils):.1f}%')
    print(f'  Min: {min(gpu_utils):.1f}%')

# CPU stats
cpu_percents = [dp['system']['cpu_percent'] for dp in data if dp['system']]
if cpu_percents:
    print(f'\nCPU Utilization:')
    print(f'  Average: {sum(cpu_percents)/len(cpu_percents):.1f}%')
    print(f'  Peak: {max(cpu_percents):.1f}%')

# Memory stats
mem_percents = [dp['system']['memory_percent'] for dp in data if dp['system']]
if mem_percents:
    print(f'\nMemory Utilization:')
    print(f'  Average: {sum(mem_percents)/len(mem_percents):.1f}%')
    print(f'  Peak: {max(mem_percents):.1f}%')

# Bottleneck analysis
avg_gpu = sum(gpu_utils)/len(gpu_utils) if gpu_utils else 0
avg_cpu = sum(cpu_percents)/len(cpu_percents) if cpu_percents else 0
avg_mem = sum(mem_percents)/len(mem_percents) if mem_percents else 0

print(f'\nBOTTLENECK ANALYSIS:')
if avg_gpu < 10:
    print(f'  - GPU utilization is very low ({avg_gpu:.1f}%) - GPU is NOT the bottleneck')
if avg_cpu > 80:
    print(f'  - CPU utilization is high ({avg_cpu:.1f}%) - CPU may be a bottleneck')
if avg_mem > 90:
    print(f'  - Memory utilization is high ({avg_mem:.1f}%) - Memory may be a bottleneck')
if avg_gpu < 10 and avg_cpu < 50 and avg_mem < 80:
    print(f'  - Low resource utilization suggests bottleneck is likely:')
    print(f'    * Network latency (WebSocket communication)')
    print(f'    * Data serialization/deserialization')
    print(f'    * JAX compilation overhead')
"
else
    echo "WARNING: Monitoring file not found!"
fi

echo ""
echo "============================================================"
echo "Results saved:"
echo "  Performance: ${PERF_LOG}"
echo "  Monitoring: ${MONITOR_FILE}"
echo "============================================================"

