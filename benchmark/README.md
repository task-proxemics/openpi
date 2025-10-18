# OpenPI Performance Benchmarking

This directory contains tools for benchmarking the OpenPI policy server with comprehensive system monitoring.

## Directory Structure

```
benchmark/
├── results/           # GPU/CPU/memory monitoring data (JSON)
├── logs/             # Performance test results (text logs)
├── run_test.sh       # Main benchmark script (RECOMMENDED)
├── simple_performance_test.py  # Performance test client
└── system_monitor.py # System resource monitoring
```

## Quick Start

### Run a Quick Test (50 iterations)
```bash
cd /workspace/openpi_jax/benchmark
./run_test.sh 50
```

### Run a Standard Test (100 iterations)
```bash
./run_test.sh 100
```

### Run a Comprehensive Test (200 iterations)
```bash
./run_test.sh 200
```

## What Gets Measured

### Performance Metrics
- **Inference time**: Latency per action prediction
- **Throughput**: Actions per second
- **Percentiles**: 50th, 90th, 95th, 99th latency percentiles

### System Monitoring
- **GPU Utilization**: Real-time GPU usage during inference
- **CPU Utilization**: CPU usage across all cores
- **Memory Usage**: RAM utilization percentage
- **System Load**: Linux load average

## How It Works

1. **Start Monitoring**: The script starts `system_monitor.py` in the background
2. **Run Performance Test**: Executes `simple_performance_test.py` with specified iterations
3. **Stop Monitoring**: Gracefully stops monitoring and saves data
4. **Analyze Results**: Automatically analyzes monitoring data and identifies bottlenecks

## Output Files

All results are automatically timestamped and saved:

- `logs/performance_YYYYMMDD_HHMMSS.log` - Performance test results
- `results/monitor_YYYYMMDD_HHMMSS.json` - System monitoring data

## Bottleneck Analysis

The script automatically identifies potential bottlenecks:

- **Low GPU utilization** → Bottleneck is NOT the GPU
- **High CPU utilization (>80%)** → CPU may be the bottleneck
- **High memory utilization (>90%)** → Memory may be the bottleneck
- **Low resource utilization** → Likely network/serialization overhead

## Manual Testing

If you want to run components separately:

### Performance Test Only
```bash
python simple_performance_test.py --iterations 100
```

### Monitoring Only
```bash
python system_monitor.py --duration 60 --interval 1.0 --output results/monitor.json
```

## Requirements

- OpenPI server must be running on `localhost:8000`
- Dependencies: `websockets`, `msgpack-numpy`, `psutil`, `numpy`

## Typical Results (Jetson Thor)

- **Throughput**: ~12-13 actions/second
- **Latency**: ~80ms average (72-93ms range)
- **GPU Utilization**: <5% (GPU is not the bottleneck)
- **CPU Utilization**: ~8-10% average

The low resource utilization suggests the bottleneck is likely:
- WebSocket communication overhead
- Data serialization/deserialization (MessagePack ↔ JSON)
- JAX compilation/initialization overhead

