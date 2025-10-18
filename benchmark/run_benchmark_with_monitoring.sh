#!/bin/bash
# Run OpenPI performance test with system monitoring

# Default values
ITERATIONS=200
MONITOR_INTERVAL=1.0
OUTPUT_DIR="./benchmark_results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations|-n)
            ITERATIONS="$2"
            shift 2
            ;;
        --quick)
            ITERATIONS=50
            shift
            ;;
        --monitor-interval|-i)
            MONITOR_INTERVAL="$2"
            shift 2
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --iterations, -n NUM     Number of inference iterations (default: 200)"
            echo "  --quick                  Run quick test with 50 iterations"
            echo "  --monitor-interval, -i   Monitoring interval in seconds (default: 1.0)"
            echo "  --output-dir, -o DIR     Output directory for results (default: ./benchmark_results)"
            echo "  --help, -h               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PERFORMANCE_LOG="$OUTPUT_DIR/performance_$TIMESTAMP.log"
MONITOR_LOG="$OUTPUT_DIR/monitor_$TIMESTAMP.json"

echo "Starting OpenPI performance benchmark with monitoring..."
echo "Iterations: $ITERATIONS"
echo "Monitor interval: ${MONITOR_INTERVAL}s"
echo "Performance log: $PERFORMANCE_LOG"
echo "Monitor log: $MONITOR_LOG"
echo ""

# Start system monitoring in background
echo "Starting system monitoring..."
python3 benchmark/system_monitor.py \
    --interval "$MONITOR_INTERVAL" \
    --output "$MONITOR_LOG" &
MONITOR_PID=$!

# Give monitor a moment to start
sleep 2

# Run performance test
echo "Starting performance test..."
python3 benchmark/simple_performance_test.py --iterations "$ITERATIONS" > "$PERFORMANCE_LOG" 2>&1
PERFORMANCE_EXIT_CODE=$?

# Stop monitoring
echo "Stopping system monitoring..."
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

# Show results
echo ""
echo "Benchmark completed!"
echo "Performance results:"
cat "$PERFORMANCE_LOG"

echo ""
echo "Monitor data saved to: $MONITOR_LOG"
echo "Performance log saved to: $PERFORMANCE_LOG"

exit $PERFORMANCE_EXIT_CODE


