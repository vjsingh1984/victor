#!/bin/bash
# Comprehensive Benchmarking Script for Victor AI
#
# This script runs all performance benchmarks including tool selection,
# cache performance, bootstrap, memory, and workflow performance.
#
# Usage:
#   ./scripts/benchmark_all.sh [options]
#
# Options:
#   --baseline FILE      Baseline JSON file for comparison (optional)
#   --output FILE        Output directory for results (default: /tmp/victor_benchmarks)
#   --iterations N       Number of iterations for averaging (default: 10)
#   --warmup N           Number of warmup iterations (default: 3)
#   --skip-tool-sel      Skip tool selection benchmarks
#   --skip-cache         Skip cache performance benchmarks
#   --skip-bootstrap     Skip bootstrap benchmarks
#   --skip-memory        Skip memory benchmarks
#   --skip-workflow      Skip workflow benchmarks
#   --parallel           Run benchmarks in parallel (experimental)
#   --verbose            Enable verbose output
#   --help               Show this help message

set -euo pipefail

# Default values
OUTPUT_DIR="/tmp/victor_benchmarks"
BASELINE_FILE=""
ITERATIONS=10
WARMUP=3
SKIP_TOOL_SEL=false
SKIP_CACHE=false
SKIP_BOOTSTRAP=false
SKIP_MEMORY=false
SKIP_WORKFLOW=false
PARALLEL=false
VERBOSE=false
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${CYAN}[VERBOSE]${NC} $*"
    fi
}

log_section() {
    echo ""
    echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  $*${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline)
            BASELINE_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --skip-tool-sel)
            SKIP_TOOL_SEL=true
            shift
            ;;
        --skip-cache)
            SKIP_CACHE=true
            shift
            ;;
        --skip-bootstrap)
            SKIP_BOOTSTRAP=true
            shift
            ;;
        --skip-memory)
            SKIP_MEMORY=true
            shift
            ;;
        --skip-workflow)
            SKIP_WORKFLOW=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --baseline FILE      Baseline JSON file for comparison (optional)"
            echo "  --output FILE        Output directory for results (default: /tmp/victor_benchmarks)"
            echo "  --iterations N       Number of iterations for averaging (default: 10)"
            echo "  --warmup N           Number of warmup iterations (default: 3)"
            echo "  --skip-tool-sel      Skip tool selection benchmarks"
            echo "  --skip-cache         Skip cache performance benchmarks"
            echo "  --skip-bootstrap     Skip bootstrap benchmarks"
            echo "  --skip-memory        Skip memory benchmarks"
            echo "  --skip-workflow      Skip workflow benchmarks"
            echo "  --parallel           Run benchmarks in parallel (experimental)"
            echo "  --verbose            Enable verbose output"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

log_section "Victor AI Comprehensive Benchmarking"
log_info "Output directory: $OUTPUT_DIR"
log_info "Iterations: $ITERATIONS"
log_info "Warmup iterations: $WARMUP"
if [[ -n "$BASELINE_FILE" ]]; then
    log_info "Baseline file: $BASELINE_FILE"
fi
log_info ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is required but not installed"
    exit 1
fi

# Check if victor is installed
if ! python3 -c "import victor" 2>/dev/null; then
    log_error "Victor is not installed. Run: pip install -e ."
    exit 1
fi

# Load baseline if provided
BASELINE_LOADED=false
if [[ -n "$BASELINE_FILE" && -f "$BASELINE_FILE" ]]; then
    log_info "Loading baseline from $BASELINE_FILE"
    BASELINE_LOADED=true
fi

# Initialize results JSON
RESULTS_FILE="$OUTPUT_DIR/benchmark_results.json"
cat > "$RESULTS_FILE" << EOF
{
  "timestamp": "$TIMESTAMP",
  "iterations": $ITERATIONS,
  "warmup_iterations": $WARMUP,
  "benchmarks": {}
}
EOF

# Function to add benchmark result to JSON
add_benchmark() {
    local name="$1"
    local metric="$2"
    local value="$3"
    local unit="$4"

    python3 << EOF
import json

with open("$RESULTS_FILE", "r") as f:
    data = json.load(f)

if "$name" not in data["benchmarks"]:
    data["benchmarks"]["$name"] = {}

keys = "$metric".split(".")
current = data["benchmarks"]["$name"]
for k in keys[:-1]:
    if k not in current:
        current[k] = {}
    current = current[k]

current[keys[-1]] = {
    "value": $value,
    "unit": "$unit"
}

with open("$RESULTS_FILE", "w") as f:
    json.dump(data, f, indent=2)
EOF
}

# Function to compare with baseline
compare_baseline() {
    local name="$1"
    local metric="$2"
    local value="$3"

    if [[ "$BASELINE_LOADED" == "false" ]]; then
        return
    fi

    BASELINE_VALUE=$(python3 << EOF
import json
import sys

try:
    with open("$BASELINE_FILE", "r") as f:
        data = json.load(f)

    keys = "$metric".split(".")
    current = data["metrics"]
    for k in keys:
        current = current[k]

    print(current["value"])
except:
    sys.exit(1)
EOF
)

    if [[ $? -eq 0 ]]; then
        IMPROVEMENT=$(python3 << EOF
baseline = float("$BASELINE_VALUE")
current = float("$value")
improvement = ((baseline - current) / baseline) * 100
print(f"{improvement:+.1f}")
EOF
)
        echo -e " ${GREEN}(${IMPROVEMENT}% vs baseline)${NC}"
    fi
}

################################################################################
# Tool Selection Benchmarks
################################################################################

if [[ "$SKIP_TOOL_SEL" == "false" ]]; then
    log_section "Tool Selection Benchmarks"

    # Cold cache benchmark
    log_info "Running cold cache benchmark..."

    COLD_CACHE_TIMES=()
    for i in $(seq 1 $((ITERATIONS + WARMUP))); do
        if [[ $i -le $WARMUP ]]; then
            log_verbose "Warmup iteration $i/$WARMUP"
        else
            log_verbose "Cold cache iteration $((i - WARMUP))/$ITERATIONS"
        fi

        START=$(python3 -c "import time; print(time.time())")

        python3 << EOF
import sys
import asyncio
sys.path.insert(0, "$PROJECT_ROOT")

from victor.core.bootstrap import bootstrap_container
from victor.config.settings import Settings
settings = Settings()
from victor.agent.service_provider import configure_orchestrator_services
from victor.agent.protocols import ToolCoordinatorProtocol
from victor.agent.tool_selection.context import AgentToolSelectionContext

async def run():
    container = bootstrap_container()
    configure_orchestrator_services(container, settings)
    coordinator = container.get(ToolCoordinatorProtocol)

    context = AgentToolSelectionContext(
        query="Analyze Python code for security vulnerabilities and performance issues",
        max_tools=10,
    )

    await coordinator.select_and_execute(context)

asyncio.run(run())
EOF

        END=$(python3 -c "import time; print(time.time())")
        ELAPSED=$(python3 -c "print($END - $START)")

        if [[ $i -gt $WARMUP ]]; then
            COLD_CACHE_TIMES+=("$ELAPSED")
        fi
    done

    COLD_AVG=$(python3 << EOF
times = ${COLD_CACHE_TIMES[@]}
avg = sum(times) / len(times)
print(f"{avg:.3f}")
EOF
)

    COLD_P50=$(python3 << EOF
times = sorted(${COLD_CACHE_TIMES[@]})
p50 = times[len(times) // 2]
print(f"{p50:.3f}")
EOF
)

    COLD_P95=$(python3 << EOF
times = sorted(${COLD_CACHE_TIMES[@]})
idx = int(len(times) * 0.95)
p95 = times[idx]
print(f"{p95:.3f}")
EOF
)

    COLD_P99=$(python3 << EOF
times = sorted(${COLD_CACHE_TIMES[@]})
idx = int(len(times) * 0.99)
p99 = times[idx]
print(f"{p99:.3f}")
EOF
)

    echo -n "Cold Cache: avg=${COLD_AVG}ms, p50=${COLD_P50}ms, p95=${COLD_P95}ms, p99=${COLD_P99}ms"
    compare_baseline "tool_selection" "cold_cache.average" "$COLD_AVG"
    echo ""

    add_benchmark "tool_selection" "cold_cache.average" "$COLD_AVG" "ms"
    add_benchmark "tool_selection" "cold_cache.p50" "$COLD_P50" "ms"
    add_benchmark "tool_selection" "cold_cache.p95" "$COLD_P95" "ms"
    add_benchmark "tool_selection" "cold_cache.p99" "$COLD_P99" "ms"

    # Warm cache benchmark
    log_info "Running warm cache benchmark..."

    WARM_CACHE_TIMES=()
    for i in $(seq 1 $((ITERATIONS + WARMUP))); do
        if [[ $i -le $WARMUP ]]; then
            log_verbose "Warmup iteration $i/$WARMUP"
        else
            log_verbose "Warm cache iteration $((i - WARMUP))/$ITERATIONS"
        fi

        START=$(python3 -c "import time; print(time.time())")

        python3 << EOF
import sys
import asyncio
sys.path.insert(0, "$PROJECT_ROOT")

from victor.core.bootstrap import bootstrap_container
from victor.config.settings import Settings
settings = Settings()
from victor.agent.service_provider import configure_orchestrator_services
from victor.agent.protocols import ToolCoordinatorProtocol
from victor.agent.tool_selection.context import AgentToolSelectionContext

async def run():
    container = bootstrap_container()
    configure_orchestrator_services(container, settings)
    coordinator = container.get(ToolCoordinatorProtocol)

    # Run same query multiple times to test cache
    context = AgentToolSelectionContext(
        query="Analyze Python code for security vulnerabilities and performance issues",
        max_tools=10,
    )

    await coordinator.select_and_execute(context)

asyncio.run(run())
EOF

        END=$(python3 -c "import time; print(time.time())")
        ELAPSED=$(python3 -c "print($END - $START)")

        if [[ $i -gt $WARMUP ]]; then
            WARM_CACHE_TIMES+=("$ELAPSED")
        fi
    done

    WARM_AVG=$(python3 << EOF
times = ${WARM_CACHE_TIMES[@]}
avg = sum(times) / len(times)
print(f"{avg:.3f}")
EOF
)

    WARM_P50=$(python3 << EOF
times = sorted(${WARM_CACHE_TIMES[@]})
p50 = times[len(times) // 2]
print(f"{p50:.3f}")
EOF
)

    WARM_P95=$(python3 << EOF
times = sorted(${WARM_CACHE_TIMES[@]})
idx = int(len(times) * 0.95)
p95 = times[idx]
print(f"{p95:.3f}")
EOF
)

    echo -n "Warm Cache: avg=${WARM_AVG}ms, p50=${WARM_P50}ms, p95=${WARM_P95}ms"
    SPEEDUP=$(python3 << EOF
cold = $COLD_AVG
warm = $WARM_AVG
speedup = cold / warm
print(f"{speedup:.2f}x")
EOF
)
    echo -e " ${GREEN}(${SPEEDUP} speedup)${NC}"
    echo ""

    add_benchmark "tool_selection" "warm_cache.average" "$WARM_AVG" "ms"
    add_benchmark "tool_selection" "warm_cache.p50" "$WARM_P50" "ms"
    add_benchmark "tool_selection" "warm_cache.p95" "$WARM_P95" "ms"

    # Cache hit rate
    log_info "Measuring cache hit rate..."

    CACHE_STATS=$(python3 << EOF
import sys
sys.path.insert(0, "$PROJECT_ROOT")

from victor.core.bootstrap import bootstrap_container
from victor.agent.service_provider import configure_orchestrator_services
from victor.config.settings import Settings
settings = Settings()
from victor.agent.coordinators.tool_coordinator import ToolCoordinator
from victor.core.registries import UniversalRegistry, CacheStrategy

# Get cache registry
cache = UniversalRegistry.get_registry("tool_selection", CacheStrategy.LRU)

stats = cache.get_stats()
hit_rate = stats.get("hit_rate", 0.0)
total_hits = stats.get("total_accesses", 0) - stats.get("total_misses", 0)
total_misses = stats.get("total_misses", 0)

print(f"{hit_rate:.2f},{total_hits},{total_misses}")
EOF
)

    HIT_RATE=$(echo "$CACHE_STATS" | cut -d',' -f1)
    TOTAL_HITS=$(echo "$CACHE_STATS" | cut -d',' -f2)
    TOTAL_MISSES=$(echo "$CACHE_STATS" | cut -d',' -f3)

    echo -n "Cache Hit Rate: ${HIT_RATE}% (${TOTAL_HITS} hits, ${TOTAL_MISSES} misses)"
    if [[ $(python3 -c "print(float('$HIT_RATE') >= 70.0)") == "True" ]]; then
        echo -e " ${GREEN}(✓ Target met)${NC}"
    else
        echo -e " ${YELLOW}(! Below 70% target)${NC}"
    fi
    echo ""

    add_benchmark "cache" "hit_rate" "$HIT_RATE" "percent"
    add_benchmark "cache" "total_hits" "$TOTAL_HITS" "count"
    add_benchmark "cache" "total_misses" "$TOTAL_MISSES" "count"
fi

################################################################################
# Cache Performance Benchmarks
################################################################################

if [[ "$SKIP_CACHE" == "false" ]]; then
    log_section "Cache Performance Benchmarks"

    # Cache key generation
    log_info "Benchmarking cache key generation..."

    CACHE_KEY_TIMES=()
    for i in $(seq 1 $((ITERATIONS + WARMUP))); do
        if [[ $i -le $WARMUP ]]; then
            log_verbose "Warmup iteration $i/$WARMUP"
        else
            log_verbose "Cache key iteration $((i - WARMUP))/$ITERATIONS"
        fi

        START=$(python3 -c "import time; print(time.time())")

        python3 << EOF
import sys
sys.path.insert(0, "$PROJECT_ROOT")

from victor.core.bootstrap import bootstrap_container
from victor.agent.service_provider import configure_orchestrator_services
from victor.config.settings import Settings
settings = Settings()
from victor.agent.coordinators.tool_coordinator import ToolCoordinator

container = bootstrap_container()
configure_orchestrator_services(container, settings)

# Generate 100 cache keys
for i in range(100):
    key = ToolCoordinator._generate_selection_key(
        query=f"Test query {i}",
        max_tools=10,
        conversation_hash=f"conv_{i % 5}",
        pending_actions_hash=f"pending_{i % 3}",
    )
EOF

        END=$(python3 -c "import time; print(time.time())")
        ELAPSED=$(python3 -c "print(($END - $START) * 1000000)")

        if [[ $i -gt $WARMUP ]]; then
            CACHE_KEY_TIMES+=("$ELAPSED")
        fi
    done

    CACHE_KEY_AVG=$(python3 << EOF
times = ${CACHE_KEY_TIMES[@]}
avg = sum(times) / len(times)
print(f"{avg:.1f}")
EOF
)

    echo -n "Cache Key Generation: avg=${CACHE_KEY_AVG}μs per 100 keys"
    PER_KEY=$(python3 << EOF
avg = $CACHE_KEY_AVG
per_key = avg / 100
print(f"{per_key:.2f}")
EOF
)
    echo " (${PER_KEY}μs per key)"
    echo ""

    add_benchmark "cache" "key_generation_100" "$CACHE_KEY_AVG" "microseconds"
    add_benchmark "cache" "key_generation_single" "$PER_KEY" "microseconds"
fi

################################################################################
# Bootstrap Benchmarks
################################################################################

if [[ "$SKIP_BOOTSTRAP" == "false" ]]; then
    log_section "Bootstrap Benchmarks"

    log_info "Benchmarking bootstrap time..."

    BOOTSTRAP_TIMES=()
    for i in $(seq 1 $((ITERATIONS + WARMUP))); do
        if [[ $i -le $WARMUP ]]; then
            log_verbose "Warmup iteration $i/$WARMUP"
        else
            log_verbose "Bootstrap iteration $((i - WARMUP))/$ITERATIONS"
        fi

        START=$(python3 -c "import time; print(time.time())")

        python3 << EOF
import sys
sys.path.insert(0, "$PROJECT_ROOT")

from victor.core.bootstrap import bootstrap_container

container = bootstrap_container()
EOF

        END=$(python3 -c "import time; print(time.time())")
        ELAPSED=$(python3 -c "print($END - $START)")

        if [[ $i -gt $WARMUP ]]; then
            BOOTSTRAP_TIMES+=("$ELAPSED")
        fi
    done

    BOOTSTRAP_AVG=$(python3 << EOF
times = ${BOOTSTRAP_TIMES[@]}
avg = sum(times) / len(times)
print(f"{avg:.3f}")
EOF
)

    BOOTSTRAP_P50=$(python3 << EOF
times = sorted(${BOOTSTRAP_TIMES[@]})
p50 = times[len(times) // 2]
print(f"{p50:.3f}")
EOF
)

    BOOTSTRAP_P95=$(python3 << EOF
times = sorted(${BOOTSTRAP_TIMES[@]})
idx = int(len(times) * 0.95)
p95 = times[idx]
print(f"{p95:.3f}")
EOF
)

    echo -n "Bootstrap Time: avg=${BOOTSTRAP_AVG}ms, p50=${BOOTSTRAP_P50}ms, p95=${BOOTSTRAP_P95}ms"
    compare_baseline "bootstrap" "time.average" "$BOOTSTRAP_AVG"
    echo ""

    add_benchmark "bootstrap" "time.average" "$BOOTSTRAP_AVG" "ms"
    add_benchmark "bootstrap" "time.p50" "$BOOTSTRAP_P50" "ms"
    add_benchmark "bootstrap" "time.p95" "$BOOTSTRAP_P95" "ms"
fi

################################################################################
# Memory Benchmarks
################################################################################

if [[ "$SKIP_MEMORY" == "false" ]]; then
    log_section "Memory Benchmarks"

    log_info "Benchmarking memory usage..."

    MEMORY_USAGE=$(python3 << EOF
import sys
import tracemalloc
sys.path.insert(0, "$PROJECT_ROOT")

tracemalloc.start()

from victor.core.bootstrap import bootstrap_container
from victor.agent.service_provider import configure_orchestrator_services
from victor.config.settings import Settings
settings = Settings()

container = bootstrap_container()
configure_orchestrator_services(container, settings)

# Simulate some work
from victor.agent.protocols import ToolCoordinatorProtocol
from victor.agent.tool_selection.context import AgentToolSelectionContext
import asyncio

async def simulate_work():
    coordinator = container.get(ToolCoordinatorProtocol)

    for i in range(10):
        context = AgentToolSelectionContext(
            query=f"Test query {i}",
            max_tools=5,
        )
        await coordinator.select_and_execute(context)

asyncio.run(simulate_work())

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

current_mb = current / 1024 / 1024
peak_mb = peak / 1024 / 1024

print(f"{current_mb:.2f},{peak_mb:.2f}")
EOF
)

    MEMORY_CURRENT=$(echo "$MEMORY_USAGE" | cut -d',' -f1)
    MEMORY_PEAK=$(echo "$MEMORY_USAGE" | cut -d',' -f2)

    echo -n "Memory Usage: current=${MEMORY_CURRENT}MB, peak=${MEMORY_PEAK}MB"
    compare_baseline "memory" "current" "$MEMORY_CURRENT"
    echo ""

    add_benchmark "memory" "current" "$MEMORY_CURRENT" "MB"
    add_benchmark "memory" "peak" "$MEMORY_PEAK" "MB"

    # Memory profiling snapshot
    log_info "Capturing memory profile..."

    python3 << EOF
import sys
import tracemalloc
sys.path.insert(0, "$PROJECT_ROOT")

tracemalloc.start()

from victor.core.bootstrap import bootstrap_container
from victor.agent.service_provider import configure_orchestrator_services
from victor.config.settings import Settings
settings = Settings()

container = bootstrap_container()
configure_orchestrator_services(container, settings)

# Take snapshot
snapshot = tracemalloc.take_snapshot()
tracemalloc.stop()

# Save top statistics
with open("$OUTPUT_DIR/memory_profile.txt", "w") as f:
    f.write("Top 20 memory allocations:\n")
    f.write("=" * 80 + "\n\n")

    top_stats = snapshot.statistics('lineno')

    for stat in top_stats[:20]:
        f.write(f"{stat}\n")

print("Memory profile saved to $OUTPUT_DIR/memory_profile.txt")
EOF
fi

################################################################################
# Workflow Benchmarks
################################################################################

if [[ "$SKIP_WORKFLOW" == "false" ]]; then
    log_section "Workflow Benchmarks"

    log_info "Benchmarking workflow compilation..."

    WORKFLOW_TIMES=()
    for i in $(seq 1 $((ITERATIONS + WARMUP))); do
        if [[ $i -le $WARMUP ]]; then
            log_verbose "Warmup iteration $i/$WARMUP"
        else
            log_verbose "Workflow iteration $((i - WARMUP))/$ITERATIONS"
        fi

        START=$(python3 -c "import time; print(time.time())")

        python3 << EOF
import sys
import asyncio
sys.path.insert(0, "$PROJECT_ROOT")

from victor.core.bootstrap import bootstrap_container
from victor.config.settings import Settings
settings = Settings()
from victor.agent.service_provider import configure_orchestrator_services
from victor.framework.graph import StateGraph, START, END

async def compile_workflow():
    container = bootstrap_container()
    configure_orchestrator_services(container, settings)

    # Create complex workflow
    graph = StateGraph("test_workflow")

    graph.add_node("analyze", lambda state: {**state, "stage": "analysis"})
    graph.add_node("process", lambda state: {**state, "stage": "processing"})
    graph.add_node("validate", lambda state: {**state, "stage": "validation"})
    graph.add_node("finalize", lambda state: {**state, "stage": "final"})

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "process")
    graph.add_edge("process", "validate")
    graph.add_edge("validate", "finalize")
    graph.add_edge("finalize", END)

    compiled = graph.compile()
    await compiled.ainvoke({"test": "data"})

asyncio.run(compile_workflow())
EOF

        END=$(python3 -c "import time; print(time.time())")
        ELAPSED=$(python3 -c "print($END - $START)")

        if [[ $i -gt $WARMUP ]]; then
            WORKFLOW_TIMES+=("$ELAPSED")
        fi
    done

    WORKFLOW_AVG=$(python3 << EOF
times = ${WORKFLOW_TIMES[@]}
avg = sum(times) / len(times)
print(f"{avg:.3f}")
EOF
)

    WORKFLOW_P50=$(python3 << EOF
times = sorted(${WORKFLOW_TIMES[@]})
p50 = times[len(times) // 2]
print(f"{p50:.3f}")
EOF
)

    echo -n "Workflow Compilation: avg=${WORKFLOW_AVG}ms, p50=${WORKFLOW_P50}ms"
    compare_baseline "workflow" "compilation_time" "$WORKFLOW_AVG"
    echo ""

    add_benchmark "workflow" "compilation_time" "$WORKFLOW_AVG" "ms"
    add_benchmark "workflow" "compilation_p50" "$WORKFLOW_P50" "ms"
fi

################################################################################
# Summary
################################################################################

log_section "Benchmark Summary"

python3 << EOF
import json

with open("$RESULTS_FILE", "r") as f:
    data = json.load(f)

benchmarks = data["benchmarks"]

print("╔═══════════════════════════════════════════════════════════════╗")
print("║                    BENCHMARK RESULTS                          ║")
print("╚═══════════════════════════════════════════════════════════════╝")
print()

if "tool_selection" in benchmarks:
    ts = benchmarks["tool_selection"]
    print("Tool Selection:")
    print(f"  Cold Cache (avg):   {ts['cold_cache']['average']['value']:.3f} {ts['cold_cache']['average']['unit']}")
    print(f"  Cold Cache (p95):   {ts['cold_cache']['p95']['value']:.3f} {ts['cold_cache']['p95']['unit']}")
    print(f"  Warm Cache (avg):   {ts['warm_cache']['average']['value']:.3f} {ts['warm_cache']['average']['unit']}")
    if ts['cold_cache']['average']['value'] > 0:
        speedup = ts['cold_cache']['average']['value'] / ts['warm_cache']['average']['value']
        print(f"  Cache Speedup:      {speedup:.2f}x")
    print()

if "cache" in benchmarks and "hit_rate" in benchmarks["cache"]:
    cache = benchmarks["cache"]
    print("Cache Performance:")
    print(f"  Hit Rate:           {cache['hit_rate']['value']:.1f}%")
    if 'total_hits' in cache:
        print(f"  Total Hits:         {cache['total_hits']['value']}")
        print(f"  Total Misses:       {cache['total_misses']['value']}")
    if 'key_generation_single' in cache:
        print(f"  Key Generation:     {cache['key_generation_single']['value']:.2f} {cache['key_generation_single']['unit']}")
    print()

if "bootstrap" in benchmarks:
    boot = benchmarks["bootstrap"]
    print("Bootstrap:")
    print(f"  Time (avg):         {boot['time']['average']['value']:.3f} {boot['time']['average']['unit']}")
    print(f"  Time (p95):         {boot['time']['p95']['value']:.3f} {boot['time']['p95']['unit']}")
    print()

if "memory" in benchmarks:
    mem = benchmarks["memory"]
    print("Memory:")
    print(f"  Current:            {mem['current']['value']:.2f} {mem['current']['unit']}")
    print(f"  Peak:               {mem['peak']['value']:.2f} {mem['peak']['unit']}")
    print()

if "workflow" in benchmarks:
    wf = benchmarks["workflow"]
    print("Workflow:")
    print(f"  Compilation (avg):  {wf['compilation_time']['value']:.3f} {wf['compilation_time']['unit']}")
    print()
EOF

log_success "Benchmark results saved to: $RESULTS_FILE"
log_info ""
log_info "To generate a performance report:"
log_info "  ./scripts/performance_report.sh --baseline $BASELINE_FILE --current $RESULTS_FILE --output report.md"
