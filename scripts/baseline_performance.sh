#!/bin/bash
# Performance Baseline Measurement Script for Victor AI
#
# This script establishes a performance baseline by measuring all key performance indicators.
# It records baseline metrics for future comparison and optimization tracking.
#
# Usage:
#   ./scripts/baseline_performance.sh [options]
#
# Options:
#   --output FILE      Output JSON file for baseline results (default: /tmp/victor_baseline.json)
#   --iterations N     Number of iterations for averaging (default: 10)
#   --warmup N         Number of warmup iterations (default: 3)
#   --verbose          Enable verbose output
#   --skip-cpu         Skip CPU profiling
#   --skip-memory      Skip memory profiling
#   --help             Show this help message

set -euo pipefail

# Default values
OUTPUT_FILE="/tmp/victor_baseline.json"
ITERATIONS=10
WARMUP=3
VERBOSE=false
SKIP_CPU=false
SKIP_MEMORY=false
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
        echo -e "${BLUE}[VERBOSE]${NC} $*"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            OUTPUT_FILE="$2"
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
        --verbose)
            VERBOSE=true
            shift
            ;;
        --skip-cpu)
            SKIP_CPU=true
            shift
            ;;
        --skip-memory)
            SKIP_MEMORY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --output FILE      Output JSON file for baseline results (default: /tmp/victor_baseline.json)"
            echo "  --iterations N     Number of iterations for averaging (default: 10)"
            echo "  --warmup N         Number of warmup iterations (default: 3)"
            echo "  --verbose          Enable verbose output"
            echo "  --skip-cpu         Skip CPU profiling"
            echo "  --skip-memory      Skip memory profiling"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

log_info "Victor AI Performance Baseline Measurement"
log_info "==========================================="
log_info "Output file: $OUTPUT_FILE"
log_info "Iterations: $ITERATIONS"
log_info "Warmup iterations: $WARMUP"
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

# Initialize JSON output
cat > "$OUTPUT_FILE" << EOF
{
  "timestamp": "$TIMESTAMP",
  "iterations": $ITERATIONS,
  "warmup_iterations": $WARMUP,
  "metrics": {}
}
EOF

# Function to add metric to JSON
add_metric() {
    local key="$1"
    local value="$2"
    local unit="$3"

    python3 << EOF
import json
import sys

with open("$OUTPUT_FILE", "r") as f:
    data = json.load(f)

# Build nested structure
keys = "$key".split(".")
current = data["metrics"]
for k in keys[:-1]:
    if k not in current:
        current[k] = {}
    current = current[k]

current[keys[-1]] = {
    "value": $value,
    "unit": "$unit"
}

with open("$OUTPUT_FILE", "w") as f:
    json.dump(data, f, indent=2)
EOF
}

# Measure bootstrap time
log_info "Measuring bootstrap time..."

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
from victor.config.settings import Settings
settings = Settings()

container = bootstrap_container()
EOF

    END=$(python3 -c "import time; print(time.time())")
    ELAPSED=$(python3 -c "print($END - $START)")

    if [[ $i -gt $WARMUP ]]; then
        BOOTSTRAP_TIMES+=("$ELAPSED")
    fi
done

# Calculate statistics
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

BOOTSTRAP_P99=$(python3 << EOF
times = sorted(${BOOTSTRAP_TIMES[@]})
idx = int(len(times) * 0.99)
p99 = times[idx]
print(f"{p99:.3f}")
EOF
)

log_success "Bootstrap time: avg=${BOOTSTRAP_AVG}ms, p50=${BOOTSTRAP_P50}ms, p95=${BOOTSTRAP_P95}ms, p99=${BOOTSTRAP_P99}ms"

add_metric "bootstrap.time.average" "$BOOTSTRAP_AVG" "ms"
add_metric "bootstrap.time.p50" "$BOOTSTRAP_P50" "ms"
add_metric "bootstrap.time.p95" "$BOOTSTRAP_P95" "ms"
add_metric "bootstrap.time.p99" "$BOOTSTRAP_P99" "ms"

# Measure startup time (full application startup)
log_info "Measuring startup time..."

STARTUP_TIMES=()
for i in $(seq 1 $((ITERATIONS + WARMUP))); do
    if [[ $i -le $WARMUP ]]; then
        log_verbose "Warmup iteration $i/$WARMUP"
    else
        log_verbose "Startup iteration $((i - WARMUP))/$ITERATIONS"
    fi

    START=$(python3 -c "import time; print(time.time())")

    python3 << EOF
import sys
sys.path.insert(0, "$PROJECT_ROOT")

from victor.core.bootstrap import bootstrap_container
from victor.config.settings import Settings
settings = Settings()
from victor.agent.service_provider import configure_orchestrator_services

container = bootstrap_container()
configure_orchestrator_services(container, settings)
EOF

    END=$(python3 -c "import time; print(time.time())")
    ELAPSED=$(python3 -c "print($END - $START)")

    if [[ $i -gt $WARMUP ]]; then
        STARTUP_TIMES+=("$ELAPSED")
    fi
done

STARTUP_AVG=$(python3 << EOF
times = ${STARTUP_TIMES[@]}
avg = sum(times) / len(times)
print(f"{avg:.3f}")
EOF
)

log_success "Startup time: avg=${STARTUP_AVG}ms"
add_metric "startup.time" "$STARTUP_AVG" "ms"

# Measure tool selection performance
log_info "Measuring tool selection performance..."

TOOL_SELECTION_TIMES=()
for i in $(seq 1 $((ITERATIONS + WARMUP))); do
    if [[ $i -le $WARMUP ]]; then
        log_verbose "Warmup iteration $i/$WARMUP"
    else
        log_verbose "Tool selection iteration $((i - WARMUP))/$ITERATIONS"
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

async def measure():
    container = bootstrap_container()
    configure_orchestrator_services(container, settings)
    coordinator = container.get(ToolCoordinatorProtocol)

    context = AgentToolSelectionContext(
        query="Analyze Python code for security issues",
        max_tools=10,
    )

    await coordinator.select_and_execute(context)

asyncio.run(measure())
EOF

    END=$(python3 -c "import time; print(time.time())")
    ELAPSED=$(python3 -c "print($END - $START)")

    if [[ $i -gt $WARMUP ]]; then
        TOOL_SELECTION_TIMES+=("$ELAPSED")
    fi
done

TOOL_SEL_AVG=$(python3 << EOF
times = ${TOOL_SELECTION_TIMES[@]}
avg = sum(times) / len(times)
print(f"{avg:.3f}")
EOF
)

TOOL_SEL_P50=$(python3 << EOF
times = sorted(${TOOL_SELECTION_TIMES[@]})
p50 = times[len(times) // 2]
print(f"{p50:.3f}")
EOF
)

TOOL_SEL_P95=$(python3 << EOF
times = sorted(${TOOL_SELECTION_TIMES[@]})
idx = int(len(times) * 0.95)
p95 = times[idx]
print(f"{p95:.3f}")
EOF
)

log_success "Tool selection: avg=${TOOL_SEL_AVG}ms, p50=${TOOL_SEL_P50}ms, p95=${TOOL_SEL_P95}ms"

add_metric "tool_selection.cold_cache.average" "$TOOL_SEL_AVG" "ms"
add_metric "tool_selection.cold_cache.p50" "$TOOL_SEL_P50" "ms"
add_metric "tool_selection.cold_cache.p95" "$TOOL_SEL_P95" "ms"

# Measure cache key generation
log_info "Measuring cache key generation..."

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
from victor.agent.tool_selection.context import AgentToolSelectionContext
import hashlib

container = bootstrap_container()
configure_orchestrator_services(container, settings)

context = AgentToolSelectionContext(
    query="Analyze Python code for security issues",
    max_tools=10,
)

# Generate cache key
key = ToolCoordinator._generate_selection_key(
    query=context.query,
    max_tools=context.max_tools,
    conversation_hash="abc123",
    pending_actions_hash="def456",
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

log_success "Cache key generation: avg=${CACHE_KEY_AVG}μs"
add_metric "cache.key_generation" "$CACHE_KEY_AVG" "microseconds"

# Measure memory usage
if [[ "$SKIP_MEMORY" == "false" ]]; then
    log_info "Measuring memory usage..."

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

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

# Convert to MB
current_mb = current / 1024 / 1024
peak_mb = peak / 1024 / 1024

print(f"{current_mb:.2f},{peak_mb:.2f}")
EOF
)

    MEMORY_CURRENT=$(echo "$MEMORY_USAGE" | cut -d',' -f1)
    MEMORY_PEAK=$(echo "$MEMORY_USAGE" | cut -d',' -f2)

    log_success "Memory usage: current=${MEMORY_CURRENT}MB, peak=${MEMORY_PEAK}MB"

    add_metric "memory.current" "$MEMORY_CURRENT" "MB"
    add_metric "memory.peak" "$MEMORY_PEAK" "MB"
fi

# Measure provider pool initialization
log_info "Measuring provider pool initialization..."

PROVIDER_POOL_TIMES=()
for i in $(seq 1 $((ITERATIONS + WARMUP))); do
    if [[ $i -le $WARMUP ]]; then
        log_verbose "Warmup iteration $i/$WARMUP"
    else
        log_verbose "Provider pool iteration $((i - WARMUP))/$ITERATIONS"
    fi

    START=$(python3 -c "import time; print(time.time())")

    python3 << EOF
import sys
sys.path.insert(0, "$PROJECT_ROOT")

from victor.core.bootstrap import bootstrap_container
from victor.config.settings import Settings
settings = Settings()
from victor.providers.provider_factory import ProviderFactory
from victor.core.events import MessagingEvent, create_event_backend

container = bootstrap_container()
factory = ProviderFactory(container)
EOF

    END=$(python3 -c "import time; print(time.time())")
    ELAPSED=$(python3 -c "print($END - $START)")

    if [[ $i -gt $WARMUP ]]; then
        PROVIDER_POOL_TIMES+=("$ELAPSED")
    fi
done

PROVIDER_POOL_AVG=$(python3 << EOF
times = ${PROVIDER_POOL_TIMES[@]}
avg = sum(times) / len(times)
print(f"{avg:.3f}")
EOF
)

log_success "Provider pool initialization: avg=${PROVIDER_POOL_AVG}ms"
add_metric "provider_pool.initialization_time" "$PROVIDER_POOL_AVG" "ms"

# Measure workflow compilation
log_info "Measuring workflow compilation..."

WORKFLOW_COMPILE_TIMES=()
for i in $(seq 1 $((ITERATIONS + WARMUP))); do
    if [[ $i -le $WARMUP ]]; then
        log_verbose "Warmup iteration $i/$WARMUP"
    else
        log_verbose "Workflow compile iteration $((i - WARMUP))/$ITERATIONS"
    fi

    START=$(python3 -c "import time; print(time.time())")

    python3 << EOF
import sys
sys.path.insert(0, "$PROJECT_ROOT")

from victor.core.bootstrap import bootstrap_container
from victor.config.settings import Settings
settings = Settings()
from victor.agent.service_provider import configure_orchestrator_services
from victor.framework.graph import StateGraph, START, END

async def compile_workflow():
    container = bootstrap_container()
    configure_orchestrator_services(container, settings)

    graph = StateGraph("test_workflow")
    graph.add_node("process", lambda state: state)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)

    compiled = graph.compile()
    await compiled.ainvoke({"test": "data"})

import asyncio
asyncio.run(compile_workflow())
EOF

    END=$(python3 -c "import time; print(time.time())")
    ELAPSED=$(python3 -c "print($END - $START)")

    if [[ $i -gt $WARMUP ]]; then
        WORKFLOW_COMPILE_TIMES+=("$ELAPSED")
    fi
done

WORKFLOW_COMPILE_AVG=$(python3 << EOF
times = ${WORKFLOW_COMPILE_TIMES[@]}
avg = sum(times) / len(times)
print(f"{avg:.3f}")
EOF
)

log_success "Workflow compilation: avg=${WORKFLOW_COMPILE_AVG}ms"
add_metric "workflow.compilation_time" "$WORKFLOW_COMPILE_AVG" "ms"

# Get system info
log_info "Collecting system information..."

PYTHON_VERSION=$(python3 --version)
OS_INFO=$(uname -s -r)
CPU_INFO=$(python3 << EOF
import os
count = os.cpu_count()
print(f"{count} cores")
EOF
)

python3 << EOF
import json

with open("$OUTPUT_FILE", "r") as f:
    data = json.load(f)

data["system"] = {
    "python_version": "$(echo "$PYTHON_VERSION" | sed 's/"/\\"/g')",
    "os": "$(echo "$OS_INFO" | sed 's/"/\\"/g')",
    "cpu": "$CPU_INFO"
}

with open("$OUTPUT_FILE", "w") as f:
    json.dump(data, f, indent=2)
EOF

log_success "System information collected"

# Final summary
log_info ""
log_info "Baseline Summary:"
log_info "=================="

python3 << EOF
import json

with open("$OUTPUT_FILE", "r") as f:
    data = json.load(f)

metrics = data["metrics"]

print(f"Bootstrap Time (avg):     {metrics['bootstrap']['time']['average']['value']:.3f} {metrics['bootstrap']['time']['average']['unit']}")
print(f"Startup Time:             {metrics['startup']['time']['value']:.3f} {metrics['startup']['time']['unit']}")
print(f"Tool Selection (avg):     {metrics['tool_selection']['cold_cache']['average']['value']:.3f} {metrics['tool_selection']['cold_cache']['average']['unit']}")
print(f"Cache Key Generation:     {metrics['cache']['key_generation']['value']:.1f} {metrics['cache']['key_generation']['unit']}")
if 'memory' in metrics:
    print(f"Memory (current):         {metrics['memory']['current']['value']:.2f} {metrics['memory']['current']['unit']}")
    print(f"Memory (peak):            {metrics['memory']['peak']['value']:.2f} {metrics['memory']['peak']['unit']}")
print(f"Provider Pool Init:       {metrics['provider_pool']['initialization_time']['value']:.3f} {metrics['provider_pool']['initialization_time']['unit']}")
print(f"Workflow Compilation:     {metrics['workflow']['compilation_time']['value']:.3f} {metrics['workflow']['compilation_time']['unit']}")
EOF

log_info ""
log_success "Baseline results saved to: $OUTPUT_FILE"
log_info ""
log_info "To compare with future benchmarks:"
log_info "  ./scripts/performance_report.sh --baseline $OUTPUT_FILE --output comparison.md"
