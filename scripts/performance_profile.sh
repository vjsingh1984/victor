#!/bin/bash
# Performance Profiling Script for Victor AI
#
# This script profiles Victor AI performance to identify hotspots and bottlenecks.
# It performs CPU profiling, memory profiling, and I/O profiling.
#
# Usage:
#   ./scripts/performance_profile.sh [options]
#
# Options:
#   --output DIR          Output directory for profiles (default: /tmp/victor_profiles)
#   --profile-type TYPE   Type of profiling: cpu, memory, io, all (default: all)
#   --duration SECONDS    Profiling duration (default: 30)
#   --scenario FILE       Python script with scenario to profile (optional)
#   --generate-graphs     Generate visualization graphs (requires matplotlib)
#   --help                Show this help message

set -euo pipefail

# Default values
OUTPUT_DIR="/tmp/victor_profiles"
PROFILE_TYPE="all"
DURATION=30
SCENARIO_FILE=""
GENERATE_GRAPHS=false
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

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
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --profile-type)
            PROFILE_TYPE="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --scenario)
            SCENARIO_FILE="$2"
            shift 2
            ;;
        --generate-graphs)
            GENERATE_GRAPHS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --output DIR          Output directory for profiles (default: /tmp/victor_profiles)"
            echo "  --profile-type TYPE   Type of profiling: cpu, memory, io, all (default: all)"
            echo "  --duration SECONDS    Profiling duration (default: 30)"
            echo "  --scenario FILE       Python script with scenario to profile (optional)"
            echo "  --generate-graphs     Generate visualization graphs (requires matplotlib)"
            echo "  --help                Show this help message"
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

log_section "Victor AI Performance Profiling"
log_info "Output directory: $OUTPUT_DIR"
log_info "Profile type: $PROFILE_TYPE"
log_info "Duration: ${DURATION}s"
if [[ -n "$SCENARIO_FILE" ]]; then
    log_info "Scenario: $SCENARIO_FILE"
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

################################################################################
# CPU Profiling
################################################################################

if [[ "$PROFILE_TYPE" == "cpu" || "$PROFILE_TYPE" == "all" ]]; then
    log_section "CPU Profiling"

    CPU_PROFILE_SCRIPT="$OUTPUT_DIR/cpu_profile_scenario.py"

    if [[ -n "$SCENARIO_FILE" && -f "$SCENARIO_FILE" ]]; then
        log_info "Using custom scenario: $SCENARIO_FILE"
        cp "$SCENARIO_FILE" "$CPU_PROFILE_SCRIPT"
    else
        log_info "Creating default profiling scenario..."

        cat > "$CPU_PROFILE_SCRIPT" << EOF
"""
CPU Profiling Scenario for Victor AI
"""
import sys
import asyncio
import time
sys.path.insert(0, "$PROJECT_ROOT")

from victor.core.bootstrap import bootstrap_container
from victor.config.settings import settings
from victor.agent.service_provider import configure_orchestrator_services
from victor.agent.protocols import ToolCoordinatorProtocol
from victor.agent.tool_selection.context import AgentToolSelectionContext

def main():
    """Main profiling scenario"""
    # Bootstrap
    container = bootstrap_container()
    configure_orchestrator_services(container, settings)

    # Run tool selection multiple times
    async def run_tool_selection():
        coordinator = container.get(ToolCoordinatorProtocol)

        queries = [
            "Analyze Python code for security issues",
            "Review code quality and suggest improvements",
            "Generate unit tests for this function",
            "Refactor this code for better performance",
            "Debug this error message",
        ]

        for i in range(20):
            query = queries[i % len(queries)]
            context = AgentToolSelectionContext(
                query=query,
                max_tools=10,
            )
            await coordinator.select_and_execute(context)

    asyncio.run(run_tool_selection())

if __name__ == '__main__':
    main()
EOF
    fi

    log_info "Running CPU profile (cProfile)..."

    python3 -m cProfile -o "$OUTPUT_DIR/cpu_profile.pstats" "$CPU_PROFILE_SCRIPT"

    log_success "CPU profile saved to: $OUTPUT_DIR/cpu_profile.pstats"

    # Generate human-readable report
    log_info "Generating CPU profile report..."

    python3 << EOF
import pstats
from pstats import SortKey

# Load profile
stats = pstats.Stats("$OUTPUT_DIR/cpu_profile.pstats")

# Save to text file
with open("$OUTPUT_DIR/cpu_profile.txt", "w") as f:
    f.write("CPU Profile Results\n")
    f.write("=" * 80 + "\n\n")

    # Top 20 by cumulative time
    f.write("Top 20 Functions by Cumulative Time:\n")
    f.write("-" * 80 + "\n")
    stats.stream = f
    stats.strip_dirs()
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)

    f.write("\n\n")

    # Top 20 by total time
    f.write("Top 20 Functions by Total Time:\n")
    f.write("-" * 80 + "\n")
    stats.sort_stats(SortKey.TIME)
    stats.print_stats(20)

    f.write("\n\n")

    # Callers of expensive functions
    f.write("Callers of Most Expensive Functions:\n")
    f.write("-" * 80 + "\n")
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_callers(10)

print("CPU profile report saved to: $OUTPUT_DIR/cpu_profile.txt")

# Print summary to console
print("\nTop 10 Functions by Cumulative Time:")
print("=" * 80)
stats.stream = None
stats.strip_dirs()
stats.sort_stats(SortKey.CUMULATIVE)
stats.print_stats(10)
EOF

    # Generate snakeviz visualization if available
    if command -v snakeviz &> /dev/null; then
        log_info "Generating snakeviz visualization..."
        snakeviz "$OUTPUT_DIR/cpu_profile.pstats" &
        SNAKEVIZ_PID=$!
        sleep 2
        kill $SNAKEVIZ_PID 2>/dev/null || true
        log_success "Snakeviz visualization opened in browser"
    else
        log_warning "snakeviz not installed. Install with: pip install snakeviz"
    fi
fi

################################################################################
# Memory Profiling
################################################################################

if [[ "$PROFILE_TYPE" == "memory" || "$PROFILE_TYPE" == "all" ]]; then
    log_section "Memory Profiling"

    MEMORY_PROFILE_SCRIPT="$OUTPUT_DIR/memory_profile_scenario.py"

    if [[ -n "$SCENARIO_FILE" && -f "$SCENARIO_FILE" ]]; then
        log_info "Using custom scenario: $SCENARIO_FILE"
        cp "$SCENARIO_FILE" "$MEMORY_PROFILE_SCRIPT"
    else
        log_info "Creating default profiling scenario..."

        cat > "$MEMORY_PROFILE_SCRIPT" << EOF
"""
Memory Profiling Scenario for Victor AI
"""
import sys
import asyncio
sys.path.insert(0, "$PROJECT_ROOT")

from victor.core.bootstrap import bootstrap_container
from victor.config.settings import settings
from victor.agent.service_provider import configure_orchestrator_services
from victor.agent.protocols import ToolCoordinatorProtocol
from victor.agent.tool_selection.context import AgentToolSelectionContext

def main():
    """Main profiling scenario"""
    # Bootstrap
    container = bootstrap_container()
    configure_orchestrator_services(container, settings)

    # Run tool selection multiple times
    async def run_tool_selection():
        coordinator = container.get(ToolCoordinatorProtocol)

        queries = [
            "Analyze Python code for security issues",
            "Review code quality and suggest improvements",
            "Generate unit tests for this function",
            "Refactor this code for better performance",
            "Debug this error message",
        ]

        for i in range(20):
            query = queries[i % len(queries)]
            context = AgentToolSelectionContext(
                query=query,
                max_tools=10,
            )
            await coordinator.select_and_execute(context)

    asyncio.run(run_tool_selection())

if __name__ == '__main__':
    main()
EOF
    fi

    log_info "Running memory profile (tracemalloc)..."

    python3 << EOF
import sys
import tracemalloc
sys.path.insert(0, "$PROJECT_ROOT")

# Start tracing
tracemalloc.start()

# Run the scenario
import subprocess
result = subprocess.run(
    ["python3", "$MEMORY_PROFILE_SCRIPT"],
    capture_output=True,
    text=True
)

# Take snapshot
snapshot = tracemalloc.take_snapshot()
tracemalloc.stop()

# Save top statistics
with open("$OUTPUT_DIR/memory_profile.txt", "w") as f:
    f.write("Memory Profile Results\n")
    f.write("=" * 80 + "\n\n")

    f.write("Current Size: ")
    current, peak = tracemalloc.get_traced_memory()
    f.write(f"{current / 1024 / 1024:.2f} MB\n")
    f.write(f"Peak Size: {peak / 1024 / 1024:.2f} MB\n\n")

    f.write("Top 20 Memory Allocations:\n")
    f.write("-" * 80 + "\n")

    top_stats = snapshot.statistics('lineno')

    for stat in top_stats[:20]:
        f.write(f"{stat}\n")

    f.write("\n\n")

    # Group by filename
    f.write("Top 10 Files by Memory Usage:\n")
    f.write("-" * 80 + "\n")

    stats_by_file = {}
    for stat in top_stats:
        filename = stat.traceback[0].filename
        if filename not in stats_by_file:
            stats_by_file[filename] = 0
        stats_by_file[filename] += stat.size

    sorted_files = sorted(stats_by_file.items(), key=lambda x: x[1], reverse=True)
    for filename, size in sorted_files[:10]:
        f.write(f"{size / 1024:.2f} KB  {filename}\n")

print("Memory profile saved to: $OUTPUT_DIR/memory_profile.txt")

# Print summary to console
print("\nTop 10 Memory Allocations:")
print("=" * 80)
for stat in top_stats[:10]:
    print(stat)
EOF

    # Try memory_profiler if available
    if python3 -c "import memory_profiler" 2>/dev/null; then
        log_info "Running memory_profiler..."

        python3 -m memory_profiler "$MEMORY_PROFILE_SCRIPT" > "$OUTPUT_DIR/memory_profile_detailed.txt"

        log_success "Detailed memory profile saved to: $OUTPUT_DIR/memory_profile_detailed.txt"
    else
        log_warning "memory_profiler not installed. Install with: pip install memory_profiler"
    fi
fi

################################################################################
# I/O Profiling
################################################################################

if [[ "$PROFILE_TYPE" == "io" || "$PROFILE_TYPE" == "all" ]]; then
    log_section "I/O Profiling"

    IO_PROFILE_SCRIPT="$OUTPUT_DIR/io_profile_scenario.py"

    if [[ -n "$SCENARIO_FILE" && -f "$SCENARIO_FILE" ]]; then
        log_info "Using custom scenario: $SCENARIO_FILE"
        cp "$SCENARIO_FILE" "$IO_PROFILE_SCRIPT"
    else
        log_info "Creating default profiling scenario..."

        cat > "$IO_PROFILE_SCRIPT" << EOF
"""
I/O Profiling Scenario for Victor AI
"""
import sys
import asyncio
sys.path.insert(0, "$PROJECT_ROOT")

from victor.core.bootstrap import bootstrap_container
from victor.config.settings import settings
from victor.agent.service_provider import configure_orchestrator_services
from victor.agent.protocols import ToolCoordinatorProtocol
from victor.agent.tool_selection.context import AgentToolSelectionContext

def main():
    """Main profiling scenario"""
    # Bootstrap
    container = bootstrap_container()
    configure_orchestrator_services(container, settings)

    # Run tool selection multiple times
    async def run_tool_selection():
        coordinator = container.get(ToolCoordinatorProtocol)

        queries = [
            "Analyze Python code for security issues",
            "Review code quality and suggest improvements",
            "Generate unit tests for this function",
            "Refactor this code for better performance",
            "Debug this error message",
        ]

        for i in range(20):
            query = queries[i % len(queries)]
            context = AgentToolSelectionContext(
                query=query,
                max_tools=10,
            )
            await coordinator.select_and_execute(context)

    asyncio.run(run_tool_selection())

if __name__ == '__main__':
    main()
EOF
    fi

    log_info "Running I/O profile..."

    python3 << EOF
import sys
import io
import contextlib
import time
sys.path.insert(0, "$PROJECT_ROOT")

# Capture I/O operations
io_operations = {
    'reads': [],
    'writes': [],
    'file_operations': []
}

original_open = open

def instrumented_open(file, mode='r', *args, **kwargs):
    start_time = time.time()
    result = original_open(file, mode, *args, **kwargs)
    elapsed = time.time() - start_time

    io_operations['file_operations'].append({
        'file': file,
        'mode': mode,
        'duration': elapsed
    })

    return result

# Monkey-patch open
import builtins
builtins.open = instrumented_open

# Run the scenario
import subprocess
result = subprocess.run(
    ["python3", "$IO_PROFILE_SCRIPT"],
    capture_output=True,
    text=True
)

# Restore original
builtins.open = original_open

# Save I/O profile
with open("$OUTPUT_DIR/io_profile.txt", "w") as f:
    f.write("I/O Profile Results\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Total File Operations: {len(io_operations['file_operations'])}\n\n")

    # Slowest operations
    sorted_ops = sorted(
        io_operations['file_operations'],
        key=lambda x: x['duration'],
        reverse=True
    )

    f.write("Top 20 Slowest File Operations:\n")
    f.write("-" * 80 + "\n")

    for op in sorted_ops[:20]:
        f.write(f"{op['duration']*1000:.3f} ms  {op['mode']:5s}  {op['file']}\n")

    f.write("\n\n")

    # Summary by file
    file_stats = {}
    for op in io_operations['file_operations']:
        file = op['file']
        if file not in file_stats:
            file_stats[file] = {'count': 0, 'total_time': 0}
        file_stats[file]['count'] += 1
        file_stats[file]['total_time'] += op['duration']

    f.write("File Access Summary:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'File':<60} {'Count':>10} {'Total Time':>15}\n")
    f.write("-" * 80 + "\n")

    sorted_files = sorted(
        file_stats.items(),
        key=lambda x: x[1]['total_time'],
        reverse=True
    )

    for file, stats in sorted_files[:20]:
        f.write(f"{file:<60} {stats['count']:>10} {stats['total_time']*1000:>15.3f} ms\n")

print("I/O profile saved to: $OUTPUT_DIR/io_profile.txt")

# Print summary to console
print(f"\nTotal File Operations: {len(io_operations['file_operations'])}")
print("\nTop 10 Slowest Operations:")
print("=" * 80)
for op in sorted_ops[:10]:
    print(f"{op['duration']*1000:.3f} ms  {op['mode']:5s}  {op['file']}")
EOF
fi

################################################################################
# Hotspot Analysis
################################################################################

log_section "Hotspot Analysis"

log_info "Analyzing performance hotspots..."

python3 << EOF
import re
import sys
from pathlib import Path

output_dir = "$OUTPUT_DIR"

# Parse CPU profile for hotspots
hotspots = []

try:
    with open(f"{output_dir}/cpu_profile.txt", "r") as f:
        content = f.read()

        # Extract function statistics
        in_function_list = False
        for line in content.split('\n'):
            if 'Top 20 Functions by Cumulative Time' in line:
                in_function_list = True
                continue

            if in_function_list:
                if line.startswith('-'):
                    continue
                if line.strip() == '':
                    break

                # Parse: ncalls  tottime  percall  cumtime  percall filename:lineno(function)
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        cumtime = float(parts[3])
                        function = ' '.join(parts[5:])
                        hotspots.append({
                            'type': 'cpu',
                            'value': cumtime,
                            'description': function
                        })
                    except:
                        pass
except:
    pass

# Parse memory profile for hotspots
try:
    with open(f"{output_dir}/memory_profile.txt", "r") as f:
        content = f.read()

        in_top_allocations = False
        for line in content.split('\n'):
            if 'Top 20 Memory Allocations' in line:
                in_top_allocations = True
                continue

            if in_top_allocations:
                if line.startswith('-'):
                    continue
                if line.strip() == '':
                    break

                # Parse memory allocations
                # Format: path/to/file.py:123: size bytes
                match = re.search(r'(\d+\.?\d*\s*[KMG]?iB)', line)
                if match:
                    size_str = match.group(1)
                    # Convert to bytes
                    if 'KiB' in size_str:
                        size = float(size_str.replace('KiB', '')) * 1024
                    elif 'MiB' in size_str:
                        size = float(size_str.replace('MiB', '')) * 1024 * 1024
                    else:
                        size = float(size_str.replace('bytes', ''))

                    hotspots.append({
                        'type': 'memory',
                        'value': size,
                        'description': line.strip()
                    })
except:
    pass

# Sort hotspots by value
hotspots.sort(key=lambda x: x['value'], reverse=True)

# Save hotspot analysis
with open(f"{output_dir}/hotspot_analysis.txt", "w") as f:
    f.write("Performance Hotspot Analysis\n")
    f.write("=" * 80 + "\n\n")

    cpu_hotspots = [h for h in hotspots if h['type'] == 'cpu'][:10]
    memory_hotspots = [h for h in hotspots if h['type'] == 'memory'][:10]

    if cpu_hotspots:
        f.write("Top CPU Hotspots:\n")
        f.write("-" * 80 + "\n")
        for i, spot in enumerate(cpu_hotspots, 1):
            f.write(f"{i}. {spot['value']:.3f}s - {spot['description']}\n")
        f.write("\n\n")

    if memory_hotspots:
        f.write("Top Memory Hotspots:\n")
        f.write("-" * 80 + "\n")
        for i, spot in enumerate(memory_hotspots, 1):
            f.write(f"{i}. {spot['value']/1024:.2f} KB - {spot['description']}\n")

print("Hotspot analysis saved to: $OUTPUT_DIR/hotspot_analysis.txt")

# Print summary to console
print("\nTop 5 CPU Hotspots:")
print("=" * 80)
for i, spot in enumerate([h for h in hotspots if h['type'] == 'cpu'][:5], 1):
    print(f"{i}. {spot['value']:.3f}s - {spot['description']}")

if any(h['type'] == 'memory' for h in hotspots):
    print("\nTop 5 Memory Hotspots:")
    print("=" * 80)
    for i, spot in enumerate([h for h in hotspots if h['type'] == 'memory'][:5], 1):
        print(f"{i}. {spot['value']/1024:.2f} KB - {spot['description']}")
EOF

################################################################################
# Optimization Recommendations
################################################################################

log_section "Optimization Recommendations"

log_info "Generating optimization recommendations..."

python3 << EOF
import json
from pathlib import Path

output_dir = "$OUTPUT_DIR"

recommendations = []

# Analyze CPU profile for optimization opportunities
try:
    with open(f"{output_dir}/cpu_profile.txt", "r") as f:
        content = f.read()

        # Look for patterns suggesting optimization opportunities
        if '__init__' in content:
            recommendations.append({
                'category': 'CPU',
                'priority': 'medium',
                'issue': 'Frequent object initialization',
                'recommendation': 'Consider object pooling or caching to reduce initialization overhead'
            })

        if 'hashlib' in content or 'md5' in content or 'sha' in content:
            recommendations.append({
                'category': 'CPU',
                'priority': 'low',
                'issue': 'Hash computations detected',
                'recommendation': 'Cache hash results when possible to avoid recomputation'
            })
except:
    pass

# Analyze memory profile
try:
    with open(f"{output_dir}/memory_profile.txt", "r") as f:
        content = f.read()

        # Look for memory patterns
        if 'dict' in content.lower():
            recommendations.append({
                'category': 'Memory',
                'priority': 'medium',
                'issue': 'Dictionary allocations',
                'recommendation': 'Consider using __slots__ for classes or using more memory-efficient data structures'
            })

        # Check for large memory usage
        if 'MB' in content:
            mb_values = [float(line.split('MB')[0].strip()) for line in content.split('\n') if 'MB' in line]
            if mb_values and max(mb_values) > 100:
                recommendations.append({
                    'category': 'Memory',
                    'priority': 'high',
                    'issue': 'High memory usage detected',
                    'recommendation': 'Implement memory profiling and consider lazy loading for large objects'
                })
except:
    pass

# General recommendations
recommendations.extend([
    {
        'category': 'Cache',
        'priority': 'high',
        'issue': 'Tool selection caching',
        'recommendation': 'Implement advanced caching strategies with 70-80% hit rate target'
    },
    {
        'category': 'Bootstrap',
        'priority': 'high',
        'issue': 'Bootstrap performance',
        'recommendation': 'Use lazy loading for non-essential services to reduce bootstrap time'
    },
    {
        'category': 'I/O',
        'priority': 'medium',
        'issue': 'File I/O operations',
        'recommendation': 'Batch file operations and use asynchronous I/O where possible'
    },
])

# Save recommendations
with open(f"{output_dir}/optimization_recommendations.json", "w") as f:
    json.dump(recommendations, f, indent=2)

# Save human-readable version
with open(f"{output_dir}/optimization_recommendations.txt", "w") as f:
    f.write("Optimization Recommendations\n")
    f.write("=" * 80 + "\n\n")

    # Group by priority
    high_priority = [r for r in recommendations if r['priority'] == 'high']
    medium_priority = [r for r in recommendations if r['priority'] == 'medium']
    low_priority = [r for r in recommendations if r['priority'] == 'low']

    if high_priority:
        f.write("HIGH PRIORITY:\n")
        f.write("-" * 80 + "\n")
        for rec in high_priority:
            f.write(f"\n[{rec['category']}]\n")
            f.write(f"Issue: {rec['issue']}\n")
            f.write(f"Recommendation: {rec['recommendation']}\n")
        f.write("\n\n")

    if medium_priority:
        f.write("MEDIUM PRIORITY:\n")
        f.write("-" * 80 + "\n")
        for rec in medium_priority:
            f.write(f"\n[{rec['category']}]\n")
            f.write(f"Issue: {rec['issue']}\n")
            f.write(f"Recommendation: {rec['recommendation']}\n")
        f.write("\n\n")

    if low_priority:
        f.write("LOW PRIORITY:\n")
        f.write("-" * 80 + "\n")
        for rec in low_priority:
            f.write(f"\n[{rec['category']}]\n")
            f.write(f"Issue: {rec['issue']}\n")
            f.write(f"Recommendation: {rec['recommendation']}\n")

print("Optimization recommendations saved to: $OUTPUT_DIR/optimization_recommendations.txt")

# Print summary to console
print(f"\nTotal Recommendations: {len(recommendations)}")
print(f"  High Priority: {len(high_priority)}")
print(f"  Medium Priority: {len(medium_priority)}")
print(f"  Low Priority: {len(low_priority)}")

if high_priority:
    print("\nHIGH PRIORITY RECOMMENDATIONS:")
    print("=" * 80)
    for rec in high_priority[:3]:
        print(f"\n[{rec['category']}] {rec['issue']}")
        print(f"→ {rec['recommendation']}")
EOF

################################################################################
# Summary
################################################################################

log_section "Profiling Summary"

log_info "Profiling results saved to: $OUTPUT_DIR"
log_info ""
log_info "Generated files:"
ls -lh "$OUTPUT_DIR" | awk '{print "  " $9 " (" $5 ")"}' | grep -v "^  $"
log_info ""

log_success "Performance profiling complete!"
log_info ""
log_info "Next steps:"
log_info "  1. Review hotspot analysis: $OUTPUT_DIR/hotspot_analysis.txt"
log_info "  2. Check optimization recommendations: $OUTPUT_DIR/optimization_recommendations.txt"
log_info "  3. Implement top-priority optimizations"
log_info "  4. Re-run benchmarks to validate improvements"
