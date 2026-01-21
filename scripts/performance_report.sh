#!/bin/bash
# Performance Comparison Report Generator for Victor AI
#
# This script generates comprehensive performance comparison reports,
# including before/after metrics, improvement calculations, and recommendations.
#
# Usage:
#   ./scripts/performance_report.sh [options]
#
# Options:
#   --baseline FILE       Baseline JSON file (required)
#   --current FILE        Current/benchmark JSON file (required)
#   --output FILE         Output markdown file (default: /tmp/performance_report.md)
#   --format FORMAT       Output format: markdown, html, json (default: markdown)
#   --include-graphs      Generate graphs (requires matplotlib)
#   --help                Show this help message

set -euo pipefail

# Default values
BASELINE_FILE=""
CURRENT_FILE=""
OUTPUT_FILE="/tmp/performance_report.md"
FORMAT="markdown"
INCLUDE_GRAPHS=false
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline)
            BASELINE_FILE="$2"
            shift 2
            ;;
        --current)
            CURRENT_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --include-graphs)
            INCLUDE_GRAPHS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --baseline FILE       Baseline JSON file (required)"
            echo "  --current FILE        Current/benchmark JSON file (required)"
            echo "  --output FILE         Output markdown file (default: /tmp/performance_report.md)"
            echo "  --format FORMAT       Output format: markdown, html, json (default: markdown)"
            echo "  --include-graphs      Generate graphs (requires matplotlib)"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$BASELINE_FILE" ]]; then
    log_error "Missing required argument: --baseline"
    exit 1
fi

if [[ -z "$CURRENT_FILE" ]]; then
    log_error "Missing required argument: --current"
    exit 1
fi

# Check if files exist
if [[ ! -f "$BASELINE_FILE" ]]; then
    log_error "Baseline file not found: $BASELINE_FILE"
    exit 1
fi

if [[ ! -f "$CURRENT_FILE" ]]; then
    log_error "Current file not found: $CURRENT_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

log_info "Victor AI Performance Report Generator"
log_info "======================================="
log_info "Baseline: $BASELINE_FILE"
log_info "Current:  $CURRENT_FILE"
log_info "Output:   $OUTPUT_FILE"
log_info "Format:   $FORMAT"
log_info ""

# Generate Python script to create report
PYTHON_SCRIPT="/tmp/generate_report.py"

cat > "$PYTHON_SCRIPT" << 'EOFPYTHON'
#!/usr/bin/env python3
"""
Generate performance comparison report
"""
import json
import sys
from datetime import datetime
from pathlib import Path

def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def format_value(value, unit):
    """Format value with appropriate precision"""
    try:
        val = float(value)
        if unit in ['ms', 'MB']:
            return f"{val:.3f}"
        elif unit in ['microseconds', 'percent']:
            return f"{val:.1f}"
        else:
            return str(value)
    except:
        return str(value)

def calculate_improvement(baseline_val, current_val):
    """Calculate percentage improvement"""
    try:
        baseline = float(baseline_val)
        current = float(current_val)
        if baseline == 0:
            return 0.0
        improvement = ((baseline - current) / baseline) * 100
        return improvement
    except:
        return 0.0

def get_improvement_emoji(improvement):
    """Get emoji for improvement"""
    if improvement > 20:
        return "🚀"
    elif improvement > 10:
        return "✓"
    elif improvement > 0:
        return "↑"
    elif improvement > -10:
        return "↓"
    else:
        return "⚠"

def format_improvement(improvement):
    """Format improvement with sign"""
    if improvement > 0:
        return f"+{improvement:.1f}%"
    else:
        return f"{improvement:.1f}%"

def check_sla(metric, value, unit):
    """Check if metric meets SLA"""
    slas = {
        'tool_selection_p95': {'threshold': 1.0, 'unit': 'ms', 'operator': '<'},
        'tool_selection_p99': {'threshold': 2.0, 'unit': 'ms', 'operator': '<'},
        'cache_hit_rate': {'threshold': 70.0, 'unit': 'percent', 'operator': '>'},
        'bootstrap_time': {'threshold': 700.0, 'unit': 'ms', 'operator': '<'},
        'startup_time': {'threshold': 700.0, 'unit': 'ms', 'operator': '<'},
        'memory_current': {'threshold': 2000.0, 'unit': 'MB', 'operator': '<'},
    }

    if metric not in slas:
        return None

    sla = slas[metric]
    if sla['unit'] != unit:
        return None

    value = float(value)
    threshold = sla['threshold']

    if sla['operator'] == '<':
        meets_sla = value < threshold
    else:
        meets_sla = value > threshold

    return {
        'meets': meets_sla,
        'threshold': threshold,
        'operator': sla['operator'],
        'value': value
    }

def generate_markdown_report(baseline, current, output_file):
    """Generate markdown report"""

    report_lines = []
    report_lines.append("# Victor AI Performance Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    report_lines.append("")

    # Executive Summary
    report_lines.append("## Executive Summary")
    report_lines.append("")
    report_lines.append("This report compares the current performance against the baseline to identify")
    report_lines.append("improvements and areas for optimization.")
    report_lines.append("")

    # System Information
    report_lines.append("## System Information")
    report_lines.append("")
    report_lines.append("| Property | Baseline | Current |")
    report_lines.append("|----------|----------|---------|")

    if 'system' in baseline:
        sys_info = baseline['system']
        report_lines.append(f"| Python Version | {sys_info.get('python_version', 'N/A')} | N/A |")
        report_lines.append(f"| Operating System | {sys_info.get('os', 'N/A')} | N/A |")
        report_lines.append(f"| CPU | {sys_info.get('cpu', 'N/A')} | N/A |")

    report_lines.append("")

    # Bootstrap Performance
    report_lines.append("## Bootstrap Performance")
    report_lines.append("")

    if 'metrics' in baseline and 'bootstrap' in baseline['metrics']:
        boot_base = baseline['metrics']['bootstrap']['time']
        boot_curr = current.get('metrics', {}).get('bootstrap', {}).get('time', boot_base)

        report_lines.append("Bootstrap time measures how long it takes to initialize the service container.")
        report_lines.append("")

        report_lines.append("| Metric | Baseline | Current | Improvement | SLA Status |")
        report_lines.append("|--------|----------|---------|-------------|------------|")

        for metric in ['average', 'p50', 'p95', 'p99']:
            if metric in boot_base:
                base_val = boot_base[metric]['value']
                curr_val = boot_curr.get(metric, {}).get('value', base_val)
                unit = boot_base[metric]['unit']

                improvement = calculate_improvement(base_val, curr_val)
                emoji = get_improvement_emoji(improvement)
                sla = check_sla(f'bootstrap_time', curr_val, unit)

                sla_status = "✓" if sla and sla['meets'] else "⚠" if sla else "N/A"

                report_lines.append(
                    f"| {metric.capitalize()} | "
                    f"{format_value(base_val, unit)} {unit} | "
                    f"{format_value(curr_val, unit)} {unit} | "
                    f"{emoji} {format_improvement(improvement)} | "
                    f"{sla_status} |"
                )

        report_lines.append("")
        report_lines.append("**Target:** Bootstrap time should be < 700ms")
        report_lines.append("")

    # Tool Selection Performance
    report_lines.append("## Tool Selection Performance")
    report_lines.append("")

    if 'metrics' in baseline and 'tool_selection' in baseline['metrics']:
        ts_base = baseline['metrics']['tool_selection']
        ts_curr = current.get('metrics', {}).get('tool_selection', ts_base)

        report_lines.append("Tool selection measures how long it takes to select and prepare tools")
        report_lines.append("for agent execution.")
        report_lines.append("")

        report_lines.append("| Metric | Baseline | Current | Improvement | SLA Status |")
        report_lines.append("|--------|----------|---------|-------------|------------|")

        for cache_type in ['cold_cache', 'warm_cache']:
            if cache_type in ts_base:
                cache_base = ts_base[cache_type]
                cache_curr = ts_curr.get(cache_type, cache_base)

                for metric in ['average', 'p50', 'p95']:
                    if metric in cache_base:
                        base_val = cache_base[metric]['value']
                        curr_val = cache_curr.get(metric, {}).get('value', base_val)
                        unit = cache_base[metric]['unit']

                        improvement = calculate_improvement(base_val, curr_val)
                        emoji = get_improvement_emoji(improvement)

                        sla_key = f'tool_selection_{metric}' if metric in ['p95', 'p99'] else None
                        sla = check_sla(sla_key, curr_val, unit) if sla_key else None
                        sla_status = "✓" if sla and sla['meets'] else "⚠" if sla else "N/A"

                        metric_name = f"{cache_type.replace('_', ' ').capitalize()} {metric.capitalize()}"
                        report_lines.append(
                            f"| {metric_name} | "
                            f"{format_value(base_val, unit)} {unit} | "
                            f"{format_value(curr_val, unit)} {unit} | "
                            f"{emoji} {format_improvement(improvement)} | "
                            f"{sla_status} |"
                        )

        report_lines.append("")
        report_lines.append("**Targets:**")
        report_lines.append("- Tool Selection P95: < 1ms")
        report_lines.append("- Tool Selection P99: < 2ms")
        report_lines.append("")

        # Calculate cache speedup
        if 'cold_cache' in ts_base and 'average' in ts_base['cold_cache']:
            cold_base = ts_base['cold_cache']['average']['value']
            warm_base = ts_base.get('warm_cache', {}).get('average', {})
            warm_val = warm_base.get('value', cold_base)

            if warm_val > 0:
                speedup = cold_base / warm_val
                report_lines.append(f"**Cache Speedup:** {speedup:.2f}x")
                report_lines.append("")

    # Cache Performance
    report_lines.append("## Cache Performance")
    report_lines.append("")

    if 'benchmarks' in current and 'cache' in current['benchmarks']:
        cache = current['benchmarks']['cache']

        if 'hit_rate' in cache:
            hit_rate = cache['hit_rate']['value']
            report_lines.append(f"**Cache Hit Rate:** {hit_rate:.1f}%")

            sla = check_sla('cache_hit_rate', hit_rate, 'percent')
            if sla:
                if sla['meets']:
                    report_lines.append(f"✓ Meets SLA (target: > {sla['threshold']}%)")
                else:
                    report_lines.append(f"⚠ Below SLA (target: > {sla['threshold']}%)")

            report_lines.append("")

        if 'total_hits' in cache and 'total_misses' in cache:
            hits = cache['total_hits']['value']
            misses = cache['total_misses']['value']
            total = hits + misses

            if total > 0:
                report_lines.append("| Cache Statistic | Value |")
                report_lines.append("|-----------------|-------|")
                report_lines.append(f"| Total Hits | {hits} |")
                report_lines.append(f"| Total Misses | {misses} |")
                report_lines.append(f"| Total Requests | {total} |")
                report_lines.append("")

        if 'key_generation_single' in cache:
            key_gen = cache['key_generation_single']['value']
            report_lines.append(f"**Key Generation Time:** {key_gen:.2f} microseconds")
            report_lines.append("")

    # Memory Usage
    report_lines.append("## Memory Usage")
    report_lines.append("")

    if 'metrics' in baseline and 'memory' in baseline['metrics']:
        mem_base = baseline['metrics']['memory']
        mem_curr = current.get('metrics', {}).get('memory', mem_base)

        report_lines.append("| Metric | Baseline | Current | Improvement | SLA Status |")
        report_lines.append("|--------|----------|---------|-------------|------------|")

        for metric in ['current', 'peak']:
            if metric in mem_base:
                base_val = mem_base[metric]['value']
                curr_val = mem_curr.get(metric, {}).get('value', base_val)
                unit = mem_base[metric]['unit']

                improvement = calculate_improvement(base_val, curr_val)
                emoji = get_improvement_emoji(improvement)
                sla = check_sla(f'memory_{metric}', curr_val, unit)
                sla_status = "✓" if sla and sla['meets'] else "⚠" if sla else "N/A"

                report_lines.append(
                    f"| {metric.capitalize()} | "
                    f"{format_value(base_val, unit)} {unit} | "
                    f"{format_value(curr_val, unit)} {unit} | "
                    f"{emoji} {format_improvement(improvement)} | "
                    f"{sla_status} |"
                )

        report_lines.append("")
        report_lines.append("**Target:** Memory usage should be < 2GB")
        report_lines.append("")

    # Workflow Performance
    report_lines.append("## Workflow Performance")
    report_lines.append("")

    if 'metrics' in baseline and 'workflow' in baseline['metrics']:
        wf_base = baseline['metrics']['workflow']
        wf_curr = current.get('metrics', {}).get('workflow', wf_base)

        if 'compilation_time' in wf_base:
            base_val = wf_base['compilation_time']['value']
            curr_val = wf_curr.get('compilation_time', {}).get('value', base_val)
            unit = wf_base['compilation_time']['unit']

            improvement = calculate_improvement(base_val, curr_val)
            emoji = get_improvement_emoji(improvement)

            report_lines.append("| Metric | Baseline | Current | Improvement |")
            report_lines.append("|--------|----------|---------|-------------|")
            report_lines.append(
                f"| Compilation Time | "
                f"{format_value(base_val, unit)} {unit} | "
                f"{format_value(curr_val, unit)} {unit} | "
                f"{emoji} {format_improvement(improvement)} |"
            )
            report_lines.append("")

    # Summary Table
    report_lines.append("## Performance Summary")
    report_lines.append("")
    report_lines.append("| Category | Baseline | Current | Change | Status |")
    report_lines.append("|----------|----------|---------|--------|--------|")

    # Bootstrap summary
    if 'metrics' in baseline and 'bootstrap' in baseline['metrics']:
        boot_base = baseline['metrics']['bootstrap']['time']['average']['value']
        boot_curr = current.get('metrics', {}).get('bootstrap', {}).get('time', {}).get('average', {}).get('value', boot_base)
        improvement = calculate_improvement(boot_base, boot_curr)
        status = "✓" if improvement > 0 else "⚠"
        report_lines.append(
            f"| Bootstrap | "
            f"{boot_base:.3f} ms | "
            f"{boot_curr:.3f} ms | "
            f"{format_improvement(improvement)} | "
            f"{status} |"
        )

    # Tool selection summary
    if 'metrics' in baseline and 'tool_selection' in baseline['metrics']:
        ts_base = baseline['metrics']['tool_selection']['cold_cache']['average']['value']
        ts_curr = current.get('metrics', {}).get('tool_selection', {}).get('cold_cache', {}).get('average', {}).get('value', ts_base)
        improvement = calculate_improvement(ts_base, ts_curr)
        status = "✓" if improvement > 0 else "⚠"
        report_lines.append(
            f"| Tool Selection | "
            f"{ts_base:.3f} ms | "
            f"{ts_curr:.3f} ms | "
            f"{format_improvement(improvement)} | "
            f"{status} |"
        )

    # Memory summary
    if 'metrics' in baseline and 'memory' in baseline['metrics']:
        mem_base = baseline['metrics']['memory']['current']['value']
        mem_curr = current.get('metrics', {}).get('memory', {}).get('current', {}).get('value', mem_base)
        improvement = calculate_improvement(mem_base, mem_curr)
        status = "✓" if improvement > 0 else "⚠"
        report_lines.append(
            f"| Memory Usage | "
            f"{mem_base:.2f} MB | "
            f"{mem_curr:.2f} MB | "
            f"{format_improvement(improvement)} | "
            f"{status} |"
        )

    report_lines.append("")

    # Recommendations
    report_lines.append("## Recommendations")
    report_lines.append("")

    recommendations = []

    # Check bootstrap
    if 'metrics' in baseline and 'bootstrap' in baseline['metrics']:
        boot_base = baseline['metrics']['bootstrap']['time']['average']['value']
        boot_curr = current.get('metrics', {}).get('bootstrap', {}).get('time', {}).get('average', {}).get('value', boot_base)
        if boot_curr > 700:
            recommendations.append(
                f"- **Bootstrap time is {boot_curr:.0f}ms**, exceeding the 700ms target. "
                f"Consider implementing lazy loading for non-essential services."
            )
        else:
            recommendations.append(
                f"- ✓ Bootstrap time is {boot_curr:.0f}ms, meeting the 700ms target."
            )

    # Check tool selection
    if 'metrics' in baseline and 'tool_selection' in baseline['metrics']:
        ts_curr = current.get('metrics', {}).get('tool_selection', {}).get('cold_cache', {}).get('average', {})
        if 'value' in ts_curr and ts_curr['value'] > 1.0:
            recommendations.append(
                f"- **Tool selection is {ts_curr['value']:.3f}ms**, exceeding the 1ms target. "
                f"Review cache hit rates and consider increasing cache size."
            )
        else:
            recommendations.append(
                f"- ✓ Tool selection performance is within acceptable range."
            )

    # Check cache hit rate
    if 'benchmarks' in current and 'cache' in current['benchmarks']:
        if 'hit_rate' in current['benchmarks']['cache']:
            hit_rate = current['benchmarks']['cache']['hit_rate']['value']
            if hit_rate < 70:
                recommendations.append(
                    f"- **Cache hit rate is {hit_rate:.1f}%**, below the 70% target. "
                    f"Consider increasing cache TTL or size."
                )
            else:
                recommendations.append(
                    f"- ✓ Cache hit rate is {hit_rate:.1f}%, meeting the 70% target."
                )

    # Check memory
    if 'metrics' in baseline and 'memory' in baseline['metrics']:
        mem_curr = current.get('metrics', {}).get('memory', {}).get('current', {}).get('value', 0)
        if mem_curr > 2000:
            recommendations.append(
                f"- **Memory usage is {mem_curr:.0f}MB**, exceeding the 2GB target. "
                f"Consider implementing memory profiling to identify leaks."
            )
        else:
            recommendations.append(
                f"- ✓ Memory usage is {mem_curr:.0f}MB, well within the 2GB target."
            )

    if recommendations:
        for rec in recommendations:
            report_lines.append(rec)
    else:
        report_lines.append("No specific recommendations at this time.")
    report_lines.append("")

    # SLA Compliance
    report_lines.append("## SLA Compliance")
    report_lines.append("")
    report_lines.append("| Metric | Target | Current | Status |")
    report_lines.append("|--------|--------|---------|--------|")

    slas = [
        ("Tool Selection P95", "< 1ms", current.get('metrics', {}).get('tool_selection', {}).get('cold_cache', {}).get('p95', {}).get('value', 0), 1.0, '<'),
        ("Tool Selection P99", "< 2ms", current.get('metrics', {}).get('tool_selection', {}).get('cold_cache', {}).get('p99', {}).get('value', 0), 2.0, '<'),
        ("Cache Hit Rate", "> 70%", current.get('benchmarks', {}).get('cache', {}).get('hit_rate', {}).get('value', 0), 70.0, '>'),
        ("Bootstrap Time", "< 700ms", current.get('metrics', {}).get('bootstrap', {}).get('time', {}).get('average', {}).get('value', 0), 700.0, '<'),
        ("Startup Time", "< 700ms", current.get('metrics', {}).get('startup', {}).get('time', {}).get('value', 0), 700.0, '<'),
        ("Memory Usage", "< 2GB", current.get('metrics', {}).get('memory', {}).get('current', {}).get('value', 0), 2000.0, '<'),
    ]

    for name, target, current_val, threshold, operator in slas:
        val = float(current_val) if current_val else 0
        meets_sla = (val < threshold) if operator == '<' else (val > threshold)
        status = "✓ Pass" if meets_sla else "⚠ Fail"
        report_lines.append(f"| {name} | {target} | {val:.3f} | {status} |")

    report_lines.append("")

    # Footer
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("*This report was generated automatically by Victor AI performance benchmarking tools.*")
    report_lines.append("")

    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Markdown report saved to: {output_file}")

def main():
    baseline_file = sys.argv[1]
    current_file = sys.argv[2]
    output_file = sys.argv[3]

    baseline = load_json(baseline_file)
    current = load_json(current_file)

    generate_markdown_report(baseline, current, output_file)

if __name__ == '__main__':
    main()
EOFPYTHON

# Run the Python script
python3 "$PYTHON_SCRIPT" "$BASELINE_FILE" "$CURRENT_FILE" "$OUTPUT_FILE"

log_success "Performance report generated: $OUTPUT_FILE"
log_info ""
log_info "Report contents:"
log_info "================"
log_info ""

# Display the report
head -100 "$OUTPUT_FILE"

log_info ""
log_info "..."
log_info ""
log_info "(Report truncated - see full report at $OUTPUT_FILE)"
