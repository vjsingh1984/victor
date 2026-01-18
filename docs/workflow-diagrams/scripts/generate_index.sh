#!/bin/bash
# Workflow Diagrams Index Generator
# Generates statistics and reports for workflow diagrams

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_DIR="$(dirname "$SCRIPT_DIR")"

cd "$WORKFLOW_DIR"

echo "=== Victor Workflow Diagrams Report ==="
echo ""
echo "Generated: $(date +%Y-%m-%d)"
echo ""

# Total count
TOTAL=$(ls -1 *.svg 2>/dev/null | wc -l | tr -d ' ')
echo "Total Diagrams: $TOTAL"
echo ""

# Directory size
SIZE=$(du -sh . | cut -f1)
echo "Total Size: $SIZE"
echo ""

# Vertical breakdown
echo "=== By Vertical ==="
for vertical in benchmark coding core dataanalysis devops rag research; do
    count=$(ls -1 ${vertical}_*.svg 2>/dev/null | wc -l | tr -d ' ')
    if [ "$count" -gt 0 ]; then
        echo "$vertical: $count"
    fi
done
echo ""

# Variant breakdown
echo "=== Variants ==="
QUICK=$(ls -1 *_quick.svg 2>/dev/null | wc -l | tr -d ' ')
LITE=$(ls -1 *_lite.svg 2>/dev/null | wc -l | tr -d ' ')
SIMPLE=$(ls -1 *_simple.svg 2>/dev/null | wc -l | tr -d ' ')

echo "Quick: $QUICK"
echo "Lite: $LITE"
echo "Simple: $SIMPLE"
echo "Standard: $((TOTAL - QUICK - LITE - SIMPLE))"
echo ""

# Largest files
echo "=== Largest Files (Top 10) ==="
ls -lhS *.svg 2>/dev/null | head -11 | tail -10 | awk '{print $9, $5}'
echo ""

# Smallest files
echo "=== Smallest Files (Bottom 10) ==="
ls -lhS *.svg 2>/dev/null | tail -10 | awk '{print $9, $5}'
echo ""

# Naming issues
echo "=== Potential Naming Issues ==="

# Check for inconsistent separators
echo "Files with inconsistent compound word naming:"
for f in *.svg; do
    # Check for patterns like "bugfix" (no underscore in compound word)
    if [[ $f =~ [a-z]_[a-z]_[a-z] ]] || [[ $f =~ [a-z][a-z][a-rt-z][a-z] ]]; then
        # Skip known valid patterns
        if [[ ! $f =~ _quick ]] && [[ ! $f =~ _lite ]] && [[ ! $f =~ _simple ]] && [[ ! $f =~ _plan ]]; then
            echo "  - $f"
        fi
    fi
done
echo ""

# Check for missing quick variants
echo "Workflows that might need quick variants:"
for f in *.svg; do
    if [[ ! $f =~ _quick ]] && [[ ! $f =~ _lite ]] && [[ ! $f =~ _simple ]]; then
        size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null)
        # If file > 15KB, suggest quick variant
        if [ "$size" -gt 15000 ]; then
            base=$(basename "$f" .svg)
            if [ ! -f "${base}_quick.svg" ]; then
                echo "  - $f ($(stat -f%z "$f" 2>/dev/null | awk '{print $1/1024 "KB"}' || stat -c%s "$f" 2>/dev/null | awk '{print $1/1024 "KB"}'))"
            fi
        fi
    fi
done
echo ""

# Recent changes
echo "=== Recently Modified ==="
ls -lt *.svg 2>/dev/null | head -6 | tail -5 | awk '{print $9, $6, $7, $8}'
echo ""

echo "=== End Report ==="
