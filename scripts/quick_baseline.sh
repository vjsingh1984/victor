#!/bin/bash
# Quick Performance Baseline for Victor AI
# Simplified version that focuses on core metrics

set -euo pipefail

OUTPUT_FILE="${1:-/tmp/victor_baseline.json}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "Victor AI Quick Baseline Measurement"
echo "====================================="
echo "Output: $OUTPUT_FILE"
echo ""

# Initialize JSON
cat > "$OUTPUT_FILE" << EOF
{
  "timestamp": "$TIMESTAMP",
  "metrics": {}
}
EOF

# Measure bootstrap time
echo "Measuring bootstrap time..."
BOOTSTRAP_TIMES=()
for i in {1..5}; do
    START=$(python3 -c "import time; print(time.time())")
    python3 << PYEOF
import sys
sys.path.insert(0, "$PROJECT_ROOT")
from victor.core.bootstrap import bootstrap_container
container = bootstrap_container()
PYEOF
    END=$(python3 -c "import time; print(time.time())")
    BOOTSTRAP_TIMES+=($(python3 -c "print($END - $START)"))
done

BOOTSTRAP_AVG=$(python3 -c "import statistics; print(f'{statistics.mean(BOOTSTRAP_TIMES):.3f}')")
echo "Bootstrap: ${BOOTSTRAP_AVG}ms"

# Measure memory usage
echo "Measuring memory usage..."
MEMORY_USAGE=$(python3 << PYEOF
import sys
import tracemalloc
sys.path.insert(0, "$PROJECT_ROOT")

tracemalloc.start()
from victor.core.bootstrap import bootstrap_container
container = bootstrap_container()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"{current/1024/1024:.2f},{peak/1024/1024:.2f}")
PYEOF
)

MEMORY_CURRENT=$(echo "$MEMORY_USAGE" | cut -d',' -f1)
MEMORY_PEAK=$(echo "$MEMORY_USAGE" | cut -d',' -f2)
echo "Memory: current=${MEMORY_CURRENT}MB, peak=${MEMORY_PEAK}MB"

# Update JSON
python3 << PYEOF
import json

with open("$OUTPUT_FILE", "r") as f:
    data = json.load(f)

data["metrics"]["bootstrap"] = {
    "time": {
        "average": {"value": float("$BOOTSTRAP_AVG"), "unit": "ms"}
    }
}

data["metrics"]["memory"] = {
    "current": {"value": float("$MEMORY_CURRENT"), "unit": "MB"},
    "peak": {"value": float("$MEMORY_PEAK"), "unit": "MB"}
}

with open("$OUTPUT_FILE", "w") as f:
    json.dump(data, f, indent=2)
PYEOF

echo ""
echo "Baseline saved to: $OUTPUT_FILE"
echo ""
echo "Summary:"
echo "  Bootstrap Time: ${BOOTSTRAP_AVG}ms"
echo "  Memory Usage: ${MEMORY_CURRENT}MB (current), ${MEMORY_PEAK}MB (peak)"
