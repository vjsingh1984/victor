#!/bin/bash
# Monitor test failures every 5 minutes

LOG_FILE="/tmp/test_monitor.log"
SUMMARY_FILE="/tmp/test_summary.txt"
ITERATION=0

echo "Starting test monitoring - $(date)" | tee -a "$LOG_FILE"
echo "=====================================" | tee -a "$LOG_FILE"

while true; do
    ITERATION=$((ITERATION + 1))
    echo "" | tee -a "$LOG_FILE"
    echo "=== Iteration $ITERATION at $(date) ===" | tee -a "$LOG_FILE"

    # Run tests and capture summary
    python -m pytest tests/unit/ --tb=no -q 2>&1 | tee "$SUMMARY_FILE" | tee -a "$LOG_FILE"

    # Extract failure count
    FAILED=$(grep -oP '\d+(?= failed)' "$SUMMARY_FILE" | tail -1)
    PASSED=$(grep -oP '\d+(?= passed)' "$SUMMARY_FILE" | tail -1)

    if [ -z "$FAILED" ]; then
        FAILED="0"
    fi

    if [ -z "$PASSED" ]; then
        PASSED="0"
    fi

    echo "Status: $FAILED failed, $PASSED passed" | tee -a "$LOG_FILE"

    # Check for specific error patterns
    if grep -q "SyntaxError" "$SUMMARY_FILE"; then
        echo "⚠️  SYNTAX ERRORS DETECTED" | tee -a "$LOG_FILE"
    fi

    if grep -q "IndentationError" "$SUMMARY_FILE"; then
        echo "⚠️  INDENTATION ERRORS DETECTED" | tee -a "$LOG_FILE"
    fi

    if grep -q "ImportError" "$SUMMARY_FILE"; then
        echo "⚠️  IMPORT ERRORS DETECTED" | tee -a "$LOG_FILE"
    fi

    # Show top failing test files
    echo "" | tee -a "$LOG_FILE"
    echo "Top failing test files:" | tee -a "$LOG_FILE"
    grep -E "FAILED tests/unit" "$SUMMARY_FILE" | head -10 | tee -a "$LOG_FILE"

    echo "" | tee -a "$LOG_FILE"
    echo "Next check in 5 minutes (Ctrl+C to stop)..." | tee -a "$LOG_FILE"
    echo "=====================================" | tee -a "$LOG_FILE"

    sleep 300
done
