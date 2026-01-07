#!/bin/bash
# Quick verification script for dashboard fixes

echo "=========================================="
echo "Dashboard Event Loading Verification"
echo "=========================================="
echo ""

# Check 1: Event file exists and has content
echo "1. Checking event file..."
EVENT_FILE="$HOME/.victor/metrics/victor.jsonl"

if [ -f "$EVENT_FILE" ]; then
    EVENT_COUNT=$(wc -l < "$EVENT_FILE")
    FILE_SIZE=$(du -h "$EVENT_FILE" | cut -f1)
    echo "   ✓ File exists: $EVENT_FILE"
    echo "   ✓ Event count: $EVENT_COUNT"
    echo "   ✓ File size: $FILE_SIZE"

    if [ "$EVENT_COUNT" -gt 0 ]; then
        echo "   ✓ File has events!"

        # Show last 3 events
        echo ""
        echo "   Last 3 events:"
        tail -n 3 "$EVENT_FILE" | while read -r line; do
            CATEGORY=$(echo "$line" | jq -r '.category')
            NAME=$(echo "$line" | jq -r '.name')
            echo "      - [$CATEGORY] $NAME"
        done
    else
        echo "   ✗ File is empty!"
    fi
else
    echo "   ✗ File does not exist: $EVENT_FILE"
    echo "   → Run: victor chat --log-events 'test message'"
fi

echo ""
echo "=========================================="
echo "2. Reinstalling with fixes..."
echo "=========================================="

cd /Users/vijaysingh/code/glm-branch
pip install -e . --quiet

echo "   ✓ Reinstalled"
echo ""

echo "=========================================="
echo "3. Starting dashboard..."
echo "=========================================="
echo ""
echo "   Run this command in a new terminal:"
echo ""
echo "   victor dashboard"
echo ""
echo "   Expected behavior:"
echo "   • Dashboard starts"
echo "   • Loads last 100 events from file"
echo "   • Shows them in all 9 tabs"
echo "   • Monitors for new events (100ms polling)"
echo ""
echo "   In another terminal, monitor logs:"
echo "   tail -f ~/.victor/logs/victor.log | grep EventFileWatcher"
echo ""
echo "=========================================="
