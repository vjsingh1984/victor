#!/bin/bash
# Comprehensive dashboard debug test

set -e

echo "=========================================="
echo "Dashboard Debug Test"
echo "=========================================="
echo ""

echo "Step 1: Check event file"
echo "=========================================="
EVENT_FILE="$HOME/.victor/metrics/victor.jsonl"

if [ -f "$EVENT_FILE" ]; then
    EVENT_COUNT=$(wc -l < "$EVENT_FILE")
    echo "✓ Event file exists: $EVENT_FILE"
    echo "✓ Event count: $EVENT_COUNT"
    echo ""
    echo "First 2 events:"
    head -n 2 "$EVENT_FILE" | jq -c '{category, name, session_id}'
    echo ""
else
    echo "✗ Event file not found: $EVENT_FILE"
    echo "  Run: victor chat --log-events 'test message'"
    exit 1
fi

echo ""
echo "Step 2: Reinstall Victor"
echo "=========================================="
cd /Users/vijaysingh/code/glm-branch
pip install -e . --quiet
echo "✓ Reinstalled"
echo ""

echo "Step 3: Clear old logs"
echo "=========================================="
LOG_FILE="$HOME/.victor/logs/victor.log"
> "$LOG_FILE"
echo "✓ Cleared log file: $LOG_FILE"
echo ""

echo "Step 4: Start monitoring logs (background)"
echo "=========================================="
# Start tail in background, save PID
tail -f "$LOG_FILE" > /tmp/victor_dashboard_debug.log 2>&1 &
TAIL_PID=$!
echo "✓ Started log monitoring (PID: $TAIL_PID)"
echo "  Log output: /tmp/victor_dashboard_debug.log"
echo ""

echo "Step 5: Instructions"
echo "=========================================="
echo ""
echo "In a NEW terminal, run:"
echo ""
echo "  victor dashboard --log-level DEBUG"
echo ""
echo "Then in THIS terminal, press any key to analyze logs..."
echo ""
read -p "Press Enter after dashboard starts..."

echo ""
echo "=========================================="
echo "Analyzing Logs..."
echo "=========================================="
echo ""

# Give dashboard time to start
sleep 2

# Count key log messages
echo "EventFileWatcher logs:"
grep -c "\[EventFileWatcher\]" /tmp/victor_dashboard_debug.log || echo "  0"
echo ""

echo "Dashboard logs:"
grep -c "\[Dashboard\]" /tmp/victor_dashboard_debug.log || echo "  0"
echo ""

echo "EventBus.publish logs:"
grep -c "\[EventBus.publish\]" /tmp/victor_dashboard_debug.log || echo "  0"
echo ""

echo "=========================================="
echo "Key Events:"
echo "=========================================="
echo ""

echo "1. File Watcher Initialization:"
grep "\[EventFileWatcher\] Mounting" /tmp/victor_dashboard_debug.log || echo "  NOT FOUND"
echo ""

echo "2. Event File Reading:"
grep "\[EventFileWatcher\] Read.*lines from file" /tmp/victor_dashboard_debug.log || echo "  NOT FOUND"
echo ""

echo "3. Events Parsed:"
grep "\[EventFileWatcher\] PARSED" /tmp/victor_dashboard_debug.log | head -5 || echo "  NOT FOUND"
echo ""

echo "4. Events Published:"
grep "\[EventFileWatcher\] PUBLISHED" /tmp/victor_dashboard_debug.log | head -5 || echo "  NOT FOUND"
echo ""

echo "5. EventBus Publish:"
grep "\[EventBus.publish\] Publishing event:" /tmp/victor_dashboard_debug.log | head -5 || echo "  NOT FOUND"
echo ""

echo "6. Dashboard Subscription:"
grep "\[Dashboard\] Subscribing to EventBus" /tmp/victor_dashboard_debug.log || echo "  NOT FOUND"
echo ""

echo "7. Dashboard Receiving Events:"
grep "\[Dashboard.handle_event\] RECEIVED" /tmp/victor_dashboard_debug.log | head -5 || echo "  NOT FOUND"
echo ""

echo "8. Dashboard Processing Events:"
grep "\[Dashboard._process_event\] PROCESSING" /tmp/victor_dashboard_debug.log | head -5 || echo "  NOT FOUND"
echo ""

echo ""
echo "=========================================="
echo "Full Log Sample (last 50 lines):"
echo "=========================================="
tail -50 /tmp/victor_dashboard_debug.log
echo ""

# Cleanup
kill $TAIL_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
echo ""
echo "Full debug log saved to: /tmp/victor_dashboard_debug.log"
echo ""
echo "To manually monitor logs:"
echo "  tail -f $LOG_FILE | grep -E '(EventFileWatcher|Dashboard|EventBus)'"
echo ""
