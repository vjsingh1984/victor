#!/bin/bash
# Test dashboard with live events

set -e

echo "=========================================="
echo "Dashboard Live Event Test"
echo "=========================================="
echo ""

echo "Step 1: Clear old event file"
EVENT_FILE="$HOME/.victor/metrics/victor.jsonl"
> "$EVENT_FILE"
echo "✓ Cleared event file"
echo ""

echo "Step 2: Start monitoring logs"
LOG_FILE="$HOME/.victor/logs/victor.log"
> "$LOG_FILE"

echo "In another terminal, run:"
echo "  tail -f $LOG_FILE | grep -E '(Dashboard|EventFileWatcher)'"
echo ""
read -p "Press Enter when you're ready to start dashboard..."
echo ""

echo "Step 3: Instructions"
echo ""
echo "1. Start dashboard: victor dashboard --log-level DEBUG"
echo ""
echo "2. In THIS terminal, generate test events:"
echo "   victor chat --no-tui --log-events 'List Python files'"
echo ""
echo "3. Watch dashboard for events (should appear within 60 seconds)"
echo ""
echo "Expected:"
echo "  - Agent generates ~5-10 events"
echo "  - Events flushed to JSONL (10 events or 60 seconds)"
echo "  - File watcher detects new content (100ms polling)"
echo "  - Dashboard displays events"
echo ""
