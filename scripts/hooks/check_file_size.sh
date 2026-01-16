#!/bin/bash
# Check that Python files are not too large
# Usage: check_file_size.sh <file_path>

set -e

FILE="$1"
MAX_LINES=500

if [ ! -f "$FILE" ]; then
  echo "Error: File not found: $FILE"
  exit 1
fi

# Count lines
LINES=$(wc -l < "$FILE")

if [ "$LINES" -gt "$MAX_LINES" ]; then
  echo "Warning: $FILE has $LINES lines (max: $MAX_LINES)"
  echo "  Consider refactoring into smaller modules"
  exit 0  # Don't fail, just warn
else
  exit 0
fi
