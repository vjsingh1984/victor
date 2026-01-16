#!/bin/bash
# Check for TODO/FIXME/HACK comments in code
# Usage: check_todos.sh <file_path>

set -e

FILE="$1"

if [ ! -f "$FILE" ]; then
  echo "Error: File not found: $FILE"
  exit 1
fi

# Check for TODO/FIXME/HACK comments
PATTERNS=(
  "TODO"
  "FIXME"
  "HACK"
  "XXX"
  "NOTE.*fix"
)

FOUND=0
for pattern in "${PATTERNS[@]}"; do
  if grep -q "$pattern" "$FILE"; then
    FOUND=1
    echo "  Found $pattern in $FILE:"
    grep -n "$pattern" "$FILE" | head -5
  fi
done

if [ $FOUND -eq 0 ]; then
  exit 0
else
  echo "  Consider resolving these comments or converting to GitHub issues"
  exit 0  # Don't fail, just warn
fi
