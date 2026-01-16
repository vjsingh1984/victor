#!/bin/bash
# Run linting suite
# Usage: scripts/ci/run_lint.sh [--fix] [--strict]

set -e

FIX=""
STRICT=""
FAILED=0

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --fix)
      FIX="--fix"
      shift
      ;;
    --strict)
      STRICT="--strict"
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--fix] [--strict]"
      echo ""
      echo "Options:"
      echo "  --fix     Auto-fix issues where possible"
      echo "  --strict  Enable strict mode (fail on warnings)"
      echo "  --help    Show this help"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Running linting suite..."
echo ""

# Black formatting check
echo "1. Black formatting check..."
if black --check victor tests; then
  echo "✓ Black: OK"
else
  if [ -n "$FIX" ]; then
    echo "⚠ Black: Formatting issues found, auto-fixing..."
    black victor tests
  else
    echo "✗ Black: Formatting issues found"
    FAILED=1
  fi
fi
echo ""

# Ruff linting
echo "2. Ruff linting..."
if ruff check victor tests; then
  echo "✓ Ruff: OK"
else
  if [ -n "$FIX" ]; then
    echo "⚠ Ruff: Issues found, auto-fixing..."
    ruff check $FIX victor tests || FAILED=1
  else
    echo "✗ Ruff: Issues found"
    ruff check victor tests || FAILED=1
  fi
fi
echo ""

# MyPy type checking
echo "3. MyPy type checking..."
echo "   Checking strict-typed modules..."
if mypy victor/protocols victor/framework/coordinators victor/core/registries victor/core/verticals; then
  echo "✓ MyPy (strict): OK"
else
  echo "✗ MyPy (strict): Type errors found"
  if [ -n "$STRICT" ]; then
    FAILED=1
  fi
fi
echo ""

# Bandit security check
echo "4. Bandit security check..."
if bandit -r victor/ -ll -ii --exclude victor/test/,victor/tests/,tests/; then
  echo "✓ Bandit: OK"
else
  echo "⚠ Bandit: Security issues found (review recommended)"
fi
echo ""

# Summary
echo "Linting summary:"
if [ $FAILED -eq 0 ]; then
  echo "✓ All critical checks passed"
  exit 0
else
  echo "✗ Some checks failed"
  exit 1
fi
