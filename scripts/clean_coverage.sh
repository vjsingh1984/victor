#!/bin/bash
# Clean coverage database and artifacts
# Usage: scripts/clean_coverage.sh

echo "Cleaning coverage artifacts..."

# Remove coverage database files
rm -f .coverage
rm -f .coverage.*

# Remove coverage HTML reports
rm -rf htmlcov/

# Remove coverage XML reports
rm -f coverage.xml

echo "✓ Coverage artifacts cleaned successfully"
echo ""
echo "To run tests without coverage:"
echo "  pytest tests/ --no-cov"
echo ""
echo "To run tests with fresh coverage:"
echo "  pytest tests/"
