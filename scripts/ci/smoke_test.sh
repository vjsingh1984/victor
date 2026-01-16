#!/bin/bash
# Run smoke tests against deployed environment
# Usage: scripts/ci/smoke_test.sh <environment>

set -e

ENVIRONMENT=""
BASE_URL=""
TIMEOUT=30

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    staging|production)
      ENVIRONMENT="$1"
      case $ENVIRONMENT in
        staging)
          BASE_URL="${STAGING_URL:-https://staging.victor.example.com}"
          ;;
        production)
          BASE_URL="${PRODUCTION_URL:-https://victor.example.com}"
          ;;
      esac
      shift
      ;;
    --url)
      BASE_URL="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 <environment> [--url URL] [--timeout SECONDS]"
      echo ""
      echo "Arguments:"
      echo "  environment     Deployment environment (staging|production)"
      echo ""
      echo "Options:"
      echo "  --url URL       Base URL to test (default: auto-detected)"
      echo "  --timeout       Request timeout in seconds (default: 30)"
      echo "  --help          Show this help"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ -z "$ENVIRONMENT" ]; then
  echo "Error: Environment must be specified (staging|production)"
  exit 1
fi

echo "Running smoke tests against $ENVIRONMENT"
echo "  URL: $BASE_URL"
echo "  Timeout: ${TIMEOUT}s"
echo ""

FAILED=0

# Test 1: Health check
echo "Test 1: Health check endpoint..."
if curl -f -s --max-time "$TIMEOUT" "${BASE_URL}/health" | grep -q "healthy"; then
  echo "  ✓ Health check passed"
else
  echo "  ✗ Health check failed"
  FAILED=1
fi

# Test 2: Version check
echo "Test 2: Version endpoint..."
if curl -f -s --max-time "$TIMEOUT" "${BASE_URL}/version" | grep -q "version"; then
  echo "  ✓ Version check passed"
else
  echo "  ✗ Version check failed"
  FAILED=1
fi

# Test 3: API availability
echo "Test 3: API availability..."
if curl -f -s --max-time "$TIMEOUT" "${BASE_URL}/api/v1/" | grep -q "API"; then
  echo "  ✓ API available"
else
  echo "  ✗ API unavailable"
  FAILED=1
fi

# Test 4: CLI basic functionality
echo "Test 4: CLI basic functionality..."
if victor --version &> /dev/null; then
  echo "  ✓ CLI available"
else
  echo "  ✗ CLI unavailable"
  FAILED=1
fi

echo ""
if [ $FAILED -eq 0 ]; then
  echo "✓ All smoke tests passed!"
  exit 0
else
  echo "✗ Some smoke tests failed"
  exit 1
fi
