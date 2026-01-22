#!/bin/bash
################################################################################
# Test script for secrets configuration
#
# Validates that the configure_secrets.sh script works correctly
# without actually creating Kubernetes resources.
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TESTS_PASSED=0
TESTS_FAILED=0

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

log_test() {
    echo ""
    echo -e "${BLUE}=== TEST: $1 ===${NC}"
}

cleanup() {
    rm -f /tmp/test_secrets_*
}

trap cleanup EXIT

################################################################################
# Test 1: Script exists and is executable
################################################################################

log_test "Script exists and is executable"

if [[ -x "${SCRIPT_DIR}/configure_secrets.sh" ]]; then
    log_success "configure_secrets.sh is executable"
else
    log_error "configure_secrets.sh is not executable"
fi

if [[ -x "${SCRIPT_DIR}/verify_secrets.sh" ]]; then
    log_success "verify_secrets.sh is executable"
else
    log_error "verify_secrets.sh is not executable"
fi

################################################################################
# Test 2: Template file exists
################################################################################

log_test "Template file exists"

if [[ -f "${PROJECT_ROOT}/deployment/secrets.env.template" ]]; then
    log_success "secrets.env.template exists"

    # Check for required variables in template
    required_vars=(
        "DB_PASSWORD"
        "ANTHROPIC_API_KEY"
        "ENCRYPTION_KEY"
        "JWT_SECRET"
        "SMTP_HOST"
        "GRAFANA_ADMIN_PASSWORD"
    )

    for var in "${required_vars[@]}"; do
        if grep -q "$var" "${PROJECT_ROOT}/deployment/secrets.env.template"; then
            log_success "Template contains $var"
        else
            log_error "Template missing $var"
        fi
    done
else
    log_error "secrets.env.template not found"
fi

################################################################################
# Test 3: Script syntax is valid
################################################################################

log_test "Script syntax validation"

if bash -n "${SCRIPT_DIR}/configure_secrets.sh" 2>&1; then
    log_success "configure_secrets.sh has valid syntax"
else
    log_error "configure_secrets.sh has syntax errors"
fi

if bash -n "${SCRIPT_DIR}/verify_secrets.sh" 2>&1; then
    log_success "verify_secrets.sh has valid syntax"
else
    log_error "verify_secrets.sh has syntax errors"
fi

################################################################################
# Test 4: Help message works
################################################################################

log_test "Help message display"

if "${SCRIPT_DIR}/configure_secrets.sh" --help > /dev/null 2>&1; then
    log_success "configure_secrets.sh --help works"
else
    log_error "configure_secrets.sh --help failed"
fi

if "${SCRIPT_DIR}/verify_secrets.sh" --help > /dev/null 2>&1; then
    log_success "verify_secrets.sh --help works"
else
    log_error "verify_secrets.sh --help failed"
fi

################################################################################
# Test 5: Password generation functions
################################################################################

log_test "Password generation"

# Source the script to access functions
source "${SCRIPT_DIR}/configure_secrets.sh" 2>/dev/null || true

# Test password generation
test_password=$(generate_password 16 2>/dev/null || echo "")
if [[ ${#test_password} -eq 16 ]]; then
    log_success "generate_password generates correct length"
else
    log_error "generate_password failed"
fi

# Test encryption key generation
test_encryption=$(generate_encryption_key 2>/dev/null || echo "")
if [[ ${#test_encryption} -gt 0 ]]; then
    log_success "generate_encryption_key works"
else
    log_error "generate_encryption_key failed"
fi

# Test JWT secret generation
test_jwt=$(generate_jwt_secret 2>/dev/null || echo "")
if [[ ${#test_jwt} -gt 0 ]]; then
    log_success "generate_jwt_secret works"
else
    log_error "generate_jwt_secret failed"
fi

################################################################################
# Test 6: Template file has security warnings
################################################################################

log_test "Security warnings in template"

if grep -qi "never commit" "${PROJECT_ROOT}/deployment/secrets.env.template"; then
    log_success "Template has 'never commit' warning"
else
    log_error "Template missing security warning"
fi

if grep -qi "gitignore" "${PROJECT_ROOT}/deployment/secrets.env.template"; then
    log_success "Template mentions .gitignore"
else
    log_error "Template doesn't mention .gitignore"
fi

################################################################################
# Test 7: Documentation exists
################################################################################

log_test "Documentation files"

if [[ -f "${PROJECT_ROOT}/deployment/docs/SECRETS_MANAGEMENT.md" ]]; then
    log_success "SECRETS_MANAGEMENT.md exists"

    # Check for key sections
    required_sections=(
        "Quick Start"
        "configure_secrets.sh"
        "verify_secrets.sh"
        "Security Best Practices"
    )

    for section in "${required_sections[@]}"; do
        if grep -q "$section" "${PROJECT_ROOT}/deployment/docs/SECRETS_MANAGEMENT.md"; then
            log_success "Docs contain section: $section"
        else
            log_error "Docs missing section: $section"
        fi
    done
else
    log_error "SECRETS_MANAGEMENT.md not found"
fi

if [[ -f "${SCRIPT_DIR}/README_SECRETS.md" ]]; then
    log_success "README_SECRETS.md exists"
else
    log_error "README_SECRETS.md not found"
fi

################################################################################
# Test 8: Verify script doesn't expose secrets
################################################################################

log_test "Security: No secret logging"

if ! grep -q "echo.*secret" "${SCRIPT_DIR}/configure_secrets.sh" | grep -v "echo.*exists"; then
    log_success "configure_secrets.sh doesn't log secrets"
else
    log_error "Potential secret logging found"
fi

# Check for verification that secrets are not echoed
if grep -q "Does NOT log secret values" "${SCRIPT_DIR}/configure_secrets.sh"; then
    log_success "Script explicitly states it doesn't log secrets"
else
    log_error "Script doesn't document security behavior"
fi

################################################################################
# Test 9: Dry-run mode support
################################################################################

log_test "Dry-run mode"

if grep -q -- "--dry-run" "${SCRIPT_DIR}/configure_secrets.sh"; then
    log_success "configure_secrets.sh supports --dry-run"
else
    log_error "configure_secrets.sh missing --dry-run"
fi

################################################################################
# Test 10: Multiple source support
################################################################################

log_test "Multiple secret sources"

supported_sources=("env" "aws-ssm" "azure-keyvault")
for source in "${supported_sources[@]}"; do
    if grep -q "$source" "${SCRIPT_DIR}/configure_secrets.sh"; then
        log_success "Supports source: $source"
    else
        log_error "Missing source: $source"
    fi
done

################################################################################
# Test 11: Examples script
################################################################################

log_test "Examples script"

if [[ -x "${SCRIPT_DIR}/secrets_examples.sh" ]]; then
    log_success "secrets_examples.sh exists and is executable"

    if "${SCRIPT_DIR}/secrets_examples.sh" > /dev/null 2>&1; then
        log_success "secrets_examples.sh runs without errors"
    else
        log_error "secrets_examples.sh has errors"
    fi
else
    log_error "secrets_examples.sh not found or not executable"
fi

################################################################################
# Summary
################################################################################

echo ""
echo "===================="
echo "Test Results Summary"
echo "===================="
echo ""
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"
echo ""

if [[ $TESTS_FAILED -eq 0 ]]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Please review the errors above.${NC}"
    exit 1
fi
