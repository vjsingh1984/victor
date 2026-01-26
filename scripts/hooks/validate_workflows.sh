#!/usr/bin/env bash
# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Pre-commit hook to validate YAML workflow files
#
# This script validates workflow YAML files before git commit by:
# 1. Checking YAML syntax
# 2. Validating workflow structure
# 3. Compiling workflows to check for errors
# 4. Checking for common errors (unknown node types, missing references)
#
# Usage:
#   ./scripts/hooks/validate_workflows.sh victor/coding/workflows/feature.yaml
#   ./scripts/hooks/validate_workflows.sh victor/coding/workflows/*.yaml
#   ./scripts/hooks/validate_workflows.sh --ci victor/*/workflows/*.yaml  # CI mode
#
# Environment variables:
#   VICTOR_SKIP_VALIDATION=1    - Skip validation (use with caution)
#   VICTOR_VERBOSE_VALIDATION=1 - Enable verbose output
#   VICTOR_CACHE_VALIDATION=1   - Enable caching for faster validation
#   VICTOR_CI_MODE=1            - CI mode (JSON output, exit codes only)
#

set -eou pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $*"
}

check_python() {
    if ! command -v "$PYTHON_BIN" &> /dev/null; then
        log_error "Python not found: $PYTHON_BIN"
        log_info "Install Python or set PYTHON_BIN environment variable"
        exit 1
    fi
}

# =============================================================================
# CI Mode Support
# =============================================================================

# Global variables for CI mode
CI_MODE=false
JSON_OUTPUT_FILE="validation-results.json"
SUMMARY_OUTPUT_FILE="validation-summary.txt"

# Initialize CI output files
init_ci_output() {
    if [[ "$CI_MODE" == true ]]; then
        # Start JSON output
        echo '{"version": "1.0", "validation_results": [' > "$JSON_OUTPUT_FILE"
        echo "" > "$SUMMARY_OUTPUT_FILE"
    fi
}

# Add validation result to JSON output
add_validation_result() {
    local file="$1"
    local status="$2"
    local message="$3"
    local errors="$4"

    if [[ "$CI_MODE" == true ]]; then
        # Escape strings for JSON
        local escaped_file=$(echo "$file" | sed 's/"/\\"/g')
        local escaped_message=$(echo "$message" | sed 's/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')
        local escaped_errors=$(echo "$errors" | sed 's/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')

        # Add comma if not first entry
        if [[ $(wc -c < "$JSON_OUTPUT_FILE") -gt 50 ]]; then
            echo "," >> "$JSON_OUTPUT_FILE"
        fi

        cat >> "$JSON_OUTPUT_FILE" << EOF
  {
    "file": "$escaped_file",
    "status": "$status",
    "message": "$escaped_message",
    "errors": "$escaped_errors",
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  }
EOF
    fi
}

# Finalize CI output
finalize_ci_output() {
    local total_files="$1"
    local total_errors="$2"
    local total_warnings="$3"
    local elapsed_time="$4"

    if [[ "$CI_MODE" == true ]]; then
        # Close JSON array
        echo ']}' >> "$JSON_OUTPUT_FILE"

        # Add summary stats to JSON
        local temp_file="${JSON_OUTPUT_FILE}.tmp"
        python3 - <<EOF
import json
import sys

with open('$JSON_OUTPUT_FILE', 'r') as f:
    data = json.load(f)

# Add summary
data['summary'] = {
    'total_files': $total_files,
    'total_errors': $total_errors,
    'total_warnings': $total_warnings,
    'elapsed_time_seconds': $elapsed_time,
    'validation_timestamp': '$(date -u +"%Y-%m-%dT%H:%M:%SZ")'
}

with open('$temp_file', 'w') as f:
    json.dump(data, f, indent=2)
EOF

        mv "$temp_file" "$JSON_OUTPUT_FILE"

        # Generate human-readable summary
        cat > "$SUMMARY_OUTPUT_FILE" << EOF
Workflow Validation Summary
===========================
Total Files:      $total_files
Errors:           $total_errors
Warnings:         $total_warnings
Elapsed Time:     ${elapsed_time}s
Validation Time:  $(date -u +"%Y-%m-%dT%H:%M:%SZ")

Exit Code:        $([ $total_errors -eq 0 ] && echo "0 (success)" || echo "1 (failed)")
EOF

        # Add Sarif output for GitHub annotations
        if command -v jq &> /dev/null; then
            # Convert to Sarif format if jq is available
            python3 - <<EOF
import json
import sys

try:
    with open('$JSON_OUTPUT_FILE', 'r') as f:
        data = json.load(f)

    sarif = {
        "version": "2.1.0",
        "\$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "Victor Workflow Validator",
                    "version": "1.0.0",
                    "informationUri": "https://github.com/vijay-singh-iitg/victor"
                }
            },
            "results": []
        }]
    }

    for result in data.get('validation_results', []):
        if result['status'] == 'failed':
            sarif['runs'][0]['results'].append({
                "ruleId": "workflow-validation-error",
                "level": "error",
                "message": {
                    "text": result['message']
                },
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": result['file']
                        }
                    }
                }]
                })
            elif result['status'] == 'warning':
                sarif['runs'][0]['results'].append({
                    "ruleId": "workflow-validation-warning",
                    "level": "warning",
                    "message": {
                        "text": result['message']
                    },
                    "locations": [{
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": result['file']
                            }
                        }
                    }]
                    })

    with open('validation-results.sarif', 'w') as f:
        json.dump(sarif, f, indent=2)

except Exception as e:
    print(f"Warning: Could not generate Sarif output: {e}", file=sys.stderr)
EOF
        fi
    fi
}

# =============================================================================
# Validation Functions
# =============================================================================

validate_yaml_syntax() {
    local file="$1"

    log_info "Checking YAML syntax: $file"

    # Normalize path for Windows
    local normalized_file="$file"
    if [[ "$(uname -s)" == MINGW* ]] || [[ "$(uname -s)" == MSYS* ]]; then
        normalized_file=$(echo "$file" | sed 's|\\|/|g' | sed 's|^\([A-Za-z]\):|/\1|')
    fi

    # Use Python to check YAML syntax with path normalization
    if ! "$PYTHON_BIN" -c "
import sys
import os
import yaml

file_path = os.path.normpath('$normalized_file')
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        yaml.safe_load(f)
    sys.exit(0)
except Exception as e:
    print(f'YAML syntax error: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1; then
        log_error "Invalid YAML syntax in: $file"
        return 1
    fi

    log_success "YAML syntax valid: $file"
    return 0
}

validate_workflow_compilation() {
    local file="$1"
    local verbose="${VICTOR_VERBOSE_VALIDATION:-0}"

    log_info "Validating workflow compilation: $file"

    # Detect Windows and normalize path
    local normalized_file="$file"
    if [[ "$(uname -s)" == MINGW* ]] || [[ "$(uname -s)" == MSYS* ]]; then
        # Convert Windows path to Unix-style for Git Bash
        normalized_file=$(echo "$file" | sed 's|\\|/|g' | sed 's|^\([A-Za-z]\):|/\1|')
    fi

    # Use Python to call workflow validation directly
    # Pass file as argument instead of embedding in code to avoid path escaping issues
    local python_cmd="
import sys
import os

# Normalize project root for cross-platform compatibility
project_root = os.path.normpath('${PROJECT_ROOT}')
sys.path.insert(0, project_root)

from victor.ui.commands.workflow import workflow_app
import io
from contextlib import redirect_stdout, redirect_stderr

# Get file path from environment to avoid escaping issues
file_path = os.path.normpath('${normalized_file}')

# Capture output
stdout_capture = io.StringIO()
stderr_capture = io.StringIO()

try:
    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        sys.argv = ['workflow', 'validate', file_path]
        if ${verbose}:
            sys.argv.append('--verbose')
        workflow_app()
    sys.exit(0)
except SystemExit as e:
    if e.code != 0:
        print(stderr_capture.getvalue(), file=sys.stderr)
        print(stdout_capture.getvalue(), file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
except Exception as e:
    print(f'Validation error: {e}', file=sys.stderr)
    sys.exit(1)
"

    # Run validation, capture output
    if ! output=$("$PYTHON_BIN" -c "$python_cmd" 2>&1); then
        log_error "Workflow validation failed: $file"
        echo "$output" | sed 's/^/  /'
        return 1
    fi

    # Check for warnings in output (ignore "Unknown node type" warnings for HITL nodes)
    if echo "$output" | grep -v "Unknown node type.*HITLNode" | grep -q "Warning:"; then
        log_warning "Workflow validated with warnings: $file"
        if [[ "$verbose" == "1" ]]; then
            echo "$output" | grep "Warning:" | sed 's/^/  /'
        fi
    else
        log_success "Workflow validated: $file"
    fi

    return 0
}

# =============================================================================
# Caching
# =============================================================================

get_file_hash() {
    local file="$1"
    # Use md5 or sha256 depending on availability
    if command -v md5 &> /dev/null; then
        md5 -q "$file"
    elif command -v md5sum &> /dev/null; then
        md5sum "$file" | cut -d' ' -f1
    else
        # Fallback to modification time (Windows-compatible)
        # Try BSD stat first (macOS), then GNU stat (Linux)
        stat -f "%m" "$file" 2>/dev/null || stat -c "%Y" "$file" 2>/dev/null || stat "%Y" "$file" 2>/dev/null
    fi
}

check_cache() {
    local file="$1"
    local cache_dir="${PROJECT_ROOT}/.cache/workflow-validation"
    local cache_file="${cache_dir}/$(basename "$file").hash"

    if [[ "${VICTOR_CACHE_VALIDATION:-0}" != "1" ]]; then
        return 1
    fi

    if [[ -f "$cache_file" ]]; then
        local cached_hash
        local current_hash
        cached_hash=$(cat "$cache_file")
        current_hash=$(get_file_hash "$file")

        if [[ "$cached_hash" == "$current_hash" ]]; then
            return 0
        fi
    fi

    return 1
}

update_cache() {
    local file="$1"
    local cache_dir="${PROJECT_ROOT}/.cache/workflow-validation"

    if [[ "${VICTOR_CACHE_VALIDATION:-0}" != "1" ]]; then
        return 0
    fi

    mkdir -p "$cache_dir"
    get_file_hash "$file" > "${cache_dir}/$(basename "$file").hash"
}

# =============================================================================
# Main Validation Logic
# =============================================================================

validate_workflow_file() {
    local file="$1"
    local total_errors=0
    local total_warnings=0

    # Check if file exists
    if [[ ! -f "$file" ]]; then
        log_error "File not found: $file"
        return 1
    fi

    # Check file extension
    if [[ "$file" != *.yaml && "$file" != *.yml ]]; then
        log_warning "Skipping non-YAML file: $file"
        return 0
    fi

    # Check cache
    if check_cache "$file"; then
        log_info "Using cached validation result: $file"
        return 0
    fi

    # Validate YAML syntax
    if ! validate_yaml_syntax "$file"; then
        ((total_errors++))
    fi

    # Validate workflow compilation
    if ! validate_workflow_compilation "$file"; then
        ((total_errors++))
    fi

    # Update cache if validation passed
    if [[ $total_errors -eq 0 ]]; then
        update_cache "$file"
    fi

    return $total_errors
}

# =============================================================================
# Main Entry Point
# =============================================================================

main() {
    # Check for --ci flag
    if [[ "${1:-}" == "--ci" ]]; then
        CI_MODE=true
        VICTOR_CI_MODE=1
        shift  # Remove --ci from arguments
    fi

    # Check if validation is disabled
    if [[ "${VICTOR_SKIP_VALIDATION:-0}" == "1" ]]; then
        log_info "VICTOR_SKIP_VALIDATION=1, skipping workflow validation"
        exit 0
    fi

    # Check Python availability
    check_python

    # Initialize CI output files
    init_ci_output

    # Check if files provided
    if [[ $# -eq 0 ]]; then
        log_error "No files provided for validation"
        log_info "Usage: $0 [--ci] <workflow.yaml> [<workflow2.yaml> ...]"
        exit 1
    fi

    # Track overall status
    local total_files=0
    local total_errors=0
    local total_warnings=0
    local failed_files=()
    local start_time=$(date +%s)

    # Validate each file
    for file in "$@"; do
        # Convert to absolute path
        if [[ ! -f "$file" ]]; then
            # Try relative to project root
            file="${PROJECT_ROOT}/${file}"
        fi

        ((total_files++))

        # Track file-specific errors
        local file_errors=0
        local error_messages=""

        # Validate YAML syntax
        if ! validate_yaml_syntax "$file" 2>&1; then
            ((file_errors++))
            error_messages+="YAML syntax error\n"
        fi

        # Validate workflow compilation
        if ! validate_workflow_compilation "$file" 2>&1; then
            ((file_errors++))
            error_messages+="Workflow compilation error\n"
        fi

        # Add to CI results
        if [[ $file_errors -gt 0 ]]; then
            ((total_errors++))
            failed_files+=("$file")
            add_validation_result "$file" "failed" "Validation failed" "$error_messages"
        else
            add_validation_result "$file" "passed" "Validation successful" ""
        fi
    done

    local end_time=$(date +%s)
    local elapsed_time=$((end_time - start_time))

    # Finalize CI output
    finalize_ci_output "$total_files" "$total_errors" "$total_warnings" "$elapsed_time"

    # Print summary
    echo ""
    log_info "Validation complete: $total_files file(s) in ${elapsed_time}s"

    if [[ $total_errors -gt 0 ]]; then
        log_error "Validation failed for $total_errors file(s):"
        for file in "${failed_files[@]}"; do
            echo "  - $file"
        done
        exit 1
    fi

    log_success "All workflows validated successfully"
    exit 0
}

# Run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
