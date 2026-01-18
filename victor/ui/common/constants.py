"""Named constants for array/string indexing to improve maintainability.

This module provides semantic names for commonly used array/string indices
throughout the UI codebase, replacing magic numbers with meaningful constants.
"""

# =============================================================================
# General Array/String Indices
# =============================================================================

FIRST_ARG_INDEX = 0
SECOND_ARG_INDEX = 1
THIRD_ARG_INDEX = 2

FIRST_MATCH_INDEX = 0

# =============================================================================
# Session Database Row Indices
# Used when querying sessions from SQLite database
# Schema: id, name, provider, model, created_at, updated_at, data
# =============================================================================

SESSION_ID_INDEX = 0
SESSION_NAME_INDEX = 1
SESSION_PROVIDER_INDEX = 2
SESSION_MODEL_INDEX = 3
SESSION_CREATED_AT_INDEX = 4
SESSION_UPDATED_AT_INDEX = 5
SESSION_DATA_INDEX = 6

# =============================================================================
# UI Component Indices
# =============================================================================

CURSOR_ROW_INDEX = 0  # Row index for cursor position
CURSOR_COL_INDEX = 1  # Column index for cursor position

# =============================================================================
# Statistics/Analytics Indices
# =============================================================================

CI_LOWER_BOUND_INDEX = 0  # Lower bound of confidence interval
CI_UPPER_BOUND_INDEX = 1  # Upper bound of confidence interval

# =============================================================================
# Provider/Model Split Indices
# =============================================================================

PROVIDER_PART_INDEX = 0  # When splitting provider:model string
MODEL_PART_INDEX = 1  # When splitting provider:model string

# =============================================================================
# Session ID Format Indices
# Used when splitting session_id strings like "session:abc123"
# =============================================================================

SESSION_TYPE_INDEX = 0  # e.g., "session" or "project"
SESSION_UUID_INDEX = 1  # e.g., "abc123-def456"
