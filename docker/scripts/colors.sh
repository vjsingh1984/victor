#!/bin/bash
# Color definitions for Docker scripts
# Usage: source docker/scripts/colors.sh
#
# This provides consistent color output across all Victor Docker scripts.
# If sourcing fails, scripts should define empty fallbacks.

# Basic colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'

# Modifiers
BOLD='\033[1m'
DIM='\033[2m'

# Reset
NC='\033[0m'  # No Color

# Export for use in subshells
export GREEN BLUE YELLOW CYAN RED MAGENTA BOLD DIM NC
