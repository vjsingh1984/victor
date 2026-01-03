#!/usr/bin/env bash
set -euo pipefail

target="${1:-src/app.py}"
profile_arg=()
if [ -n "${VICTOR_PROFILE:-}" ]; then
  profile_arg=(--profile "$VICTOR_PROFILE")
fi

victor chat "${profile_arg[@]}" "Refactor ${target} for clarity and maintainability. Preserve behavior. Summarize changes."
