#!/usr/bin/env bash
set -euo pipefail

target="${1:-src/app.py}"
profile_arg=()
if [ -n "${VICTOR_PROFILE:-}" ]; then
  profile_arg=(--profile "$VICTOR_PROFILE")
fi

victor chat "${profile_arg[@]}" "Write unit tests for ${target}. Use the existing test framework and follow local conventions. Focus on edge cases."
