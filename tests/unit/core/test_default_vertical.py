# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for configurable DEFAULT_VERTICAL."""

from __future__ import annotations

import importlib
import os
from unittest.mock import patch


class TestDefaultVertical:
    def test_default_is_coding(self):
        """Default value is 'coding' when no env var set."""
        env = {k: v for k, v in os.environ.items() if k != "VICTOR_DEFAULT_VERTICAL"}
        with patch.dict(os.environ, env, clear=True):
            import victor.core.constants

            importlib.reload(victor.core.constants)
            assert victor.core.constants.DEFAULT_VERTICAL == "coding"

    def test_env_var_override(self):
        """VICTOR_DEFAULT_VERTICAL env var overrides the default."""
        with patch.dict(os.environ, {"VICTOR_DEFAULT_VERTICAL": "research"}):
            import victor.core.constants

            importlib.reload(victor.core.constants)
            assert victor.core.constants.DEFAULT_VERTICAL == "research"

    def test_env_var_empty_keeps_coding(self):
        """Empty env var falls back to 'coding'."""
        with patch.dict(os.environ, {"VICTOR_DEFAULT_VERTICAL": ""}):
            import victor.core.constants

            importlib.reload(victor.core.constants)
            assert victor.core.constants.DEFAULT_VERTICAL == "coding"
