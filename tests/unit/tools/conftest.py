# Copyright 2025 Vijaykumar Singh <singhv@gmail.com>
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

"""Test fixtures for tools tests."""

import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def disable_sandbox_mode_for_tools():
    """
    Disable sandbox mode for all tools tests.

    In PLAN/EXPLORE modes, file writes are restricted to .victor/sandbox/.
    This fixture disables that restriction for unit tests so they can write
    to temp directories without errors.
    """
    with patch("victor.tools.filesystem.get_sandbox_path", return_value=None):
        yield
