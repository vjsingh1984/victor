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

"""pytest configuration and fixtures."""

import sys

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "requires_victor_coding: marks tests that require victor-coding package"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests that require victor-coding if not installed."""
    # Check if victor_coding is importable
    try:
        __import__("victor_coding")
        victor_coding_available = True
    except ImportError:
        victor_coding_available = False

    # Skip tests that require victor_coding if not available
    for item in items:
        # Skip tests in files that require victor_coding
        if not victor_coding_available:
            # Skip tests in specific test files
            file_path = str(item.fspath)
            if any(
                x in file_path
                for x in [
                    "test_file_editor_tool",
                    "test_lsp_tool",
                    "test_code_intelligence_tool",
                    "test_lsp.",
                    "lsp_write_enhancer",
                ]
            ):
                item.add_marker(
                    pytest.mark.skipif(
                        not victor_coding_available,
                        reason="victor-coding package not installed"
                    )
                )
