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

"""Test fixtures for cache module tests."""

import pytest

from victor.agent.cache.file_watcher import FileWatcher
from victor.agent.cache.dependency_extractor import DependencyExtractor


@pytest.fixture
async def file_watcher() -> FileWatcher:
    """Create a file watcher instance for testing."""
    watcher = FileWatcher()
    yield watcher
    # Cleanup
    if watcher.is_running():
        await watcher.stop()


@pytest.fixture
def dependency_extractor() -> DependencyExtractor:
    """Create a dependency extractor instance for testing."""
    return DependencyExtractor()
