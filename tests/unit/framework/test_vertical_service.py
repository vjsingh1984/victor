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

"""Tests for shared framework vertical integration service."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from victor.framework.vertical_integration import IntegrationResult
from victor.framework.vertical_service import (
    apply_vertical_configuration,
    clear_vertical_integration_pipeline_cache,
    get_vertical_integration_pipeline,
)


class DummyVertical:
    """Minimal vertical type for service-level tests."""

    name = "dummy"


class TestVerticalService:
    """Tests for framework-level vertical integration service."""

    def test_get_vertical_integration_pipeline_returns_singleton(self):
        """Service should reuse one pipeline instance by default."""
        p1 = get_vertical_integration_pipeline(reset=True)
        p2 = get_vertical_integration_pipeline()
        assert p1 is p2

    def test_get_vertical_integration_pipeline_thread_safe_singleton(self):
        """Concurrent access should return one singleton pipeline instance."""
        get_vertical_integration_pipeline(reset=True)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(get_vertical_integration_pipeline) for _ in range(24)]
            instances = [f.result() for f in futures]

        assert len({id(instance) for instance in instances}) == 1

    def test_apply_vertical_configuration_uses_shared_pipeline(self):
        """apply_vertical_configuration should delegate to shared pipeline."""
        orchestrator = MagicMock()
        expected = IntegrationResult(success=True, vertical_name="dummy")

        with patch(
            "victor.framework.vertical_service.get_vertical_integration_pipeline"
        ) as mock_get:
            pipeline = MagicMock()
            pipeline.apply.return_value = expected
            mock_get.return_value = pipeline

            result = apply_vertical_configuration(orchestrator, DummyVertical, source="sdk")

        assert result is expected
        pipeline.apply.assert_called_once_with(orchestrator, DummyVertical)

    def test_clear_vertical_integration_pipeline_cache_delegates_to_pipeline(self):
        """Cache clear helper should delegate to shared pipeline clear_cache()."""
        with patch(
            "victor.framework.vertical_service.get_vertical_integration_pipeline"
        ) as mock_get:
            pipeline = MagicMock()
            mock_get.return_value = pipeline

            clear_vertical_integration_pipeline_cache()

        pipeline.clear_cache.assert_called_once_with()
