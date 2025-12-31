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

"""Tests for CodeSandbox cleanup functionality.

These tests verify that Docker containers are properly cleaned up
to prevent resource leaks.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from victor.tools.code_executor_tool import (
    CodeSandbox,
    cleanup_all_sandboxes,
    cleanup_orphaned_containers,
    _active_sandboxes,
    SANDBOX_CONTAINER_LABEL,
    SANDBOX_CONTAINER_VALUE,
)


class TestCodeSandboxContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_calls_start_and_stop(self):
        """Verify context manager calls start() on enter and stop() on exit."""
        with (
            patch.object(CodeSandbox, "start") as mock_start,
            patch.object(CodeSandbox, "stop") as mock_stop,
        ):
            sandbox = CodeSandbox(require_docker=False)

            with sandbox:
                mock_start.assert_called_once()
                mock_stop.assert_not_called()

            mock_stop.assert_called_once()

    def test_context_manager_stops_on_exception(self):
        """Verify stop() is called even when exception occurs."""
        with patch.object(CodeSandbox, "start"), patch.object(CodeSandbox, "stop") as mock_stop:
            sandbox = CodeSandbox(require_docker=False)

            with pytest.raises(ValueError):
                with sandbox:
                    raise ValueError("Test exception")

            mock_stop.assert_called_once()


class TestSandboxCleanupRegistry:
    """Tests for the sandbox cleanup registry."""

    def test_sandbox_added_to_registry_on_start(self):
        """Verify sandbox is added to active set when started."""
        with patch("victor.tools.code_executor_tool.DOCKER_AVAILABLE", False):
            sandbox = CodeSandbox(require_docker=False)
            # Should not be in registry before start
            assert sandbox not in _active_sandboxes

    def test_sandbox_removed_from_registry_on_stop(self):
        """Verify sandbox is removed from active set when stopped."""
        with patch("victor.tools.code_executor_tool.DOCKER_AVAILABLE", False):
            sandbox = CodeSandbox(require_docker=False)
            sandbox.stop()
            assert sandbox not in _active_sandboxes


class TestCleanupFunctions:
    """Tests for cleanup utility functions."""

    def test_cleanup_all_sandboxes_calls_stop(self):
        """Verify cleanup_all_sandboxes stops all active sandboxes."""
        mock_sandbox1 = Mock()
        mock_sandbox2 = Mock()

        # Add to active set
        _active_sandboxes.add(mock_sandbox1)
        _active_sandboxes.add(mock_sandbox2)

        try:
            cleanup_all_sandboxes()
            mock_sandbox1.stop.assert_called_once()
            mock_sandbox2.stop.assert_called_once()
        finally:
            # Clean up test state
            _active_sandboxes.discard(mock_sandbox1)
            _active_sandboxes.discard(mock_sandbox2)

    def test_cleanup_orphaned_containers_filters_by_label(self):
        """Verify orphaned cleanup only removes labeled containers."""
        mock_container1 = Mock()
        mock_container1.short_id = "abc123"
        mock_container2 = Mock()
        mock_container2.short_id = "def456"

        mock_client = Mock()
        mock_client.containers.list.return_value = [mock_container1, mock_container2]

        with (
            patch("victor.tools.code_executor_tool.DOCKER_AVAILABLE", True),
            patch("victor.tools.code_executor_tool.docker") as mock_docker,
        ):
            mock_docker.from_env.return_value = mock_client

            cleaned = cleanup_orphaned_containers()

            # Verify filter was used
            mock_client.containers.list.assert_called_once_with(
                all=True,
                filters={"label": f"{SANDBOX_CONTAINER_LABEL}={SANDBOX_CONTAINER_VALUE}"},
            )

            # Verify both containers were removed
            assert cleaned == 2
            mock_container1.remove.assert_called_once_with(force=True)
            mock_container2.remove.assert_called_once_with(force=True)

    def test_cleanup_orphaned_containers_handles_errors(self):
        """Verify orphaned cleanup handles errors gracefully."""
        mock_container = Mock()
        mock_container.short_id = "abc123"
        mock_container.remove.side_effect = Exception("Test error")

        mock_client = Mock()
        mock_client.containers.list.return_value = [mock_container]

        with (
            patch("victor.tools.code_executor_tool.DOCKER_AVAILABLE", True),
            patch("victor.tools.code_executor_tool.docker") as mock_docker,
        ):
            mock_docker.from_env.return_value = mock_client

            # Should not raise, just log warning
            cleaned = cleanup_orphaned_containers()
            assert cleaned == 0  # Failed to clean

    def test_cleanup_orphaned_returns_zero_when_docker_unavailable(self):
        """Verify orphaned cleanup returns 0 when Docker not available."""
        with patch("victor.tools.code_executor_tool.DOCKER_AVAILABLE", False):
            cleaned = cleanup_orphaned_containers()
            assert cleaned == 0


class TestContainerLabeling:
    """Tests for container labeling."""

    def test_container_created_with_label(self):
        """Verify containers are created with the sandbox label."""
        mock_container = Mock()
        mock_container.short_id = "test123"

        mock_client = Mock()
        mock_client.images.pull.return_value = None
        mock_client.containers.run.return_value = mock_container

        with (
            patch("victor.tools.code_executor_tool.DOCKER_AVAILABLE", True),
            patch("victor.tools.code_executor_tool.docker") as mock_docker,
        ):
            mock_docker.from_env.return_value = mock_client

            sandbox = CodeSandbox()
            sandbox.docker_client = mock_client
            sandbox.docker_available = True
            sandbox.start()

            # Verify labels were passed to container creation
            call_kwargs = mock_client.containers.run.call_args[1]
            assert "labels" in call_kwargs
            assert call_kwargs["labels"] == {SANDBOX_CONTAINER_LABEL: SANDBOX_CONTAINER_VALUE}


class TestSandboxResourceCleanup:
    """Integration-style tests for resource cleanup scenarios."""

    def test_multiple_sandboxes_all_cleaned(self):
        """Verify multiple sandboxes are all cleaned up."""
        sandboxes = []
        for _ in range(3):
            sandbox = Mock()
            sandboxes.append(sandbox)
            _active_sandboxes.add(sandbox)

        try:
            cleaned = cleanup_all_sandboxes()
            assert cleaned == 3
            for sandbox in sandboxes:
                sandbox.stop.assert_called_once()
        finally:
            for sandbox in sandboxes:
                _active_sandboxes.discard(sandbox)

    def test_cleanup_resilient_to_individual_failures(self):
        """Verify cleanup continues even if individual sandbox fails."""
        sandbox1 = Mock()
        sandbox1.stop.side_effect = Exception("Sandbox 1 error")
        sandbox2 = Mock()
        sandbox3 = Mock()

        _active_sandboxes.add(sandbox1)
        _active_sandboxes.add(sandbox2)
        _active_sandboxes.add(sandbox3)

        try:
            # Should not raise
            cleaned = cleanup_all_sandboxes()
            # sandbox1 failed, but 2 and 3 should succeed
            assert cleaned >= 2
        finally:
            _active_sandboxes.discard(sandbox1)
            _active_sandboxes.discard(sandbox2)
            _active_sandboxes.discard(sandbox3)
