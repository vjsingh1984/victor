from __future__ import annotations

"""Tests for Docker tool operation whitelist security."""

import pytest
from unittest.mock import patch, MagicMock

from victor.tools.docker_tool import docker, SAFE_OPERATIONS, DANGEROUS_OPERATIONS


class TestDockerOperationWhitelist:
    """Verify dangerous Docker operations are gated by settings."""

    @pytest.mark.asyncio
    async def test_exec_blocked_by_default(self):
        """exec should be rejected when docker_allow_dangerous_operations is False."""
        mock_settings = MagicMock()
        mock_settings.docker_allow_dangerous_operations = False

        with (
            patch("victor.tools.docker_tool.check_docker_available", return_value=True),
            patch(
                "victor.tools.docker_tool.get_settings",
                return_value=mock_settings,
                create=True,
            ),
        ):
            # Patch the import inside the function
            with patch.dict(
                "sys.modules",
                {"victor.config.settings": MagicMock(get_settings=lambda: mock_settings)},
            ):
                result = await docker(
                    operation="exec",
                    resource_id="my-container",
                    options={"command": "ls"},
                )

        assert result["success"] is False
        assert "restricted" in result["error"]

    @pytest.mark.asyncio
    async def test_ps_allowed_by_default(self):
        """ps (a safe operation) should not be blocked regardless of settings."""
        with (
            patch("victor.tools.docker_tool.check_docker_available", return_value=True),
            patch(
                "victor.tools.docker_tool._run_docker_command_async",
                return_value=(True, "", ""),
            ),
        ):
            result = await docker(operation="ps")

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_exec_allowed_when_setting_enabled(self):
        """exec should proceed when docker_allow_dangerous_operations is True."""
        mock_settings = MagicMock()
        mock_settings.docker_allow_dangerous_operations = True

        with (
            patch("victor.tools.docker_tool.check_docker_available", return_value=True),
            patch.dict(
                "sys.modules",
                {"victor.config.settings": MagicMock(get_settings=lambda: mock_settings)},
            ),
            patch(
                "victor.tools.docker_tool._run_docker_command_async",
                return_value=(True, "output", ""),
            ),
        ):
            result = await docker(
                operation="exec",
                resource_id="my-container",
                options={"command": "ls -la"},
            )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_all_dangerous_ops_blocked_by_default(self):
        """Every operation in DANGEROUS_OPERATIONS should be blocked by default."""
        mock_settings = MagicMock()
        mock_settings.docker_allow_dangerous_operations = False

        for op in DANGEROUS_OPERATIONS:
            with (
                patch("victor.tools.docker_tool.check_docker_available", return_value=True),
                patch.dict(
                    "sys.modules",
                    {"victor.config.settings": MagicMock(get_settings=lambda: mock_settings)},
                ),
            ):
                result = await docker(
                    operation=op,
                    resource_id="test-resource",
                    options={"command": "echo hi"},
                )
            assert result["success"] is False, f"{op} should be blocked"
            assert "restricted" in result["error"], f"{op} error should mention restricted"

    def test_safe_and_dangerous_sets_are_disjoint(self):
        """SAFE_OPERATIONS and DANGEROUS_OPERATIONS must not overlap."""
        assert SAFE_OPERATIONS.isdisjoint(DANGEROUS_OPERATIONS)
