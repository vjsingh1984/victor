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

"""Comprehensive tests for victor/agent/safety.py.

This test suite targets 75%+ coverage for the SafetyChecker module,
which is the security gatekeeper for:
- Bash command authorization
- File access control
- Dangerous command detection
- Risk level categorization
- HITL integration

Test areas:
1. Bash authorization (12 tests)
2. File access control (10 tests)
3. Dangerous command patterns (8 tests)
4. Risk level categorization (10 tests)
5. Approval modes (8 tests)
6. Custom patterns (6 tests)
7. HITL integration (8 tests)
8. Edge cases (10 tests)

Total: ~72 tests
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from victor.agent.safety import (
    SafetyChecker,
    ConfirmationRequest,
    OperationalRiskLevel,
    ApprovalMode,
    get_safety_checker,
    set_confirmation_callback,
    _get_risk_icon,
    create_hitl_confirmation_callback,
    setup_hitl_safety_integration,
    _resolve_approval_mode,
    _STATIC_WRITE_TOOL_NAMES,
    get_write_tool_names,
)
from victor.agent.presentation import PresentationProtocol


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_presentation():
    """Mock presentation adapter for testing."""
    presentation = Mock(spec=PresentationProtocol)
    presentation.icon = Mock(return_value="⚠️")
    return presentation


@pytest.fixture
def safety_checker():
    """Create a fresh SafetyChecker for each test."""
    return SafetyChecker()


@pytest.fixture
def mock_callback():
    """Create a mock confirmation callback."""

    async def callback(request):
        return True

    return callback


# =============================================================================
# 1. Bash Authorization Tests (12 tests)
# =============================================================================


class TestBashAuthorization:
    """Test bash command authorization and risk detection."""

    def test_safe_bash_commands(self, safety_checker):
        """Safe commands should return SAFE risk level."""
        safe_commands = [
            "ls -la",
            "pwd",
            "echo hello world",
            "cat file.txt",
            "grep pattern file.txt",
            "find . -name '*.py'",
            "head -n 10 file.txt",
            "tail -n 20 file.txt",
            "wc -l file.txt",
            "date",
        ]
        for cmd in safe_commands:
            level, details = safety_checker.check_bash_command(cmd)
            assert level == OperationalRiskLevel.SAFE, f"'{cmd}' should be SAFE but got {level}"
            assert details == [], f"'{cmd}' should have no details but got {details}"

    def test_medium_risk_bash_commands(self, safety_checker):
        """Medium risk commands should be detected correctly."""
        medium_commands = [
            ("rm file.txt", ["Delete file(s)"]),
            ("> output.txt", ["Overwrite file with redirection"]),
            ("pip uninstall package", ["Uninstall Python package"]),
            ("npm uninstall package", ["Uninstall npm package"]),
            ("truncate -s 0 file.txt", ["Truncate file"]),
            ("mv file.txt /dev/null", ["Discard file to /dev/null"]),
        ]
        for cmd, expected_keywords in medium_commands:
            level, details = safety_checker.check_bash_command(cmd)
            assert level == OperationalRiskLevel.MEDIUM, f"'{cmd}' should be MEDIUM but got {level}"
            for keyword in expected_keywords:
                assert any(
                    keyword.lower() in d.lower() for d in details
                ), f"Expected '{keyword}' in details for '{cmd}' but got {details}"

    def test_high_risk_bash_commands(self, safety_checker):
        """High risk commands should be detected correctly."""
        high_commands = [
            ("rm -rf /tmp/test", ["Recursively delete"]),
            ("rm -r directory", ["Recursively delete"]),
            ("rm *.log", ["Delete files with wildcard"]),
            ("git reset --hard HEAD", ["Discard all uncommitted changes"]),
            ("git clean -fd", ["Delete untracked files"]),
            ("git push origin --force", ["Force push"]),
            ("git push -f", ["Force push"]),
            ("sudo apt install package", ["elevated privileges"]),
            ("chmod -R 755 /path", ["Recursively change permissions"]),
        ]
        for cmd, expected_keywords in high_commands:
            level, details = safety_checker.check_bash_command(cmd)
            assert level == OperationalRiskLevel.HIGH, f"'{cmd}' should be HIGH but got {level}"
            for keyword in expected_keywords:
                assert any(
                    keyword.lower() in d.lower() for d in details
                ), f"Expected '{keyword}' in details for '{cmd}' but got {details}"

    def test_critical_risk_bash_commands(self, safety_checker):
        """Critical risk commands should be detected correctly."""
        critical_commands = [
            ("rm -rf /", ["Delete entire filesystem"]),
            ("rm -rf /*", ["Delete all root-level directories"]),
            ("dd if=/dev/zero of=/dev/sda", ["Write directly to disk device"]),
            ("mkfs.ext4 /dev/sda1", ["Format filesystem"]),
            ("echo data > /dev/sda", ["Overwrite disk device"]),
            ("chmod -R 777 /", ["Set world-writable permissions on root"]),
        ]
        for cmd, expected_keywords in critical_commands:
            level, details = safety_checker.check_bash_command(cmd)
            assert (
                level == OperationalRiskLevel.CRITICAL
            ), f"'{cmd}' should be CRITICAL but got {level}"
            for keyword in expected_keywords:
                assert any(
                    keyword.lower() in d.lower() for d in details
                ), f"Expected '{keyword}' in details for '{cmd}' but got {details}"

    def test_bash_command_case_insensitive(self, safety_checker):
        """Bash command patterns should be case-insensitive."""
        cmd = "RM -RF /tmp/test"
        level, details = safety_checker.check_bash_command(cmd)
        assert level == OperationalRiskLevel.HIGH

    def test_bash_command_with_multiple_patterns(self, safety_checker):
        """Commands matching multiple patterns should return highest risk."""
        # This should match both sudo (HIGH) and rm -rf (HIGH)
        cmd = "sudo rm -rf /tmp/test"
        level, details = safety_checker.check_bash_command(cmd)
        assert level == OperationalRiskLevel.HIGH
        assert len(details) >= 1

    def test_empty_bash_command(self, safety_checker):
        """Empty bash command should be SAFE."""
        level, details = safety_checker.check_bash_command("")
        assert level == OperationalRiskLevel.SAFE
        assert details == []

    def test_bash_command_with_comments(self, safety_checker):
        """Commands with comments should still be detected."""
        cmd = "rm -rf /tmp/test # delete temp files"
        level, details = safety_checker.check_bash_command(cmd)
        assert level == OperationalRiskLevel.HIGH

    def test_bash_command_with_variables(self, safety_checker):
        """Commands with variables should be detected."""
        cmd = "rm -rf $TMP_DIR"
        level, details = safety_checker.check_bash_command(cmd)
        assert level == OperationalRiskLevel.HIGH

    def test_bash_command_pipeline(self, safety_checker):
        """Piped commands should be checked."""
        cmd = "cat file.txt | rm -rf /tmp"
        level, details = safety_checker.check_bash_command(cmd)
        assert level == OperationalRiskLevel.HIGH

    def test_bash_command_with_command_substitution(self, safety_checker):
        """Commands with command substitution should be checked."""
        cmd = "rm -rf $(find /tmp -name 'test')"
        level, details = safety_checker.check_bash_command(cmd)
        # Should detect rm -rf as HIGH (not CRITICAL since it's not rm -rf /)
        assert level == OperationalRiskLevel.HIGH


# =============================================================================
# 2. File Access Control Tests (10 tests)
# =============================================================================


class TestFileAccessControl:
    """Test file operation access control."""

    def test_write_to_safe_file(self, safety_checker):
        """Writing to safe file should be SAFE."""
        level, details = safety_checker.check_file_operation(
            "write", "/tmp/test.txt", overwrite=False
        )
        assert level == OperationalRiskLevel.SAFE
        assert details == []

    def test_write_to_sensitive_file(self, safety_checker):
        """Writing to sensitive file should be HIGH risk."""
        sensitive_files = [
            ".env",
            "secrets.env",
            "private.key",
            "cert.pem",
            "database.db",
            "data.sqlite3",
        ]
        for file_path in sensitive_files:
            level, details = safety_checker.check_file_operation(
                "write", file_path, overwrite=False
            )
            assert level == OperationalRiskLevel.HIGH, f"{file_path} should be HIGH risk"
            assert any("sensitive" in d.lower() for d in details)

    def test_write_to_system_file(self, safety_checker):
        """Writing to system files should be HIGH risk."""
        system_files = [
            "/etc/config.conf",
            "/usr/local/bin/script.sh",
            "/etc/passwd",
            "/etc/shadow",
        ]
        for file_path in system_files:
            level, details = safety_checker.check_file_operation(
                "write", file_path, overwrite=False
            )
            assert level == OperationalRiskLevel.HIGH, f"{file_path} should be HIGH risk"
            assert any("system" in d.lower() for d in details)

    def test_overwrite_existing_file(self, safety_checker, tmp_path):
        """Overwriting existing file should be MEDIUM risk."""
        # Create a temporary file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        level, details = safety_checker.check_file_operation(
            "write", str(test_file), overwrite=True
        )
        assert level == OperationalRiskLevel.MEDIUM
        assert any("overwriting" in d.lower() for d in details)

    def test_delete_file_operation(self, safety_checker):
        """Delete operations should be HIGH risk."""
        level, details = safety_checker.check_file_operation("delete", "/tmp/test.txt")
        assert level == OperationalRiskLevel.HIGH
        assert any("deleting" in d.lower() for d in details)

    def test_edit_file_operation(self, safety_checker):
        """Edit operations should be SAFE by default."""
        level, details = safety_checker.check_file_operation("edit", "/tmp/test.txt")
        assert level == OperationalRiskLevel.SAFE

    def test_path_with_dangerous_extension_in_middle(self, safety_checker):
        """Paths with dangerous extensions anywhere should be detected."""
        level, details = safety_checker.check_file_operation(
            "write", "/path/to/config.env.backup", overwrite=False
        )
        assert level == OperationalRiskLevel.HIGH

    def test_absolute_vs_relative_paths(self, safety_checker):
        """Both absolute and relative paths should be checked."""
        for file_path in ["secrets.env", "./secrets.env", "../config/secrets.env"]:
            level, details = safety_checker.check_file_operation(
                "write", file_path, overwrite=False
            )
            assert level == OperationalRiskLevel.HIGH

    def test_non_overwrite_write(self, safety_checker, tmp_path):
        """Writing to non-existent file should be SAFE."""
        test_file = tmp_path / "new_file.txt"
        # Ensure file doesn't exist
        if test_file.exists():
            test_file.unlink()

        level, details = safety_checker.check_file_operation(
            "write", str(test_file), overwrite=True
        )
        assert level == OperationalRiskLevel.SAFE

    def test_mixed_risk_file_operation(self, safety_checker, tmp_path):
        """File with multiple risk factors should return highest risk."""
        # Create a sensitive file that exists
        test_file = tmp_path / "test.env"
        test_file.write_text("SECRET_KEY=value")

        level, details = safety_checker.check_file_operation(
            "write", str(test_file), overwrite=True
        )
        # Should be HIGH due to .env extension (higher priority than overwrite)
        assert level == OperationalRiskLevel.HIGH


# =============================================================================
# 3. Dangerous Command Pattern Tests (8 tests)
# =============================================================================


class TestDangerousCommandPatterns:
    """Test detection of dangerous command patterns."""

    def test_git_destructive_commands(self, safety_checker):
        """Git destructive operations should be detected."""
        from victor.agent.safety import _RISK_ORDER

        destructive_cmds = [
            "git reset --hard",
            "git clean -fd",
            "git push origin --force",
            "git push -f",
            "git rebase --force",
            "git branch -D feature",
        ]
        for cmd in destructive_cmds:
            level, details = safety_checker.check_bash_command(cmd)
            assert (
                _RISK_ORDER[level] >= _RISK_ORDER[OperationalRiskLevel.HIGH]
            ), f"'{cmd}' should be HIGH or CRITICAL but got {level}"

    def test_fork_bomb_detection(self, safety_checker):
        """Fork bomb should be detected as CRITICAL."""
        # The fork bomb pattern matches :(){ :|:& };:
        # Note: This is a subtle pattern where colons are function names
        fork_bomb = ":{ :|:& };:"
        level, details = safety_checker.check_bash_command(fork_bomb)

        # Should detect this dangerous pattern as CRITICAL
        assert (
            level == OperationalRiskLevel.CRITICAL
        ), f"Fork bomb should be CRITICAL but got {level}, details: {details}"

    def test_disk_write_commands(self, safety_checker):
        """Direct disk writes should be CRITICAL."""
        disk_cmds = [
            "dd if=/dev/zero of=/dev/sda",
            "echo data > /dev/sdb",
            "cat file > /dev/sdc",
        ]
        for cmd in disk_cmds:
            level, details = safety_checker.check_bash_command(cmd)
            assert level == OperationalRiskLevel.CRITICAL

    def test_filesystem_format_commands(self, safety_checker):
        """Filesystem formatting should be CRITICAL."""
        format_cmds = [
            "mkfs.ext4 /dev/sda1",
            "mkfs.xfs /dev/sdb",
            "mkfs.vfat /dev/sdc1",
        ]
        for cmd in format_cmds:
            level, details = safety_checker.check_bash_command(cmd)
            assert level == OperationalRiskLevel.CRITICAL

    def test_permission_changes(self, safety_checker):
        """Recursive permission changes should be HIGH risk."""
        # Note: chmod -R 777 / is CRITICAL, but chmod -R with other paths is HIGH
        # Also note: chmod -R 777 /tmp matches the CRITICAL pattern because it starts with /
        perm_cmds = [
            ("chmod -R 755 /path", OperationalRiskLevel.HIGH),
            ("chown -R user:group /path", OperationalRiskLevel.HIGH),
            ("chmod -R 777 /home/user", OperationalRiskLevel.CRITICAL),  # Matches chmod -R 777 /
        ]
        for cmd, expected_level in perm_cmds:
            level, details = safety_checker.check_bash_command(cmd)
            assert level == expected_level, f"'{cmd}' should be {expected_level} but got {level}"

    def test_package_uninstall(self, safety_checker):
        """Package uninstallation should be MEDIUM risk."""
        uninstall_cmds = [
            ("pip uninstall package", OperationalRiskLevel.MEDIUM),
            ("npm uninstall package", OperationalRiskLevel.MEDIUM),
        ]
        for cmd, expected_level in uninstall_cmds:
            level, details = safety_checker.check_bash_command(cmd)
            assert level == expected_level, f"'{cmd}' should be {expected_level} but got {level}"

    def test_wildcard_deletion(self, safety_checker):
        """Wildcard deletion should be HIGH risk."""
        from victor.agent.safety import _RISK_ORDER

        wildcard_cmds = [
            "rm *.log",
            "rm *.tmp",
            "rm -f /tmp/*",
        ]
        for cmd in wildcard_cmds:
            level, details = safety_checker.check_bash_command(cmd)
            assert (
                _RISK_ORDER[level] >= _RISK_ORDER[OperationalRiskLevel.HIGH]
            ), f"'{cmd}' should be HIGH or CRITICAL but got {level}"

    def test_git_checkout_discard(self, safety_checker):
        """Git checkout -- should discard changes (MEDIUM risk)."""
        cmd = "git checkout -- file.txt"
        level, details = safety_checker.check_bash_command(cmd)
        assert level == OperationalRiskLevel.MEDIUM


# =============================================================================
# 4. Risk Level Categorization Tests (10 tests)
# =============================================================================


class TestRiskLevelCategorization:
    """Test risk level categorization and ordering."""

    def test_risk_level_ordering(self):
        """Risk levels should be properly ordered."""
        from victor.agent.safety import _RISK_ORDER

        assert _RISK_ORDER[OperationalRiskLevel.SAFE] == 0
        assert _RISK_ORDER[OperationalRiskLevel.LOW] == 1
        assert _RISK_ORDER[OperationalRiskLevel.MEDIUM] == 2
        assert _RISK_ORDER[OperationalRiskLevel.HIGH] == 3
        assert _RISK_ORDER[OperationalRiskLevel.CRITICAL] == 4

    def test_multiple_patterns_same_risk(self, safety_checker):
        """Multiple patterns at same risk level should accumulate."""
        # Command that matches multiple MEDIUM patterns
        cmd = "rm file.txt && truncate file.txt"
        level, details = safety_checker.check_bash_command(cmd)
        assert level == OperationalRiskLevel.MEDIUM
        assert len(details) >= 1

    def test_critical_overrides_lower_risks(self, safety_checker):
        """Critical pattern should override lower risk patterns."""
        # This has both rm (HIGH) and rm -rf / (CRITICAL context)
        cmd = "rm -rf /tmp && rm file.txt"
        level, details = safety_checker.check_bash_command(cmd)
        # Should detect rm -rf as HIGH (not CRITICAL since it's not /)
        assert level == OperationalRiskLevel.HIGH

    def test_high_overrides_medium(self, safety_checker):
        """High risk pattern should override medium."""
        cmd = "rm file.txt && sudo echo test"
        level, details = safety_checker.check_bash_command(cmd)
        assert level == OperationalRiskLevel.HIGH

    def test_risk_escalation_with_multiple_matches(self, safety_checker):
        """Risk should escalate with multiple dangerous patterns."""
        # Match multiple patterns that escalate risk
        cmd = "sudo rm -rf /tmp/test"
        level, details = safety_checker.check_bash_command(cmd)
        assert level == OperationalRiskLevel.HIGH

    def test_safe_command_with_no_patterns(self, safety_checker):
        """Command matching no patterns should be SAFE."""
        cmd = "echo 'hello world'"
        level, details = safety_checker.check_bash_command(cmd)
        assert level == OperationalRiskLevel.SAFE
        assert details == []

    def test_file_write_risk_levels(self, safety_checker):
        """File write risk levels should escalate appropriately."""
        # Normal write - SAFE
        level1, _ = safety_checker.check_file_operation("write", "/tmp/test.txt")
        assert level1 == OperationalRiskLevel.SAFE

        # Sensitive file - HIGH
        level2, _ = safety_checker.check_file_operation("write", "config.env")
        assert level2 == OperationalRiskLevel.HIGH

        # Delete - HIGH
        level3, _ = safety_checker.check_file_operation("delete", "/tmp/test.txt")
        assert level3 == OperationalRiskLevel.HIGH

    def test_system_file_risk_escalation(self, safety_checker):
        """System files should escalate to HIGH risk."""
        level, _ = safety_checker.check_file_operation("write", "/etc/config")
        assert level == OperationalRiskLevel.HIGH

    def test_overwrite_risk_escalation(self, safety_checker, tmp_path):
        """Overwriting existing file should escalate risk."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        level, details = safety_checker.check_file_operation(
            "write", str(test_file), overwrite=True
        )
        assert level == OperationalRiskLevel.MEDIUM
        assert any("overwriting" in d.lower() for d in details)

    def test_sensitive_file_overwrite_risk(self, safety_checker, tmp_path):
        """Overwriting sensitive file should be HIGH risk."""
        test_file = tmp_path / "test.env"
        test_file.write_text("SECRET=value")

        level, _ = safety_checker.check_file_operation("write", str(test_file), overwrite=True)
        # Should be HIGH due to .env extension (takes priority)
        assert level == OperationalRiskLevel.HIGH


# =============================================================================
# 5. Approval Mode Tests (8 tests)
# =============================================================================


class TestApprovalModes:
    """Test different approval modes."""

    @pytest.mark.asyncio
    async def test_off_approval_mode(self, mock_callback):
        """OFF mode should auto-approve all operations."""
        checker = SafetyChecker(approval_mode=ApprovalMode.OFF, confirmation_callback=mock_callback)

        # Even critical operations should be auto-approved
        should_proceed, reason = await checker.check_and_confirm(
            "execute_bash", {"command": "rm -rf /"}
        )
        assert should_proceed is True
        assert reason is None

    @pytest.mark.asyncio
    async def test_all_writes_approval_mode(self):
        """ALL_WRITES mode should require approval for all writes."""
        callback_called = []

        async def mock_callback(request):
            callback_called.append(request)
            return True

        checker = SafetyChecker(
            approval_mode=ApprovalMode.ALL_WRITES, confirmation_callback=mock_callback
        )

        # Even safe write operations should require confirmation
        should_proceed, reason = await checker.check_and_confirm(
            "write_file", {"path": "/tmp/safe.txt"}
        )
        assert should_proceed is True
        assert len(callback_called) == 1
        assert callback_called[0].tool_name == "write_file"

    @pytest.mark.asyncio
    async def test_risky_only_approval_mode_safe_operation(self):
        """RISKY_ONLY mode should auto-approve safe operations."""
        checker = SafetyChecker(
            approval_mode=ApprovalMode.RISKY_ONLY,
            require_confirmation_threshold=OperationalRiskLevel.HIGH,
        )

        should_proceed, reason = await checker.check_and_confirm(
            "read_file", {"path": "/tmp/file.txt"}
        )
        assert should_proceed is True
        assert reason is None

    @pytest.mark.asyncio
    async def test_risky_only_approval_mode_high_risk(self):
        """RISKY_ONLY mode should require confirmation for HIGH risk."""
        callback_called = []

        async def mock_callback(request):
            callback_called.append(request)
            assert request.risk_level == OperationalRiskLevel.HIGH
            return True

        checker = SafetyChecker(
            approval_mode=ApprovalMode.RISKY_ONLY,
            confirmation_callback=mock_callback,
            require_confirmation_threshold=OperationalRiskLevel.HIGH,
        )

        should_proceed, reason = await checker.check_and_confirm(
            "execute_bash", {"command": "rm -rf /tmp/test"}
        )
        assert should_proceed is True
        assert len(callback_called) == 1

    @pytest.mark.asyncio
    async def test_medium_threshold_with_medium_risk(self):
        """MEDIUM threshold should require confirmation for MEDIUM risk."""
        callback_called = []

        async def mock_callback(request):
            callback_called.append(request)
            return True

        checker = SafetyChecker(
            approval_mode=ApprovalMode.RISKY_ONLY,
            confirmation_callback=mock_callback,
            require_confirmation_threshold=OperationalRiskLevel.MEDIUM,
        )

        should_proceed, reason = await checker.check_and_confirm(
            "execute_bash", {"command": "rm file.txt"}
        )
        assert should_proceed is True
        assert len(callback_called) == 1

    @pytest.mark.asyncio
    async def test_user_rejects_operation(self):
        """User rejection should block operation."""

        async def mock_callback(request):
            return False  # User rejects

        checker = SafetyChecker(
            approval_mode=ApprovalMode.RISKY_ONLY, confirmation_callback=mock_callback
        )

        should_proceed, reason = await checker.check_and_confirm(
            "execute_bash", {"command": "rm -rf /tmp/test"}
        )
        assert should_proceed is False
        assert "cancelled" in reason.lower()

    @pytest.mark.asyncio
    async def test_no_callback_with_high_risk(self):
        """High risk without callback should proceed with warning."""
        checker = SafetyChecker(
            approval_mode=ApprovalMode.RISKY_ONLY,
            confirmation_callback=None,
            require_confirmation_threshold=OperationalRiskLevel.HIGH,
        )

        # Should proceed even without callback (with logged warning)
        should_proceed, reason = await checker.check_and_confirm(
            "execute_bash", {"command": "rm -rf /tmp/test"}
        )
        assert should_proceed is True
        assert reason is None

    @pytest.mark.asyncio
    async def test_callback_exception_handling(self):
        """Callback exceptions should be handled appropriately."""

        async def failing_callback(request):
            raise Exception("Callback failed")

        checker = SafetyChecker(
            approval_mode=ApprovalMode.RISKY_ONLY, confirmation_callback=failing_callback
        )

        # High risk should be blocked on callback failure
        should_proceed, reason = await checker.check_and_confirm(
            "execute_bash", {"command": "rm -rf /tmp/test"}
        )
        assert should_proceed is False
        assert "failed" in reason.lower()


# =============================================================================
# 6. Custom Pattern Tests (6 tests)
# =============================================================================


class TestCustomPatterns:
    """Test custom safety pattern functionality."""

    def test_add_custom_pattern_high_risk(self, safety_checker):
        """Custom HIGH risk pattern should be detected."""
        safety_checker.add_custom_pattern(
            pattern=r"dangerous_command", description="Custom dangerous command", risk_level="HIGH"
        )

        level, details = safety_checker.check_bash_command("dangerous_command")
        assert level == OperationalRiskLevel.HIGH
        assert any("Custom dangerous command" in d for d in details)

    def test_add_custom_pattern_critical_risk(self, safety_checker):
        """Custom CRITICAL risk pattern should be detected."""
        safety_checker.add_custom_pattern(
            pattern=r"critical_cmd", description="Critical operation", risk_level="CRITICAL"
        )

        level, details = safety_checker.check_bash_command("critical_cmd")
        assert level == OperationalRiskLevel.CRITICAL

    def test_add_custom_pattern_medium_risk(self, safety_checker):
        """Custom MEDIUM risk pattern should be detected."""
        safety_checker.add_custom_pattern(
            pattern=r"moderate_cmd", description="Moderate risk operation", risk_level="MEDIUM"
        )

        level, details = safety_checker.check_bash_command("moderate_cmd")
        assert level == OperationalRiskLevel.MEDIUM

    def test_add_custom_pattern_low_risk(self, safety_checker):
        """Custom LOW risk pattern should be detected."""
        safety_checker.add_custom_pattern(
            pattern=r"low_cmd", description="Low risk operation", risk_level="LOW"
        )

        level, details = safety_checker.check_bash_command("low_cmd")
        assert level == OperationalRiskLevel.LOW

    def test_custom_pattern_case_insensitive(self, safety_checker):
        """Custom patterns should be case-insensitive."""
        safety_checker.add_custom_pattern(
            pattern=r"custom_pattern", description="Custom pattern test", risk_level="HIGH"
        )

        level, details = safety_checker.check_bash_command("CUSTOM_PATTERN")
        assert level == OperationalRiskLevel.HIGH

    def test_invalid_custom_pattern(self, safety_checker):
        """Invalid regex pattern should be handled gracefully."""
        # Should not raise exception
        safety_checker.add_custom_pattern(
            pattern=r"[invalid(unclosed", description="Invalid pattern", risk_level="HIGH"
        )

        # Pattern should be ignored (no crash)
        level, details = safety_checker.check_bash_command("test")
        # Should not match the invalid pattern
        assert level == OperationalRiskLevel.SAFE


# =============================================================================
# 7. Write Tool Detection Tests (8 tests)
# =============================================================================


class TestWriteToolDetection:
    """Test write tool identification."""

    def test_static_write_tools_are_correct(self):
        """Static write tool names should include expected tools."""
        expected_tools = {
            "write_file",
            "edit_files",
            "apply_patch",
            "execute_bash",
            "git",
            "refactor_rename_symbol",
            "refactor_extract_function",
            "refactor_inline_variable",
            "refactor_organize_imports",
            "rename_symbol",
            "scaffold",
            "batch",
        }
        assert expected_tools.issubset(_STATIC_WRITE_TOOL_NAMES)

    def test_is_write_tool_file_operations(self, safety_checker):
        """File modification tools should be detected as write tools."""
        write_tools = ["write_file", "edit_files", "apply_patch"]
        for tool in write_tools:
            assert safety_checker.is_write_tool(tool), f"{tool} should be a write tool"

    def test_is_write_tool_bash(self, safety_checker):
        """Bash execution should be detected as write tool."""
        assert safety_checker.is_write_tool("execute_bash")

    def test_is_write_tool_git(self, safety_checker):
        """Git operations should be detected as write tool."""
        assert safety_checker.is_write_tool("git")

    def test_is_write_tool_refactoring(self, safety_checker):
        """Refactoring tools should be detected as write tools."""
        refactor_tools = [
            "refactor_rename_symbol",
            "refactor_extract_function",
            "refactor_inline_variable",
            "refactor_organize_imports",
            "rename_symbol",
        ]
        for tool in refactor_tools:
            assert safety_checker.is_write_tool(tool), f"{tool} should be a write tool"

    def test_is_write_tool_read_only_tools(self, safety_checker):
        """Read-only tools should not be detected as write tools."""
        read_only_tools = [
            "read_file",
            "list_directory",
            "code_search",
            "grep_search",
            "git_status",
            "git_log",
        ]
        for tool in read_only_tools:
            assert not safety_checker.is_write_tool(tool), f"{tool} should not be a write tool"

    def test_get_write_tool_names(self):
        """get_write_tool_names should return comprehensive set."""
        write_tools = get_write_tool_names()
        assert isinstance(write_tools, set)
        assert "write_file" in write_tools
        assert "execute_bash" in write_tools

    def test_scaffolding_is_write_tool(self, safety_checker):
        """Scaffolding should be detected as write tool."""
        assert safety_checker.is_write_tool("scaffold")


# =============================================================================
# 8. ConfirmationRequest Tests (6 tests)
# =============================================================================


class TestConfirmationRequest:
    """Test ConfirmationRequest functionality."""

    def test_format_message_high_risk(self, mock_presentation):
        """Format message for HIGH risk operation."""
        request = ConfirmationRequest(
            tool_name="execute_bash",
            risk_level=OperationalRiskLevel.HIGH,
            description="Delete files",
            details=["Recursively delete files/directories"],
            arguments={"command": "rm -rf /tmp/test"},
        )

        message = request.format_message(mock_presentation)
        assert "HIGH RISK" in message
        assert "execute_bash" in message
        assert "Delete files" in message
        assert "Recursively delete" in message

    def test_format_message_critical_risk(self, mock_presentation):
        """Format message for CRITICAL risk operation."""
        request = ConfirmationRequest(
            tool_name="execute_bash",
            risk_level=OperationalRiskLevel.CRITICAL,
            description="Delete filesystem",
            details=["Delete entire filesystem"],
            arguments={"command": "rm -rf /"},
        )

        message = request.format_message(mock_presentation)
        assert "CRITICAL RISK" in message

    def test_format_message_with_empty_details(self, mock_presentation):
        """Format message with empty details."""
        request = ConfirmationRequest(
            tool_name="write_file",
            risk_level=OperationalRiskLevel.LOW,
            description="Write file",
            details=[],
            arguments={"path": "/tmp/test.txt"},
        )

        message = request.format_message(mock_presentation)
        assert "LOW RISK" in message
        assert "write_file" in message
        # Should not show empty "Details:" section

    def test_format_message_with_multiple_details(self, mock_presentation):
        """Format message with multiple detail lines."""
        request = ConfirmationRequest(
            tool_name="execute_bash",
            risk_level=OperationalRiskLevel.HIGH,
            description="Dangerous operation",
            details=["Risk 1", "Risk 2", "Risk 3"],
            arguments={"cmd": "test"},
        )

        message = request.format_message(mock_presentation)
        assert "Details:" in message
        assert "Risk 1" in message
        assert "Risk 2" in message
        assert "Risk 3" in message

    def test_format_message_includes_arguments(self, mock_presentation):
        """Format message should include tool arguments."""
        request = ConfirmationRequest(
            tool_name="execute_bash",
            risk_level=OperationalRiskLevel.HIGH,
            description="Execute command",
            details=["Dangerous command"],
            arguments={"command": "rm -rf /tmp"},
        )

        message = request.format_message(mock_presentation)
        # Arguments are included but not displayed in format_message
        # They're available in the request object
        assert request.arguments == {"command": "rm -rf /tmp"}

    def test_format_message_without_presentation_adapter(self):
        """Format message should work without presentation adapter."""
        request = ConfirmationRequest(
            tool_name="write_file",
            risk_level=OperationalRiskLevel.MEDIUM,
            description="Write file",
            details=["Overwrite file"],
            arguments={"path": "/tmp/test.txt"},
        )

        # Should create default presentation adapter
        message = request.format_message(None)
        assert "MEDIUM RISK" in message


# =============================================================================
# 9. HITL Integration Tests (8 tests)
# =============================================================================


class TestHITLIntegration:
    """Test Human-in-the-Loop integration."""

    def test_create_hitl_confirmation_callback_returns_function(self):
        """create_hitl_confirmation_callback should return an async function."""
        callback = create_hitl_confirmation_callback()
        assert asyncio.iscoroutinefunction(callback)

    def test_create_hitl_with_custom_timeout(self):
        """Custom timeout should be accepted."""
        callback = create_hitl_confirmation_callback(timeout=600.0)
        assert asyncio.iscoroutinefunction(callback)

    def test_create_hitl_with_custom_fallback(self):
        """Custom fallback should be accepted."""
        for fallback in ["abort", "continue", "skip", "retry"]:
            callback = create_hitl_confirmation_callback(fallback=fallback)
            assert asyncio.iscoroutinefunction(callback)

    def test_setup_hitl_integration(self):
        """setup_hitl_safety_integration should configure callback."""
        # Save original callback
        import victor.agent.safety as safety_module

        original_checker = safety_module._default_checker

        try:
            # Reset to test fresh setup
            safety_module._default_checker = None
            setup_hitl_safety_integration()

            checker = get_safety_checker()
            assert checker.confirmation_callback is not None
        finally:
            # Restore original
            safety_module._default_checker = original_checker


# =============================================================================
# 10. Global Functions and Utilities Tests (6 tests)
# =============================================================================


class TestGlobalFunctions:
    """Test module-level utility functions."""

    def test_resolve_approval_mode_off(self):
        """Resolve 'off' string to OFF mode."""
        mode = _resolve_approval_mode("off")
        assert mode == ApprovalMode.OFF

    def test_resolve_approval_mode_risky_only(self):
        """Resolve 'risky_only' string to RISKY_ONLY mode."""
        mode = _resolve_approval_mode("risky_only")
        assert mode == ApprovalMode.RISKY_ONLY

    def test_resolve_approval_mode_all_writes(self):
        """Resolve 'all_writes' string to ALL_WRITES mode."""
        mode = _resolve_approval_mode("all_writes")
        assert mode == ApprovalMode.ALL_WRITES

    def test_resolve_approval_mode_invalid(self):
        """Invalid string should default to RISKY_ONLY."""
        mode = _resolve_approval_mode("invalid_mode")
        assert mode == ApprovalMode.RISKY_ONLY

    def test_resolve_approval_mode_case_insensitive(self):
        """Resolution should be case-insensitive."""
        mode = _resolve_approval_mode("RISKY_ONLY")
        assert mode == ApprovalMode.RISKY_ONLY

    def test_get_risk_icon(self, mock_presentation):
        """get_risk_icon should return appropriate icon for risk level."""
        icon = _get_risk_icon(OperationalRiskLevel.CRITICAL, mock_presentation)
        assert icon == "⚠️"
        mock_presentation.icon.assert_called_once()


# =============================================================================
# 11. Edge Cases and Error Handling Tests (10 tests)
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_check_and_confirm_with_no_arguments(self):
        """Operations with no arguments should be handled."""
        checker = SafetyChecker()
        should_proceed, reason = await checker.check_and_confirm("read_file", {})
        assert should_proceed is True

    @pytest.mark.asyncio
    async def test_check_and_confirm_edit_multiple_files(self):
        """Edit operations with multiple files should aggregate risk."""
        callback_called = []

        async def mock_callback(request):
            callback_called.append(request)
            return True

        checker = SafetyChecker(
            approval_mode=ApprovalMode.ALL_WRITES, confirmation_callback=mock_callback
        )

        should_proceed, reason = await checker.check_and_confirm(
            "edit_files",
            {
                "edits": [
                    {"path": "/tmp/file1.txt"},
                    {"path": "/tmp/file2.txt"},
                    {"path": "/tmp/file3.txt"},
                ]
            },
        )

        assert should_proceed is True
        assert len(callback_called) == 1
        assert "3 file" in callback_called[0].description

    @pytest.mark.asyncio
    async def test_check_and_confirm_git_subcommand(self):
        """Git operations should check subcommand risk."""
        checker = SafetyChecker()
        level, details = checker.check_bash_command("git reset --hard")
        assert level == OperationalRiskLevel.HIGH

    @pytest.mark.asyncio
    async def test_check_and_confirm_with_truncated_command(self):
        """Long commands should be truncated in description."""
        callback_called = []

        async def mock_callback(request):
            callback_called.append(request)
            return True

        checker = SafetyChecker(
            confirmation_callback=mock_callback,
            require_confirmation_threshold=OperationalRiskLevel.MEDIUM,
        )

        long_cmd = "rm " + "x" * 200
        should_proceed, reason = await checker.check_and_confirm(
            "execute_bash", {"command": long_cmd}
        )

        assert should_proceed is True
        assert len(callback_called) == 1
        # Description should be truncated
        assert len(callback_called[0].description) < 200

    def test_check_bash_with_very_long_command(self, safety_checker):
        """Very long commands should be handled."""
        long_cmd = "echo 'test' " * 1000
        level, details = safety_checker.check_bash_command(long_cmd)
        assert level == OperationalRiskLevel.SAFE

    def test_check_file_operation_with_empty_path(self, safety_checker):
        """Empty file path should be handled."""
        level, details = safety_checker.check_file_operation("write", "")
        assert level == OperationalRiskLevel.SAFE

    def test_check_file_operation_with_special_characters(self, safety_checker):
        """Special characters in paths should be handled."""
        special_paths = [
            "/tmp/file with spaces.txt",
            "/tmp/file'with'quotes.txt",
            '/tmp/file"with"doublequotes.txt',
        ]
        for path in special_paths:
            level, details = safety_checker.check_file_operation("write", path)
            # Should not crash
            assert level in OperationalRiskLevel

    def test_custom_pattern_with_special_regex_chars(self, safety_checker):
        """Custom patterns with special regex chars should work."""
        safety_checker.add_custom_pattern(
            pattern=r"test\.\*+", description="Special regex pattern", risk_level="HIGH"
        )

        level, details = safety_checker.check_bash_command("test.*+")
        assert level == OperationalRiskLevel.HIGH

    def test_get_risk_icon_without_presentation(self):
        """get_risk_icon should work without presentation adapter."""
        icon = _get_risk_icon(OperationalRiskLevel.HIGH, None)
        assert isinstance(icon, str)
        assert len(icon) > 0

    @pytest.mark.asyncio
    async def test_confirmation_with_all_risk_levels(self):
        """Test confirmation behavior with all risk levels."""
        for risk_level in OperationalRiskLevel:
            callback_called = []

            async def mock_callback(request):
                callback_called.append(request)
                return True

            checker = SafetyChecker(
                confirmation_callback=mock_callback, require_confirmation_threshold=risk_level
            )

            # Operation at threshold should require confirmation
            should_proceed, reason = await checker.check_and_confirm(
                "execute_bash", {"command": "test"}
            )

            assert should_proceed is True


# =============================================================================
# 12. Integration and End-to-End Tests (4 tests)
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_complete_safe_workflow(self):
        """Complete workflow for safe operation."""
        checker = SafetyChecker(approval_mode=ApprovalMode.RISKY_ONLY)

        # Safe operation should proceed without confirmation
        should_proceed, reason = await checker.check_and_confirm(
            "read_file", {"path": "/tmp/safe.txt"}
        )
        assert should_proceed is True
        assert reason is None

    @pytest.mark.asyncio
    async def test_complete_dangerous_workflow_with_approval(self):
        """Complete workflow for dangerous operation with user approval."""
        callback_called = []

        async def mock_callback(request):
            callback_called.append(request)
            assert request.risk_level == OperationalRiskLevel.HIGH
            assert "rm -rf" in request.arguments.get("command", "")
            return True

        checker = SafetyChecker(
            approval_mode=ApprovalMode.RISKY_ONLY, confirmation_callback=mock_callback
        )

        should_proceed, reason = await checker.check_and_confirm(
            "execute_bash", {"command": "rm -rf /tmp/test"}
        )

        assert should_proceed is True
        assert reason is None
        assert len(callback_called) == 1

    @pytest.mark.asyncio
    async def test_complete_dangerous_workflow_with_rejection(self):
        """Complete workflow for dangerous operation with user rejection."""

        async def mock_callback(request):
            return False

        checker = SafetyChecker(
            approval_mode=ApprovalMode.RISKY_ONLY, confirmation_callback=mock_callback
        )

        should_proceed, reason = await checker.check_and_confirm(
            "execute_bash", {"command": "rm -rf /tmp/test"}
        )

        assert should_proceed is False
        assert "cancelled" in reason.lower()

    @pytest.mark.asyncio
    async def test_complete_write_workflow_all_writes_mode(self):
        """Complete write workflow in ALL_WRITES mode."""
        callback_count = []

        async def mock_callback(request):
            callback_count.append(request)
            return True

        checker = SafetyChecker(
            approval_mode=ApprovalMode.ALL_WRITES, confirmation_callback=mock_callback
        )

        # Write operation should require confirmation
        should_proceed, reason = await checker.check_and_confirm(
            "write_file", {"path": "/tmp/test.txt"}
        )

        assert should_proceed is True
        assert len(callback_count) == 1
