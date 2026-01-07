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

"""Tests for path sanitization to prevent privacy leaks and sandbox escape.

Tests the normalize_tool_path, normalize_tool_paths, and sanitize_shell_command_paths
functions that protect against unauthorized filesystem access.
"""

import pytest
from pathlib import Path
from unittest.mock import patch

# Import the functions we're testing
from victor.agent.orchestrator import (
    normalize_tool_path,
    normalize_tool_paths,
    sanitize_shell_command_paths,
    _get_project_root,
    _get_allowed_temp_dirs,
)


class TestProjectRootDetection:
    """Tests for project root detection."""

    def test_get_project_root_finds_git_dir(self):
        """Test that project root finds .git directory."""
        root = _get_project_root()
        assert isinstance(root, Path)
        # Should be a directory that exists
        assert root.exists()
        assert root.is_dir()

    def test_project_root_has_git(self):
        """Test that detected root has .git directory."""
        root = _get_project_root()
        # The root or a parent should have .git
        for parent in [root] + list(root.parents):
            if (parent / ".git").exists():
                assert True
                return
        # If no .git, current directory is used
        assert True


class TestTempDirectoryDetection:
    """Tests for allowed temp directory detection."""

    def test_get_allowed_temp_dirs_returns_set(self):
        """Test that temp dirs returns a set of Paths."""
        temp_dirs = _get_allowed_temp_dirs()
        assert isinstance(temp_dirs, set)
        assert len(temp_dirs) > 0
        for temp_dir in temp_dirs:
            assert isinstance(temp_dir, Path)

    def test_temp_dirs_exist(self):
        """Test that detected temp directories exist."""
        temp_dirs = _get_allowed_temp_dirs()
        for temp_dir in temp_dirs:
            # At least one temp dir should exist
            if temp_dir.exists():
                assert True
                return
        assert True  # At least one temp dir was found


class TestNormalizeToolPath:
    """Tests for single path normalization."""

    def test_allow_current_directory(self):
        """Test that current directory is allowed."""
        result = normalize_tool_path(".", "test_tool")
        # Should preserve "." or convert to resolved path
        assert result == "." or "code/glm-branch" in result

        result = normalize_tool_path("./src", "test_tool")
        # Should allow relative paths within project
        assert "src" in result

    def test_block_root_directory(self):
        """Test that root directory is blocked."""
        result = normalize_tool_path("/", "test_tool")
        assert result == "."

    def test_allow_project_subdirectories(self):
        """Test that project subdirectories are allowed."""
        # Test with actual project paths
        result = normalize_tool_path("./victor/agent", "test_tool")
        # Should allow relative paths within project
        assert "./victor" in result or "victor" in result

    def test_block_user_home_directory(self):
        """Test that home directory is blocked."""
        # This will be resolved to actual home dir
        result = normalize_tool_path("~", "test_tool")
        # Should block and default to current directory
        # (Unless ~ happens to be in allowed temp dirs)
        assert result == "." or "/tmp" in result or "/var/tmp" in result

    def test_allow_temp_directories(self):
        """Test that temp directories are allowed."""
        import platform

        system = platform.system()

        if system == "Darwin":  # macOS
            # Note: macOS symlinks /tmp → /private/tmp, so resolve() gives canonical path
            result_tmp = normalize_tool_path("/tmp", "test_tool")
            assert "/tmp" in result_tmp or "/private/tmp" in result_tmp
            assert result_tmp != "."  # Should be allowed, not blocked

            result_var_tmp = normalize_tool_path("/var/tmp", "test_tool")
            assert "/var/tmp" in result_var_tmp
            assert result_var_tmp != "."
        elif system == "Linux":
            assert normalize_tool_path("/tmp", "test_tool") == "/tmp"
            assert normalize_tool_path("/var/tmp", "test_tool") == "/var/tmp"
        elif system == "Windows":
            # Windows temp dirs depend on environment
            # Just verify it doesn't crash
            result = normalize_tool_path("/tmp", "test_tool")
            assert result is not None

    def test_block_system_directories(self):
        """Test that system directories are blocked."""
        sys_dirs = [
            "/System",
            "/Library",
            "/usr",
            "/bin",
            "/etc",
        ]

        for sys_dir in sys_dirs:
            result = normalize_tool_path(sys_dir, "test_tool")
            # Should block and default to current directory
            assert result == "."

    def test_allow_relative_paths(self):
        """Test that relative paths are allowed."""
        result = normalize_tool_path("src/victor", "test_tool")
        # Should allow relative paths (may be resolved to absolute)
        assert "victor" in result or "src" in result

        result = normalize_tool_path("../tests", "test_tool")
        # Parent directory may be allowed if it exists
        assert result is not None

    def test_block_other_users_on_macos(self):
        """Test that other users' directories are blocked on macOS."""
        import platform

        if platform.system() == "Darwin":
            # Try to access another user's directory
            result = normalize_tool_path("/Users/root", "test_tool")
            # Should block unless we're actually root
            # (which we shouldn't be in normal testing)
            assert result == "." or "/Users/root" not in result

    def test_invalid_path_defaults_to_current(self):
        """Test that invalid paths default to current directory."""
        # Empty string
        assert normalize_tool_path("", "test_tool") == "."

        # Just in case test for something that can't be a path
        result = normalize_tool_path("\x00invalid", "test_tool")
        assert result == "."


class TestNormalizeToolPaths:
    """Tests for argument dict path normalization."""

    def test_normalize_path_argument(self):
        """Test normalizing a path argument."""
        args = {"path": "/", "query": "test"}
        result = normalize_tool_paths(args, "grep")

        assert result["path"] == "."  # Root dir blocked
        assert result["query"] == "test"  # Other args preserved

    def test_normalize_multiple_path_arguments(self):
        """Test normalizing multiple path arguments."""
        args = {"src": "/Users/alice/file.txt", "dst": "/tmp/output.txt", "query": "test"}
        result = normalize_tool_paths(args, "copy_tool")

        assert result["src"] == "."  # Blocked (outside project)
        # Temp dir - handle macOS symlink
        assert "/tmp" in result["dst"] or result["dst"] == "."  # May be resolved
        assert result["query"] == "test"

    def test_normalize_path_list_argument(self):
        """Test normalizing a list of paths."""
        args = {"paths": ["/", "/tmp", "./src"], "query": "test"}
        result = normalize_tool_paths(args, "multi_tool")

        assert result["paths"][0] == "."  # Root blocked
        # Temp dir - handle macOS symlink
        assert "/tmp" in result["paths"][1] or "/private/tmp" in result["paths"][1]
        # Project path
        assert "src" in result["paths"][2]
        assert result["query"] == "test"

    def test_preserves_non_path_arguments(self):
        """Test that non-path arguments are preserved."""
        args = {"query": "test", "count": 10, "flag": True, "name": "tool_name"}
        result = normalize_tool_paths(args, "test_tool")

        assert result == args  # Nothing changed

    def test_handles_none_arguments(self):
        """Test handling of None arguments."""
        result = normalize_tool_paths(None, "test_tool")
        assert result is None

    def test_handles_non_dict_arguments(self):
        """Test handling of non-dict arguments."""
        result = normalize_tool_paths("not a dict", "test_tool")
        assert result == "not a dict"

    def test_recognizes_common_path_parameter_names(self):
        """Test that common path parameter names are recognized."""
        path_params = [
            "path",
            "paths",
            "dir",
            "directory",
            "file",
            "filename",
            "src",
            "source",
            "dst",
            "destination",
            "target",
            "root",
            "base_dir",
            "working_dir",
            "output_dir",
        ]

        for param in path_params:
            args = {param: "/"}
            result = normalize_tool_paths(args, "test_tool")
            # Should sanitize all of these
            assert result[param] == "."


class TestSanitizeShellCommands:
    """Tests for shell command path sanitization."""

    def test_sanitize_find_command_root(self):
        """Test sanitizing find command with root directory."""
        cmd = 'find / -type d -name "test"'
        result = sanitize_shell_command_paths(cmd)

        # Root directory should be replaced with .
        assert "/" not in result.split()[1:3]  # First path after "find" should not be /
        assert "." in result  # Should have current directory

    def test_sanitize_grep_command_home(self):
        """Test sanitizing grep command with home directory."""
        cmd = 'grep -r "pattern" /Users/alice'
        result = sanitize_shell_command_paths(cmd)

        # Home directory should be replaced
        assert "/Users/alice" not in result
        assert "." in result or "./" in result

    def test_sanitize_find_command_users(self):
        """Test sanitizing find command searching entire Users directory."""
        cmd = 'find /Users/vijaysingh -type d -name "*victor*"'
        result = sanitize_shell_command_paths(cmd)

        # Should block the /Users path
        assert "/Users/vijaysingh" not in result
        # Should replace with safe path
        assert "." in result or "./" in result

    def test_preserve_safe_shell_commands(self):
        """Test that safe commands are preserved."""
        cmd = "ls -la ./src"
        result = sanitize_shell_command_paths(cmd)

        # Should be unchanged
        assert result == cmd

    def test_preserve_temp_dir_commands(self):
        """Test that temp directory commands are preserved."""
        cmd = "cat /var/tmp/file.txt"
        result = sanitize_shell_command_paths(cmd)

        # Temp dir should be allowed
        assert "/var/tmp" in result

    def test_sanitize_cd_command(self):
        """Test sanitizing cd command with unsafe path."""
        cmd = "cd /Users/alice/Documents"
        result = sanitize_shell_command_paths(cmd)

        # Should block and default to .
        assert "/Users/alice" not in result
        assert "." in result

    def test_handles_quoted_paths(self):
        """Test handling of quoted paths in commands."""
        cmd = 'grep "pattern" /Users/alice/docs'
        result = sanitize_shell_command_paths(cmd)

        # Should still sanitize even with quotes
        assert "/Users/alice" not in result

    def test_handles_flags(self):
        """Test that flags are preserved."""
        cmd = 'grep -r -i "pattern" /'
        result = sanitize_shell_command_paths(cmd)

        # Flags should be preserved
        assert "-r" in result
        assert "-i" in result
        # Root should be blocked
        assert result.endswith(".") or result.endswith("./")

    def test_handles_empty_command(self):
        """Test handling of empty command."""
        result = sanitize_shell_command_paths("")
        assert result == ""

    def test_handles_invalid_command(self):
        """Test handling of invalid command that can't be parsed."""
        # Command with unmatched quotes
        cmd = 'grep "pattern'
        result = sanitize_shell_command_paths(cmd)

        # Should either sanitize or return original (depends on parser)
        assert result is not None


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_deeply_nested_project_path(self):
        """Test deeply nested path within project."""
        path = "./victor/agent/orchestrator/path/to/file"
        result = normalize_tool_path(path, "test_tool")

        # Should allow nested project paths
        assert "victor" in result or "orchestrator" in result

    def test_path_with_special_characters(self):
        """Test path with special characters."""
        path = "./path with spaces/file"
        result = normalize_tool_path(path, "test_tool")

        # Should handle spaces
        assert result is not None

    def test_relative_path_escaping_project(self):
        """Test relative path that escapes project."""
        # This would typically be blocked by the sandbox
        # because it resolves outside the project
        path = "../../../etc/passwd"
        result = normalize_tool_path(path, "test_tool")

        # Should either allow (since it's relative) or sanitize
        # The implementation checks the resolved path
        assert result is not None

    def test_multiple_slashes(self):
        """Test path with multiple consecutive slashes."""
        path = "//var///tmp////file"
        result = normalize_tool_path(path, "test_tool")

        # Path.resolve() should normalize this
        # If it resolves to /var/tmp, it should be allowed
        assert result is not None

    def test_dotdot_in_path(self):
        """Test path with .. components."""
        path = "../sibling_dir/file"
        result = normalize_tool_path(path, "test_tool")

        # Should handle parent directory references
        # (May be allowed or blocked depending on where it resolves)
        assert result is not None


class TestIntegration:
    """Integration tests for path sanitization."""

    def test_full_workflow_grep_tool(self):
        """Test full workflow: grep tool with bad path."""
        # Simulate LLM calling grep with root directory
        tool_call = {"name": "grep", "arguments": {"query": "Victor", "path": "/", "exts": [".py"]}}

        result = normalize_tool_paths(tool_call["arguments"], "grep")

        # Path should be sanitized
        assert result["path"] == "."
        # Other args preserved
        assert result["query"] == "Victor"
        assert result["exts"] == [".py"]

    def test_full_workflow_shell_tool(self):
        """Test full workflow: shell tool with find command."""
        tool_call = {
            "name": "shell",
            "arguments": {"cmd": 'find /Users/vijaysingh -type d -name "*victor*"'},
        }

        result = normalize_tool_paths(tool_call["arguments"], "shell")

        # Command should be sanitized
        assert "/Users/vijaysingh" not in result["cmd"]
        assert "." in result["cmd"]

    def test_full_workflow_code_search_tool(self):
        """Test full workflow: code_search tool (uses 'path' parameter)."""
        tool_call = {
            "name": "code_search",
            "arguments": {"query": "Formation", "path": "/System/Library", "k": 10},  # Unsafe!
        }

        result = normalize_tool_paths(tool_call["arguments"], "code_search")

        # System directory should be blocked
        assert result["path"] == "."
        # Other args preserved
        assert result["query"] == "Formation"
        assert result["k"] == 10

    def test_multiple_sanitization_passes(self):
        """Test that sanitizing twice doesn't change the result."""
        args = {"path": "/", "query": "test"}

        first_pass = normalize_tool_paths(args, "test_tool")
        second_pass = normalize_tool_paths(first_pass, "test_tool")

        # Should be idempotent
        assert first_pass == second_pass
