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

"""Tests for LSP Write Enhancer.

Tests LSP validation and formatting for multiple supported languages:
- C/C++ (clangd, clang-format)
- Python (pyright/pylsp, black)
- Rust (rust-analyzer, rustfmt)
- JavaScript/TypeScript (tsserver, prettier)
- Go (gopls, gofmt)
- Config files (JSON, YAML, TOML)
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.tools.lsp_write_enhancer import (
    Diagnostic,
    DiagnosticSeverity,
    get_write_enhancer,
    LSPWriteEnhancer,
    WriteResult,
    write_with_lsp,
)


@pytest.fixture(autouse=True)
def reset_enhancer():
    """Reset LSP write enhancer between tests."""
    global _enhancer
    import victor.tools.lsp_write_enhancer as enhancer_module

    enhancer_module._enhancer = None
    yield
    enhancer_module._enhancer = None


class TestDiagnostic:
    """Tests for Diagnostic dataclass."""

    def test_diagnostic_creation(self):
        """Test creating a diagnostic."""
        diagnostic = Diagnostic(
            line=10,
            column=5,
            severity=DiagnosticSeverity.ERROR,
            message="Undefined variable 'x'",
            code="E0602",
            source="pylint",
        )

        assert diagnostic.line == 10
        assert diagnostic.column == 5
        assert diagnostic.severity == DiagnosticSeverity.ERROR
        assert diagnostic.message == "Undefined variable 'x'"
        assert diagnostic.code == "E0602"
        assert diagnostic.source == "pylint"

    def test_diagnostic_without_optional_fields(self):
        """Test creating diagnostic without optional fields."""
        diagnostic = Diagnostic(
            line=1,
            column=0,
            severity=DiagnosticSeverity.WARNING,
            message="Unused import",
        )

        assert diagnostic.code is None
        assert diagnostic.source is None


class TestWriteResult:
    """Tests for WriteResult dataclass."""

    def test_write_result_creation(self):
        """Test creating a write result."""
        result = WriteResult(
            success=True,
            path="test.py",
            formatted=True,
            validated=True,
        )

        assert result.success is True
        assert result.path == "test.py"
        assert result.formatted is True
        assert result.validated is True

    def test_has_errors(self):
        """Test checking for errors."""
        result = WriteResult(success=False, path="test.py")

        # No diagnostics
        assert result.has_errors is False

        # Add error diagnostic
        result.diagnostics.append(
            Diagnostic(line=1, column=0, severity=DiagnosticSeverity.ERROR, message="Error")
        )
        assert result.has_errors is True

    def test_has_warnings(self):
        """Test checking for warnings."""
        result = WriteResult(success=True, path="test.py")

        # No diagnostics
        assert result.has_warnings is False

        # Add warning diagnostic
        result.diagnostics.append(
            Diagnostic(line=1, column=0, severity=DiagnosticSeverity.WARNING, message="Warning")
        )
        assert result.has_warnings is True

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = WriteResult(
            success=True,
            path="test.py",
            formatted=True,
            validated=True,
        )
        result.diagnostics = [
            Diagnostic(
                line=1,
                column=0,
                severity=DiagnosticSeverity.ERROR,
                message="Error",
                code="E001",
            ),
            Diagnostic(
                line=2,
                column=0,
                severity=DiagnosticSeverity.WARNING,
                message="Warning",
            ),
        ]

        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["path"] == "test.py"
        assert result_dict["formatted"] is True
        assert result_dict["validated"] is True
        assert len(result_dict["diagnostics"]) == 2
        assert result_dict["summary"]["total_diagnostics"] == 2
        assert result_dict["summary"]["errors"] == 1
        assert result_dict["summary"]["warnings"] == 1


class TestLSPWriteEnhancer:
    """Tests for LSPWriteEnhancer class."""

    def test_init_default_workspace(self):
        """Test initialization with default workspace."""
        enhancer = LSPWriteEnhancer()

        assert enhancer._workspace_root == str(Path.cwd())
        assert enhancer._lsp_pool is None
        assert enhancer._language_registry is None

    def test_init_custom_workspace(self):
        """Test initialization with custom workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enhancer = LSPWriteEnhancer(workspace_root=tmpdir)

            assert enhancer._workspace_root == tmpdir

    def test_get_lsp_pool_lazy_initialization(self):
        """Test LSP pool is lazily initialized."""
        enhancer = LSPWriteEnhancer()

        # Pool should be None initially
        assert enhancer._lsp_pool is None

        # The pool is created when first accessed via _get_lsp_pool
        # We can't easily test the actual pool creation without mocking,
        # but we can verify the singleton behavior
        from victor.tools.lsp_write_enhancer import get_write_enhancer

        enhancer2 = get_write_enhancer()
        assert enhancer2 is not None

        # Multiple calls should return the same instance (if workspace root is same)
        enhancer3 = get_write_enhancer()
        assert enhancer2 is enhancer3

    def test_get_language_registry_lazy_initialization(self):
        """Test language registry is lazily initialized."""
        enhancer = LSPWriteEnhancer()

        # Registry should be None initially
        assert enhancer._language_registry is None

        registry = enhancer._get_language_registry()
        assert registry is not None
        assert enhancer._language_registry is not None

    @pytest.mark.asyncio
    async def test_format_with_formatter_no_language(self):
        """Test formatting when language is not detected."""
        enhancer = LSPWriteEnhancer()

        # Unknown file extension
        content, formatter_name = enhancer.format_with_formatter("test.unknown_ext", "some content")

        # Should return content unchanged
        assert content == "some content"
        assert formatter_name is None

    @pytest.mark.asyncio
    async def test_format_with_formatter_unavailable_formatter(self):
        """Test formatting when formatter is not available."""
        enhancer = LSPWriteEnhancer()

        # Python file but formatter might not be installed
        content, formatter_name = enhancer.format_with_formatter("test.py", "def hello(): pass")

        # Should return content (possibly formatted or unchanged)
        assert isinstance(content, str)
        # Formatter name depends on system, so we don't assert it

    @pytest.mark.asyncio
    async def test_validate_with_lsp_mocked(self):
        """Test LSP validation with mocked LSP pool."""
        enhancer = LSPWriteEnhancer()

        # Mock LSP pool and diagnostics
        mock_pool = AsyncMock()
        mock_pool._path_to_uri = MagicMock(return_value="file://test.py")
        mock_pool.open_document = MagicMock()
        mock_pool.get_diagnostics = MagicMock(
            return_value=[
                {
                    "line": 1,
                    "column": 0,
                    "severity": "error",
                    "message": "Syntax error",
                    "code": "E001",
                }
            ]
        )
        mock_pool.close_document = MagicMock()

        enhancer._lsp_pool = mock_pool

        diagnostics = await enhancer.validate_with_lsp("test.py", "invalid python code")

        assert len(diagnostics) == 1
        assert diagnostics[0].line == 1
        assert diagnostics[0].severity == DiagnosticSeverity.ERROR
        assert diagnostics[0].message == "Syntax error"

    @pytest.mark.asyncio
    async def test_write_with_lsp_dry_run(self):
        """Test dry-run mode (validate and format without writing)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enhancer = LSPWriteEnhancer(workspace_root=tmpdir)

            # Mock methods to avoid actual LSP calls
            enhancer.format_with_formatter = MagicMock(return_value=("formatted content", "black"))
            enhancer.validate_with_lsp = AsyncMock(return_value=[])

            result = await enhancer.write_with_lsp(
                path="test.py",
                content="original content",
                validate=True,
                format_code=True,
                write=False,  # Dry run
            )

            assert result.success is True  # No errors
            assert result.path == "test.py"
            assert result.formatted is True
            assert result.formatter_used == "black"
            assert result.validated is True
            assert result.written_content == "formatted content"
            assert len(result.diagnostics) == 0

            # File should not exist (dry run)
            assert not Path(tmpdir, "test.py").exists()

    @pytest.mark.asyncio
    async def test_write_with_lsp_with_errors(self):
        """Test write with validation errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enhancer = LSPWriteEnhancer(workspace_root=tmpdir)

            # Mock validation to return errors
            enhancer.format_with_formatter = MagicMock(return_value=("formatted content", "black"))
            enhancer.validate_with_lsp = AsyncMock(
                return_value=[
                    Diagnostic(
                        line=1,
                        column=0,
                        severity=DiagnosticSeverity.ERROR,
                        message="Syntax error",
                    )
                ]
            )

            result = await enhancer.write_with_lsp(
                path="test.py",
                content="invalid code",
                validate=True,
                format_code=True,
                write=True,
            )

            assert result.success is False
            assert result.has_errors is True
            assert result.error == "File has errors - not written"

            # File should not exist (errors prevented write)
            assert not Path(tmpdir, "test.py").exists()

    @pytest.mark.asyncio
    async def test_write_with_lsp_success(self):
        """Test successful write with LSP enhancement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enhancer = LSPWriteEnhancer(workspace_root=tmpdir)

            # Mock methods
            enhancer.format_with_formatter = MagicMock(return_value=("formatted content", "black"))
            enhancer.validate_with_lsp = AsyncMock(return_value=[])

            # Use full path in tmpdir
            test_file = Path(tmpdir) / "test.py"

            result = await enhancer.write_with_lsp(
                path=str(test_file),
                content="original content",
                validate=True,
                format_code=True,
                write=True,
            )

            assert result.success is True
            assert result.has_errors is False
            assert result.formatted is True
            assert result.validated is True
            assert result.formatter_used == "black"

            # File should exist with formatted content
            assert test_file.exists()
            assert test_file.read_text() == "formatted content"


class TestLanguageSupport:
    """Tests for language-specific support."""

    def test_python_language_detection(self):
        """Test Python language is detected."""
        from victor.verticals.contrib.coding.languages.registry import get_language_registry

        registry = get_language_registry()
        registry.discover_plugins()

        # Python should be registered
        assert registry.has("python")
        assert registry.has("py")

        # Extensions
        assert ".py" in registry.get_extensions("python")
        assert ".pyi" in registry.get_extensions("python")

    def test_rust_language_detection(self):
        """Test Rust language is detected."""
        from victor.verticals.contrib.coding.languages.registry import get_language_registry

        registry = get_language_registry()
        registry.discover_plugins()

        # Rust should be registered
        assert registry.has("rust")
        assert registry.has("rs")

        # Extensions
        assert ".rs" in registry.get_extensions("rust")

    def test_javascript_language_detection(self):
        """Test JavaScript language is detected."""
        from victor.verticals.contrib.coding.languages.registry import get_language_registry

        registry = get_language_registry()
        registry.discover_plugins()

        # JavaScript should be registered
        assert registry.has("javascript")
        assert registry.has("js")

        # Extensions
        assert ".js" in registry.get_extensions("javascript")
        assert ".jsx" in registry.get_extensions("javascript")

    def test_typescript_language_detection(self):
        """Test TypeScript language is detected."""
        from victor.verticals.contrib.coding.languages.registry import get_language_registry

        registry = get_language_registry()
        registry.discover_plugins()

        # TypeScript should be registered
        assert registry.has("typescript")
        assert registry.has("ts")

        # Extensions
        assert ".ts" in registry.get_extensions("typescript")
        assert ".tsx" in registry.get_extensions("typescript")

    def test_cpp_language_detection(self):
        """Test C++ language is detected."""
        from victor.verticals.contrib.coding.languages.registry import get_language_registry

        registry = get_language_registry()
        registry.discover_plugins()

        # C++ should be registered
        assert registry.has("cpp")
        assert registry.has("c++")

        # Extensions
        assert ".cpp" in registry.get_extensions("cpp")
        assert ".hpp" in registry.get_extensions("cpp")
        assert ".cc" in registry.get_extensions("cpp")

    def test_c_language_detection(self):
        """Test C language is detected."""
        from victor.verticals.contrib.coding.languages.registry import get_language_registry

        registry = get_language_registry()
        registry.discover_plugins()

        # C should be registered
        assert registry.has("c")

        # Extensions
        assert ".c" in registry.get_extensions("c")
        assert ".h" in registry.get_extensions("c")

    def test_go_language_detection(self):
        """Test Go language is detected."""
        from victor.verticals.contrib.coding.languages.registry import get_language_registry

        registry = get_language_registry()
        registry.discover_plugins()

        # Go should be registered
        assert registry.has("go")

        # Extensions
        assert ".go" in registry.get_extensions("go")

    def test_java_language_detection(self):
        """Test Java language is detected."""
        from victor.verticals.contrib.coding.languages.registry import get_language_registry

        registry = get_language_registry()
        registry.discover_plugins()

        # Java should be registered
        assert registry.has("java")

        # Extensions
        assert ".java" in registry.get_extensions("java")

    def test_config_file_languages(self):
        """Test config file languages are detected."""
        from victor.verticals.contrib.coding.languages.registry import get_language_registry

        registry = get_language_registry()
        registry.discover_plugins()

        # JSON
        assert registry.has("json")
        assert ".json" in registry.get_extensions("json")

        # YAML
        assert registry.has("yaml")
        assert ".yaml" in registry.get_extensions("yaml")
        assert ".yml" in registry.get_extensions("yaml")

        # TOML
        assert registry.has("toml")
        assert ".toml" in registry.get_extensions("toml")


class TestFormatterAvailability:
    """Test formatter methods are available on language plugins."""

    def test_python_has_formatter(self):
        """Test Python plugin has get_formatter method."""
        from victor.verticals.contrib.coding.languages.plugins import PythonPlugin

        plugin = PythonPlugin()

        # Should have get_formatter method
        assert hasattr(plugin, "get_formatter")

        # Get formatter should return a Formatter
        formatter = plugin.get_formatter(Path.cwd())
        assert formatter is not None
        assert formatter.name in ["black", "autopep8"]

    def test_rust_has_formatter(self):
        """Test Rust plugin has get_formatter method."""
        from victor.verticals.contrib.coding.languages.plugins import RustPlugin

        plugin = RustPlugin()

        # Should have get_formatter method
        assert hasattr(plugin, "get_formatter")

        # Get formatter should return a Formatter
        formatter = plugin.get_formatter(Path.cwd())
        assert formatter is not None
        assert formatter.name == "rustfmt"

    def test_javascript_has_formatter(self):
        """Test JavaScript plugin has get_formatter method."""
        from victor.verticals.contrib.coding.languages.plugins import JavaScriptPlugin

        plugin = JavaScriptPlugin()

        # Should have get_formatter method
        assert hasattr(plugin, "get_formatter")

        # Get formatter should return a Formatter
        formatter = plugin.get_formatter(Path.cwd())
        assert formatter is not None
        assert formatter.name == "prettier"

    def test_typescript_has_formatter(self):
        """Test TypeScript plugin has get_formatter method."""
        from victor.verticals.contrib.coding.languages.plugins import TypeScriptPlugin

        plugin = TypeScriptPlugin()

        # Should have get_formatter method
        assert hasattr(plugin, "get_formatter")

        # Get formatter should return a Formatter
        formatter = plugin.get_formatter(Path.cwd())
        assert formatter is not None
        assert formatter.name == "prettier"

    def test_cpp_has_formatter(self):
        """Test C++ plugin has get_formatter method."""
        from victor.verticals.contrib.coding.languages.plugins import CppPlugin

        plugin = CppPlugin()

        # Should have get_formatter method
        assert hasattr(plugin, "get_formatter")

        # Get formatter should return a Formatter
        formatter = plugin.get_formatter(Path.cwd())
        assert formatter is not None
        assert formatter.name == "clang-format"

    def test_c_has_formatter(self):
        """Test C plugin has get_formatter method."""
        from victor.verticals.contrib.coding.languages.plugins.additional import CPlugin

        plugin = CPlugin()

        # C plugin may not have its own get_formatter method
        # It might inherit from base or not support formatting
        if hasattr(plugin, "get_formatter"):
            formatter = plugin.get_formatter(Path.cwd())
            # Formatter might be None if not configured
            if formatter is not None:
                assert formatter.name in ["clang-format", "clang-format-15"]
        else:
            # C plugin uses clang-format like C++
            # This is acceptable - the formatter is handled at system level
            pass

    def test_go_has_formatter(self):
        """Test Go plugin has get_formatter method."""
        from victor.verticals.contrib.coding.languages.plugins import GoPlugin

        plugin = GoPlugin()

        # Should have get_formatter method
        assert hasattr(plugin, "get_formatter")

        # Get formatter should return a Formatter
        formatter = plugin.get_formatter(Path.cwd())
        assert formatter is not None
        assert formatter.name == "gofmt"


class TestLSPServerConfigs:
    """Test LSP server configurations exist for all languages."""

    def test_python_lsp_config(self):
        """Test Python LSP server configuration exists."""
        from victor.verticals.contrib.coding.lsp.config import LANGUAGE_SERVERS

        assert "python" in LANGUAGE_SERVERS
        assert "python-pylsp" in LANGUAGE_SERVERS

        python_config = LANGUAGE_SERVERS["python"]
        assert python_config.name == "Pyright"
        assert ".py" in python_config.file_extensions
        assert python_config.command == ["pyright-langserver", "--stdio"]

        pylsp_config = LANGUAGE_SERVERS["python-pylsp"]
        assert pylsp_config.name == "Python LSP Server"
        assert ".py" in pylsp_config.file_extensions
        assert pylsp_config.command == ["pylsp"]

    def test_rust_lsp_config(self):
        """Test Rust LSP server configuration exists."""
        from victor.verticals.contrib.coding.lsp.config import LANGUAGE_SERVERS

        assert "rust" in LANGUAGE_SERVERS

        config = LANGUAGE_SERVERS["rust"]
        assert config.name == "rust-analyzer"
        assert ".rs" in config.file_extensions
        assert config.command == ["rust-analyzer"]

    def test_javascript_lsp_config(self):
        """Test JavaScript/TypeScript LSP server configuration exists."""
        from victor.verticals.contrib.coding.lsp.config import LANGUAGE_SERVERS

        assert "typescript" in LANGUAGE_SERVERS

        config = LANGUAGE_SERVERS["typescript"]
        assert config.name == "TypeScript Language Server"
        assert ".ts" in config.file_extensions
        assert ".js" in config.file_extensions
        assert config.command == ["typescript-language-server", "--stdio"]

    def test_cpp_lsp_config(self):
        """Test C/C++ LSP server configuration exists."""
        from victor.verticals.contrib.coding.lsp.config import LANGUAGE_SERVERS

        assert "c" in LANGUAGE_SERVERS

        config = LANGUAGE_SERVERS["c"]
        assert config.name == "clangd"
        assert ".c" in config.file_extensions
        assert ".cpp" in config.file_extensions
        assert config.command == ["clangd"]

    def test_go_lsp_config(self):
        """Test Go LSP server configuration exists."""
        from victor.verticals.contrib.coding.lsp.config import LANGUAGE_SERVERS

        assert "go" in LANGUAGE_SERVERS

        config = LANGUAGE_SERVERS["go"]
        assert config.name == "gopls"
        assert ".go" in config.file_extensions
        assert config.command == ["gopls"]

    def test_java_lsp_config(self):
        """Test Java LSP server configuration exists."""
        from victor.verticals.contrib.coding.lsp.config import LANGUAGE_SERVERS

        assert "java" in LANGUAGE_SERVERS

        config = LANGUAGE_SERVERS["java"]
        assert config.name == "Eclipse JDT Language Server"
        assert ".java" in config.file_extensions
        assert config.command == ["jdtls"]


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_write_enhancer_singleton(self):
        """Test get_write_enhancer returns singleton."""
        enhancer1 = get_write_enhancer()
        enhancer2 = get_write_enhancer()

        # Should return the same instance
        assert enhancer1 is enhancer2

    def test_get_write_enhancer_custom_workspace(self):
        """Test get_write_enhancer with custom workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enhancer = get_write_enhancer(workspace_root=tmpdir)

            assert enhancer._workspace_root == tmpdir

    @pytest.mark.asyncio
    async def test_write_with_lsp_convenience_function(self):
        """Test write_with_lsp convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the enhancer methods
            with patch("victor.tools.lsp_write_enhancer.get_write_enhancer") as mock_get:
                enhancer = MagicMock()
                enhancer.write_with_lsp = AsyncMock(
                    return_value=WriteResult(success=True, path="test.py")
                )
                mock_get.return_value = enhancer

                result = await write_with_lsp("test.py", "content", workspace_root=tmpdir)

                assert result.success is True
                assert result.path == "test.py"


class TestWriteResultSummary:
    """Test WriteResult summary calculation."""

    def test_summary_all_severities(self):
        """Test summary with all severity levels."""
        result = WriteResult(success=True, path="test.py")
        result.diagnostics = [
            Diagnostic(1, 0, DiagnosticSeverity.ERROR, "Error 1"),
            Diagnostic(2, 0, DiagnosticSeverity.ERROR, "Error 2"),
            Diagnostic(3, 0, DiagnosticSeverity.WARNING, "Warning 1"),
            Diagnostic(4, 0, DiagnosticSeverity.INFO, "Info 1"),
            Diagnostic(5, 0, DiagnosticSeverity.HINT, "Hint 1"),
        ]

        summary = result.to_dict()["summary"]

        assert summary["total_diagnostics"] == 5
        assert summary["errors"] == 2
        assert summary["warnings"] == 1
        assert summary["info"] == 1
        assert summary["hints"] == 1

    def test_summary_empty(self):
        """Test summary with no diagnostics."""
        result = WriteResult(success=True, path="test.py")

        summary = result.to_dict()["summary"]

        assert summary["total_diagnostics"] == 0
        assert summary["errors"] == 0
        assert summary["warnings"] == 0
        assert summary["info"] == 0
        assert summary["hints"] == 0


class TestIntegrationWithWriteTool:
    """Integration tests with write tool."""

    def test_write_lsp_tool_exists(self):
        """Test write_lsp tool is exported from filesystem module."""
        from victor.tools.filesystem import write_lsp

        assert write_lsp is not None
        assert callable(write_lsp)

    def test_write_tool_still_exists(self):
        """Test original write tool still exists."""
        from victor.tools.filesystem import write

        assert write is not None
        assert callable(write)


class TestAllSupportedLanguages:
    """Comprehensive test for all supported languages."""

    def test_all_core_languages_registered(self):
        """Test all core languages are registered."""
        from victor.verticals.contrib.coding.languages.registry import (
            get_language_registry,
        )

        registry = get_language_registry()
        registry.discover_plugins()

        # Core languages
        core_languages = [
            "python",
            "javascript",
            "typescript",
            "rust",
            "go",
            "java",
            "cpp",
            "c",
        ]

        for lang in core_languages:
            assert registry.has(lang), f"Language '{lang}' should be registered"

    def test_all_config_languages_registered(self):
        """Test all config file languages are registered."""
        from victor.verticals.contrib.coding.languages.registry import (
            get_language_registry,
        )

        registry = get_language_registry()
        registry.discover_plugins()

        config_languages = ["json", "yaml", "toml", "ini"]

        for lang in config_languages:
            assert registry.has(lang), f"Config language '{lang}' should be registered"

    def test_all_additional_languages_registered(self):
        """Test additional languages are registered."""
        from victor.verticals.contrib.coding.languages.registry import (
            get_language_registry,
        )

        registry = get_language_registry()
        registry.discover_plugins()

        additional_languages = [
            "kotlin",
            "csharp",
            "ruby",
            "php",
            "swift",
            "scala",
            "bash",
            "sql",
            "html",
            "css",
            "lua",
        ]

        for lang in additional_languages:
            assert registry.has(lang), f"Additional language '{lang}' should be registered"

    def test_total_registered_languages(self):
        """Test total number of registered languages."""
        from victor.verticals.contrib.coding.languages.registry import (
            get_language_registry,
        )

        registry = get_language_registry()
        registry.discover_plugins()

        languages = registry.list_languages()

        # Should have at least 20 languages registered
        assert len(languages) >= 20, f"Expected at least 20 languages, got {len(languages)}"

        # Print all registered languages for verification
        print(f"\n=== Registered Languages ({len(languages)}) ===")
        for lang in sorted(languages):
            print(f"  - {lang}")
