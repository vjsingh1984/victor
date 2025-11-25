"""Testing tool for automated test generation.

Features:
- Generate unit tests for functions and classes
- Create test fixtures and mock data
- Analyze code to suggest test cases
- Generate test file scaffolds
- pytest-based test generation
"""

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import logging

from victor.tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class TestingTool(BaseTool):
    """Tool for automated test generation."""

    @property
    def name(self) -> str:
        """Get tool name."""
        return "testing"

    @property
    def description(self) -> str:
        """Get tool description."""
        return """Automated test generation and scaffolding.

Generate comprehensive test suites for Python code:
- Generate unit tests for functions and classes
- Create test fixtures and mock data
- Suggest test cases based on code analysis
- Generate pytest-compatible test files
- Create test scaffolds for projects

Operations:
- generate_tests: Generate unit tests for a file
- generate_fixture: Create test fixture
- analyze_coverage: Suggest missing test cases
- scaffold: Create test file structure
- mock_data: Generate mock data for testing

Example workflows:
1. Generate tests for a function:
   testing(operation="generate_tests", file="app.py", target="process_data")

2. Create test fixture:
   testing(operation="generate_fixture", name="sample_users", type="list")

3. Analyze test coverage gaps:
   testing(operation="analyze_coverage", file="app.py")

4. Scaffold test file:
   testing(operation="scaffold", file="app.py")
"""

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameters."""
        return self.convert_parameters_to_schema(
        [
            ToolParameter(
                name="operation",
                type="string",
                description="Operation: generate_tests, generate_fixture, analyze_coverage, scaffold, mock_data",
                required=True,
            ),
            ToolParameter(
                name="file",
                type="string",
                description="Source file path",
                required=False,
            ),
            ToolParameter(
                name="target",
                type="string",
                description="Target function/class name for test generation",
                required=False,
            ),
            ToolParameter(
                name="output",
                type="string",
                description="Output test file path",
                required=False,
            ),
            ToolParameter(
                name="name",
                type="string",
                description="Fixture or mock data name",
                required=False,
            ),
            ToolParameter(
                name="type",
                type="string",
                description="Data type for fixtures (list, dict, object, etc.)",
                required=False,
            ),
        ]
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute testing operation.

        Args:
            operation: Testing operation to perform
            **kwargs: Operation-specific parameters

        Returns:
            Tool result with generated tests
        """
        operation = kwargs.get("operation")

        if not operation:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: operation",
            )

        try:
            if operation == "generate_tests":
                return await self._generate_tests(kwargs)
            elif operation == "generate_fixture":
                return await self._generate_fixture(kwargs)
            elif operation == "analyze_coverage":
                return await self._analyze_coverage(kwargs)
            elif operation == "scaffold":
                return await self._scaffold_tests(kwargs)
            elif operation == "mock_data":
                return await self._generate_mock_data(kwargs)
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown operation: {operation}",
                )

        except Exception as e:
            logger.exception("Test generation failed")
            return ToolResult(
                success=False, output="", error=f"Test generation error: {str(e)}"
            )

    async def _generate_tests(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Generate unit tests for a file or function."""
        file_path = kwargs.get("file")
        target = kwargs.get("target")
        output_path = kwargs.get("output")

        if not file_path:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: file",
            )

        file_obj = Path(file_path)
        if not file_obj.exists():
            return ToolResult(
                success=False, output="", error=f"File not found: {file_path}"
            )

        # Read and parse file
        content = file_obj.read_text()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Syntax error in file: {e}",
            )

        # Find functions and classes
        if target:
            # Generate tests for specific target
            functions = self._find_functions(tree, target)
            classes = self._find_classes(tree, target)
        else:
            # Generate tests for all
            functions = self._find_all_functions(tree)
            classes = self._find_all_classes(tree)

        # Generate test code
        test_code = self._build_test_file(file_obj, functions, classes)

        # Determine output path
        if not output_path:
            # Default: tests/test_<filename>.py
            output_path = f"tests/test_{file_obj.name}"

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(test_code)

        # Build report
        report = []
        report.append("Test Generation Complete")
        report.append("=" * 70)
        report.append("")
        report.append(f"Source: {file_path}")
        report.append(f"Output: {output_path}")
        report.append(f"Functions: {len(functions)}")
        report.append(f"Classes: {len(classes)}")
        report.append("")
        report.append("Generated test file:")
        report.append("-" * 70)
        report.append(test_code[:1000])  # Show first 1000 chars
        if len(test_code) > 1000:
            report.append("...")
            report.append(f"\n({len(test_code) - 1000} more characters)")

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _generate_fixture(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Generate test fixture."""
        name = kwargs.get("name")
        data_type = kwargs.get("type", "dict")

        if not name:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: name",
            )

        # Generate fixture code
        fixture_code = self._build_fixture(name, data_type)

        report = []
        report.append(f"Test Fixture: {name}")
        report.append("=" * 70)
        report.append("")
        report.append("Generated fixture:")
        report.append(fixture_code)
        report.append("")
        report.append("Usage:")
        report.append(f"  def test_example({name}):")
        report.append(f"      # Use {name} in your test")
        report.append(f"      assert {name} is not None")

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _analyze_coverage(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Analyze code and suggest missing test cases."""
        file_path = kwargs.get("file")

        if not file_path:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: file",
            )

        file_obj = Path(file_path)
        if not file_obj.exists():
            return ToolResult(
                success=False, output="", error=f"File not found: {file_path}"
            )

        # Read and parse file
        content = file_obj.read_text()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Syntax error in file: {e}",
            )

        # Analyze code
        analysis = self._analyze_code(tree)

        # Build report
        report = []
        report.append("Test Coverage Analysis")
        report.append("=" * 70)
        report.append("")
        report.append(f"File: {file_path}")
        report.append("")

        report.append(f"Functions: {analysis['function_count']}")
        report.append(f"Classes: {analysis['class_count']}")
        report.append(f"Branches: {analysis['branch_count']}")
        report.append("")

        if analysis["functions"]:
            report.append("Suggested test cases:")
            report.append("")
            for func in analysis["functions"][:10]:
                report.append(f"📝 {func['name']}:")
                for test_case in func["suggested_tests"]:
                    report.append(f"   • {test_case}")
                report.append("")

        if len(analysis["functions"]) > 10:
            report.append(f"... and {len(analysis['functions']) - 10} more functions")

        report.append("Recommendations:")
        report.append("  • Test all public functions")
        report.append("  • Cover edge cases and error conditions")
        report.append("  • Test branch conditions (if/else)")
        report.append("  • Validate input/output types")
        report.append("  • Test exception handling")

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _scaffold_tests(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Create test file scaffold."""
        file_path = kwargs.get("file")
        output_path = kwargs.get("output")

        if not file_path:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: file",
            )

        file_obj = Path(file_path)

        # Determine output path
        if not output_path:
            output_path = f"tests/test_{file_obj.name}"

        # Generate scaffold
        scaffold_code = self._build_scaffold(file_obj)

        # Write scaffold
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(scaffold_code)

        report = []
        report.append("Test Scaffold Created")
        report.append("=" * 70)
        report.append("")
        report.append(f"Source: {file_path}")
        report.append(f"Output: {output_path}")
        report.append("")
        report.append("Scaffold structure:")
        report.append(scaffold_code)

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _generate_mock_data(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Generate mock data for testing."""
        name = kwargs.get("name", "mock_data")
        data_type = kwargs.get("type", "dict")

        # Generate mock data
        mock_code = self._build_mock_data(name, data_type)

        report = []
        report.append(f"Mock Data: {name}")
        report.append("=" * 70)
        report.append("")
        report.append("Generated mock data:")
        report.append(mock_code)

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    def _find_functions(self, tree: ast.AST, name: str) -> List[Dict[str, Any]]:
        """Find specific function."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                functions.append(self._analyze_function(node))
        return functions

    def _find_classes(self, tree: ast.AST, name: str) -> List[Dict[str, Any]]:
        """Find specific class."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == name:
                classes.append(self._analyze_class(node))
        return classes

    def _find_all_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find all functions."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private functions
                if not node.name.startswith("_"):
                    functions.append(self._analyze_function(node))
        return functions

    def _find_all_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find all classes."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Skip private classes
                if not node.name.startswith("_"):
                    classes.append(self._analyze_class(node))
        return classes

    def _analyze_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze function for test generation."""
        params = []
        for arg in node.args.args:
            params.append(arg.arg)

        # Detect return type
        has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))

        return {
            "name": node.name,
            "params": params,
            "has_return": has_return,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
        }

    def _analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze class for test generation."""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if not item.name.startswith("_") or item.name == "__init__":
                    methods.append(self._analyze_function(item))

        return {
            "name": node.name,
            "methods": methods,
        }

    def _build_test_file(
        self, source_file: Path, functions: List[Dict[str, Any]], classes: List[Dict[str, Any]]
    ) -> str:
        """Build complete test file."""
        lines = []

        # Header
        lines.append(f'"""Tests for {source_file.name}."""')
        lines.append("")
        lines.append("import pytest")
        lines.append(f"from {source_file.stem} import *")
        lines.append("")
        lines.append("")

        # Generate test fixtures
        lines.append("# Fixtures")
        lines.append("")
        lines.append("@pytest.fixture")
        lines.append("def sample_data():")
        lines.append('    """Sample test data."""')
        lines.append("    return {")
        lines.append('        "name": "Test User",')
        lines.append('        "email": "test@example.com",')
        lines.append('        "age": 25,')
        lines.append("    }")
        lines.append("")
        lines.append("")

        # Generate function tests
        if functions:
            lines.append("# Function Tests")
            lines.append("")

            for func in functions:
                test_name = f"test_{func['name']}"

                # Basic test
                lines.append(f"def {test_name}():")
                lines.append(f'    """Test {func["name"]} function."""')

                # Create test body
                param_str = ", ".join([f"'{p}'" for p in func["params"][:3]])
                if param_str:
                    lines.append(f"    # Arrange")
                    for param in func["params"][:3]:
                        lines.append(f"    {param} = None  # TODO: Set test value")
                    lines.append("")

                lines.append(f"    # Act")
                call_params = ", ".join(func["params"][:3])
                if func["has_return"]:
                    if func["is_async"]:
                        lines.append(f"    # result = await {func['name']}({call_params})")
                        lines.append(f"    result = None  # TODO: Call async function")
                    else:
                        lines.append(f"    result = {func['name']}({call_params})")
                else:
                    if func["is_async"]:
                        lines.append(f"    # await {func['name']}({call_params})")
                        lines.append(f"    pass  # TODO: Call async function")
                    else:
                        lines.append(f"    {func['name']}({call_params})")

                lines.append("")
                lines.append(f"    # Assert")
                if func["has_return"]:
                    lines.append(f"    assert result is not None  # TODO: Add specific assertions")
                else:
                    lines.append(f"    # TODO: Add assertions")

                lines.append("")
                lines.append("")

                # Edge case test
                lines.append(f"def {test_name}_edge_cases():")
                lines.append(f'    """Test {func["name"]} with edge cases."""')
                lines.append("    # TODO: Test with None, empty values, extreme values")
                lines.append("    pass")
                lines.append("")
                lines.append("")

                # Error case test
                lines.append(f"def {test_name}_errors():")
                lines.append(f'    """Test {func["name"]} error handling."""')
                lines.append("    # TODO: Test error conditions")
                lines.append("    # with pytest.raises(ValueError):")
                lines.append(f"    #     {func['name']}(invalid_input)")
                lines.append("    pass")
                lines.append("")
                lines.append("")

        # Generate class tests
        if classes:
            lines.append("# Class Tests")
            lines.append("")

            for cls in classes:
                lines.append(f"class Test{cls['name']}:")
                lines.append(f'    """Tests for {cls["name"]} class."""')
                lines.append("")

                # Setup fixture
                lines.append("    @pytest.fixture")
                lines.append(f"    def instance(self):")
                lines.append(f'        """Create {cls["name"]} instance for testing."""')
                lines.append(f"        return {cls['name']}()  # TODO: Add constructor params")
                lines.append("")

                # Generate method tests
                for method in cls["methods"][:5]:  # Limit to 5 methods
                    test_name = f"test_{method['name']}"

                    lines.append(f"    def {test_name}(self, instance):")
                    lines.append(f'        """Test {method["name"]} method."""')

                    if method["params"] and method["params"][0] == "self":
                        params = method["params"][1:]
                    else:
                        params = method["params"]

                    if params:
                        lines.append(f"        # Arrange")
                        for param in params[:3]:
                            lines.append(f"        {param} = None  # TODO: Set test value")
                        lines.append("")

                    lines.append(f"        # Act")
                    call_params = ", ".join(params[:3])
                    if method["has_return"]:
                        lines.append(f"        result = instance.{method['name']}({call_params})")
                    else:
                        lines.append(f"        instance.{method['name']}({call_params})")

                    lines.append("")
                    lines.append(f"        # Assert")
                    if method["has_return"]:
                        lines.append(f"        assert result is not None  # TODO: Add assertions")
                    else:
                        lines.append(f"        # TODO: Add assertions")

                    lines.append("")

                lines.append("")

        return "\n".join(lines)

    def _build_fixture(self, name: str, data_type: str) -> str:
        """Build pytest fixture."""
        lines = []
        lines.append("@pytest.fixture")
        lines.append(f"def {name}():")
        lines.append(f'    """Test fixture for {name}."""')

        if data_type == "list":
            lines.append("    return [")
            lines.append('        {"id": 1, "name": "Item 1"},')
            lines.append('        {"id": 2, "name": "Item 2"},')
            lines.append('        {"id": 3, "name": "Item 3"},')
            lines.append("    ]")
        elif data_type == "dict":
            lines.append("    return {")
            lines.append('        "key1": "value1",')
            lines.append('        "key2": "value2",')
            lines.append('        "key3": "value3",')
            lines.append("    }")
        elif data_type == "object":
            lines.append("    class MockObject:")
            lines.append('        def __init__(self):')
            lines.append('            self.attr1 = "value1"')
            lines.append('            self.attr2 = "value2"')
            lines.append("")
            lines.append("    return MockObject()")
        else:
            lines.append('    return "test_value"')

        return "\n".join(lines)

    def _build_scaffold(self, source_file: Path) -> str:
        """Build test file scaffold."""
        lines = []

        lines.append(f'"""Tests for {source_file.name}."""')
        lines.append("")
        lines.append("import pytest")
        lines.append("")
        lines.append("")
        lines.append("# Add your fixtures here")
        lines.append("")
        lines.append("")
        lines.append("# Add your test functions here")
        lines.append("")
        lines.append("")
        lines.append("def test_placeholder():")
        lines.append('    """Placeholder test."""')
        lines.append("    assert True")
        lines.append("")

        return "\n".join(lines)

    def _build_mock_data(self, name: str, data_type: str) -> str:
        """Build mock data."""
        if data_type == "list":
            return f'''{name} = [
    {{"id": 1, "name": "Mock 1", "active": True}},
    {{"id": 2, "name": "Mock 2", "active": False}},
    {{"id": 3, "name": "Mock 3", "active": True}},
]'''
        elif data_type == "dict":
            return f'''{name} = {{
    "name": "Test User",
    "email": "test@example.com",
    "age": 25,
    "active": True,
}}'''
        elif data_type == "user":
            return f'''{name} = {{
    "id": 123,
    "username": "testuser",
    "email": "test@example.com",
    "first_name": "Test",
    "last_name": "User",
    "created_at": "2025-01-01T00:00:00Z",
}}'''
        else:
            return f'{name} = "mock_value"'

    def _analyze_code(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code for coverage suggestions."""
        functions = []
        classes = []
        branch_count = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith("_"):
                    func_info = self._analyze_function(node)
                    func_info["suggested_tests"] = self._suggest_tests(node)
                    functions.append(func_info)

            elif isinstance(node, ast.ClassDef):
                if not node.name.startswith("_"):
                    classes.append(self._analyze_class(node))

            elif isinstance(node, (ast.If, ast.For, ast.While)):
                branch_count += 1

        return {
            "function_count": len(functions),
            "class_count": len(classes),
            "branch_count": branch_count,
            "functions": functions,
            "classes": classes,
        }

    def _suggest_tests(self, node: ast.FunctionDef) -> List[str]:
        """Suggest test cases for a function."""
        suggestions = []

        # Always suggest happy path
        suggestions.append("Test normal/happy path")

        # Suggest edge cases based on parameters
        if node.args.args:
            suggestions.append("Test with None parameters")
            suggestions.append("Test with empty values")

        # Suggest tests based on code structure
        has_if = any(isinstance(n, ast.If) for n in ast.walk(node))
        has_loop = any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(node))
        has_except = any(isinstance(n, ast.Try) for n in ast.walk(node))

        if has_if:
            suggestions.append("Test all branch conditions")

        if has_loop:
            suggestions.append("Test with empty collection")
            suggestions.append("Test with large dataset")

        if has_except:
            suggestions.append("Test error/exception handling")

        return suggestions
