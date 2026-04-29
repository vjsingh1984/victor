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

"""Unit tests for language-specific edge detection handlers."""

import pytest
from pathlib import Path
from typing import Any

from victor.core.graph_rag.language_handlers import (
    get_edge_handler,
    _VictorCodingPluginAdapter,
    EdgeHandlerRegistry,
)


def _parse_code(code: str, language_module: Any, language_attr: str = "language") -> Any:
    """Helper to parse code using tree-sitter with new API.

    Args:
        code: Source code to parse
        language_module: Tree-sitter language module
        language_attr: Attribute name for language function (default: "language")

    Returns:
        Parsed tree-sitter tree
    """
    import tree_sitter as ts

    lang_func = getattr(language_module, language_attr)
    lang_obj = lang_func()
    ts_language = ts.Language(lang_obj) if not isinstance(lang_obj, ts.Language) else lang_obj
    parser = ts.Parser(ts_language)
    return parser.parse(bytes(code, "utf-8"))


@pytest.mark.unit
class TestLanguageHandlerRegistry:
    """Test language handler registry."""

    def test_all_handlers_registered(self):
        """Test that all handlers are registered via victor_coding."""
        # victor_coding provides handlers for these languages
        expected = {"python", "py", "javascript", "js", "typescript", "ts",
                   "tsx", "jsx", "go", "golang", "rust", "rs", "java", "cpp", "c++",
                   "csharp", "cs", "kotlin", "kt", "swift", "c", "ruby", "rb", "php"}
        # Check that we can get handlers for all expected languages
        for lang in expected:
            handler = get_edge_handler(lang)
            assert handler is not None, f"No handler found for {lang}"
            assert isinstance(handler, _VictorCodingPluginAdapter), f"Handler for {lang} should use victor_coding adapter"

    def test_get_python_handler(self):
        """Test getting Python handler."""
        handler = get_edge_handler("python")
        assert handler is not None
        assert isinstance(handler, _VictorCodingPluginAdapter)

    def test_get_javascript_handler(self):
        """Test getting JavaScript handler."""
        handler = get_edge_handler("javascript")
        assert handler is not None
        assert isinstance(handler, _VictorCodingPluginAdapter)

    def test_get_typescript_handler(self):
        """Test getting TypeScript handler."""
        handler = get_edge_handler("typescript")
        assert handler is not None
        assert isinstance(handler, _VictorCodingPluginAdapter)

    def test_get_go_handler(self):
        """Test getting Go handler."""
        handler = get_edge_handler("go")
        assert handler is not None
        assert isinstance(handler, _VictorCodingPluginAdapter)

    def test_get_rust_handler(self):
        """Test getting Rust handler."""
        handler = get_edge_handler("rust")
        assert handler is not None
        assert isinstance(handler, _VictorCodingPluginAdapter)

    def test_get_java_handler(self):
        """Test getting Java handler."""
        handler = get_edge_handler("java")
        assert handler is not None
        assert isinstance(handler, _VictorCodingPluginAdapter)

    def test_get_cpp_handler(self):
        """Test getting C++ handler."""
        handler = get_edge_handler("cpp")
        assert handler is not None
        assert isinstance(handler, _VictorCodingPluginAdapter)

    def test_get_csharp_handler(self):
        """Test getting C# handler."""
        handler = get_edge_handler("csharp")
        assert handler is not None
        assert isinstance(handler, _VictorCodingPluginAdapter)

    def test_get_kotlin_handler(self):
        """Test getting Kotlin handler."""
        handler = get_edge_handler("kotlin")
        assert handler is not None
        assert isinstance(handler, _VictorCodingPluginAdapter)

    def test_get_swift_handler(self):
        """Test getting Swift handler."""
        handler = get_edge_handler("swift")
        assert handler is not None
        assert isinstance(handler, _VictorCodingPluginAdapter)

    def test_get_c_handler(self):
        """Test getting C handler."""
        handler = get_edge_handler("c")
        assert handler is not None
        assert isinstance(handler, _VictorCodingPluginAdapter)

    def test_get_ruby_handler(self):
        """Test getting Ruby handler."""
        handler = get_edge_handler("ruby")
        assert handler is not None
        assert isinstance(handler, _VictorCodingPluginAdapter)

    def test_get_php_handler(self):
        """Test getting PHP handler."""
        handler = get_edge_handler("php")
        assert handler is not None
        assert isinstance(handler, _VictorCodingPluginAdapter)


@pytest.mark.unit
class TestTypeScriptEdgeHandler:
    """Test TypeScript edge detection handler."""

    @pytest.fixture
    def handler(self):
        from victor_coding.languages.registry import get_language_registry, get_plugin_by_language
        from victor.core.graph_rag.language_handlers import _VictorCodingPluginAdapter

        # Ensure plugins are discovered
        registry = get_language_registry()
        if not registry.list_languages():
            registry.discover_plugins()

        plugin = get_plugin_by_language('typescript')
        return _VictorCodingPluginAdapter(plugin)

    @pytest.mark.asyncio
    async def test_detect_simple_calls(self, handler, tmp_path):
        """Test detecting simple function calls in TypeScript."""
        import tree_sitter_typescript

        code = """
function foo(): string {
    return "foo";
}

function bar(): string {
    return foo();  // bar calls foo
}
"""
        test_file = tmp_path / "test.ts"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_typescript, "language_typescript")
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert ("bar", "foo") in call_edges

    @pytest.mark.asyncio
    async def test_detect_method_calls(self, handler, tmp_path):
        """Test detecting method calls in TypeScript."""
        import tree_sitter_typescript

        code = """
class Calculator {
    add(a: number, b: number): number {
        return a + b;
    }
}

function compute(calc: Calculator): number {
    return calc.add(1, 2);  // compute calls calc.add
}
"""
        test_file = tmp_path / "test.ts"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_typescript, "language_typescript")
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert any(caller == "compute" and "add" in callee
                  for caller, callee in call_edges)


@pytest.mark.unit
class TestCppEdgeHandler:
    """Test C++ edge detection handler."""

    @pytest.fixture
    def handler(self):
        from victor_coding.languages.registry import get_language_registry, get_plugin_by_language
        from victor.core.graph_rag.language_handlers import _VictorCodingPluginAdapter

        # Ensure plugins are discovered
        registry = get_language_registry()
        if not registry.list_languages():
            registry.discover_plugins()

        plugin = get_plugin_by_language('cpp')
        return _VictorCodingPluginAdapter(plugin)

    @pytest.mark.asyncio
    async def test_detect_simple_calls(self, handler, tmp_path):
        """Test detecting simple function calls in C++."""
        import tree_sitter_cpp

        code = """
#include <string>

std::string foo() {
    return "foo";
}

std::string bar() {
    return foo();  // bar calls foo
}
"""
        test_file = tmp_path / "test.cpp"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_cpp)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert ("bar", "foo") in call_edges

    @pytest.mark.asyncio
    async def test_detect_method_calls(self, handler, tmp_path):
        """Test detecting method calls in C++."""
        import tree_sitter_cpp

        code = """
class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }
};

int compute(Calculator* calc) {
    return calc->add(1, 2);  // compute calls calc.add
}
"""
        test_file = tmp_path / "test.cpp"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_cpp)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert any(caller == "compute" and "add" in callee
                  for caller, callee in call_edges)

    @pytest.mark.asyncio
    async def test_detect_namespace_calls(self, handler, tmp_path):
        """Test detecting namespace-qualified calls in C++."""
        import tree_sitter_cpp

        code = """
namespace std {
    void helper() {}
}

void caller() {
    std::helper();  // caller calls std::helper
}
"""
        test_file = tmp_path / "test.cpp"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_cpp)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert any(caller == "caller" and "helper" in callee
                  for caller, callee in call_edges)


@pytest.mark.unit
class TestCEdgeHandler:
    """Test C edge detection handler."""

    @pytest.fixture
    def handler(self):
        from victor_coding.languages.registry import get_language_registry, get_plugin_by_language
        from victor.core.graph_rag.language_handlers import _VictorCodingPluginAdapter

        # Ensure plugins are discovered
        registry = get_language_registry()
        if not registry.list_languages():
            registry.discover_plugins()

        plugin = get_plugin_by_language('c')
        return _VictorCodingPluginAdapter(plugin)

    @pytest.mark.asyncio
    async def test_detect_simple_calls(self, handler, tmp_path):
        """Test detecting simple function calls in C."""
        import tree_sitter_c

        code = """
void foo() {
}

void bar() {
    foo();  // bar calls foo
}
"""
        test_file = tmp_path / "test.c"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_c)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert ("bar", "foo") in call_edges


@pytest.mark.unit
class TestCSharpEdgeHandler:
    """Test C# edge detection handler."""

    @pytest.fixture
    def handler(self):
        from victor_coding.languages.registry import get_language_registry, get_plugin_by_language
        from victor.core.graph_rag.language_handlers import _VictorCodingPluginAdapter

        # Ensure plugins are discovered
        registry = get_language_registry()
        if not registry.list_languages():
            registry.discover_plugins()

        plugin = get_plugin_by_language('csharp')
        return _VictorCodingPluginAdapter(plugin)

    @pytest.mark.asyncio
    async def test_detect_simple_calls(self, handler, tmp_path):
        """Test detecting simple method calls in C#."""
        import tree_sitter_c_sharp

        code = """
class Program {
    static void Foo() {
    }

    static void Bar() {
        Foo();  // Bar calls Foo
    }
}
"""
        test_file = tmp_path / "test.cs"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_c_sharp)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert ("Bar", "Foo") in call_edges

    @pytest.mark.asyncio
    async def test_detect_method_calls(self, handler, tmp_path):
        """Test detecting method calls in C#."""
        import tree_sitter_c_sharp

        code = """
class Calculator {
    public int Add(int a, int b) {
        return a + b;
    }
}

class Program {
    static int Compute(Calculator calc) {
        return calc.Add(1, 2);  // Compute calls calc.Add
    }
}
"""
        test_file = tmp_path / "test.cs"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_c_sharp)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert any(caller == "Compute" and "Add" in callee
                  for caller, callee in call_edges)


@pytest.mark.unit
class TestKotlinEdgeHandler:
    """Test Kotlin edge detection handler."""

    @pytest.fixture
    def handler(self):
        from victor_coding.languages.registry import get_language_registry, get_plugin_by_language
        from victor.core.graph_rag.language_handlers import _VictorCodingPluginAdapter

        # Ensure plugins are discovered
        registry = get_language_registry()
        if not registry.list_languages():
            registry.discover_plugins()

        plugin = get_plugin_by_language('kotlin')
        return _VictorCodingPluginAdapter(plugin)

    @pytest.mark.asyncio
    async def test_detect_simple_calls(self, handler, tmp_path):
        """Test detecting simple function calls in Kotlin."""
        import tree_sitter_kotlin

        code = """
fun foo() {
}

fun bar() {
    foo()  // bar calls foo
}
"""
        test_file = tmp_path / "test.kt"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_kotlin)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert ("bar", "foo") in call_edges

    @pytest.mark.asyncio
    async def test_detect_method_calls(self, handler, tmp_path):
        """Test detecting method calls in Kotlin."""
        import tree_sitter_kotlin

        code = """
class Calculator {
    fun add(a: Int, b: Int): Int {
        return a + b
    }
}

fun compute(calc: Calculator): Int {
    return calc.add(1, 2)  // compute calls calc.add
}
"""
        test_file = tmp_path / "test.kt"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_kotlin)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert any(caller == "compute" and "add" in callee
                  for caller, callee in call_edges)


@pytest.mark.unit
class TestSwiftEdgeHandler:
    """Test Swift edge detection handler."""

    @pytest.fixture
    def handler(self):
        from victor_coding.languages.registry import get_language_registry, get_plugin_by_language
        from victor.core.graph_rag.language_handlers import _VictorCodingPluginAdapter

        # Ensure plugins are discovered
        registry = get_language_registry()
        if not registry.list_languages():
            registry.discover_plugins()

        plugin = get_plugin_by_language('swift')
        return _VictorCodingPluginAdapter(plugin)

    @pytest.mark.asyncio
    async def test_detect_simple_calls(self, handler, tmp_path):
        """Test detecting simple function calls in Swift."""
        import tree_sitter_swift

        code = """
func foo() {
}

func bar() {
    foo()  // bar calls foo
}
"""
        test_file = tmp_path / "test.swift"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_swift)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert ("bar", "foo") in call_edges

    @pytest.mark.asyncio
    async def test_detect_method_calls(self, handler, tmp_path):
        """Test detecting method calls in Swift."""
        import tree_sitter_swift

        code = """
class Calculator {
    func add(_ a: Int, _ b: Int) -> Int {
        return a + b
    }
}

func compute(calc: Calculator) -> Int {
    return calc.add(1, 2)  // compute calls calc.add
}
"""
        test_file = tmp_path / "test.swift"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_swift)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert any(caller == "compute" and "add" in callee
                  for caller, callee in call_edges)


@pytest.mark.unit
class TestRubyEdgeHandler:
    """Test Ruby edge detection handler."""

    @pytest.fixture
    def handler(self):
        from victor_coding.languages.registry import get_language_registry, get_plugin_by_language
        from victor.core.graph_rag.language_handlers import _VictorCodingPluginAdapter

        # Ensure plugins are discovered
        registry = get_language_registry()
        if not registry.list_languages():
            registry.discover_plugins()

        plugin = get_plugin_by_language('ruby')
        return _VictorCodingPluginAdapter(plugin)

    @pytest.mark.asyncio
    async def test_detect_simple_calls(self, handler, tmp_path):
        """Test detecting simple method calls in Ruby."""
        import tree_sitter_ruby

        code = """
def foo
end

def bar
  foo  # bar calls foo
end
"""
        test_file = tmp_path / "test.rb"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_ruby)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert ("bar", "foo") in call_edges

    @pytest.mark.asyncio
    async def test_detect_method_calls(self, handler, tmp_path):
        """Test detecting method calls with receiver in Ruby."""
        import tree_sitter_ruby

        code = """
class Calculator
  def add(a, b)
    a + b
  end
end

def compute(calc)
  calc.add(1, 2)  # compute calls calc.add
end
"""
        test_file = tmp_path / "test.rb"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_ruby)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert any(caller == "compute" and "add" in callee
                  for caller, callee in call_edges)


@pytest.mark.unit
class TestPhpEdgeHandler:
    """Test PHP edge detection handler."""

    @pytest.fixture
    def handler(self):
        from victor_coding.languages.registry import get_language_registry, get_plugin_by_language
        from victor.core.graph_rag.language_handlers import _VictorCodingPluginAdapter

        # Ensure plugins are discovered
        registry = get_language_registry()
        if not registry.list_languages():
            registry.discover_plugins()

        plugin = get_plugin_by_language('php')
        return _VictorCodingPluginAdapter(plugin)

    @pytest.mark.asyncio
    async def test_detect_simple_calls(self, handler, tmp_path):
        """Test detecting simple function calls in PHP."""
        import tree_sitter_php

        code = """
<?php
function foo() {
}

function bar() {
    foo();  // bar calls foo
}
"""
        test_file = tmp_path / "test.php"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_php, "language_php")
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert ("bar", "foo") in call_edges

    @pytest.mark.asyncio
    async def test_detect_method_calls(self, handler, tmp_path):
        """Test detecting method calls in PHP."""
        import tree_sitter_php

        code = """
<?php
class Calculator {
    public function add($a, $b) {
        return $a + $b;
    }
}

function compute($calc) {
    return $calc->add(1, 2);  // compute calls calc.add
}
"""
        test_file = tmp_path / "test.php"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_php, "language_php")
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert any(caller == "compute" and "add" in callee
                  for caller, callee in call_edges)

    @pytest.mark.asyncio
    async def test_detect_static_calls(self, handler, tmp_path):
        """Test detecting static method calls in PHP."""
        import tree_sitter_php

        code = """
<?php
class Helper {
    public static function process() {
    }
}

function caller() {
    Helper::process();  // caller calls Helper.process
}
"""
        test_file = tmp_path / "test.php"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_php, "language_php")
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert any(caller == "caller" and "process" in callee
                  for caller, callee in call_edges)

