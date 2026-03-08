"""Tests for victor.core.utils.ast_helpers."""

import ast
import textwrap

import pytest

from victor.core.utils.ast_helpers import (
    STDLIB_MODULES,
    SymbolSummary,
    build_signature,
    extract_base_classes,
    extract_imports,
    extract_parameters,
    extract_symbols,
    get_annotation_str,
    get_decorator_name,
    is_stdlib_module,
)

# =============================================================================
# Fixtures
# =============================================================================

SAMPLE_CODE = textwrap.dedent("""\
    import os
    import os.path
    from pathlib import Path
    from typing import List, Optional

    class Base:
        \"\"\"A base class.\"\"\"
        pass

    class Child(Base, os.PathLike):
        \"\"\"A child class.\"\"\"

        def method(self, x: int, y: str = "hello") -> bool:
            \"\"\"Do something.\"\"\"
            return True

    async def top_func(a, b: float) -> None:
        \"\"\"An async function.\"\"\"
        pass

    @staticmethod
    def plain_func():
        pass
""")


@pytest.fixture
def sample_tree():
    return ast.parse(SAMPLE_CODE)


# =============================================================================
# STDLIB_MODULES
# =============================================================================


class TestStdlibModules:
    def test_contains_common_modules(self):
        for mod in ("os", "sys", "json", "typing", "pathlib", "asyncio"):
            assert mod in STDLIB_MODULES

    def test_contains_third_party_exclusions(self):
        for mod in ("numpy", "pandas", "pydantic", "pytest"):
            assert mod in STDLIB_MODULES

    def test_is_frozenset(self):
        assert isinstance(STDLIB_MODULES, frozenset)


class TestIsStdlibModule:
    def test_exact_match(self):
        assert is_stdlib_module("os") is True
        assert is_stdlib_module("json") is True

    def test_dotted_module(self):
        assert is_stdlib_module("os.path") is True
        assert is_stdlib_module("collections.abc") is True

    def test_non_stdlib(self):
        assert is_stdlib_module("victor") is False
        assert is_stdlib_module("my_package") is False


# =============================================================================
# get_annotation_str
# =============================================================================


class TestGetAnnotationStr:
    def test_none(self):
        assert get_annotation_str(None) is None

    def test_simple_name(self):
        ann = ast.parse("int", mode="eval").body
        result = get_annotation_str(ann)
        assert result == "int"

    def test_complex_annotation(self):
        ann = ast.parse("Optional[List[str]]", mode="eval").body
        assert get_annotation_str(ann) == "Optional[List[str]]"


# =============================================================================
# get_decorator_name
# =============================================================================


class TestGetDecoratorName:
    def test_name_decorator(self):
        node = ast.Name(id="staticmethod")
        assert get_decorator_name(node) == "staticmethod"

    def test_attribute_decorator(self):
        code = textwrap.dedent("""\
            class Foo:
                @app.route
                def bar(self): pass
        """)
        tree = ast.parse(code)
        cls = tree.body[0]
        func = cls.body[0]
        dec = func.decorator_list[0]
        assert get_decorator_name(dec) == "app.route"

    def test_call_decorator(self):
        code = textwrap.dedent("""\
            @app.route("/")
            def index(): pass
        """)
        tree = ast.parse(code)
        func = tree.body[0]
        dec = func.decorator_list[0]
        assert get_decorator_name(dec) == "app.route"


# =============================================================================
# extract_imports
# =============================================================================


class TestExtractImports:
    def test_top_level_only(self, sample_tree):
        imports = extract_imports(sample_tree, top_level_only=True)
        assert "os" in imports
        assert "pathlib" in imports
        assert "typing" in imports

    def test_top_level_deduplicates_os(self, sample_tree):
        imports = extract_imports(sample_tree, top_level_only=True)
        # "import os" and "import os.path" both produce "os"
        assert imports.count("os") == 2  # both occurrences kept (caller dedupes)

    def test_fully_qualified(self, sample_tree):
        imports = extract_imports(sample_tree, top_level_only=False)
        assert "os" in imports
        assert "os.path" in imports
        assert "pathlib.Path" in imports
        assert "typing.List" in imports
        assert "typing.Optional" in imports

    def test_empty_tree(self):
        tree = ast.parse("")
        assert extract_imports(tree) == []


# =============================================================================
# extract_base_classes
# =============================================================================


class TestExtractBaseClasses:
    def test_simple_base(self, sample_tree):
        child_cls = [
            n for n in ast.walk(sample_tree) if isinstance(n, ast.ClassDef) and n.name == "Child"
        ][0]
        bases = extract_base_classes(child_cls)
        assert "Base" in bases
        # os.PathLike should be unparsed as dotted name
        assert any("PathLike" in b for b in bases)

    def test_no_bases(self, sample_tree):
        base_cls = [
            n for n in ast.walk(sample_tree) if isinstance(n, ast.ClassDef) and n.name == "Base"
        ][0]
        bases = extract_base_classes(base_cls)
        assert bases == []

    def test_subscript_base(self):
        code = "class Foo(Generic[T]): pass"
        tree = ast.parse(code)
        cls = tree.body[0]
        bases = extract_base_classes(cls)
        assert "Generic" in bases


# =============================================================================
# build_signature
# =============================================================================


class TestBuildSignature:
    def test_method_signature(self, sample_tree):
        funcs = [
            n
            for n in ast.walk(sample_tree)
            if isinstance(n, ast.FunctionDef) and n.name == "method"
        ]
        sig = build_signature(funcs[0])
        assert sig == "method(self, x, y)"

    def test_no_args(self):
        code = "def foo(): pass"
        tree = ast.parse(code)
        func = tree.body[0]
        assert build_signature(func) == "foo()"


# =============================================================================
# extract_parameters
# =============================================================================


class TestExtractParameters:
    def test_without_annotations(self, sample_tree):
        funcs = [
            n
            for n in ast.walk(sample_tree)
            if isinstance(n, ast.FunctionDef) and n.name == "method"
        ]
        params = extract_parameters(funcs[0], include_annotations=False)
        assert params == ["self", "x", "y"]

    def test_with_annotations(self, sample_tree):
        funcs = [
            n
            for n in ast.walk(sample_tree)
            if isinstance(n, ast.FunctionDef) and n.name == "method"
        ]
        params = extract_parameters(funcs[0], include_annotations=True)
        assert params[0] == "self"  # no annotation
        assert params[1] == "x: int"
        assert params[2] == "y: str"


# =============================================================================
# extract_symbols
# =============================================================================


class TestExtractSymbols:
    def test_basic_extraction(self, sample_tree):
        symbols = extract_symbols(sample_tree, enrich=False)
        names = [s.name for s in symbols]
        assert "Base" in names
        assert "Child" in names
        assert "top_func" in names
        assert "plain_func" in names

    def test_class_has_bases(self, sample_tree):
        symbols = extract_symbols(sample_tree, enrich=False)
        child = [s for s in symbols if s.name == "Child"][0]
        assert "Base" in child.base_classes

    def test_function_has_signature(self, sample_tree):
        symbols = extract_symbols(sample_tree, enrich=False)
        method = [s for s in symbols if s.name == "method"][0]
        assert method.signature == "method(self, x, y)"

    def test_enriched_async(self, sample_tree):
        symbols = extract_symbols(sample_tree, enrich=True)
        top_func = [s for s in symbols if s.name == "top_func"][0]
        assert top_func.is_async is True
        assert top_func.return_type == "None"

    def test_enriched_parameters(self, sample_tree):
        symbols = extract_symbols(sample_tree, enrich=True)
        method = [s for s in symbols if s.name == "method"][0]
        assert "x: int" in method.parameters

    def test_enriched_decorators(self, sample_tree):
        symbols = extract_symbols(sample_tree, enrich=True)
        plain = [s for s in symbols if s.name == "plain_func"][0]
        assert "staticmethod" in plain.decorators

    def test_enriched_class_decorators(self):
        code = textwrap.dedent("""\
            @dataclass
            class Foo:
                x: int = 0
        """)
        tree = ast.parse(code)
        symbols = extract_symbols(tree, enrich=True)
        foo = symbols[0]
        assert foo.type == "class"
        assert "dataclass" in foo.decorators

    def test_docstrings_extracted(self, sample_tree):
        symbols = extract_symbols(sample_tree, enrich=False)
        base = [s for s in symbols if s.name == "Base"][0]
        assert base.docstring == "A base class."

    def test_empty_tree(self):
        tree = ast.parse("")
        assert extract_symbols(tree) == []


# =============================================================================
# SymbolSummary dataclass
# =============================================================================


class TestSymbolSummary:
    def test_defaults(self):
        s = SymbolSummary(name="foo", type="function", line_number=1)
        assert s.docstring is None
        assert s.signature is None
        assert s.base_classes == []
        assert s.is_async is False
        assert s.parameters == []
        assert s.return_type is None
        assert s.decorators == []
