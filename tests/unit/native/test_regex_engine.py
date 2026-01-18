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

"""
Unit tests for regex_engine module.
"""

import pytest

# Check if native module is available
try:
    from victor.native.rust.regex_engine import compile_language_patterns

    NATIVE_AVAILABLE = compile_language_patterns is not None
except ImportError:
    NATIVE_AVAILABLE = False

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_import_regex_engine():
    """Test that regex_engine module can be imported."""
    from victor.native.rust.regex_engine import (
        compile_language_patterns,
        list_supported_languages,
        get_language_categories,
        CompiledRegexSet,
        MatchResult,
    )

    assert compile_language_patterns is not None
    assert list_supported_languages is not None
    assert get_language_categories is not None
    assert CompiledRegexSet is not None
    assert MatchResult is not None


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_list_supported_languages():
    """Test listing supported languages."""
    from victor.native.rust.regex_engine import list_supported_languages

    languages = list_supported_languages()
    assert isinstance(languages, list)
    assert len(languages) > 0
    assert "python" in languages
    assert "rust" in languages
    assert "javascript" in languages


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_get_language_categories():
    """Test getting language categories."""
    from victor.native.rust.regex_engine import get_language_categories

    # Test Python categories
    python_categories = get_language_categories("python")
    assert isinstance(python_categories, list)
    assert len(python_categories) > 0
    assert "function" in python_categories
    assert "class" in python_categories
    assert "import" in python_categories

    # Test Rust categories
    rust_categories = get_language_categories("rust")
    assert isinstance(rust_categories, list)
    assert "function" in rust_categories
    assert "struct" in rust_categories


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_compile_python_patterns():
    """Test compiling Python patterns."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("python")
    assert regex_set is not None
    assert regex_set.language == "python"
    assert regex_set.pattern_count() > 0


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_compile_rust_patterns():
    """Test compiling Rust patterns."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("rust")
    assert regex_set is not None
    assert regex_set.language == "rust"
    assert regex_set.pattern_count() > 0


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_python_function_detection():
    """Test Python function definition detection."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("python")

    code = """
def my_function(param1, param2):
    return param1 + param2

async def async_function():
    pass
"""

    matches = regex_set.match_all(code)
    assert len(matches) > 0

    function_matches = [m for m in matches if m.category == "function"]
    assert len(function_matches) >= 2


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_python_class_detection():
    """Test Python class definition detection."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("python")

    code = """
class MyClass:
    def __init__(self):
        self.value = 42

class AnotherClass(BaseClass):
    pass
"""

    matches = regex_set.match_all(code)
    class_matches = [m for m in matches if m.category == "class"]
    assert len(class_matches) >= 2


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_python_import_detection():
    """Test Python import statement detection."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("python")

    code = """
import os
import sys
from pathlib import Path
from collections import defaultdict
"""

    matches = regex_set.match_all(code)
    import_matches = [m for m in matches if m.category == "import"]
    assert len(import_matches) >= 4


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_python_decorator_detection():
    """Test Python decorator detection."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("python")

    code = """
@property
def my_property(self):
    return self._value

@classmethod
def from_dict(cls, data):
    pass

@decorator_with_args(arg1, arg2)
def decorated_function():
    pass
"""

    matches = regex_set.match_all(code)
    decorator_matches = [m for m in matches if m.category == "decorator"]
    assert len(decorator_matches) >= 3


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_rust_struct_detection():
    """Test Rust struct definition detection."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("rust")

    code = """
pub struct MyStruct {
    field1: i32,
    field2: String,
}

struct PrivateStruct {
    value: bool,
}
"""

    matches = regex_set.match_all(code)
    struct_matches = [m for m in matches if m.category == "struct"]
    assert len(struct_matches) >= 2


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_rust_function_detection():
    """Test Rust function definition detection."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("rust")

    code = """
pub fn public_function() -> i32 {
    42
}

async fn async_function() -> Result<()> {
    Ok(())
}

unsafe extern "C" fn extern_function() {
    unimplemented!()
}
"""

    matches = regex_set.match_all(code)
    function_matches = [m for m in matches if m.category == "function"]
    assert len(function_matches) >= 3


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_rust_macro_detection():
    """Test Rust macro detection."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("rust")

    code = """
vec![1, 2, 3]
println!("Hello, {}", name)
macro_rules! my_macro {
    () => {};
}
"""

    matches = regex_set.match_all(code)
    macro_matches = [m for m in matches if m.category == "macro"]
    assert len(macro_matches) >= 3


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_typescript_interface_detection():
    """Test TypeScript interface definition detection."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("typescript")

    code = """
interface MyInterface {
    method(): void;
    property: string;
}

interface ExtendedInterface extends BaseInterface {
    extraMethod(): void;
}
"""

    matches = regex_set.match_all(code)
    interface_matches = [m for m in matches if m.category == "interface"]
    assert len(interface_matches) >= 2


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_typescript_class_detection():
    """Test TypeScript class definition detection."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("typescript")

    code = """
class MyClass implements MyInterface {
    method() {
        console.log("hello");
    }
    property = "test";
}

class GenericClass<T> {
    value: T;
}
"""

    matches = regex_set.match_all(code)
    class_matches = [m for m in matches if m.category == "class"]
    assert len(class_matches) >= 2


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_contains_any():
    """Test contains_any method."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("python")

    assert regex_set.contains_any("def foo():")
    assert regex_set.contains_any("class Foo:")
    assert regex_set.contains_any("import os")
    assert not regex_set.contains_any("just random text")


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_matched_pattern_names():
    """Test matched_pattern_names method."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("python")

    code = """
def function1():
    pass

class MyClass:
    pass

import os
"""

    names = regex_set.matched_pattern_names(code)
    assert isinstance(names, list)
    assert "function_def" in names
    assert "class_def" in names
    assert "import_statement" in names


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_count_by_pattern():
    """Test count_by_pattern method."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("python")

    code = """
def func1():
    pass

def func2():
    pass

def func3():
    pass
"""

    counts = regex_set.count_by_pattern(code)
    assert isinstance(counts, dict)
    assert counts.get("function_def", 0) >= 3


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_list_patterns():
    """Test list_patterns method."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("python")
    patterns = regex_set.list_patterns()

    assert isinstance(patterns, list)
    assert len(patterns) > 0
    assert "function_def" in patterns
    assert "class_def" in patterns


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_patterns_by_category():
    """Test patterns_by_category method."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("python")

    function_patterns = regex_set.patterns_by_category("function")
    assert isinstance(function_patterns, list)
    assert len(function_patterns) > 0

    class_patterns = regex_set.patterns_by_category("class")
    assert isinstance(class_patterns, list)
    assert len(class_patterns) > 0


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_list_categories():
    """Test list_categories method."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("python")
    categories = regex_set.list_categories()

    assert isinstance(categories, list)
    assert len(categories) > 0
    assert "function" in categories
    assert "class" in categories
    assert "import" in categories


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_pattern_filtering():
    """Test filtering patterns by category."""
    from victor.native.rust.regex_engine import compile_language_patterns

    # Compile only function patterns
    regex_set = compile_language_patterns("python", ["function"])

    code = """
def my_function():
    pass

class MyClass:
    pass
"""

    matches = regex_set.match_all(code)

    # Should only have function matches
    for match in matches:
        assert match.category == "function"


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_match_result_attributes():
    """Test MatchResult attributes."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("python")

    code = "def my_function():\n    pass"
    matches = regex_set.match_all(code)

    assert len(matches) > 0

    match = matches[0]
    assert hasattr(match, "pattern_id")
    assert hasattr(match, "pattern_name")
    assert hasattr(match, "category")
    assert hasattr(match, "start_byte")
    assert hasattr(match, "end_byte")
    assert hasattr(match, "matched_text")
    assert hasattr(match, "line_number")
    assert hasattr(match, "column_number")

    # Check types
    assert isinstance(match.pattern_id, int)
    assert isinstance(match.pattern_name, str)
    assert isinstance(match.category, str)
    assert isinstance(match.start_byte, int)
    assert isinstance(match.end_byte, int)
    assert isinstance(match.matched_text, str)
    assert isinstance(match.line_number, int)
    assert isinstance(match.column_number, int)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_match_result_line_number():
    """Test that line numbers are correct."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("python")

    code = """# Line 1
# Line 2
def function_on_line_3():
    pass
"""

    matches = regex_set.match_all(code)
    function_matches = [m for m in matches if m.category == "function"]

    assert len(function_matches) > 0
    assert function_matches[0].line_number == 3


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_unsupported_language():
    """Test that unsupported language raises error."""
    from victor.native.rust.regex_engine import compile_language_patterns

    with pytest.raises(ValueError) as exc_info:
        compile_language_patterns("nonexistent_language")

    assert "Unsupported language" in str(exc_info.value)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_go_import_detection():
    """Test Go import statement detection."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("go")

    code = """
import (
    "fmt"
    "os"
    "github.com/user/package"
)

import "single"
"""

    matches = regex_set.match_all(code)
    import_matches = [m for m in matches if m.category == "import"]
    assert len(import_matches) >= 2


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_java_annotation_detection():
    """Test Java annotation detection."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("java")

    code = """
@Override
public String toString() {
    return "test";
}

@Deprecated
@SuppressWarnings("unchecked")
public void oldMethod() {
}
"""

    matches = regex_set.match_all(code)
    annotation_matches = [m for m in matches if m.category == "annotation"]
    assert len(annotation_matches) >= 3


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="victor_native module not built")
def test_cpp_include_detection():
    """Test C++ include directive detection."""
    from victor.native.rust.regex_engine import compile_language_patterns

    regex_set = compile_language_patterns("cpp")

    code = """
#include <iostream>
#include <vector>
#include "myheader.h"
"""

    matches = regex_set.match_all(code)
    include_matches = [m for m in matches if m.category == "import"]
    assert len(include_matches) >= 3
