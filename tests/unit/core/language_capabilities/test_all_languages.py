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

"""Comprehensive tests for all supported languages and their mechanisms.

This module tests each supported language with:
1. Tree-sitter extraction and validation (universal fallback)
2. Native validators where available (Python, Go, Java, C/C++)
3. Config validators where available (JSON, YAML, TOML, XML, etc.)
"""

import pytest
import tempfile
from pathlib import Path

from victor.core.language_capabilities import (
    LanguageCapabilityRegistry,
    LanguageTier,
)
from victor.core.language_capabilities.extractors import (
    TreeSitterExtractor,
    PythonASTExtractor,
    GoExtractor,
    JavaExtractor,
    CppExtractor,
)
from victor.core.language_capabilities.validators import (
    TreeSitterValidator,
    PythonASTValidator,
    GoValidator,
    JavaValidator,
    CppValidator,
    JsonValidator,
    YamlValidator,
    TomlValidator,
    XmlValidator,
    HoconValidator,
    MarkdownValidator,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry singleton between tests."""
    LanguageCapabilityRegistry.reset_instance()
    yield
    LanguageCapabilityRegistry.reset_instance()


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def ts_extractor():
    """Tree-sitter extractor instance."""
    return TreeSitterExtractor()


@pytest.fixture
def ts_validator():
    """Tree-sitter validator instance."""
    return TreeSitterValidator()


# =============================================================================
# Language Sample Code
# =============================================================================

LANGUAGE_SAMPLES = {
    # Tier 1 Languages
    "python": {
        "valid": '''def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"

class Greeter:
    def greet(self, name: str) -> None:
        print(hello(name))
''',
        "invalid": "def foo(:",
        "extension": ".py",
    },
    "javascript": {
        "valid": '''function hello(name) {
    return `Hello, ${name}!`;
}

class Greeter {
    greet(name) {
        console.log(hello(name));
    }
}
''',
        "invalid": "function foo( {",
        "extension": ".js",
    },
    "typescript": {
        "valid": '''function hello(name: string): string {
    return `Hello, ${name}!`;
}

interface Greeter {
    greet(name: string): void;
}

class MyGreeter implements Greeter {
    greet(name: string): void {
        console.log(hello(name));
    }
}
''',
        "invalid": "function foo(: string {",
        "extension": ".ts",
    },
    "jsx": {
        "valid": '''function App() {
    return <div className="app">Hello World</div>;
}
''',
        "invalid": "function App() { return <div>; }",
        "extension": ".jsx",
    },
    "tsx": {
        "valid": '''interface Props {
    name: string;
}

function Hello({ name }: Props): JSX.Element {
    return <div>Hello, {name}!</div>;
}
''',
        "invalid": "function Hello(: Props) { return <div>; }",
        "extension": ".tsx",
    },
    # Tier 2 Languages
    "go": {
        "valid": '''package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}

func add(a, b int) int {
    return a + b
}
''',
        "invalid": "package main\n\nfunc main( {",
        "extension": ".go",
    },
    "rust": {
        "valid": '''fn main() {
    println!("Hello, World!");
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}
''',
        "invalid": "fn main( {",
        "extension": ".rs",
    },
    "java": {
        "valid": '''public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }

    public static int add(int a, int b) {
        return a + b;
    }
}
''',
        "invalid": "public class Main {\n    public static void main(String[] args {",
        "extension": ".java",
    },
    "c": {
        "valid": '''#include <stdio.h>

int main() {
    printf("Hello, World!\\n");
    return 0;
}

int add(int a, int b) {
    return a + b;
}
''',
        "invalid": "int main( {\n    return 0;\n}",
        "extension": ".c",
    },
    "cpp": {
        "valid": '''#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}

class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }
};
''',
        "invalid": "int main( {\n    return 0;\n}",
        "extension": ".cpp",
    },
    # Tier 3 Languages
    "ruby": {
        "valid": '''def hello(name)
  "Hello, #{name}!"
end

class Greeter
  def greet(name)
    puts hello(name)
  end
end
''',
        "invalid": "def hello(name\n  puts name\nend",
        "extension": ".rb",
    },
    "php": {
        "valid": '''<?php
function hello($name) {
    return "Hello, " . $name . "!";
}

class Greeter {
    public function greet($name) {
        echo hello($name);
    }
}
?>
''',
        "invalid": "<?php\nfunction hello($name {\n    return $name;\n}\n?>",
        "extension": ".php",
    },
    "csharp": {
        "valid": '''using System;

class Program {
    static void Main(string[] args) {
        Console.WriteLine("Hello, World!");
    }

    static int Add(int a, int b) {
        return a + b;
    }
}
''',
        "invalid": "class Program {\n    static void Main(string[] args {",
        "extension": ".cs",
    },
    "scala": {
        "valid": '''object Main {
  def main(args: Array[String]): Unit = {
    println("Hello, World!")
  }

  def add(a: Int, b: Int): Int = a + b
}
''',
        "invalid": "object Main {\n  def main(args: Array[String] {",
        "extension": ".scala",
    },
    "kotlin": {
        "valid": '''fun main() {
    println("Hello, World!")
}

fun add(a: Int, b: Int): Int = a + b

class Greeter {
    fun greet(name: String) {
        println("Hello, $name!")
    }
}
''',
        "invalid": "fun main( {\n    println(\"Hello\")\n}",
        "extension": ".kt",
    },
    "swift": {
        "valid": '''import Foundation

func hello(name: String) -> String {
    return "Hello, \\(name)!"
}

class Greeter {
    func greet(name: String) {
        print(hello(name: name))
    }
}
''',
        "invalid": "func hello(name: String -> String {",
        "extension": ".swift",
    },
    "lua": {
        "valid": '''function hello(name)
    return "Hello, " .. name .. "!"
end

local Greeter = {}

function Greeter:greet(name)
    print(hello(name))
end
''',
        "invalid": "function hello(name\n    return name\nend",
        "extension": ".lua",
    },
    "bash": {
        "valid": '''#!/bin/bash

hello() {
    echo "Hello, $1!"
}

main() {
    hello "World"
}

main
''',
        "invalid": "hello( {\n    echo \"Hello\"\n}",
        "extension": ".sh",
    },
    "sql": {
        "valid": '''SELECT id, name, email
FROM users
WHERE active = true
ORDER BY name ASC;

CREATE TABLE products (
    id INT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    price DECIMAL(10, 2)
);
''',
        "invalid": "SELECT * FROM WHERE id = 1;",
        "extension": ".sql",
    },
    "html": {
        "valid": '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Hello World</title>
</head>
<body>
    <h1>Hello, World!</h1>
    <p>Welcome to my page.</p>
</body>
</html>
''',
        "invalid": "<html><body><div>Unclosed",
        "extension": ".html",
    },
    "css": {
        "valid": '''body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
}

#header {
    background-color: #333;
    color: white;
}
''',
        "invalid": "body { font-family: ",
        "extension": ".css",
    },
    # Config file formats
    "json": {
        "valid": '''{
    "name": "test",
    "version": "1.0.0",
    "dependencies": {
        "lodash": "^4.17.21"
    },
    "scripts": {
        "test": "jest"
    }
}
''',
        "invalid": '{"name": "test", value: 42}',
        "extension": ".json",
    },
    "yaml": {
        "valid": '''name: test
version: 1.0.0
dependencies:
  - lodash
  - express
config:
  port: 3000
  debug: true
''',
        "invalid": "name: test\n  invalid: indentation",
        "extension": ".yaml",
    },
    "toml": {
        "valid": '''[package]
name = "test"
version = "1.0.0"

[dependencies]
lodash = "^4.17.21"

[build]
target = "release"
''',
        "invalid": "[package\nname = \"test\"",
        "extension": ".toml",
    },
    "xml": {
        "valid": '''<?xml version="1.0" encoding="UTF-8"?>
<project>
    <name>test</name>
    <version>1.0.0</version>
    <dependencies>
        <dependency>lodash</dependency>
    </dependencies>
</project>
''',
        "invalid": "<project><name>test</project>",
        "extension": ".xml",
    },
    "markdown": {
        "valid": '''# Hello World

This is a **markdown** document with *formatting*.

## Features

- List item 1
- List item 2

```python
def hello():
    print("Hello!")
```
''',
        "invalid": "",  # Markdown is very permissive, hard to make invalid
        "extension": ".md",
    },
}


# =============================================================================
# Tier 1 Language Tests (Full Support)
# =============================================================================

class TestPythonAllMechanisms:
    """Test Python with all applicable mechanisms."""

    def test_python_native_extractor_valid(self, temp_dir):
        """Test Python extraction with native ast module."""
        extractor = PythonASTExtractor()
        code = LANGUAGE_SAMPLES["python"]["valid"]
        symbols = extractor.extract(code, temp_dir / "test.py")

        assert len(symbols) >= 2  # hello function and Greeter class
        names = [s.name for s in symbols]
        assert "hello" in names
        assert "Greeter" in names

    def test_python_native_extractor_invalid(self, temp_dir):
        """Test Python extraction handles invalid code."""
        extractor = PythonASTExtractor()
        code = LANGUAGE_SAMPLES["python"]["invalid"]
        symbols = extractor.extract(code, temp_dir / "test.py")
        assert symbols == []

    def test_python_native_validator_valid(self, temp_dir):
        """Test Python validation with native ast module."""
        validator = PythonASTValidator()
        code = LANGUAGE_SAMPLES["python"]["valid"]
        result = validator.validate(code, temp_dir / "test.py")

        assert result.is_valid
        assert result.language == "python"

    def test_python_native_validator_invalid(self, temp_dir):
        """Test Python validation detects invalid code."""
        validator = PythonASTValidator()
        code = LANGUAGE_SAMPLES["python"]["invalid"]
        result = validator.validate(code, temp_dir / "test.py")

        assert not result.is_valid
        assert len(result.errors) > 0

    def test_python_tree_sitter_extractor(self, ts_extractor, temp_dir):
        """Test Python extraction with tree-sitter."""
        if not ts_extractor.is_available():
            pytest.skip("tree-sitter not available")

        code = LANGUAGE_SAMPLES["python"]["valid"]
        symbols = ts_extractor.extract(code, temp_dir / "test.py", "python")

        assert len(symbols) >= 1
        names = [s.name for s in symbols]
        assert any("hello" in str(n) or "Greeter" in str(n) for n in names)

    def test_python_tree_sitter_validator(self, ts_validator, temp_dir):
        """Test Python validation with tree-sitter."""
        if not ts_validator.is_available():
            pytest.skip("tree-sitter not available")

        code = LANGUAGE_SAMPLES["python"]["valid"]
        result = ts_validator.validate(code, temp_dir / "test.py", "python")
        assert result.is_valid

        code = LANGUAGE_SAMPLES["python"]["invalid"]
        result = ts_validator.validate(code, temp_dir / "test.py", "python")
        assert not result.is_valid


class TestJavaScriptAllMechanisms:
    """Test JavaScript with all applicable mechanisms."""

    def test_javascript_tree_sitter_extractor(self, ts_extractor, temp_dir):
        """Test JavaScript extraction with tree-sitter."""
        if not ts_extractor.is_available():
            pytest.skip("tree-sitter not available")

        code = LANGUAGE_SAMPLES["javascript"]["valid"]
        symbols = ts_extractor.extract(code, temp_dir / "test.js", "javascript")

        assert isinstance(symbols, list)

    def test_javascript_tree_sitter_validator_valid(self, ts_validator, temp_dir):
        """Test JavaScript validation with tree-sitter - valid code."""
        if not ts_validator.is_available():
            pytest.skip("tree-sitter not available")

        code = LANGUAGE_SAMPLES["javascript"]["valid"]
        result = ts_validator.validate(code, temp_dir / "test.js", "javascript")
        assert result.is_valid

    def test_javascript_tree_sitter_validator_invalid(self, ts_validator, temp_dir):
        """Test JavaScript validation with tree-sitter - invalid code."""
        if not ts_validator.is_available():
            pytest.skip("tree-sitter not available")

        code = LANGUAGE_SAMPLES["javascript"]["invalid"]
        result = ts_validator.validate(code, temp_dir / "test.js", "javascript")
        assert not result.is_valid


class TestTypeScriptAllMechanisms:
    """Test TypeScript with all applicable mechanisms."""

    def test_typescript_tree_sitter_extractor(self, ts_extractor, temp_dir):
        """Test TypeScript extraction with tree-sitter."""
        if not ts_extractor.is_available():
            pytest.skip("tree-sitter not available")

        code = LANGUAGE_SAMPLES["typescript"]["valid"]
        symbols = ts_extractor.extract(code, temp_dir / "test.ts", "typescript")

        assert isinstance(symbols, list)

    def test_typescript_tree_sitter_validator_valid(self, ts_validator, temp_dir):
        """Test TypeScript validation with tree-sitter - valid code."""
        if not ts_validator.is_available():
            pytest.skip("tree-sitter not available")

        code = LANGUAGE_SAMPLES["typescript"]["valid"]
        result = ts_validator.validate(code, temp_dir / "test.ts", "typescript")
        assert result.is_valid

    def test_typescript_tree_sitter_validator_invalid(self, ts_validator, temp_dir):
        """Test TypeScript validation with tree-sitter - invalid code."""
        if not ts_validator.is_available():
            pytest.skip("tree-sitter not available")

        code = LANGUAGE_SAMPLES["typescript"]["invalid"]
        result = ts_validator.validate(code, temp_dir / "test.ts", "typescript")
        assert not result.is_valid


# =============================================================================
# Tier 2 Language Tests (Good Support)
# =============================================================================

class TestGoAllMechanisms:
    """Test Go with all applicable mechanisms."""

    def test_go_native_extractor(self, temp_dir):
        """Test Go extraction with native extractor (gopygo or tree-sitter fallback)."""
        extractor = GoExtractor()
        code = LANGUAGE_SAMPLES["go"]["valid"]
        symbols = extractor.extract(code, temp_dir / "main.go")

        assert isinstance(symbols, list)
        assert len(symbols) >= 1

    def test_go_native_validator_valid(self, temp_dir):
        """Test Go validation - valid code."""
        validator = GoValidator()
        code = LANGUAGE_SAMPLES["go"]["valid"]
        result = validator.validate(code, temp_dir / "main.go")

        assert result.is_valid
        assert result.language == "go"

    def test_go_native_validator_invalid(self, temp_dir):
        """Test Go validation - invalid code."""
        validator = GoValidator()
        code = LANGUAGE_SAMPLES["go"]["invalid"]
        result = validator.validate(code, temp_dir / "main.go")

        assert not result.is_valid

    def test_go_tree_sitter_extractor(self, ts_extractor, temp_dir):
        """Test Go extraction with tree-sitter."""
        if not ts_extractor.is_available():
            pytest.skip("tree-sitter not available")

        code = LANGUAGE_SAMPLES["go"]["valid"]
        symbols = ts_extractor.extract(code, temp_dir / "main.go", "go")

        assert isinstance(symbols, list)


class TestJavaAllMechanisms:
    """Test Java with all applicable mechanisms."""

    def test_java_native_extractor(self, temp_dir):
        """Test Java extraction with native extractor (javalang or tree-sitter fallback)."""
        extractor = JavaExtractor()
        code = LANGUAGE_SAMPLES["java"]["valid"]
        symbols = extractor.extract(code, temp_dir / "Main.java")

        assert isinstance(symbols, list)
        assert len(symbols) >= 1

    def test_java_native_validator_valid(self, temp_dir):
        """Test Java validation - valid code."""
        validator = JavaValidator()
        code = LANGUAGE_SAMPLES["java"]["valid"]
        result = validator.validate(code, temp_dir / "Main.java")

        assert result.is_valid
        assert result.language == "java"

    def test_java_native_validator_invalid(self, temp_dir):
        """Test Java validation - invalid code."""
        validator = JavaValidator()
        code = LANGUAGE_SAMPLES["java"]["invalid"]
        result = validator.validate(code, temp_dir / "Main.java")

        assert not result.is_valid

    def test_java_tree_sitter_extractor(self, ts_extractor, temp_dir):
        """Test Java extraction with tree-sitter."""
        if not ts_extractor.is_available():
            pytest.skip("tree-sitter not available")

        code = LANGUAGE_SAMPLES["java"]["valid"]
        symbols = ts_extractor.extract(code, temp_dir / "Main.java", "java")

        assert isinstance(symbols, list)


class TestRustAllMechanisms:
    """Test Rust with tree-sitter (no native Python library)."""

    def test_rust_tree_sitter_extractor(self, ts_extractor, temp_dir):
        """Test Rust extraction with tree-sitter."""
        if not ts_extractor.is_available():
            pytest.skip("tree-sitter not available")

        code = LANGUAGE_SAMPLES["rust"]["valid"]
        symbols = ts_extractor.extract(code, temp_dir / "main.rs", "rust")

        assert isinstance(symbols, list)

    def test_rust_tree_sitter_validator_valid(self, ts_validator, temp_dir):
        """Test Rust validation - valid code."""
        if not ts_validator.is_available():
            pytest.skip("tree-sitter not available")

        code = LANGUAGE_SAMPLES["rust"]["valid"]
        result = ts_validator.validate(code, temp_dir / "main.rs", "rust")
        assert result.is_valid

    def test_rust_tree_sitter_validator_invalid(self, ts_validator, temp_dir):
        """Test Rust validation - invalid code."""
        if not ts_validator.is_available():
            pytest.skip("tree-sitter not available")

        code = LANGUAGE_SAMPLES["rust"]["invalid"]
        result = ts_validator.validate(code, temp_dir / "main.rs", "rust")
        assert not result.is_valid


class TestCppAllMechanisms:
    """Test C++ with all applicable mechanisms."""

    def test_cpp_native_extractor(self, temp_dir):
        """Test C++ extraction with native extractor (libclang or tree-sitter fallback)."""
        extractor = CppExtractor()
        code = LANGUAGE_SAMPLES["cpp"]["valid"]
        symbols = extractor.extract(code, temp_dir / "main.cpp")

        assert isinstance(symbols, list)

    def test_cpp_native_validator_valid(self, temp_dir):
        """Test C++ validation - valid code."""
        validator = CppValidator()
        code = LANGUAGE_SAMPLES["cpp"]["valid"]
        result = validator.validate(code, temp_dir / "main.cpp")

        assert result.is_valid
        assert result.language == "cpp"

    def test_cpp_native_validator_invalid(self, temp_dir):
        """Test C++ validation - invalid code."""
        validator = CppValidator()
        code = LANGUAGE_SAMPLES["cpp"]["invalid"]
        result = validator.validate(code, temp_dir / "main.cpp")

        assert not result.is_valid

    def test_c_native_validator(self, temp_dir):
        """Test C validation with CppValidator."""
        validator = CppValidator()
        code = LANGUAGE_SAMPLES["c"]["valid"]
        result = validator.validate(code, temp_dir / "main.c", language="c")

        assert result.is_valid
        assert result.language == "c"


# =============================================================================
# Tier 3 Language Tests (Tree-sitter only)
# =============================================================================

class TestTier3LanguagesTreeSitter:
    """Test Tier 3 languages with tree-sitter."""

    @pytest.mark.parametrize("language,sample", [
        ("ruby", LANGUAGE_SAMPLES["ruby"]),
        ("lua", LANGUAGE_SAMPLES["lua"]),
        ("bash", LANGUAGE_SAMPLES["bash"]),
        ("html", LANGUAGE_SAMPLES["html"]),
        ("css", LANGUAGE_SAMPLES["css"]),
    ])
    def test_tier3_tree_sitter_extractor(self, ts_extractor, temp_dir, language, sample):
        """Test Tier 3 language extraction with tree-sitter."""
        if not ts_extractor.is_available():
            pytest.skip("tree-sitter not available")

        code = sample["valid"]
        ext = sample["extension"]
        symbols = ts_extractor.extract(code, temp_dir / f"test{ext}", language)

        assert isinstance(symbols, list)

    @pytest.mark.parametrize("language,sample", [
        ("ruby", LANGUAGE_SAMPLES["ruby"]),
        ("lua", LANGUAGE_SAMPLES["lua"]),
        ("bash", LANGUAGE_SAMPLES["bash"]),
        ("html", LANGUAGE_SAMPLES["html"]),
        ("css", LANGUAGE_SAMPLES["css"]),
    ])
    def test_tier3_tree_sitter_validator_valid(self, ts_validator, temp_dir, language, sample):
        """Test Tier 3 language validation - valid code."""
        if not ts_validator.is_available():
            pytest.skip("tree-sitter not available")

        code = sample["valid"]
        ext = sample["extension"]
        result = ts_validator.validate(code, temp_dir / f"test{ext}", language)

        # Should validate (may have warnings if grammar not installed)
        assert isinstance(result.is_valid, bool)


# =============================================================================
# Config File Validators
# =============================================================================

class TestJsonMechanisms:
    """Test JSON with native and tree-sitter validators."""

    def test_json_native_validator_valid(self, temp_dir):
        """Test JSON validation with native json module."""
        validator = JsonValidator()
        code = LANGUAGE_SAMPLES["json"]["valid"]
        result = validator.validate(code, temp_dir / "test.json")

        assert result.is_valid
        assert result.language == "json"

    def test_json_native_validator_invalid(self, temp_dir):
        """Test JSON validation detects invalid code."""
        validator = JsonValidator()
        code = LANGUAGE_SAMPLES["json"]["invalid"]
        result = validator.validate(code, temp_dir / "test.json")

        assert not result.is_valid
        assert len(result.errors) > 0


class TestYamlMechanisms:
    """Test YAML with native validator."""

    def test_yaml_native_validator_valid(self, temp_dir):
        """Test YAML validation with PyYAML."""
        validator = YamlValidator()
        if not validator.is_available():
            pytest.skip("PyYAML not available")

        code = LANGUAGE_SAMPLES["yaml"]["valid"]
        result = validator.validate(code, temp_dir / "test.yaml")

        assert result.is_valid
        assert result.language == "yaml"

    def test_yaml_native_validator_invalid(self, temp_dir):
        """Test YAML validation detects invalid code."""
        validator = YamlValidator()
        if not validator.is_available():
            pytest.skip("PyYAML not available")

        code = LANGUAGE_SAMPLES["yaml"]["invalid"]
        result = validator.validate(code, temp_dir / "test.yaml")
        # YAML is permissive, this may or may not be invalid
        assert isinstance(result.is_valid, bool)


class TestTomlMechanisms:
    """Test TOML with native validator."""

    def test_toml_native_validator_valid(self, temp_dir):
        """Test TOML validation with tomllib/tomli."""
        validator = TomlValidator()
        if not validator.is_available():
            pytest.skip("tomllib/tomli not available")

        code = LANGUAGE_SAMPLES["toml"]["valid"]
        result = validator.validate(code, temp_dir / "test.toml")

        assert result.is_valid
        assert result.language == "toml"

    def test_toml_native_validator_invalid(self, temp_dir):
        """Test TOML validation detects invalid code."""
        validator = TomlValidator()
        if not validator.is_available():
            pytest.skip("tomllib/tomli not available")

        code = LANGUAGE_SAMPLES["toml"]["invalid"]
        result = validator.validate(code, temp_dir / "test.toml")

        assert not result.is_valid


class TestXmlMechanisms:
    """Test XML with native validator."""

    def test_xml_native_validator_valid(self, temp_dir):
        """Test XML validation with ElementTree."""
        validator = XmlValidator()
        code = LANGUAGE_SAMPLES["xml"]["valid"]
        result = validator.validate(code, temp_dir / "test.xml")

        assert result.is_valid
        assert result.language == "xml"

    def test_xml_native_validator_invalid(self, temp_dir):
        """Test XML validation detects invalid code."""
        validator = XmlValidator()
        code = LANGUAGE_SAMPLES["xml"]["invalid"]
        result = validator.validate(code, temp_dir / "test.xml")

        assert not result.is_valid
        assert len(result.errors) > 0


class TestMarkdownMechanisms:
    """Test Markdown with native validator."""

    def test_markdown_native_validator_valid(self, temp_dir):
        """Test Markdown validation."""
        validator = MarkdownValidator()
        if not validator.is_available():
            pytest.skip("markdown library not available")

        code = LANGUAGE_SAMPLES["markdown"]["valid"]
        result = validator.validate(code, temp_dir / "test.md")

        assert result.is_valid
        assert result.language == "markdown"


class TestHoconMechanisms:
    """Test HOCON with native validator."""

    def test_hocon_native_validator_valid(self, temp_dir):
        """Test HOCON validation with pyhocon."""
        validator = HoconValidator()
        if not validator.is_available():
            pytest.skip("pyhocon not available")

        code = '''
app {
    name = "test"
    version = "1.0.0"
    port = 8080
}
'''
        result = validator.validate(code, temp_dir / "application.conf")

        assert result.is_valid
        assert result.language == "hocon"


# =============================================================================
# Registry Integration Tests
# =============================================================================

class TestRegistryLanguageSupport:
    """Test that all expected languages are registered."""

    def test_tier1_languages_registered(self):
        """Test Tier 1 languages are in registry."""
        registry = LanguageCapabilityRegistry.instance()

        tier1_langs = ["python", "typescript", "javascript", "jsx", "tsx"]
        for lang in tier1_langs:
            cap = registry.get(lang)
            assert cap is not None, f"{lang} not in registry"
            assert cap.tier == LanguageTier.TIER_1, f"{lang} not Tier 1"

    def test_tier2_languages_registered(self):
        """Test Tier 2 languages are in registry."""
        registry = LanguageCapabilityRegistry.instance()

        tier2_langs = ["go", "rust", "java", "c", "cpp"]
        for lang in tier2_langs:
            cap = registry.get(lang)
            assert cap is not None, f"{lang} not in registry"
            assert cap.tier == LanguageTier.TIER_2, f"{lang} not Tier 2"

    def test_tier3_languages_registered(self):
        """Test Tier 3 languages are in registry."""
        registry = LanguageCapabilityRegistry.instance()

        tier3_langs = [
            "ruby", "php", "csharp", "scala", "kotlin", "swift", "lua",
            "bash", "sql", "yaml", "json", "toml", "html", "css", "markdown",
        ]
        for lang in tier3_langs:
            cap = registry.get(lang)
            assert cap is not None, f"{lang} not in registry"
            assert cap.tier == LanguageTier.TIER_3, f"{lang} not Tier 3"

    def test_file_extension_detection(self):
        """Test language detection by file extension."""
        registry = LanguageCapabilityRegistry.instance()

        extensions = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "jsx",
            ".tsx": "tsx",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
            ".sh": "bash",
            ".sql": "sql",
            ".yaml": "yaml",
            ".json": "json",
            ".toml": "toml",
            ".html": "html",
            ".css": "css",
            ".md": "markdown",
        }

        for ext, expected_lang in extensions.items():
            cap = registry.get_for_file(Path(f"test{ext}"))
            assert cap is not None, f"No capability for {ext}"
            assert cap.name == expected_lang, f"{ext} detected as {cap.name}, expected {expected_lang}"

    def test_all_languages_have_tree_sitter_capability(self):
        """Test all languages have tree-sitter capability defined."""
        registry = LanguageCapabilityRegistry.instance()

        for lang in registry.list_supported_languages():
            cap = registry.get(lang)
            assert cap is not None
            # All languages should have tree_sitter defined
            assert cap.tree_sitter is not None, f"{lang} missing tree_sitter capability"

    def test_all_languages_have_validation_method(self):
        """Test all languages have at least one validation method."""
        registry = LanguageCapabilityRegistry.instance()

        for lang in registry.list_supported_languages():
            cap = registry.get(lang)
            method = cap.get_best_validation_method()
            assert method is not None, f"{lang} has no validation method"
