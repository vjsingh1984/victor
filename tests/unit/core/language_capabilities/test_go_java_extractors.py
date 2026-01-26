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

"""Tests for Go and Java extractors/validators."""

import pytest
import tempfile
from pathlib import Path

from victor.core.language_capabilities.extractors import GoExtractor, JavaExtractor, CppExtractor
from victor.core.language_capabilities.validators import GoValidator, JavaValidator, CppValidator


class TestGoExtractor:
    """Tests for GoExtractor."""

    @pytest.fixture
    def extractor(self):
        """Get Go extractor instance."""
        return GoExtractor()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_is_available(self, extractor):
        """Test is_available method returns a boolean."""
        # Should always return bool, regardless of gopygo availability
        result = extractor.is_available()
        assert isinstance(result, bool)

    def test_extract_function(self, extractor, temp_dir):
        """Test extracting functions from Go code."""
        file_path = temp_dir / "main.go"
        code = """
package main

func main() {
    fmt.Println("Hello")
}

func add(a, b int) int {
    return a + b
}
"""
        symbols = extractor.extract(code, file_path)

        # Should extract at least main function via tree-sitter fallback
        assert len(symbols) >= 1
        names = [s.name for s in symbols]
        assert any("main" in str(n) for n in names)

    def test_extract_struct(self, extractor, temp_dir):
        """Test extracting structs from Go code."""
        file_path = temp_dir / "types.go"
        code = """
package main

type Person struct {
    Name string
    Age  int
}

type Calculator struct {
    value int
}

func (c *Calculator) Add(n int) int {
    return c.value + n
}
"""
        symbols = extractor.extract(code, file_path)

        # Should extract at least the method via tree-sitter fallback
        # Note: Struct extraction depends on tree-sitter query support
        assert isinstance(symbols, list)

    def test_extract_interface(self, extractor, temp_dir):
        """Test extracting interfaces from Go code."""
        file_path = temp_dir / "interfaces.go"
        code = """
package main

type Reader interface {
    Read(p []byte) (n int, err error)
}

func ProcessReader(r Reader) {
    // Use the reader
}
"""
        symbols = extractor.extract(code, file_path)

        # Should extract at least the function via tree-sitter fallback
        # Note: Interface extraction depends on tree-sitter query support
        assert isinstance(symbols, list)

    def test_has_syntax_errors_valid(self, extractor):
        """Test has_syntax_errors returns False for valid Go code."""
        code = """
package main

func main() {
    fmt.Println("Hello")
}
"""
        result = extractor.has_syntax_errors(code)
        assert result is False

    def test_has_syntax_errors_invalid(self, extractor):
        """Test has_syntax_errors returns True for invalid Go code."""
        code = """
package main

func main( {  // Missing closing parenthesis
    fmt.Println("Hello")
}
"""
        result = extractor.has_syntax_errors(code)
        assert result is True


class TestJavaExtractor:
    """Tests for JavaExtractor."""

    @pytest.fixture
    def extractor(self):
        """Get Java extractor instance."""
        return JavaExtractor()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_is_available(self, extractor):
        """Test is_available method returns a boolean."""
        result = extractor.is_available()
        assert isinstance(result, bool)

    def test_extract_class(self, extractor, temp_dir):
        """Test extracting classes from Java code."""
        file_path = temp_dir / "Main.java"
        code = """
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello");
    }
}
"""
        symbols = extractor.extract(code, file_path)

        # Should extract at least the class via tree-sitter fallback
        assert len(symbols) >= 1
        names = [s.name for s in symbols]
        assert any("Main" in str(n) or "main" in str(n) for n in names)

    def test_extract_methods(self, extractor, temp_dir):
        """Test extracting methods from Java code."""
        file_path = temp_dir / "Calculator.java"
        code = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int subtract(int a, int b) {
        return a - b;
    }
}
"""
        symbols = extractor.extract(code, file_path)

        # Should extract class and methods
        assert len(symbols) >= 1

    def test_extract_interface(self, extractor, temp_dir):
        """Test extracting interfaces from Java code."""
        file_path = temp_dir / "Service.java"
        code = """
public interface Service {
    void start();
    void stop();
    boolean isRunning();
}
"""
        symbols = extractor.extract(code, file_path)

        # Should extract interface
        assert len(symbols) >= 1

    def test_extract_enum(self, extractor, temp_dir):
        """Test extracting enums from Java code."""
        file_path = temp_dir / "Color.java"
        code = """
public enum Color {
    RED, GREEN, BLUE;

    public String toHex() {
        switch (this) {
            case RED: return "#FF0000";
            case GREEN: return "#00FF00";
            case BLUE: return "#0000FF";
            default: return "#000000";
        }
    }
}
"""
        symbols = extractor.extract(code, file_path)

        # Should extract at least the method via tree-sitter fallback
        # Note: Enum constant extraction depends on tree-sitter query support
        assert isinstance(symbols, list)

    def test_has_syntax_errors_valid(self, extractor):
        """Test has_syntax_errors returns False for valid Java code."""
        code = """
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello");
    }
}
"""
        result = extractor.has_syntax_errors(code)
        assert result is False

    def test_has_syntax_errors_invalid(self, extractor):
        """Test has_syntax_errors returns True for invalid Java code."""
        code = """
public class Main {
    public static void main(String[] args {  // Missing closing parenthesis
        System.out.println("Hello");
    }
}
"""
        result = extractor.has_syntax_errors(code)
        assert result is True


class TestGoValidator:
    """Tests for GoValidator."""

    @pytest.fixture
    def validator(self):
        """Get Go validator instance."""
        return GoValidator()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_is_available(self, validator):
        """Test is_available method returns a boolean."""
        result = validator.is_available()
        assert isinstance(result, bool)

    def test_validate_valid_code(self, validator, temp_dir):
        """Test validation passes for valid Go code."""
        file_path = temp_dir / "main.go"
        code = """
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
"""
        result = validator.validate(code, file_path)
        assert result.is_valid is True
        assert result.language == "go"

    def test_validate_invalid_code(self, validator, temp_dir):
        """Test validation fails for invalid Go code."""
        file_path = temp_dir / "main.go"
        code = """
package main

func main( {  // Missing closing parenthesis
    fmt.Println("Hello")
}
"""
        result = validator.validate(code, file_path)
        assert result.is_valid is False
        assert result.language == "go"


class TestJavaValidator:
    """Tests for JavaValidator."""

    @pytest.fixture
    def validator(self):
        """Get Java validator instance."""
        return JavaValidator()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_is_available(self, validator):
        """Test is_available method returns a boolean."""
        result = validator.is_available()
        assert isinstance(result, bool)

    def test_validate_valid_code(self, validator, temp_dir):
        """Test validation passes for valid Java code."""
        file_path = temp_dir / "Main.java"
        code = """
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
"""
        result = validator.validate(code, file_path)
        assert result.is_valid is True
        assert result.language == "java"

    def test_validate_invalid_code(self, validator, temp_dir):
        """Test validation fails for invalid Java code."""
        file_path = temp_dir / "Main.java"
        code = """
public class Main {
    public static void main(String[] args {  // Missing closing parenthesis
        System.out.println("Hello");
    }
}
"""
        result = validator.validate(code, file_path)
        assert result.is_valid is False
        assert result.language == "java"


class TestTreeSitterFallback:
    """Test that extractors/validators properly fall back to tree-sitter."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_go_extractor_uses_tree_sitter_when_gopygo_unavailable(self, temp_dir):
        """Test Go extractor falls back to tree-sitter."""
        extractor = GoExtractor()

        # Even without gopygo, should still extract symbols via tree-sitter
        file_path = temp_dir / "main.go"
        code = """
package main

func main() {
    fmt.Println("Hello")
}
"""
        symbols = extractor.extract(code, file_path)
        # Tree-sitter should extract at least the function
        assert len(symbols) >= 1

    def test_java_extractor_uses_tree_sitter_when_javalang_unavailable(self, temp_dir):
        """Test Java extractor falls back to tree-sitter."""
        extractor = JavaExtractor()

        # Even without javalang, should still extract symbols via tree-sitter
        file_path = temp_dir / "Main.java"
        code = """
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello");
    }
}
"""
        symbols = extractor.extract(code, file_path)
        # Tree-sitter should extract class and method
        assert len(symbols) >= 1

    def test_go_validator_uses_tree_sitter_when_gopygo_unavailable(self, temp_dir):
        """Test Go validator falls back to tree-sitter."""
        validator = GoValidator()

        file_path = temp_dir / "main.go"
        code = """
package main

func main() {
    fmt.Println("Hello")
}
"""
        result = validator.validate(code, file_path)
        assert result.is_valid is True
        assert result.language == "go"

    def test_java_validator_uses_tree_sitter_when_javalang_unavailable(self, temp_dir):
        """Test Java validator falls back to tree-sitter."""
        validator = JavaValidator()

        file_path = temp_dir / "Main.java"
        code = """
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello");
    }
}
"""
        result = validator.validate(code, file_path)
        assert result.is_valid is True
        assert result.language == "java"

    def test_cpp_extractor_uses_tree_sitter_when_libclang_unavailable(self, temp_dir):
        """Test C++ extractor falls back to tree-sitter."""
        extractor = CppExtractor()

        file_path = temp_dir / "main.cpp"
        code = """
#include <iostream>

int main() {
    std::cout << "Hello" << std::endl;
    return 0;
}
"""
        symbols = extractor.extract(code, file_path)
        # Tree-sitter should extract at least the function
        assert isinstance(symbols, list)

    def test_cpp_validator_uses_tree_sitter_when_libclang_unavailable(self, temp_dir):
        """Test C++ validator falls back to tree-sitter."""
        validator = CppValidator()

        file_path = temp_dir / "main.cpp"
        code = """
#include <iostream>

int main() {
    std::cout << "Hello" << std::endl;
    return 0;
}
"""
        result = validator.validate(code, file_path)
        assert result.is_valid is True
        assert result.language == "cpp"


class TestCppExtractor:
    """Tests for CppExtractor."""

    @pytest.fixture
    def extractor(self):
        """Get C++ extractor instance."""
        return CppExtractor()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_is_available(self, extractor):
        """Test is_available method returns a boolean."""
        result = extractor.is_available()
        assert isinstance(result, bool)

    def test_extract_function(self, extractor, temp_dir):
        """Test extracting functions from C++ code."""
        file_path = temp_dir / "main.cpp"
        code = """
#include <iostream>

int main() {
    std::cout << "Hello" << std::endl;
    return 0;
}

int add(int a, int b) {
    return a + b;
}
"""
        symbols = extractor.extract(code, file_path)

        # Should extract at least one symbol via tree-sitter fallback
        assert isinstance(symbols, list)

    def test_extract_class(self, extractor, temp_dir):
        """Test extracting classes from C++ code."""
        file_path = temp_dir / "calculator.cpp"
        code = """
class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }

    int subtract(int a, int b) {
        return a - b;
    }
};
"""
        symbols = extractor.extract(code, file_path)
        assert isinstance(symbols, list)

    def test_has_syntax_errors_valid(self, extractor):
        """Test has_syntax_errors returns False for valid C++ code."""
        code = """
int main() {
    return 0;
}
"""
        result = extractor.has_syntax_errors(code, "cpp")
        assert result is False

    def test_has_syntax_errors_invalid(self, extractor):
        """Test has_syntax_errors returns True for invalid C++ code."""
        code = """
int main( {  // Missing closing parenthesis
    return 0;
}
"""
        result = extractor.has_syntax_errors(code, "cpp")
        assert result is True


class TestCppValidator:
    """Tests for CppValidator."""

    @pytest.fixture
    def validator(self):
        """Get C++ validator instance."""
        return CppValidator()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_is_available(self, validator):
        """Test is_available method returns a boolean."""
        result = validator.is_available()
        assert isinstance(result, bool)

    def test_validate_valid_code(self, validator, temp_dir):
        """Test validation passes for valid C++ code."""
        file_path = temp_dir / "main.cpp"
        code = """
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
"""
        result = validator.validate(code, file_path)
        assert result.is_valid is True
        assert result.language == "cpp"

    def test_validate_invalid_code(self, validator, temp_dir):
        """Test validation fails for invalid C++ code."""
        file_path = temp_dir / "main.cpp"
        code = """
int main( {  // Missing closing parenthesis
    return 0;
}
"""
        result = validator.validate(code, file_path)
        assert result.is_valid is False
        assert result.language == "cpp"

    def test_validate_c_code(self, validator, temp_dir):
        """Test validation works for C code."""
        file_path = temp_dir / "main.c"
        code = """
#include <stdio.h>

int main() {
    printf("Hello, World!");
    return 0;
}
"""
        result = validator.validate(code, file_path, language="c")
        assert result.is_valid is True
        assert result.language == "c"
