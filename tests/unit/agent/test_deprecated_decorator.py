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

"""Tests for the @deprecated decorator."""

import warnings

from victor.agent.decorators import deprecated, deprecated_property, deprecated_class


class TestDeprecatedDecorator:
    """Tests for the @deprecated decorator."""

    def test_deprecated_function_issues_warning(self):
        """Test that @deprecated decorator issues a DeprecationWarning."""

        @deprecated(version="0.5.0", replacement="new_func()")
        def old_func():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()

            # Check that a warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "old_func is deprecated" in str(w[0].message)
            assert "0.5.0" in str(w[0].message)
            assert "new_func()" in str(w[0].message)

            # Check function still works
            assert result == "result"

    def test_deprecated_with_remove_version(self):
        """Test that remove_version is included in warning."""

        @deprecated(version="0.5.0", remove_version="0.7.0")
        def old_func():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_func()

            assert len(w) == 1
            assert "will be removed in 0.7.0" in str(w[0].message)

    def test_deprecated_with_reason(self):
        """Test that reason is included in warning."""

        @deprecated(version="0.5.0", reason="This is obsolete")
        def old_func():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_func()

            assert len(w) == 1
            assert "This is obsolete" in str(w[0].message)

    def test_deprecated_method(self):
        """Test that @deprecated works on methods."""

        class TestClass:
            @deprecated(version="0.5.0", replacement="new_method()")
            def old_method(self):
                return "result"

        obj = TestClass()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = obj.old_method()

            assert len(w) == 1
            # The decorator may or may not include the class name depending on implementation
            assert "old_method is deprecated" in str(w[0].message)
            assert result == "result"

    def test_deprecated_preserves_function_signature(self):
        """Test that @deprecated preserves original function signature."""

        @deprecated(version="0.5.0")
        def func_with_args(a: int, b: str = "default") -> str:
            return f"{a}:{b}"

        # Function should work normally
        assert func_with_args(1) == "1:default"
        assert func_with_args(2, "custom") == "2:custom"

        # Name and docstring should be preserved
        assert func_with_args.__name__ == "func_with_args"


class TestDeprecatedProperty:
    """Tests for the @deprecated_property decorator."""

    def test_deprecated_property_issues_warning(self):
        """Test that @deprecated_property issues warning on access."""

        class TestClass:
            @deprecated_property(version="0.5.0", replacement="new_attr", remove_version="0.7.0")
            def old_attr(self):
                return "value"

        obj = TestClass()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = obj.old_attr

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Property 'old_attr' is deprecated" in str(w[0].message)
            assert "use 'new_attr' instead" in str(w[0].message)
            assert "will be removed in 0.7.0" in str(w[0].message)
            assert result == "value"


class TestDeprecatedClass:
    """Tests for the @deprecated_class decorator."""

    def test_deprecated_class_issues_warning_on_instantiation(self):
        """Test that @deprecated_class issues warning when instantiated."""

        @deprecated_class(version="0.5.0", replacement="NewClass", remove_version="0.7.0")
        class OldClass:
            def __init__(self):
                self.value = "initialized"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = OldClass()

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Class 'OldClass' is deprecated" in str(w[0].message)
            assert "use 'NewClass' instead" in str(w[0].message)
            assert obj.value == "initialized"

    def test_deprecated_class_no_warning_on_import(self):
        """Test that @deprecated_class doesn't warn on import."""

        @deprecated_class(version="0.5.0")
        class OldClass:
            pass

        # Importing/defining the class should not issue a warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Just reference the class, don't instantiate
            cls = OldClass
            assert len(w) == 0
