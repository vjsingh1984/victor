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

"""Tests for VerticalBase class - LSP compliance and core functionality."""

import pytest

from victor.core.verticals.base import VerticalBase, VerticalConfig


class ConcreteVertical(VerticalBase):
    """Concrete implementation of VerticalBase for testing."""

    name = "test_vertical"
    description = "A test vertical for unit testing"

    @classmethod
    def get_tools(cls) -> list[str]:
        return ["read", "write", "grep"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are a test assistant."


class TestGetExtensionsLSPCompliance:
    """Test suite for get_extensions() LSP compliance.

    LSP (Liskov Substitution Principle) requires that:
    1. The return type must be consistent (never None if declared as VerticalExtensions)
    2. Subclasses must be substitutable for the base class
    3. All code paths must return valid objects
    """

    def setup_method(self):
        """Clear caches before each test."""
        ConcreteVertical.clear_config_cache(clear_all=True)

    def test_get_extensions_never_returns_none(self):
        """Verify that get_extensions() NEVER returns None.

        This is the primary LSP compliance test. The method must return
        a valid VerticalExtensions object in all scenarios.
        """
        from victor.core.verticals.lazy_extensions import LazyVerticalExtensions
        from victor.core.verticals.protocols import VerticalExtensions

        extensions = ConcreteVertical.get_extensions(use_cache=False)

        # Primary assertion: must not be None
        assert extensions is not None, "get_extensions() must never return None"

        # Type assertion: must be LazyVerticalExtensions (which wraps VerticalExtensions)
        assert isinstance(
            extensions, (LazyVerticalExtensions, VerticalExtensions)
        ), f"Expected LazyVerticalExtensions or VerticalExtensions, got {type(extensions)}"

    def test_get_extensions_returns_valid_object_on_exception(self):
        """Verify get_extensions() returns valid VerticalExtensions even on exceptions.

        This tests the exception handler path to ensure LSP compliance
        when extension getter methods fail.
        """
        from victor.core.verticals.lazy_extensions import LazyVerticalExtensions
        from victor.core.verticals.protocols import VerticalExtensions

        class FailingVertical(VerticalBase):
            """Vertical whose extension getter raises an exception."""

            name = "failing_vertical"
            description = "A vertical that fails during extension loading"

            @classmethod
            def get_tools(cls) -> list[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Failing prompt"

            @classmethod
            def get_safety_extension(cls):
                # Simulate an error during extension loading
                raise RuntimeError("Simulated extension loading failure")

        # Force cache bypass to trigger fresh extension loading
        FailingVertical.clear_config_cache(clear_all=True)

        extensions = FailingVertical.get_extensions(use_cache=False)

        # Must still return a valid object, not None (LSP compliance)
        assert extensions is not None, "get_extensions() returned None on exception - LSP violation"
        assert isinstance(
            extensions, (LazyVerticalExtensions, VerticalExtensions)
        ), f"Expected LazyVerticalExtensions or VerticalExtensions, got {type(extensions)}"

        # Verify the returned extensions have proper default values
        assert extensions.middleware == []
        assert extensions.safety_extensions == []
        assert extensions.prompt_contributors == []

    def test_get_extensions_has_proper_default_values(self):
        """Verify that returned VerticalExtensions has proper default values.

        All fields should have sensible defaults (empty lists, None for optionals).
        """

        extensions = ConcreteVertical.get_extensions(use_cache=False)

        # Verify default values for list fields
        assert isinstance(extensions.middleware, list), "middleware should be a list"
        assert isinstance(extensions.safety_extensions, list), "safety_extensions should be a list"
        assert isinstance(
            extensions.prompt_contributors, list
        ), "prompt_contributors should be a list"

        # Verify optional fields can be None (this is allowed)
        # But the container itself must exist
        assert hasattr(extensions, "mode_config_provider")
        assert hasattr(extensions, "tool_dependency_provider")
        assert hasattr(extensions, "workflow_provider")
        assert hasattr(extensions, "service_provider")
        assert hasattr(extensions, "rl_config_provider")
        assert hasattr(extensions, "team_spec_provider")
        assert hasattr(extensions, "enrichment_strategy")

    def test_get_extensions_caching_works(self):
        """Verify that caching works correctly.

        Note: With lazy loading enabled, each call to get_extensions()
        creates a new LazyVerticalExtensions wrapper. However, the underlying
        extensions are cached and shared between wrappers. This test verifies
        that accessing the extensions triggers loading only once.
        """
        # Disable lazy loading to test object identity caching
        extensions1 = ConcreteVertical.get_extensions(use_cache=True, use_lazy=False)
        # Second call - should return cached
        extensions2 = ConcreteVertical.get_extensions(use_cache=True, use_lazy=False)

        assert extensions1 is extensions2, "Caching should return same object"

    def test_get_extensions_cache_bypass_creates_new_object(self):
        """Verify that use_cache=False creates a new object."""
        # First call
        extensions1 = ConcreteVertical.get_extensions(use_cache=True)
        # Second call with cache bypass
        extensions2 = ConcreteVertical.get_extensions(use_cache=False)

        # Should be different objects (though may have same content)
        assert extensions1 is not extensions2, "Cache bypass should create new object"

    def test_get_extensions_consistent_type_across_subclasses(self):
        """Verify LSP: subclasses return the same type as base class would."""
        from victor.core.verticals.lazy_extensions import LazyVerticalExtensions
        from victor.core.verticals.protocols import VerticalExtensions

        class AnotherVertical(VerticalBase):
            name = "another_vertical"
            description = "Another test vertical"

            @classmethod
            def get_tools(cls) -> list[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Another prompt"

        extensions1 = ConcreteVertical.get_extensions(use_cache=False)
        extensions2 = AnotherVertical.get_extensions(use_cache=False)

        # Both must return LazyVerticalExtensions or VerticalExtensions (LSP)
        assert isinstance(extensions1, (LazyVerticalExtensions, VerticalExtensions))
        assert isinstance(extensions2, (LazyVerticalExtensions, VerticalExtensions))


class TestVerticalBaseConfig:
    """Tests for VerticalBase.get_config() method."""

    def setup_method(self):
        """Clear caches before each test."""
        ConcreteVertical.clear_config_cache(clear_all=True)

    def test_get_config_returns_vertical_config(self):
        """Verify get_config returns a VerticalConfig object."""
        config = ConcreteVertical.get_config()

        assert isinstance(config, VerticalConfig)
        assert config.system_prompt == "You are a test assistant."

    def test_get_config_caching(self):
        """Verify config caching works correctly."""
        config1 = ConcreteVertical.get_config(use_cache=True)
        config2 = ConcreteVertical.get_config(use_cache=True)

        assert config1 is config2, "Cached configs should be same object"


class TestGetCachedExtension:
    """Tests for VerticalBase._get_cached_extension() helper method."""

    def setup_method(self):
        """Clear caches before each test."""
        ConcreteVertical.clear_config_cache(clear_all=True)

    def test_caches_extension_on_first_call(self):
        """Verify that factory is called once and result is cached."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"created": call_count}

        result1 = ConcreteVertical._get_cached_extension("test_key", factory)
        result2 = ConcreteVertical._get_cached_extension("test_key", factory)

        assert call_count == 1, "Factory should be called only once"
        assert result1 is result2, "Should return same cached object"
        assert result1["created"] == 1

    def test_different_keys_cache_separately(self):
        """Verify that different keys maintain separate cache entries."""
        result1 = ConcreteVertical._get_cached_extension("key_a", lambda: "value_a")
        result2 = ConcreteVertical._get_cached_extension("key_b", lambda: "value_b")

        assert result1 == "value_a"
        assert result2 == "value_b"
        assert result1 != result2

    def test_different_verticals_cache_separately(self):
        """Verify that different vertical subclasses have isolated caches."""

        class AnotherVertical(VerticalBase):
            name = "another"
            description = "Another"

            @classmethod
            def get_tools(cls) -> list[str]:
                return []

            @classmethod
            def get_system_prompt(cls) -> str:
                return ""

        # Both use same key but should be cached separately
        result1 = ConcreteVertical._get_cached_extension("shared_key", lambda: "concrete")
        result2 = AnotherVertical._get_cached_extension("shared_key", lambda: "another")

        assert result1 == "concrete"
        assert result2 == "another"

    def test_clear_config_cache_clears_extensions(self):
        """Verify that clear_config_cache() also clears extension cache."""
        # Populate cache
        result1 = ConcreteVertical._get_cached_extension("test_key", lambda: "original")
        assert result1 == "original"

        # Clear cache
        ConcreteVertical.clear_config_cache()

        # Should call factory again
        result2 = ConcreteVertical._get_cached_extension("test_key", lambda: "new_value")
        assert result2 == "new_value"


class TestDefaultStageDefinitions:
    """Tests for VerticalBase default stage definitions."""

    def test_has_seven_stages(self):
        """Verify default implementation has 7 stages."""
        stages = ConcreteVertical.get_stages()
        assert len(stages) == 7

    def test_required_stages_present(self):
        """Verify all required stages are defined."""
        stages = ConcreteVertical.get_stages()
        required = {
            "INITIAL",
            "PLANNING",
            "READING",
            "ANALYSIS",
            "EXECUTION",
            "VERIFICATION",
            "COMPLETION",
        }
        assert set(stages.keys()) == required

    def test_stages_have_comprehensive_keywords(self):
        """Verify stages have comprehensive keyword lists."""
        stages = ConcreteVertical.get_stages()

        # Each stage should have at least 8 keywords
        for stage_name, stage in stages.items():
            assert (
                len(stage.keywords) >= 8
            ), f"{stage_name} has too few keywords: {len(stage.keywords)}"

    def test_stages_have_valid_transitions(self):
        """Verify stage transitions form valid workflow graph."""
        stages = ConcreteVertical.get_stages()

        # All next_stages should reference valid stage names
        for stage_name, stage in stages.items():
            for next_stage in stage.next_stages:
                assert next_stage in stages, f"{stage_name} references invalid stage: {next_stage}"

        # COMPLETION should have no next stages (terminal state)
        assert len(stages["COMPLETION"].next_stages) == 0

        # INITIAL should be able to reach other stages
        assert len(stages["INITIAL"].next_stages) > 0

    def test_stages_have_descriptions(self):
        """Verify all stages have non-empty descriptions."""
        stages = ConcreteVertical.get_stages()

        for stage_name, stage in stages.items():
            assert stage.description, f"{stage_name} has empty description"
            assert len(stage.description) > 20, f"{stage_name} description too short"


class TestStrictExtensionLoading:
    """Tests for strict extension loading in VerticalBase.get_extensions().

    These tests verify the LSP-compliant error handling behavior where:
    - strict=True raises ExtensionLoadError on any failure
    - strict=False collects errors and returns partial extensions
    - required_extensions cause failures even in non-strict mode
    """

    def setup_method(self):
        """Clear caches before each test."""
        ConcreteVertical.clear_config_cache(clear_all=True)

    def test_strict_mode_raises_on_any_error(self):
        """Verify strict=True raises ExtensionLoadError on any extension failure."""
        from victor.core.errors import ExtensionLoadError

        class StrictFailingVertical(VerticalBase):
            """Vertical that fails during extension loading."""

            name = "strict_failing"
            description = "A vertical that fails in strict mode"
            strict_extension_loading = True  # Enable strict mode at class level

            @classmethod
            def get_tools(cls) -> list[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Strict failing prompt"

            @classmethod
            def get_safety_extension(cls):
                raise RuntimeError("Simulated safety extension failure")

        StrictFailingVertical.clear_config_cache(clear_all=True)

        # Disable lazy loading for strict mode tests to ensure immediate error
        with pytest.raises(ExtensionLoadError) as exc_info:
            StrictFailingVertical.get_extensions(use_cache=False, use_lazy=False)

        assert exc_info.value.extension_type == "safety"
        assert exc_info.value.vertical_name == "strict_failing"
        assert "Simulated safety extension failure" in str(exc_info.value.original_error)

    def test_strict_parameter_overrides_class_setting(self):
        """Verify strict parameter overrides class-level strict_extension_loading."""
        from victor.core.errors import ExtensionLoadError

        class NonStrictVertical(VerticalBase):
            """Vertical with strict mode disabled at class level."""

            name = "non_strict"
            description = "A non-strict vertical"
            strict_extension_loading = False  # Disabled at class level

            @classmethod
            def get_tools(cls) -> list[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Non-strict prompt"

            @classmethod
            def get_middleware(cls):
                raise RuntimeError("Middleware failure")

        NonStrictVertical.clear_config_cache(clear_all=True)

        # Should NOT raise because class-level is non-strict
        extensions = NonStrictVertical.get_extensions(use_cache=False)
        assert extensions is not None
        assert extensions.middleware == []

        # Clear cache and test with strict=True override
        NonStrictVertical.clear_config_cache(clear_all=True)

        # Disable lazy loading for strict mode tests to ensure immediate error
        # Should raise because we override with strict=True
        with pytest.raises(ExtensionLoadError) as exc_info:
            NonStrictVertical.get_extensions(use_cache=False, strict=True, use_lazy=False)

        assert exc_info.value.extension_type == "middleware"

    def test_required_extensions_fail_even_in_non_strict_mode(self):
        """Verify required_extensions failures raise even when strict=False."""
        from victor.core.errors import ExtensionLoadError

        class RequiredExtVertical(VerticalBase):
            """Vertical with required extensions."""

            name = "required_ext"
            description = "A vertical with required extensions"
            strict_extension_loading = False  # Non-strict mode
            required_extensions = {"safety"}  # Safety is required

            @classmethod
            def get_tools(cls) -> list[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Required ext prompt"

            @classmethod
            def get_safety_extension(cls):
                raise RuntimeError("Critical safety failure")

        RequiredExtVertical.clear_config_cache(clear_all=True)

        # Disable lazy loading for required extensions test
        with pytest.raises(ExtensionLoadError) as exc_info:
            RequiredExtVertical.get_extensions(use_cache=False, use_lazy=False)

        assert exc_info.value.extension_type == "safety"
        assert exc_info.value.is_required is True

    def test_non_strict_mode_returns_partial_extensions(self):
        """Verify non-strict mode returns partial extensions with failed components empty."""
        from victor.core.verticals.lazy_extensions import LazyVerticalExtensions
        from victor.core.verticals.protocols import VerticalExtensions

        class PartialFailVertical(VerticalBase):
            """Vertical where some extensions fail."""

            name = "partial_fail"
            description = "A vertical with partial failures"
            strict_extension_loading = False

            @classmethod
            def get_tools(cls) -> list[str]:
                return ["read", "write"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Partial fail prompt"

            @classmethod
            def get_middleware(cls):
                # This succeeds
                return []

            @classmethod
            def get_safety_extension(cls):
                # This fails (non-required)
                raise RuntimeError("Safety failure")

            @classmethod
            def get_prompt_contributor(cls):
                # This succeeds
                return None

        PartialFailVertical.clear_config_cache(clear_all=True)

        # Disable lazy loading to get actual VerticalExtensions
        extensions = PartialFailVertical.get_extensions(use_cache=False, use_lazy=False)

        # Should return valid extensions despite the failure
        assert extensions is not None
        assert isinstance(extensions, (LazyVerticalExtensions, VerticalExtensions))

        # Failed extension should have empty/None value
        assert extensions.safety_extensions == []

        # Successful extensions should work normally
        assert extensions.middleware == []

    def test_extension_load_error_has_proper_attributes(self):
        """Verify ExtensionLoadError has all expected attributes."""
        from victor.core.errors import ExtensionLoadError, ErrorCategory, ErrorSeverity

        original = RuntimeError("Original error")
        error = ExtensionLoadError(
            message="Test error",
            extension_type="safety",
            vertical_name="test_vertical",
            original_error=original,
            is_required=True,
        )

        assert error.extension_type == "safety"
        assert error.vertical_name == "test_vertical"
        assert error.original_error is original
        assert error.is_required is True
        assert error.category == ErrorCategory.CONFIG_INVALID
        assert error.severity == ErrorSeverity.CRITICAL  # Critical because is_required=True

        # Check details dict
        assert error.details["extension_type"] == "safety"
        assert error.details["vertical_name"] == "test_vertical"
        assert error.details["is_required"] is True
        assert error.details["original_error_type"] == "RuntimeError"

    def test_non_required_extension_error_has_warning_severity(self):
        """Verify non-required extension errors have WARNING severity."""
        from victor.core.errors import ExtensionLoadError, ErrorSeverity

        error = ExtensionLoadError(
            message="Test error",
            extension_type="enrichment",
            vertical_name="test_vertical",
            original_error=RuntimeError("Test"),
            is_required=False,
        )

        assert error.severity == ErrorSeverity.WARNING

    def test_backward_compatibility_default_non_strict(self):
        """Verify backward compatibility: default is non-strict mode."""
        # ConcreteVertical should have default strict_extension_loading=False
        assert ConcreteVertical.strict_extension_loading is False
        assert ConcreteVertical.required_extensions == set()

    def test_multiple_extension_failures_reports_first_critical(self):
        """Verify that with multiple failures, the first critical one is raised."""
        from victor.core.errors import ExtensionLoadError

        class MultiFailVertical(VerticalBase):
            """Vertical with multiple extension failures."""

            name = "multi_fail"
            description = "Multiple failures"
            strict_extension_loading = True  # All failures are critical

            @classmethod
            def get_tools(cls) -> list[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Multi fail"

            @classmethod
            def get_middleware(cls):
                raise RuntimeError("Middleware failure")

            @classmethod
            def get_safety_extension(cls):
                raise RuntimeError("Safety failure")

        MultiFailVertical.clear_config_cache(clear_all=True)

        # Disable lazy loading for strict mode test
        with pytest.raises(ExtensionLoadError) as exc_info:
            MultiFailVertical.get_extensions(use_cache=False, use_lazy=False)

        # First extension in loading order should be the one raised
        # (middleware is loaded before safety based on the order in get_extensions)
        assert exc_info.value.extension_type == "middleware"


class TestIntegrationResultExtensionErrors:
    """Tests for IntegrationResult extension error tracking."""

    def test_add_extension_error_creates_error_info(self):
        """Verify add_extension_error creates proper ExtensionLoadErrorInfo."""
        from victor.framework.vertical_integration import IntegrationResult

        result = IntegrationResult()
        result.add_extension_error(
            extension_type="safety",
            vertical_name="test_vertical",
            error_message="Test error",
            is_required=False,
            original_exception=RuntimeError("Original"),
        )

        assert len(result.extension_errors) == 1
        error = result.extension_errors[0]
        assert error.extension_type == "safety"
        assert error.vertical_name == "test_vertical"
        assert error.error_message == "Test error"
        assert error.is_required is False
        assert error.original_exception_type == "RuntimeError"

    def test_required_extension_error_sets_failed_status(self):
        """Verify required extension errors set validation_status to failed."""
        from victor.framework.vertical_integration import IntegrationResult

        result = IntegrationResult()
        assert result.validation_status == "success"

        result.add_extension_error(
            extension_type="safety",
            vertical_name="test_vertical",
            error_message="Critical failure",
            is_required=True,
        )

        assert result.success is False
        assert result.validation_status == "failed"

    def test_non_required_extension_error_sets_partial_status(self):
        """Verify non-required extension errors set validation_status to partial."""
        from victor.framework.vertical_integration import IntegrationResult

        result = IntegrationResult()
        assert result.validation_status == "success"

        result.add_extension_error(
            extension_type="enrichment",
            vertical_name="test_vertical",
            error_message="Non-critical failure",
            is_required=False,
        )

        assert result.success is True  # Still success for non-required
        assert result.validation_status == "partial"

    def test_to_dict_includes_extension_errors(self):
        """Verify to_dict includes extension_errors and validation_status."""
        from victor.framework.vertical_integration import IntegrationResult

        result = IntegrationResult()
        result.add_extension_error(
            extension_type="safety",
            vertical_name="test_vertical",
            error_message="Test error",
            is_required=False,
        )

        data = result.to_dict()

        assert "validation_status" in data
        assert data["validation_status"] == "partial"
        assert "extension_errors" in data
        assert len(data["extension_errors"]) == 1
        assert data["extension_errors"][0]["extension_type"] == "safety"

    def test_from_dict_restores_extension_errors(self):
        """Verify from_dict properly restores extension_errors."""
        from victor.framework.vertical_integration import IntegrationResult

        original = IntegrationResult()
        original.add_extension_error(
            extension_type="middleware",
            vertical_name="test_vertical",
            error_message="Middleware error",
            is_required=True,
        )

        data = original.to_dict()
        restored = IntegrationResult.from_dict(data)

        assert len(restored.extension_errors) == 1
        assert restored.extension_errors[0].extension_type == "middleware"
        assert restored.extension_errors[0].is_required is True
        assert restored.validation_status == "failed"

    def test_has_extension_errors_method(self):
        """Verify has_extension_errors method works correctly."""
        from victor.framework.vertical_integration import IntegrationResult

        result = IntegrationResult()
        assert result.has_extension_errors() is False

        result.add_extension_error(
            extension_type="safety",
            vertical_name="test",
            error_message="Error",
        )
        assert result.has_extension_errors() is True

    def test_get_required_extension_failures_method(self):
        """Verify get_required_extension_failures filters correctly."""
        from victor.framework.vertical_integration import IntegrationResult

        result = IntegrationResult()
        result.add_extension_error(
            extension_type="safety",
            vertical_name="test",
            error_message="Required failure",
            is_required=True,
        )
        result.add_extension_error(
            extension_type="enrichment",
            vertical_name="test",
            error_message="Optional failure",
            is_required=False,
        )

        required = result.get_required_extension_failures()
        assert len(required) == 1
        assert required[0].extension_type == "safety"
