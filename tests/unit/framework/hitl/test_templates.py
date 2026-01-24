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

"""Tests for HITL template system."""

import pytest

from victor.framework.hitl.templates import (
    PromptTemplate,
    PromptTemplateRegistry,
    TemplateContext,
    get_prompt_template,
    get_registry,
    list_templates,
    register_template,
    render_template,
)


# =============================================================================
# TemplateContext Tests
# =============================================================================


class TestTemplateContext:
    """Tests for TemplateContext."""

    def test_empty_context(self):
        """TemplateContext should initialize empty."""
        context = TemplateContext()

        assert context.variables == {}
        assert context.metadata == {}

    def test_with_variable(self):
        """with_variable should add a variable."""
        context = TemplateContext()
        new_context = context.with_variable("key", "value")

        assert new_context.variables == {"key": "value"}
        assert context.variables == {}  # Original unchanged

    def test_with_variables(self):
        """with_variables should add multiple variables."""
        context = TemplateContext()
        new_context = context.with_variables(a=1, b=2)

        assert new_context.variables == {"a": 1, "b": 2}

    def test_merge(self):
        """merge should combine contexts."""
        ctx1 = TemplateContext(variables={"a": 1})
        ctx2 = TemplateContext(variables={"b": 2})

        merged = ctx1.merge(ctx2)

        assert merged.variables == {"a": 1, "b": 2}

    def test_merge_with_overwrite(self):
        """merge should overwrite on conflict."""
        ctx1 = TemplateContext(variables={"a": 1, "b": 2})
        ctx2 = TemplateContext(variables={"b": 3, "c": 4})

        merged = ctx1.merge(ctx2)

        assert merged.variables == {"a": 1, "b": 3, "c": 4}


# =============================================================================
# PromptTemplate Tests
# =============================================================================


class TestPromptTemplate:
    """Tests for PromptTemplate."""

    def test_template_initialization(self):
        """PromptTemplate should initialize correctly."""
        template = PromptTemplate(
            name="test",
            template_string="Hello $name",
        )

        assert template.name == "test"
        assert template.template_string == "Hello $name"

    def test_render_with_kwargs(self):
        """render should substitute variables from kwargs."""
        template = PromptTemplate(
            name="test",
            template_string="Hello $name, welcome to $place",
        )

        result = template.render(name="Alice", place="Wonderland")

        assert result == "Hello Alice, welcome to Wonderland"

    def test_render_with_context(self):
        """render should substitute variables from context."""
        template = PromptTemplate(
            name="test",
            template_string="Action: $action on $target",
        )

        context = TemplateContext(variables={"action": "deploy", "target": "production"})
        result = template.render(context)

        assert result == "Action: deploy on production"

    def test_render_with_both_context_and_kwargs(self):
        """render should merge context and kwargs."""
        template = PromptTemplate(
            name="test",
            template_string="$a + $b = $c",
        )

        context = TemplateContext(variables={"a": 1, "b": 2})
        result = template.render(context, c=3)

        assert result == "1 + 2 = 3"

    def test_render_missing_required_variable(self):
        """render should raise error for missing required variables."""
        template = PromptTemplate(
            name="test",
            template_string="$required_var",
            required_variables=["required_var"],
        )

        with pytest.raises(ValueError, match="Missing required"):
            template.render()

    def test_render_safely(self):
        """render should not fail on unknown variables."""
        template = PromptTemplate(
            name="test",
            template_string="Hello $name, $unknown_var",
        )

        result = template.render(name="Alice")

        assert "Hello Alice" in result
        assert "$unknown_var" in result  # Not substituted

    def test_validate_context_pass(self):
        """validate_context should pass with all required variables."""
        template = PromptTemplate(
            name="test",
            template_string="$a $b",
            required_variables=["a", "b"],
        )

        context = TemplateContext(variables={"a": 1, "b": 2})
        is_valid, error = template.validate_context(context)

        assert is_valid is True
        assert error is None

    def test_validate_context_fail(self):
        """validate_context should fail without required variables."""
        template = PromptTemplate(
            name="test",
            template_string="$a $b",
            required_variables=["a", "b"],
        )

        context = TemplateContext(variables={"a": 1})
        is_valid, error = template.validate_context(context)

        assert is_valid is False
        assert "b" in error

    def test_extract_variables(self):
        """extract_variables should find all template variables."""
        template = PromptTemplate(
            name="test",
            template_string="$var1 and ${var2} but not $var3",
        )

        variables = template.extract_variables()

        assert set(variables) == {"var1", "var2", "var3"}


# =============================================================================
# PromptTemplateRegistry Tests
# =============================================================================


class TestPromptTemplateRegistry:
    """Tests for PromptTemplateRegistry."""

    def test_registry_is_singleton(self):
        """Registry should be a singleton."""
        registry1 = PromptTemplateRegistry()
        registry2 = PromptTemplateRegistry()

        assert registry1 is registry2

    def test_default_templates_registered(self):
        """Registry should have default templates."""
        registry = PromptTemplateRegistry()

        # Check for some default templates
        assert registry.get("approval.default") is not None
        assert registry.get("input.default") is not None
        assert registry.get("choice.default") is not None
        assert registry.get("review.default") is not None
        assert registry.get("confirmation.default") is not None

    def test_register_custom_template(self):
        """register should add custom template."""
        registry = PromptTemplateRegistry()

        template = PromptTemplate(
            name="custom.test",
            template_string="Custom: $value",
        )

        registry.register(template)

        retrieved = registry.get("custom.test")
        assert retrieved is not None
        assert retrieved.template_string == "Custom: $value"

    def test_register_overwrites_existing(self):
        """register should overwrite existing template."""
        registry = PromptTemplateRegistry()

        template1 = PromptTemplate(
            name="test",
            template_string="First: $value",
        )
        template2 = PromptTemplate(
            name="test",
            template_string="Second: $value",
        )

        registry.register(template1)
        registry.register(template2)

        retrieved = registry.get("test")
        assert retrieved.template_string == "Second: $value"

    def test_get_returns_none_for_unknown(self):
        """get should return None for unknown template."""
        registry = PromptTemplateRegistry()

        assert registry.get("unknown.template") is None

    def test_list_all_templates(self):
        """list_templates should return all template names."""
        registry = PromptTemplateRegistry()

        templates = registry.list_templates()

        assert len(templates) > 0
        assert "approval.default" in templates
        assert "input.default" in templates

    def test_list_templates_by_category(self):
        """list_templates should filter by category."""
        registry = PromptTemplateRegistry()

        approval_templates = registry.list_templates(category="approval")

        assert "approval.default" in approval_templates
        assert "input.default" not in approval_templates

    def test_render_by_name(self):
        """render should render template by name."""
        registry = PromptTemplateRegistry()

        result = registry.render(
            "approval.default",
            action="deploy to production",
        )

        assert "deploy to production" in result

    def test_render_unknown_template_raises(self):
        """render unknown template should raise error."""
        registry = PromptTemplateRegistry()

        with pytest.raises(ValueError, match="Template not found"):
            registry.render("unknown.template")

    def test_create_from_string(self):
        """create_from_string should create and register template."""
        registry = PromptTemplateRegistry()

        template = registry.create_from_string(
            name="test.dynamic",
            template_string="Dynamic: $value",
        )

        assert template.name == "test.dynamic"
        assert template.optional_variables == ["value"]

        retrieved = registry.get("test.dynamic")
        assert retrieved is template


# =============================================================================
# Global Registry Functions Tests
# =============================================================================


class TestGlobalRegistryFunctions:
    """Tests for global registry functions."""

    def test_get_registry_returns_instance(self):
        """get_registry should return registry instance."""
        registry = get_registry()

        assert isinstance(registry, PromptTemplateRegistry)

    def test_get_registry_is_singleton(self):
        """get_registry should return same instance."""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2

    def test_register_template_global(self):
        """register_template should add to global registry."""
        template = PromptTemplate(
            name="test.global",
            template_string="Global: $value",
        )

        register_template(template)

        retrieved = get_prompt_template("test.global")
        assert retrieved is not None

    def test_get_prompt_template(self):
        """get_prompt_template should retrieve from global registry."""
        # Default template should exist
        template = get_prompt_template("approval.default")

        assert template is not None
        assert template.name == "approval.default"

    def test_render_template_global(self):
        """render_template should render using global registry."""
        result = render_template(
            "approval.default",
            action="test action",
        )

        assert "test action" in result

    def test_list_templates_global(self):
        """list_templates should list from global registry."""
        templates = list_templates()

        assert len(templates) > 0

    def test_list_templates_by_category_global(self):
        """list_templates should filter by category."""
        approval_templates = list_templates(category="approval")

        assert all(t.startswith("approval.") for t in approval_templates)
        assert len(approval_templates) > 0


# =============================================================================
# Built-in Template Tests
# =============================================================================


class TestBuiltinTemplates:
    """Tests for built-in templates."""

    def test_approval_deployment_template(self):
        """approval.deployment should render with all variables."""
        result = render_template(
            "approval.deployment",
            action="Deploy",
            environment="production",
            version="1.2.3",
        )

        assert "Deploy" in result
        assert "production" in result
        assert "1.2.3" in result

    def test_approval_destructive_template(self):
        """approval.destructive should render with warning."""
        result = render_template(
            "approval.destructive",
            action="Delete",
            target="database",
            risk="HIGH",
        )

        assert "Delete" in result
        assert "database" in result
        assert "HIGH" in result
        assert "CANNOT be undone" in result

    def test_input_deployment_notes_template(self):
        """input.deployment_notes should render with version."""
        result = render_template(
            "input.deployment_notes",
            version="2.0.0",
        )

        assert "2.0.0" in result
        assert "deployment notes" in result.lower()

    def test_review_code_template(self):
        """review.code should render with file and diff."""
        result = render_template(
            "review.code",
            file="src/main.py",
            diff="+ def new_function():\n+     pass",
        )

        assert "src/main.py" in result
        assert "new_function" in result

    def test_confirmation_deployment_template(self):
        """confirmation.deployment should render with deployment info."""
        result = render_template(
            "confirmation.deployment",
            environment="staging",
            version="0.5.0",
            services="web, api, worker",
        )

        assert "staging" in result
        assert "0.5.0" in result
        assert "web, api, worker" in result
