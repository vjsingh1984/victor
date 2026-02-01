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

"""Unit tests for PromptBuilderTemplate.

TDD-first tests for Phase 2.3: Create PromptBuilderTemplate.
These tests verify:
1. Template Method pattern for consistent prompt structure
2. Hook methods for vertical customization
3. Common prompt patterns (grounding, rules, checklist)
4. Integration with existing PromptBuilder
"""



class TestPromptBuilderTemplateExists:
    """Tests for PromptBuilderTemplate module existence."""

    def test_template_class_exists(self):
        """PromptBuilderTemplate class should exist."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        assert PromptBuilderTemplate is not None

    def test_template_is_importable_from_framework(self):
        """Template should be importable from framework package."""
        from victor.framework import PromptBuilderTemplate

        assert PromptBuilderTemplate is not None


class TestPromptBuilderTemplateBasics:
    """Tests for basic PromptBuilderTemplate functionality."""

    def test_template_has_build_method(self):
        """Template should have build() method."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        template = PromptBuilderTemplate()
        assert hasattr(template, "build")
        assert callable(template.build)

    def test_template_has_get_prompt_builder_method(self):
        """Template should have get_prompt_builder() method."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        template = PromptBuilderTemplate()
        assert hasattr(template, "get_prompt_builder")
        assert callable(template.get_prompt_builder)

    def test_get_prompt_builder_returns_builder(self):
        """get_prompt_builder() should return a PromptBuilder."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate
        from victor.framework.prompt_builder import PromptBuilder

        template = PromptBuilderTemplate()
        builder = template.get_prompt_builder()

        assert isinstance(builder, PromptBuilder)


class TestTemplateHookMethods:
    """Tests for Template Method pattern hook methods."""

    def test_has_get_grounding_hook(self):
        """Template should have get_grounding() hook method."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        template = PromptBuilderTemplate()
        assert hasattr(template, "get_grounding")
        assert callable(template.get_grounding)

    def test_has_get_rules_hook(self):
        """Template should have get_rules() hook method."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        template = PromptBuilderTemplate()
        assert hasattr(template, "get_rules")
        assert callable(template.get_rules)

    def test_has_get_checklist_hook(self):
        """Template should have get_checklist() hook method."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        template = PromptBuilderTemplate()
        assert hasattr(template, "get_checklist")
        assert callable(template.get_checklist)

    def test_has_get_vertical_prompt_hook(self):
        """Template should have get_vertical_prompt() hook method."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        template = PromptBuilderTemplate()
        assert hasattr(template, "get_vertical_prompt")
        assert callable(template.get_vertical_prompt)

    def test_default_grounding_returns_none(self):
        """Default get_grounding() should return None."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        template = PromptBuilderTemplate()
        result = template.get_grounding()

        assert result is None

    def test_default_rules_returns_empty_list(self):
        """Default get_rules() should return empty list."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        template = PromptBuilderTemplate()
        result = template.get_rules()

        assert result == []

    def test_default_checklist_returns_empty_list(self):
        """Default get_checklist() should return empty list."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        template = PromptBuilderTemplate()
        result = template.get_checklist()

        assert result == []

    def test_default_vertical_prompt_returns_empty(self):
        """Default get_vertical_prompt() should return empty string."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        template = PromptBuilderTemplate()
        result = template.get_vertical_prompt()

        assert result == ""


class TestTemplateSubclassing:
    """Tests for subclassing PromptBuilderTemplate."""

    def test_subclass_can_override_grounding(self):
        """Subclass can override get_grounding()."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        class CustomTemplate(PromptBuilderTemplate):
            def get_grounding(self):
                return {
                    "template": "Context: Working on {project}.",
                    "variables": {"project": "my project"},
                    "priority": 10,
                }

        template = CustomTemplate()
        result = template.get_grounding()

        assert result is not None
        assert "template" in result
        assert "variables" in result
        assert result["priority"] == 10

    def test_subclass_can_override_rules(self):
        """Subclass can override get_rules()."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        class CustomTemplate(PromptBuilderTemplate):
            def get_rules(self):
                return [
                    "Rule 1: Follow coding standards",
                    "Rule 2: Write tests",
                ]

        template = CustomTemplate()
        result = template.get_rules()

        assert len(result) == 2
        assert "Rule 1" in result[0]

    def test_subclass_can_override_checklist(self):
        """Subclass can override get_checklist()."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        class CustomTemplate(PromptBuilderTemplate):
            def get_checklist(self):
                return [
                    "[ ] Review code",
                    "[ ] Run tests",
                    "[ ] Update documentation",
                ]

        template = CustomTemplate()
        result = template.get_checklist()

        assert len(result) == 3

    def test_subclass_can_override_vertical_prompt(self):
        """Subclass can override get_vertical_prompt()."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        class CustomTemplate(PromptBuilderTemplate):
            def get_vertical_prompt(self):
                return "You are an expert code reviewer."

        template = CustomTemplate()
        result = template.get_vertical_prompt()

        assert "expert" in result


class TestTemplateBuilderIntegration:
    """Tests for PromptBuilder integration."""

    def test_build_returns_string(self):
        """build() should return a string prompt."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        template = PromptBuilderTemplate()
        result = template.build()

        assert isinstance(result, str)

    def test_build_includes_grounding_when_set(self):
        """build() should include grounding content."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        class CustomTemplate(PromptBuilderTemplate):
            def get_grounding(self):
                return {
                    "template": "Context: Working on {project}.",
                    "variables": {"project": "TestProject"},
                    "priority": 10,
                }

        template = CustomTemplate()
        result = template.build()

        assert "TestProject" in result

    def test_build_includes_rules_when_set(self):
        """build() should include rules content."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        class CustomTemplate(PromptBuilderTemplate):
            def get_rules(self):
                return ["Always test your code"]

        template = CustomTemplate()
        result = template.build()

        assert "test your code" in result

    def test_build_includes_checklist_when_set(self):
        """build() should include checklist content."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        class CustomTemplate(PromptBuilderTemplate):
            def get_checklist(self):
                return ["Review changes"]

        template = CustomTemplate()
        result = template.build()

        assert "Review" in result

    def test_build_includes_vertical_prompt_when_set(self):
        """build() should include vertical prompt content."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        class CustomTemplate(PromptBuilderTemplate):
            vertical_name = "testing"

            def get_vertical_prompt(self):
                return "You are a testing expert."

        template = CustomTemplate()
        result = template.build()

        assert "testing expert" in result


class TestTemplateWithVerticalName:
    """Tests for vertical name integration."""

    def test_template_has_vertical_name_attribute(self):
        """Template should have vertical_name attribute."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        template = PromptBuilderTemplate()
        assert hasattr(template, "vertical_name")

    def test_default_vertical_name_is_generic(self):
        """Default vertical_name should be 'generic'."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        template = PromptBuilderTemplate()
        assert template.vertical_name == "generic"

    def test_subclass_can_set_vertical_name(self):
        """Subclass can set custom vertical_name."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        class CodingTemplate(PromptBuilderTemplate):
            vertical_name = "coding"

        template = CodingTemplate()
        assert template.vertical_name == "coding"


class TestTemplatePriorities:
    """Tests for section priority handling."""

    def test_template_has_default_priorities(self):
        """Template should define default section priorities."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        assert hasattr(PromptBuilderTemplate, "DEFAULT_GROUNDING_PRIORITY")
        assert hasattr(PromptBuilderTemplate, "DEFAULT_RULES_PRIORITY")
        assert hasattr(PromptBuilderTemplate, "DEFAULT_CHECKLIST_PRIORITY")
        assert hasattr(PromptBuilderTemplate, "DEFAULT_VERTICAL_PRIORITY")

    def test_grounding_has_high_priority(self):
        """Grounding should have higher priority (lower number)."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        assert PromptBuilderTemplate.DEFAULT_GROUNDING_PRIORITY < 50

    def test_vertical_has_medium_priority(self):
        """Vertical section should have medium priority."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        grounding = PromptBuilderTemplate.DEFAULT_GROUNDING_PRIORITY
        vertical = PromptBuilderTemplate.DEFAULT_VERTICAL_PRIORITY

        assert vertical > grounding  # Grounding comes first


class TestTemplateHookConfiguration:
    """Tests for configurable hook methods."""

    def test_get_grounding_can_return_dict_with_priority(self):
        """get_grounding() can return dict with custom priority."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        class CustomTemplate(PromptBuilderTemplate):
            def get_grounding(self):
                return {
                    "template": "Context: {ctx}",
                    "variables": {"ctx": "test"},
                    "priority": 5,
                }

        template = CustomTemplate()
        grounding = template.get_grounding()

        assert grounding["priority"] == 5

    def test_get_rules_can_return_rules_with_priority(self):
        """get_rules() can return tuple of (rules, priority)."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        class CustomTemplate(PromptBuilderTemplate):
            def get_rules(self):
                return ["Rule 1", "Rule 2"]

            def get_rules_priority(self):
                return 15

        template = CustomTemplate()
        assert hasattr(template, "get_rules_priority")


class TestTemplateExtensibility:
    """Tests for template extensibility."""

    def test_template_has_pre_build_hook(self):
        """Template should have pre_build() hook for customization."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        template = PromptBuilderTemplate()
        assert hasattr(template, "pre_build")
        assert callable(template.pre_build)

    def test_template_has_post_build_hook(self):
        """Template should have post_build() hook for customization."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        template = PromptBuilderTemplate()
        assert hasattr(template, "post_build")
        assert callable(template.post_build)

    def test_pre_build_can_modify_builder(self):
        """pre_build() can add additional sections to builder."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        class CustomTemplate(PromptBuilderTemplate):
            def pre_build(self, builder):
                builder.add_section("custom", "Custom content", priority=5)
                return builder

        template = CustomTemplate()
        result = template.build()

        assert "Custom content" in result

    def test_post_build_can_modify_result(self):
        """post_build() can modify the final prompt string."""
        from victor.framework.prompt_builder_template import PromptBuilderTemplate

        class CustomTemplate(PromptBuilderTemplate):
            def get_vertical_prompt(self):
                return "Test content"

            def post_build(self, prompt: str) -> str:
                return prompt + "\n\n[END OF PROMPT]"

        template = CustomTemplate()
        result = template.build()

        assert result.endswith("[END OF PROMPT]")
