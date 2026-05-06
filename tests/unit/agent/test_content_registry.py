"""Tests for content registry metadata wiring."""

from victor.agent.content_registry import create_default_registry
from victor.agent.prompt_section_registry import get_section_registry


class TestContentRegistry:
    """Validate alignment with the unified prompt section registry."""

    def test_static_sections_inherit_registry_metadata(self):
        registry = create_default_registry()
        section_registry = get_section_registry()

        for name in [
            "PARALLEL_READ_GUIDANCE",
            "GROUNDING_RULES_EXTENDED",
            "COMPLETION_GUIDANCE",
        ]:
            item = registry.get(name)
            section = section_registry.get(name)

            assert item is not None
            assert section is not None
            assert item.default_text == section.default_text
            assert item.evolvable is section.evolvable
            assert item.required is section.required

    def test_concise_mode_group_uses_router_compatible_name(self):
        registry = create_default_registry()

        item = registry.get("CONCISE_MODE_GUIDANCE")

        assert item is not None
        assert item.section_group == "concise_mode"
