"""Tests for the unified prompt section registry."""

from victor.agent.prompt_section_registry import get_section_registry


def test_registry_exposes_all_core_evolvable_sections() -> None:
    registry = get_section_registry()
    evolvable = registry.get_evolvable_sections()

    assert "ASI_TOOL_EFFECTIVENESS_GUIDANCE" in evolvable
    assert "GROUNDING_RULES" in evolvable
    assert "COMPLETION_GUIDANCE" in evolvable
    assert "CONCISE_MODE_GUIDANCE" in evolvable
    assert "PARALLEL_READ_GUIDANCE" in evolvable
    assert "LARGE_FILE_PAGINATION_GUIDANCE" in evolvable
    assert "GROUNDING_RULES_EXTENDED" in evolvable
    assert "FEW_SHOT_EXAMPLES" in evolvable
    assert "INIT_SYNTHESIS_RULES" in evolvable


def test_registry_resolves_aliases_for_new_prompt_sections() -> None:
    registry = get_section_registry()

    assert registry.get("parallel_reads").name == "PARALLEL_READ_GUIDANCE"
    assert registry.get("tool_output_grounding").name == "GROUNDING_RULES_EXTENDED"
    assert registry.get("synthesis_rules").name == "INIT_SYNTHESIS_RULES"
