"""Tests for the unified prompt section registry."""

from victor.agent import prompt_section_registry as registry_module
from victor.agent.prompt_section_texts import (
    ASI_TOOL_EFFECTIVENESS_GUIDANCE,
    COMPLETION_GUIDANCE,
    CONCISE_MODE_GUIDANCE,
    GROUNDING_RULES,
    GROUNDING_RULES_EXTENDED,
    LARGE_FILE_PAGINATION_GUIDANCE,
    PARALLEL_READ_GUIDANCE,
)
from victor.agent.prompt_section_registry import (
    UnifiedSectionRegistry,
    _initialize_default_sections,
    build_edge_focus_prompt_options_text,
    get_default_section_strategies,
    get_section_registry,
    get_edge_focus_selector_index,
    get_required_evolvable_section_names,
    register_prompt_contributor_sections,
)
from victor.core.verticals.protocols.prompt_provider import PromptSectionContribution


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


def test_registry_exposes_required_evolvable_section_names_in_priority_order() -> None:
    assert get_required_evolvable_section_names() == [
        "ASI_TOOL_EFFECTIVENESS_GUIDANCE",
        "COMPLETION_GUIDANCE",
        "GROUNDING_RULES",
    ]


def test_registry_resolves_aliases_for_new_prompt_sections() -> None:
    registry = get_section_registry()

    assert registry.get("parallel_reads").name == "PARALLEL_READ_GUIDANCE"
    assert registry.get("tool_output_grounding").name == "GROUNDING_RULES_EXTENDED"
    assert registry.get("synthesis_rules").name == "INIT_SYNTHESIS_RULES"


def test_runtime_registration_adds_named_contributor_sections(monkeypatch) -> None:
    fresh_registry = UnifiedSectionRegistry()
    _initialize_default_sections(fresh_registry)
    monkeypatch.setattr(registry_module, "_registry", fresh_registry)

    class _Contributor:
        def get_prompt_section_contributions(self):
            return [
                PromptSectionContribution(
                    name="CUSTOM_REVIEW_GUIDANCE",
                    text="Review for API drift first.",
                    aliases={"custom_review"},
                    category="task_hints",
                    evolvable=True,
                    required=False,
                    priority=42,
                    default_strategies=("gepa", "prefpo"),
                )
            ]

        def get_priority(self) -> int:
            return 42

    register_prompt_contributor_sections([_Contributor()])

    registered = fresh_registry.get("CUSTOM_REVIEW_GUIDANCE")
    assert registered is not None
    assert registered.default_text == "Review for API drift first."
    assert registered.evolvable is True
    assert registered.default_strategies == ("gepa", "prefpo")
    assert fresh_registry.get("custom_review").name == "CUSTOM_REVIEW_GUIDANCE"


def test_runtime_registration_falls_back_for_legacy_contributors(monkeypatch) -> None:
    fresh_registry = UnifiedSectionRegistry()
    _initialize_default_sections(fresh_registry)
    monkeypatch.setattr(registry_module, "_registry", fresh_registry)

    class _LegacyContributor:
        def get_system_prompt_section(self) -> str:
            return "Legacy contributor guidance."

        def get_priority(self) -> int:
            return 55

    register_prompt_contributor_sections([_LegacyContributor()])

    registered = fresh_registry.get("VERTICAL_LEGACYCONTRIBUTOR")
    assert registered is not None
    assert registered.default_text == "Legacy contributor guidance."
    assert registered.evolvable is False


def test_edge_focus_selector_index_maps_specialized_sections() -> None:
    selector_index = get_edge_focus_selector_index()

    assert "file_pagination" in selector_index["LARGE_FILE_PAGINATION_GUIDANCE"]
    assert "parallel_read" in selector_index["PARALLEL_READ_GUIDANCE"]
    assert "tool_guidance" in selector_index["ASI_TOOL_EFFECTIVENESS_GUIDANCE"]


def test_edge_focus_prompt_catalog_text_lists_expected_sections() -> None:
    text = build_edge_focus_prompt_options_text()

    assert '"grounding"' in text
    assert '"file_pagination"' in text
    assert '"parallel_read"' in text


def test_registry_exposes_default_section_strategies() -> None:
    strategy_map = get_default_section_strategies()

    assert strategy_map["GROUNDING_RULES"] == ["gepa", "prefpo"]
    assert strategy_map["ASI_TOOL_EFFECTIVENESS_GUIDANCE"] == [
        "gepa",
        "cot_distillation",
    ]


def test_registry_default_texts_use_canonical_prompt_section_texts() -> None:
    registry = get_section_registry()

    assert registry.get("ASI_TOOL_EFFECTIVENESS_GUIDANCE").default_text == (
        ASI_TOOL_EFFECTIVENESS_GUIDANCE
    )
    assert registry.get("GROUNDING_RULES").default_text == GROUNDING_RULES
    assert registry.get("COMPLETION_GUIDANCE").default_text == COMPLETION_GUIDANCE
    assert registry.get("CONCISE_MODE_GUIDANCE").default_text == CONCISE_MODE_GUIDANCE
    assert registry.get("PARALLEL_READ_GUIDANCE").default_text == PARALLEL_READ_GUIDANCE
    assert registry.get("LARGE_FILE_PAGINATION_GUIDANCE").default_text == (
        LARGE_FILE_PAGINATION_GUIDANCE
    )
    assert (
        registry.get("GROUNDING_RULES_EXTENDED").default_text
        == GROUNDING_RULES_EXTENDED
    )


def test_prompt_baselines_include_promoted_gepa_refinements() -> None:
    """Guard the curated improvements promoted from GEPA candidate review."""

    assert "one tool_calls block" in PARALLEL_READ_GUIDANCE
    assert "Rule of thumb: if you can name 3+ files" in PARALLEL_READ_GUIDANCE
    assert "Read error messages carefully before retrying" in CONCISE_MODE_GUIDANCE
    assert (
        "Re-reading a truncated file without parameters"
        in LARGE_FILE_PAGINATION_GUIDANCE
    )
    assert "Retry discipline: analyze the root cause" in ASI_TOOL_EFFECTIVENESS_GUIDANCE
    assert "When in doubt, call another tool. Never guess." in GROUNDING_RULES_EXTENDED
