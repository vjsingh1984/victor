"""Tests for SystemPromptPolicy."""

from victor.agent.system_prompt_policy import (
    SystemPromptPolicy,
    SystemPromptPolicyConfig,
    create_policy_from_settings,
)
from victor.framework.prompt_builder import PromptBuilder


class TestSystemPromptPolicy:
    """Verify policy enforcement and fallbacks."""

    def test_enforce_adds_required_sections(self):
        policy = SystemPromptPolicy()
        builder = PromptBuilder()

        policy.enforce(builder, context=None)

        assert builder.has_section("identity")
        assert builder.has_section("guidelines")
        identity = builder.get_section("identity")
        assert identity is not None and "Victor" in identity.content

    def test_enforce_deduplicates_sections(self):
        policy = SystemPromptPolicy()
        builder = PromptBuilder()
        builder.add_section("guidelines", "Line A\nLine B")
        builder.add_section("guidelines_duplicate", "Line A\nLine B", priority=99)

        policy.enforce(builder, context=None)

        assert builder.has_section("guidelines")
        assert not builder.has_section("guidelines_duplicate")

    def test_enforce_trims_large_sections(self):
        policy = SystemPromptPolicy(
            SystemPromptPolicyConfig(max_section_chars=50, protected_sections=("identity",))
        )
        builder = PromptBuilder()
        builder.add_section("identity", "ID", priority=10)
        builder.add_section("large_context", "X" * 200, priority=PromptBuilder.PRIORITY_CONTEXT)

        policy.enforce(builder, context=None)

        assert builder.has_section("identity")
        assert not builder.has_section("large_context")

    def test_fallback_prompt_includes_context(self):
        policy = SystemPromptPolicy()
        fallback = policy.build_fallback_prompt(
            context=type("Ctx", (), {"task_type": "debug", "message": "Fix failure"})
        )

        assert "debug" in fallback
        assert "Fix failure" in fallback

    def test_create_policy_from_settings_override(self):
        settings = type(
            "Stub",
            (),
            {
                "prompt_policy_identity": "Custom Victor",
                "prompt_policy_guidelines": "- Only run verified workflows.",
                "prompt_policy_protected_sections": ["identity", "custom"],
                "prompt_policy_max_section_chars": 100,
            },
        )()

        policy = create_policy_from_settings(settings)
        builder = PromptBuilder()
        policy.enforce(builder, context=None)

        assert builder.get_section("identity").content == "Custom Victor"
        assert builder.get_section("guidelines").content.startswith("- Only run")
        assert policy._config.max_section_chars == 100

    def test_create_policy_from_settings_disable_identity(self):
        settings = type("Stub", (), {"prompt_policy_enforce_identity": False})()
        policy = create_policy_from_settings(settings)
        builder = PromptBuilder()
        policy.enforce(builder, context=None)

        assert not builder.has_section("identity")
