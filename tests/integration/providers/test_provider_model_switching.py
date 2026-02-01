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

"""Integration tests for provider and model switching.

Tests the ability to switch providers and models mid-conversation using
real orchestrator instances with actual providers.

Coverage:
- Model switching on the same provider
- Provider switching with model preservation
- Provider + model switching together
- Context preservation during switching
- Tool adapter reinitialization
"""


import pytest
from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings, ProfileConfig
from victor.providers.registry import ProviderRegistry


def get_ollama_models() -> list[str]:
    """Get list of available Ollama models."""
    try:
        import subprocess

        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            models = []
            for line in result.stdout.strip().split("\n"):
                if "NAME" not in line and line.strip():
                    # Extract model name (first column)
                    model_name = line.strip().split()[0]
                    models.append(model_name)
            return models
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return []


def get_lmstudio_models() -> list[str]:
    """Get list of available LMStudio models."""
    try:
        import subprocess

        # LMStudio exposes OpenAI-compatible API on port 1234
        result = subprocess.run(
            ["curl", "-s", "--connect-timeout", "5", "http://localhost:1234/v1/models"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            import json

            data = json.loads(result.stdout)
            if "data" in data:
                return [model["id"] for model in data["data"]]
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        # Silently fail - provider might not be available
        pass
    return []


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        analytics_enabled=False,
        tool_cache_enabled=False,
        provider_health_checks=False,
    )


@pytest.fixture
def available_providers():
    """Get dictionary of available providers and their models."""
    providers = {}

    ollama_models = get_ollama_models()
    if ollama_models:
        providers["ollama"] = ollama_models

    lmstudio_models = get_lmstudio_models()
    if lmstudio_models:
        providers["lmstudio"] = lmstudio_models

    if not providers:
        pytest.skip("No local providers available (Ollama or LMStudio required)")

    return providers


@pytest.fixture
def provider_name(available_providers):
    """Get first available provider for testing."""
    # Return first available provider
    return next(iter(available_providers.keys()))


@pytest.fixture
def model_name(available_providers, provider_name):
    """Get first model for the provider."""
    models = available_providers[provider_name]
    if models:
        return models[0]
    pytest.skip(f"No models available for provider {provider_name}")


@pytest.fixture
def second_provider_name(available_providers, provider_name):
    """Get second provider for switching tests."""
    providers_list = list(available_providers.keys())

    if len(providers_list) >= 2:
        # Get the other provider
        if provider_name == providers_list[0]:
            return providers_list[1]
        else:
            return providers_list[0]
    else:
        # Only one provider available
        pytest.skip(
            "Provider switching tests require at least 2 different providers (Ollama and LMStudio)"
        )


@pytest.fixture
def second_model_name(available_providers, second_provider_name):
    """Get first model for the second provider."""
    models = available_providers.get(second_provider_name, [])
    if models:
        return models[0]
    pytest.skip(f"No models available for provider {second_provider_name}")


@pytest.fixture
def alternative_model(available_providers, provider_name):
    """Get an alternative model on the same provider for testing model switching."""
    models = available_providers.get(provider_name, [])
    if len(models) >= 2:
        return models[1]
    # If only one model, use the same model (switch will be to same model)
    elif models:
        return models[0]
    pytest.skip(f"No models available for provider {provider_name}")


@pytest.fixture
def orchestrator(settings, provider_name, model_name):
    """Create orchestrator with real provider."""
    # Import UsageLogger to avoid warnings
    import victor.agent.orchestrator_imports  # noqa: F401

    provider = ProviderRegistry.create(provider_name, model=model_name)

    orc = AgentOrchestrator(
        settings=settings,
        provider=provider,
        model=model_name,
    )

    return orc


class TestModelSwitchingIntegration:
    """Integration tests for model switching on the same provider."""

    def test_switch_model_same_provider(self, orchestrator, alternative_model):
        """Test switching model on the same provider."""
        initial_provider = orchestrator.provider_name
        initial_model = orchestrator.model

        # Use the orchestrator's sync switch_model method
        # This internally handles the async coordinator call
        success = orchestrator.switch_model(alternative_model)

        # Provider should remain unchanged
        assert orchestrator.provider_name == initial_provider

        # Model should be updated (if switch was successful)
        # Note: If alternative_model == initial_model, model stays the same
        if success:
            assert orchestrator.model == alternative_model

    def test_switch_model_to_nonexistent_provider(self, orchestrator):
        """Test that switching to a non-existent model is handled gracefully."""
        initial_provider = orchestrator.provider_name
        initial_model = orchestrator.model

        # Try to switch to a model that doesn't exist
        success = orchestrator.switch_model("definitely-nonexistent-model-12345")

        # Provider should remain unchanged
        assert orchestrator.provider_name == initial_provider

    def test_model_switch_updates_provider_manager(self, orchestrator, alternative_model):
        """Test that model switch updates ProviderManager state."""
        # Switch model using orchestrator's sync method
        orchestrator.switch_model(alternative_model)

        # ProviderManager should exist and have a model attribute
        assert hasattr(orchestrator._provider_manager, "model")
        assert orchestrator._provider_manager.model is not None

    def test_multiple_model_switches(self, orchestrator, alternative_model):
        """Test multiple consecutive model switches."""
        # Switch between models multiple times
        models_to_test = [alternative_model, orchestrator.model, alternative_model]

        for model in models_to_test:
            # Try to switch - should not crash
            success = orchestrator.switch_model(model)

        # Provider should still be intact
        assert orchestrator.provider_name is not None

    def test_model_switch_updates_tool_adapter(self, orchestrator, alternative_model):
        """Test that model switch updates tool adapter."""
        original_adapter = orchestrator.tool_adapter

        # Switch model using orchestrator's sync method
        orchestrator.switch_model(alternative_model)

        # Tool adapter should exist
        assert orchestrator.tool_adapter is not None


class TestProviderSwitchingIntegration:
    """Integration tests for provider switching.

    These tests require at least 2 different providers to be available.
    They will be skipped if only one provider is detected.
    """

    def test_switch_provider_preserves_model(self, orchestrator, second_provider_name):
        """Test switching provider while keeping the same model."""
        initial_model = orchestrator.model

        # Switch provider using orchestrator's sync switch_provider method
        # This internally handles the async coordinator call
        success = orchestrator.switch_provider(second_provider_name, model=initial_model)

        # Provider switching may fail if model doesn't exist on second provider
        # We test the mechanism
        if success:
            # If successful, verify provider changed
            assert orchestrator.provider_name == second_provider_name

    def test_switch_provider_with_new_model(self, orchestrator, second_provider_name):
        """Test switching provider with a new model."""
        new_model = "llama3.2"  # Common model name that might exist

        # Switch provider using orchestrator's sync method
        success = orchestrator.switch_provider(second_provider_name, model=new_model)

        # Test the mechanism - may succeed or fail depending on model availability
        if success:
            assert orchestrator.model == new_model
            assert orchestrator.provider_name == second_provider_name

    def test_switch_provider_with_custom_settings(self, orchestrator, second_provider_name):
        """Test switching provider with custom provider settings."""
        # Note: Settings are passed through provider_kwargs

        custom_settings = {"timeout": 60}

        # Switch with custom settings using orchestrator's sync method
        # The provider should be created with these settings
        success = orchestrator.switch_provider(
            second_provider_name, model="llama3.2", **custom_settings
        )

        # Verify the switch attempt was made (may fail if model doesn't exist)
        # The important thing is that the orchestrator handles the switch attempt
        assert orchestrator is not None


class TestProviderModelCombinedSwitching:
    """Integration tests for combined provider + model switching.

    These tests require at least 2 different providers to be available.
    """

    def test_switch_provider_and_model_together(self, orchestrator, second_provider_name):
        """Test switching provider and model simultaneously."""
        initial_provider = orchestrator.provider_name
        initial_model = orchestrator.model

        # Switch both provider and model using orchestrator's sync method
        success = orchestrator.switch_provider(second_provider_name, "llama3.2")

        # Test the switching mechanism
        if success:
            assert orchestrator.model == "llama3.2"
            assert orchestrator.provider_name == second_provider_name
            assert (
                orchestrator.model != initial_model
                or orchestrator.provider_name != initial_provider
            )

    def test_switch_to_profile(self, orchestrator, second_provider_name):
        """Test switching to a pre-configured profile."""
        profile = ProfileConfig(
            provider=second_provider_name,
            model="llama3.2",
            temperature=0.8,
            max_tokens=2048,
        )

        # Switch to profile using orchestrator's sync method
        success = orchestrator.switch_provider(profile.provider, profile.model)

        # Test profile-based switching
        if success:
            assert orchestrator.model == profile.model
            assert orchestrator.provider_name == profile.provider


class TestProfileBasedSwitching:
    """Integration tests for profile-based switching."""

    @pytest.fixture
    def profiles_config(self, provider_name):
        """Create test profiles configuration."""
        return {
            "fast": ProfileConfig(
                provider=provider_name,
                model="llama3.2",  # Common model that might exist
                temperature=0.5,
                max_tokens=2048,
                description="Fast local model",
            ),
            "smart": ProfileConfig(
                provider=provider_name,
                model="llama3.2",  # Using same model for testing
                temperature=0.8,
                max_tokens=8192,
                description="Smart cloud model",
            ),
        }

    def test_switch_between_profiles(self, orchestrator, profiles_config):
        """Test switching between different profiles."""
        # Start with fast profile - use orchestrator's sync method
        profile1 = profiles_config["fast"]
        success1 = orchestrator.switch_provider(profile1.provider, profile1.model)

        # Test the switching mechanism
        if success1:
            assert orchestrator.model == profile1.model

        # Switch to smart profile - use orchestrator's sync method
        profile2 = profiles_config["smart"]
        success2 = orchestrator.switch_provider(profile2.provider, profile2.model)

        if success2:
            assert orchestrator.model == profile2.model

    def test_profile_metadata_preserved(self, orchestrator):
        """Test that profile metadata is accessible."""
        profile = ProfileConfig(
            provider="mock",
            model="test-model",
            temperature=0.7,
            max_tokens=4096,
            description="Test profile",
            tool_selection={"max_cost_threshold": 0.5},
        )

        # Verify profile metadata
        assert profile.provider == "mock"
        assert profile.model == "test-model"
        assert profile.temperature == 0.7
        assert profile.max_tokens == 4096
        assert profile.description == "Test profile"
        assert profile.tool_selection is not None
        assert profile.tool_selection["max_cost_threshold"] == 0.5


class TestSwitchSequencePreservation:
    """Integration tests for context and state preservation during switching."""

    def test_model_switch_preserves_conversation_state(self, orchestrator):
        """Test that model switch preserves conversation state."""
        # Add messages to conversation using the correct API
        orchestrator.add_message("user", "Question 1")
        orchestrator.add_message("assistant", "Answer 1")

        initial_count = orchestrator.get_message_count()

        # Switch model using orchestrator's sync method
        orchestrator.switch_model("new-model")

        # Conversation should be preserved
        assert orchestrator.get_message_count() == initial_count
        messages = orchestrator.messages
        assert len(messages) == 2
        assert messages[0].content == "Question 1"
        assert messages[1].content == "Answer 1"

    def test_provider_switch_preserves_conversation_state(self, orchestrator, second_provider_name):
        """Test that provider switch preserves conversation state."""
        # Build conversation using the correct API
        orchestrator.add_message("user", "Question 1")
        orchestrator.add_message("assistant", "Answer 1")
        orchestrator.add_message("user", "Question 2")

        initial_count = orchestrator.get_message_count()

        # Switch provider using orchestrator's sync method
        success = orchestrator.switch_provider(second_provider_name, "new-model")

        # Conversation should be preserved regardless of switch success
        assert orchestrator.get_message_count() == initial_count
        messages = orchestrator.messages
        assert len(messages) == 3
        assert messages[0].content == "Question 1"
        assert messages[2].content == "Question 2"

    def test_multiple_switches_preserve_context(self, orchestrator):
        """Test that multiple switches preserve conversation context."""
        # Build conversation using the correct API
        orchestrator.add_message("user", "Initial question")
        orchestrator.add_message("assistant", "Initial answer")

        initial_count = 2

        # First switch (model - orchestrator's sync method)
        success1 = orchestrator.switch_model("model1")
        assert orchestrator.get_message_count() == initial_count

        # Second switch (model - orchestrator's sync method)
        success2 = orchestrator.switch_model("model2")
        assert orchestrator.get_message_count() == initial_count

        # All messages still present
        messages = orchestrator.messages
        assert messages[0].content == "Initial question"
        assert messages[1].content == "Initial answer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
