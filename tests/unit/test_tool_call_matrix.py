from pathlib import Path

from victor.config.model_capabilities import ToolCallingMatrix


def test_matrix_merges_yaml_and_defaults(tmp_path: Path):
    # Note: manifest_path is no longer used - ToolCallingMatrix loads from
    # profiles.yaml. Use the manifest dict directly for custom models.
    matrix = ToolCallingMatrix(
        {"ollama": ["qwen3-coder:30b", "llama3.1:8b", "custom-model"]},
        always_allow_providers=["openai"],
    )

    assert matrix.is_tool_call_supported("ollama", "qwen3-coder:30b")
    assert matrix.is_tool_call_supported("ollama", "llama3.1:8b")
    assert matrix.is_tool_call_supported("ollama", "custom-model")
    assert matrix.is_tool_call_supported("openai", "gpt-4.1")
    assert not matrix.is_tool_call_supported("ollama", "non-tool-model")
    assert "llama3.1:8b" in matrix.get_supported_models("ollama")


def test_matrix_wildcard_matching():
    matrix = ToolCallingMatrix({"ollama": ["qwen3-coder:*", "llama3.1:8b"]})

    assert matrix.is_tool_call_supported("ollama", "qwen3-coder:30b")
    assert matrix.is_tool_call_supported("ollama", "qwen3-coder:70b")
    assert matrix.is_tool_call_supported("ollama", "llama3.1:8b-q4")
    # Note: mistral:7b is in default manifest, so it IS supported
    # Test with a model that's NOT in defaults to verify negative case
    assert not matrix.is_tool_call_supported("ollama", "unsupported-model:latest")


def test_orchestrator_uses_profile_provider_name():
    from unittest.mock import MagicMock
    import tempfile

    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings
    from victor.providers.base import BaseProvider

    provider = MagicMock(spec=BaseProvider)
    provider.supports_tools.return_value = True
    provider.supports_streaming.return_value = False
    provider.name = "openai"  # Underlying class name

    settings = Settings(
        analytics_enabled=False,
        analytics_log_file="usage.log",
        tool_calling_models={"lmstudio": ["lm-tool-model"]},
    )
    settings.tool_cache_enabled = False
    settings.tool_cache_dir = tempfile.mkdtemp()

    orch_ok = AgentOrchestrator(
        settings=settings,
        provider=provider,
        model="lm-tool-model",
        provider_name="lmstudio",
    )
    assert orch_ok._model_supports_tool_calls()

    orch_block = AgentOrchestrator(
        settings=settings,
        provider=provider,
        model="unknown-model",
        provider_name="lmstudio",
    )
    assert not orch_block._model_supports_tool_calls()
