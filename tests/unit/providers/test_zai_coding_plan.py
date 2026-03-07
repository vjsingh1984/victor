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

"""TDD tests for Z.AI Coding Plan support and config strategy."""

import pytest
from unittest.mock import patch

from victor.providers.zai_provider import (
    ZAIProvider,
    ZAI_BASE_URLS,
    ZAI_MODELS,
)

# ---------------------------------------------------------------------------
# Z.AI Base URL constants
# ---------------------------------------------------------------------------


class TestZAIBaseURLs:
    """Test that Z.AI endpoint constants are defined."""

    def test_standard_url(self):
        assert ZAI_BASE_URLS["standard"] == "https://api.z.ai/api/paas/v4/"

    def test_coding_plan_url(self):
        assert ZAI_BASE_URLS["coding"] == "https://api.z.ai/api/coding/paas/v4/"

    def test_china_url(self):
        assert ZAI_BASE_URLS["china"] == "https://open.bigmodel.cn/api/paas/v4/"

    def test_anthropic_compat_url(self):
        assert ZAI_BASE_URLS["anthropic"] == "https://api.z.ai/api/anthropic/v1/"


# ---------------------------------------------------------------------------
# Z.AI Provider — Coding Plan initialization
# ---------------------------------------------------------------------------


class TestZAICodingPlan:
    """Test ZAIProvider with coding_plan=True."""

    def test_coding_plan_sets_correct_base_url(self):
        provider = ZAIProvider(
            api_key="test-key",
            coding_plan=True,
        )
        assert str(provider.client.base_url).rstrip("/") == ("https://api.z.ai/api/coding/paas/v4")

    def test_standard_default_base_url(self):
        provider = ZAIProvider(
            api_key="test-key",
        )
        # Default should be china (backward compat) or standard
        assert "paas/v4" in str(provider.client.base_url)

    def test_explicit_base_url_overrides_coding_plan(self):
        """Explicit base_url takes priority over coding_plan flag."""
        provider = ZAIProvider(
            api_key="test-key",
            base_url="https://custom.endpoint/v1/",
            coding_plan=True,
        )
        assert "custom.endpoint" in str(provider.client.base_url)

    def test_endpoint_param_standard(self):
        provider = ZAIProvider(
            api_key="test-key",
            endpoint="standard",
        )
        assert "api.z.ai/api/paas/v4" in str(provider.client.base_url)

    def test_endpoint_param_coding(self):
        provider = ZAIProvider(
            api_key="test-key",
            endpoint="coding",
        )
        assert "api.z.ai/api/coding/paas/v4" in str(provider.client.base_url)

    def test_endpoint_param_china(self):
        provider = ZAIProvider(
            api_key="test-key",
            endpoint="china",
        )
        assert "open.bigmodel.cn" in str(provider.client.base_url)

    def test_endpoint_param_anthropic(self):
        provider = ZAIProvider(
            api_key="test-key",
            endpoint="anthropic",
        )
        assert "api.z.ai/api/anthropic" in str(provider.client.base_url)


# ---------------------------------------------------------------------------
# Z.AI Provider Config Strategy
# ---------------------------------------------------------------------------


class TestZAIConfigStrategy:
    """Test ZAIConfig and ZAICodingPlanConfig strategies."""

    def test_zai_config_registered(self):
        from victor.config.provider_config_registry import get_provider_config_registry

        registry = get_provider_config_registry()
        providers = registry.list_providers()
        assert "zai" in providers

    def test_zai_coding_plan_config_registered(self):
        from victor.config.provider_config_registry import get_provider_config_registry

        registry = get_provider_config_registry()
        providers = registry.list_providers()
        assert "zai-coding-plan" in providers

    def test_zai_aliases(self):
        from victor.config.provider_config_registry import get_provider_config_registry

        registry = get_provider_config_registry()
        # zhipuai and zhipu should resolve to zai
        assert registry._aliases.get("zhipuai") == "zai"
        assert registry._aliases.get("zhipu") == "zai"

    def test_zai_default_base_url(self):
        from victor.config.provider_config_registry import ZAIConfig

        config = ZAIConfig()
        from unittest.mock import MagicMock

        settings = MagicMock()
        result = config.get_settings(settings, {})
        assert result["base_url"] == "https://api.z.ai/api/paas/v4/"

    @pytest.mark.skip(
        reason="ZAICodingPlanConfig removed - use model suffix 'glm-4.6:coding' instead"
    )
    def test_zai_coding_plan_base_url(self):
        """Test ZAI coding plan base URL."""
        # NOTE: ZAICodingPlanConfig has been removed in favor of model suffix notation.
        # Use model="glm-4.6:coding" to specify the coding plan endpoint.
        # The ZAI provider now handles this automatically via model suffix parsing.
        from victor.config.provider_config_registry import ZAICodingPlanConfig

        config = ZAICodingPlanConfig()
        from unittest.mock import MagicMock

        settings = MagicMock()
        result = config.get_settings(settings, {})
        assert result["base_url"] == "https://api.z.ai/api/coding/paas/v4/"


# ---------------------------------------------------------------------------
# Z.AI GLM-5 model support
# ---------------------------------------------------------------------------


class TestZAIModels:
    """Test model catalog completeness."""

    def test_glm5_in_catalog(self):
        assert "glm-5" in ZAI_MODELS

    def test_glm47_in_catalog(self):
        assert "glm-4.7" in ZAI_MODELS

    def test_glm45_air_in_catalog(self):
        assert "glm-4.5-air" in ZAI_MODELS

    def test_free_flash_models(self):
        assert "glm-4.7-flash" in ZAI_MODELS
        assert "glm-4.6v-flash" in ZAI_MODELS


# ---------------------------------------------------------------------------
# Z.AI Provider — _normalize_tool_calls
# ---------------------------------------------------------------------------


class TestZAINormalizeToolCalls:
    """Test _normalize_tool_calls helper."""

    @pytest.fixture
    def provider(self):
        return ZAIProvider(api_key="test-key")

    def test_none_input(self, provider):
        assert provider._normalize_tool_calls(None) is None

    def test_empty_list(self, provider):
        assert provider._normalize_tool_calls([]) is None

    def test_openai_format(self, provider):
        raw = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
            }
        ]
        result = provider._normalize_tool_calls(raw)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["arguments"] == {"city": "Beijing"}
        assert result[0]["id"] == "call_1"

    def test_already_normalized(self, provider):
        raw = [{"id": "call_1", "name": "my_tool", "arguments": {"x": 1}}]
        result = provider._normalize_tool_calls(raw)
        assert result is not None
        assert result[0]["name"] == "my_tool"

    def test_invalid_json_arguments(self, provider):
        raw = [
            {
                "id": "call_1",
                "function": {"name": "bad_tool", "arguments": "not-json"},
            }
        ]
        result = provider._normalize_tool_calls(raw)
        assert result is not None
        assert result[0]["arguments"] == {}

    def test_no_name_skipped(self, provider):
        raw = [{"id": "call_1", "function": {"arguments": "{}"}}]
        result = provider._normalize_tool_calls(raw)
        assert result is None


# ---------------------------------------------------------------------------
# Z.AI Provider — _parse_response
# ---------------------------------------------------------------------------


class TestZAIParseResponse:
    """Test _parse_response helper."""

    @pytest.fixture
    def provider(self):
        return ZAIProvider(api_key="test-key")

    def test_empty_choices(self, provider):
        result = provider._parse_response({"choices": []}, "glm-4.7")
        assert result.content == ""
        assert result.model == "glm-4.7"

    def test_basic_response(self, provider):
        raw = {
            "choices": [
                {
                    "message": {"content": "Hello", "role": "assistant"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = provider._parse_response(raw, "glm-4.7")
        assert result.content == "Hello"
        assert result.stop_reason == "stop"
        assert result.usage["total_tokens"] == 15

    def test_with_tool_calls(self, provider):
        raw = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "calc", "arguments": '{"x": 1}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        result = provider._parse_response(raw, "glm-4.7")
        assert result.tool_calls is not None
        assert result.tool_calls[0]["name"] == "calc"

    def test_with_reasoning_content(self, provider):
        raw = {
            "choices": [
                {
                    "message": {
                        "content": "Answer",
                        "reasoning_content": "Step 1...",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        result = provider._parse_response(raw, "glm-4.6")
        assert result.metadata is not None
        assert result.metadata["reasoning_content"] == "Step 1..."


# ---------------------------------------------------------------------------
# Z.AI Provider — _parse_stream_chunk
# ---------------------------------------------------------------------------


class TestZAIParseStreamChunk:
    """Test _parse_stream_chunk helper."""

    @pytest.fixture
    def provider(self):
        return ZAIProvider(api_key="test-key")

    def test_empty_choices(self, provider):
        result = provider._parse_stream_chunk({"choices": []}, [])
        assert result is None

    def test_content_chunk(self, provider):
        data = {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]}
        result = provider._parse_stream_chunk(data, [])
        assert result is not None
        assert result.content == "Hello"
        assert result.is_final is False

    def test_final_chunk(self, provider):
        data = {
            "choices": [{"delta": {"content": ""}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = provider._parse_stream_chunk(data, [])
        assert result.is_final is True
        assert result.usage["total_tokens"] == 15

    def test_tool_call_accumulation(self, provider):
        accumulated = []

        # First chunk: tool call start
        data1 = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "calc", "arguments": '{"x"'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
        result1 = provider._parse_stream_chunk(data1, accumulated)
        assert result1 is not None
        assert len(accumulated) == 1

        # Second chunk: arguments continuation
        data2 = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": ": 1}"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
        provider._parse_stream_chunk(data2, accumulated)
        assert accumulated[0]["arguments"] == '{"x": 1}'

        # Final chunk: tool_calls finish
        data3 = {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
        result3 = provider._parse_stream_chunk(data3, accumulated)
        assert result3.tool_calls is not None
        assert result3.tool_calls[0]["name"] == "calc"
        assert result3.tool_calls[0]["arguments"] == {"x": 1}

    def test_reasoning_content_in_delta(self, provider):
        data = {
            "choices": [
                {
                    "delta": {"content": "Answer", "reasoning_content": "Thinking..."},
                    "finish_reason": None,
                }
            ]
        }
        result = provider._parse_stream_chunk(data, [])
        assert result.metadata is not None
        assert result.metadata["reasoning_content"] == "Thinking..."


# ---------------------------------------------------------------------------
# Z.AI Provider — list_models
# ---------------------------------------------------------------------------


class TestZAIListModels:
    """Test list_models method."""

    @pytest.fixture
    def provider(self):
        return ZAIProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_list_models_success(self, provider):
        from unittest.mock import AsyncMock, MagicMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "glm-4.7"}, {"id": "glm-5"}]}
        mock_response.raise_for_status = MagicMock()

        provider.client.get = AsyncMock(return_value=mock_response)
        models = await provider.list_models()
        assert len(models) == 2
        assert models[0]["id"] == "glm-4.7"

    @pytest.mark.asyncio
    async def test_list_models_error(self, provider):
        from unittest.mock import AsyncMock
        from victor.providers.base import ProviderError

        provider.client.get = AsyncMock(side_effect=Exception("Network error"))
        with pytest.raises(ProviderError):
            await provider.list_models()
