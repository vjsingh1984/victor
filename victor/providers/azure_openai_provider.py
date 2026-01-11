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

"""Azure OpenAI Service provider for enterprise model access.

Azure OpenAI provides enterprise-grade access to OpenAI's models with
Azure security, compliance, and regional data residency.

Authentication:
- API Key (AZURE_OPENAI_API_KEY)
- Azure Active Directory (AAD) / Entra ID token
- Managed Identity for Azure resources

Features:
- Access to GPT-4, GPT-4o, o1 models
- Enterprise security (RBAC, Private Link, VNET)
- Content filtering and safety features
- Regional deployment options
- Native tool/function calling

References:
- https://learn.microsoft.com/en-us/azure/ai-services/openai/
- https://learn.microsoft.com/en-us/azure/ai-services/openai/reference
"""

import json
import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderError,
    ProviderTimeoutError,
    StreamChunk,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

# Azure OpenAI API versions
DEFAULT_API_VERSION = "2024-08-01-preview"

# Common Azure OpenAI and Azure AI deployments
AZURE_MODELS = {
    # OpenAI models
    "gpt-4o": {
        "description": "GPT-4o - Multimodal, fast",
        "context_window": 128000,
        "max_output": 16384,
        "supports_tools": True,
    },
    "gpt-4o-mini": {
        "description": "GPT-4o Mini - Cost effective",
        "context_window": 128000,
        "max_output": 16384,
        "supports_tools": True,
    },
    "gpt-4-turbo": {
        "description": "GPT-4 Turbo - High capability",
        "context_window": 128000,
        "max_output": 4096,
        "supports_tools": True,
    },
    "gpt-4": {
        "description": "GPT-4 - Original",
        "context_window": 8192,
        "max_output": 4096,
        "supports_tools": True,
    },
    "gpt-35-turbo": {
        "description": "GPT-3.5 Turbo - Fast, cost effective",
        "context_window": 16384,
        "max_output": 4096,
        "supports_tools": True,
    },
    "o1-preview": {
        "description": "o1 Preview - Advanced reasoning",
        "context_window": 128000,
        "max_output": 32768,
        "supports_tools": False,  # o1 doesn't support tools yet
    },
    "o1-mini": {
        "description": "o1 Mini - Fast reasoning",
        "context_window": 128000,
        "max_output": 65536,
        "supports_tools": False,
    },
    # Microsoft Phi models (Azure AI)
    "Phi-4": {
        "description": "Phi-4 - Microsoft's latest SLM (14B)",
        "context_window": 16384,
        "max_output": 4096,
        "supports_tools": True,
    },
    "Phi-3.5-mini-instruct": {
        "description": "Phi-3.5 Mini - 3.8B efficient model",
        "context_window": 131072,
        "max_output": 4096,
        "supports_tools": True,
    },
    "Phi-3.5-MoE-instruct": {
        "description": "Phi-3.5 MoE - Mixture of Experts",
        "context_window": 131072,
        "max_output": 4096,
        "supports_tools": True,
    },
    "Phi-3-medium-128k-instruct": {
        "description": "Phi-3 Medium - 14B with 128K context",
        "context_window": 131072,
        "max_output": 4096,
        "supports_tools": True,
    },
    "Phi-3-mini-128k-instruct": {
        "description": "Phi-3 Mini - 3.8B with 128K context",
        "context_window": 131072,
        "max_output": 4096,
        "supports_tools": True,
    },
}


class AzureOpenAIProvider(BaseProvider):
    """Provider for Azure OpenAI Service.

    Features:
    - Enterprise OpenAI model access
    - Azure security and compliance
    - Regional data residency
    - Native tool calling support
    """

    DEFAULT_TIMEOUT = 120

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_version: str = DEFAULT_API_VERSION,
        timeout: int = DEFAULT_TIMEOUT,
        **kwargs: Any,
    ):
        """Initialize Azure OpenAI provider.

        Args:
            api_key: Azure OpenAI API key (or set AZURE_OPENAI_API_KEY env var, or use keyring)
            endpoint: Azure OpenAI endpoint URL (or set AZURE_OPENAI_ENDPOINT)
            deployment_name: Default deployment name (or set AZURE_OPENAI_DEPLOYMENT)
            api_version: API version to use
            timeout: Request timeout
            **kwargs: Additional configuration
        """
        # Resolution order: parameter → env var → keyring → warning
        resolved_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        if not resolved_key:
            try:
                from victor.config.api_keys import get_api_key

                resolved_key = get_api_key("azure_openai") or get_api_key("azure") or ""
            except ImportError:
                pass

        if not resolved_key:
            logger.warning(
                "Azure OpenAI API key not provided. Set AZURE_OPENAI_API_KEY environment variable, "
                "use 'victor keys --set azure_openai --keyring', or pass api_key parameter."
            )

        self._api_key = resolved_key
        self._endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        self._default_deployment = deployment_name or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
        self._api_version = api_version

        if not self._endpoint:
            logger.warning(
                "Azure OpenAI endpoint not provided. Set AZURE_OPENAI_ENDPOINT environment variable."
            )

        # Clean endpoint URL
        self._endpoint = self._endpoint.rstrip("/")

        super().__init__(base_url=self._endpoint, timeout=timeout, **kwargs)

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                "api-key": self._api_key,
                "Content-Type": "application/json",
            },
        )

    @property
    def name(self) -> str:
        return "azure"

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def _get_deployment_url(
        self, model_or_deployment: str, endpoint_type: str = "chat/completions"
    ) -> str:
        """Build Azure OpenAI deployment URL.

        Args:
            model_or_deployment: Model name or deployment name
            endpoint_type: API endpoint type

        Returns:
            Full URL for the deployment endpoint
        """
        # Use the model name as deployment name (common pattern)
        deployment = model_or_deployment
        return f"{self._endpoint}/openai/deployments/{deployment}/{endpoint_type}?api-version={self._api_version}"

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send chat completion request to Azure OpenAI."""
        try:
            payload = self._build_request_payload(
                messages, model, temperature, max_tokens, tools, False, **kwargs
            )

            url = self._get_deployment_url(model)
            response = await self._execute_with_circuit_breaker(self.client.post, url, json=payload)
            response.raise_for_status()

            return self._parse_response(response.json(), model)

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"Azure OpenAI request timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            error_body = e.response.text[:500] if e.response.text else ""
            raise ProviderError(
                message=f"Azure OpenAI HTTP error {e.response.status_code}: {error_body}",
                provider=self.name,
                status_code=e.response.status_code,
            ) from e

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion from Azure OpenAI."""
        try:
            payload = self._build_request_payload(
                messages, model, temperature, max_tokens, tools, True, **kwargs
            )

            url = self._get_deployment_url(model)

            async with self.client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                accumulated_tool_calls: List[Dict[str, Any]] = []

                async for line in response.aiter_lines():
                    if not line.strip() or not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        yield StreamChunk(
                            content="",
                            tool_calls=accumulated_tool_calls if accumulated_tool_calls else None,
                            stop_reason="stop",
                            is_final=True,
                        )
                        break

                    try:
                        chunk_data = json.loads(data_str)
                        yield self._parse_stream_chunk(chunk_data, accumulated_tool_calls)
                    except json.JSONDecodeError:
                        pass

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message="Azure OpenAI stream timed out",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                message=f"Azure OpenAI streaming error {e.response.status_code}",
                provider=self.name,
                status_code=e.response.status_code,
            ) from e

    def _build_request_payload(
        self, messages, model, temperature, max_tokens, tools, stream, **kwargs
    ) -> Dict[str, Any]:
        """Build Azure OpenAI request payload (OpenAI format)."""
        formatted_messages = []
        for msg in messages:
            formatted_msg = {"role": msg.role, "content": msg.content}
            if msg.role == "tool" and hasattr(msg, "tool_call_id"):
                formatted_msg["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                formatted_msg["tool_calls"] = [
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": (
                                json.dumps(tc.get("arguments", {}))
                                if isinstance(tc.get("arguments"), dict)
                                else tc.get("arguments", "{}")
                            ),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            formatted_messages.append(formatted_msg)

        payload = {
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
            payload["tool_choice"] = "auto"

        return payload

    def _parse_response(self, result: Dict[str, Any], model: str) -> CompletionResponse:
        """Parse Azure OpenAI response."""
        choices = result.get("choices", [])
        if not choices:
            return CompletionResponse(
                content="", role="assistant", model=model, raw_response=result
            )

        choice = choices[0]
        message = choice.get("message", {})
        tool_calls = self._normalize_tool_calls(message.get("tool_calls"))

        usage = None
        if usage_data := result.get("usage"):
            usage = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }

        return CompletionResponse(
            content=message.get("content", "") or "",
            role="assistant",
            tool_calls=tool_calls,
            stop_reason=choice.get("finish_reason"),
            usage=usage,
            model=model,
            raw_response=result,
        )

    def _normalize_tool_calls(self, tool_calls) -> Optional[List[Dict[str, Any]]]:
        """Normalize tool calls to Victor format."""
        if not tool_calls:
            return None
        normalized = []
        for call in tool_calls:
            if "function" in call:
                func = call["function"]
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                normalized.append(
                    {
                        "id": call.get("id", ""),
                        "name": func.get("name", ""),
                        "arguments": args,
                    }
                )
        return normalized if normalized else None

    def _parse_stream_chunk(self, chunk_data, accumulated_tool_calls) -> StreamChunk:
        """Parse Azure OpenAI stream chunk."""
        choices = chunk_data.get("choices", [])
        if not choices:
            return StreamChunk(content="", is_final=False)

        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content", "") or ""
        finish_reason = choice.get("finish_reason")

        for tc_delta in delta.get("tool_calls", []):
            idx = tc_delta.get("index", 0)
            while len(accumulated_tool_calls) <= idx:
                accumulated_tool_calls.append({"id": "", "name": "", "arguments": ""})
            if "id" in tc_delta:
                accumulated_tool_calls[idx]["id"] = tc_delta["id"]
            if "function" in tc_delta:
                func = tc_delta["function"]
                if "name" in func:
                    accumulated_tool_calls[idx]["name"] = func["name"]
                if "arguments" in func:
                    accumulated_tool_calls[idx]["arguments"] += func["arguments"]

        final_tool_calls = None
        if finish_reason in ("tool_calls", "stop") and accumulated_tool_calls:
            final_tool_calls = []
            for tc in accumulated_tool_calls:
                if tc.get("name"):
                    args = tc.get("arguments", "{}")
                    try:
                        args = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
                        args = {}
                    final_tool_calls.append(
                        {
                            "id": tc.get("id", ""),
                            "name": tc["name"],
                            "arguments": args,
                        }
                    )

        return StreamChunk(
            content=content,
            tool_calls=final_tool_calls,
            stop_reason=finish_reason,
            is_final=finish_reason is not None,
        )

    async def close(self) -> None:
        await self.client.aclose()
