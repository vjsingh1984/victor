# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Runtime capability detection for Ollama models.

This module provides utilities to detect tool calling capabilities by
inspecting the actual model templates from the Ollama server.

The detection is authoritative - it checks the model's Modelfile template
for tool support patterns like `{{ if .Tools }}`.

Usage:
    from victor.providers.ollama_capability_detector import (
        OllamaCapabilityDetector,
        get_model_tool_support,
    )

    # Check single model
    support = get_model_tool_support("http://localhost:11434", "qwen2.5-coder:14b")
    print(support.supports_tools)  # True

    # Use detector for caching
    detector = OllamaCapabilityDetector("http://localhost:11434")
    if detector.supports_tools("llama3.1:8b"):
        print("Model has native tool calling!")
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, Optional

import httpx

logger = logging.getLogger(__name__)

# Patterns that indicate native tool support in Ollama templates
TOOL_SUPPORT_PATTERNS = [
    r"\{\{-?\s*if\s+\.Tools\s*\}\}",  # {{ if .Tools }} or {{- if .Tools }}
    r"\{\{-?\s*if\s+or\s+\.System\s+\.Tools\s*\}\}",  # {{ if or .System .Tools }}
    r"\{\{-?\s*range\s+\.Tools\s*\}\}",  # {{ range .Tools }}
    r"\{\{\s*range\s+\$\.Tools\s*\}\}",  # {{ range $.Tools }}
]

# Patterns for detecting tool response format
TOOL_FORMAT_PATTERNS = [
    (r"<tool_call>", "xml"),
    (r"<function_call>", "xml"),
    (r'"name":\s*"?<function-name>"?', "json"),
    (r'"parameters":\s*\{', "json"),
    (r'{"name":', "json"),
]


@dataclass
class ModelToolSupport:
    """Tool support information for a model."""

    model: str
    supports_tools: bool
    template_has_tools: bool
    tool_response_format: str = "unknown"  # "xml", "json", or "unknown"
    detection_method: str = "template"
    error: Optional[str] = None
    # Tool reliability rating from model_capabilities.yaml
    # Values: "high" (works reliably), "medium" (may need retries), "low" (often fails)
    tool_reliability: str = "medium"


# Known low-reliability models - these have .Tools template but don't work well
# Based on testing with Victor's tool calling system
LOW_RELIABILITY_MODELS = frozenset(
    [
        # Llama 3.3 often describes tools in prose but doesn't actually call them
        "llama3.3",
        "llama-3.3",
        # DeepSeek R1 produces truncated JSON like {"tool_call": }
        "deepseek-r1",
        # Qwen 2.5 Coder may output PARAMETER syntax instead of JSON
        "qwen2.5-coder",
        "qwen25-coder",  # Shorthand naming
    ]
)

# High-reliability models - tested and work well
HIGH_RELIABILITY_MODELS = frozenset(
    [
        # Qwen3 Coder with tools template works reliably
        "qwen3-coder-tools",
        "qwen3-coder",
        # Base Qwen3 also works well
        "qwen3",
    ]
)


class OllamaCapabilityDetector:
    """Detects Ollama model capabilities by inspecting templates.

    This class caches detection results to avoid repeated API calls.
    """

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 30):
        """Initialize the detector.

        Args:
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._cache: Dict[str, ModelToolSupport] = {}
        self._client = httpx.Client(timeout=timeout)

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()

    def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """Get model information from Ollama server.

        Args:
            model: Model name (e.g., "qwen2.5-coder:14b")

        Returns:
            Dict with model info including template, or None on error
        """
        try:
            response = self._client.post(
                f"{self.base_url}/api/show",
                json={"name": model},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"Error fetching model info for {model}: {e}")
            return None

    def _get_reliability_for_model(self, model: str) -> str:
        """Determine tool reliability rating for a model.

        Args:
            model: Model name (e.g., "qwen3-coder-tools:30b")

        Returns:
            Reliability rating: "high", "medium", or "low"
        """
        model_lower = model.lower()

        # Check high-reliability models first
        for pattern in HIGH_RELIABILITY_MODELS:
            if pattern in model_lower:
                return "high"

        # Check low-reliability models
        for pattern in LOW_RELIABILITY_MODELS:
            if pattern in model_lower:
                return "low"

        # Default to medium
        return "medium"

    def detect_tool_support(self, template: str) -> ModelToolSupport:
        """Analyze template to detect tool support patterns.

        Args:
            template: Model template string from Ollama

        Returns:
            ModelToolSupport with detection results
        """
        result = ModelToolSupport(
            model="",
            supports_tools=False,
            template_has_tools=False,
        )

        if not template:
            return result

        # Check for tool support patterns
        for pattern in TOOL_SUPPORT_PATTERNS:
            if re.search(pattern, template):
                result.template_has_tools = True
                result.supports_tools = True
                break

        # Detect tool response format
        for pattern, format_name in TOOL_FORMAT_PATTERNS:
            if re.search(pattern, template):
                result.tool_response_format = format_name
                break

        return result

    def get_tool_support(self, model: str, use_cache: bool = True) -> ModelToolSupport:
        """Check if a model supports native tool calling.

        Args:
            model: Model name (e.g., "qwen2.5-coder:14b")
            use_cache: Whether to use cached results

        Returns:
            ModelToolSupport with detection results
        """
        # Check cache
        if use_cache and model in self._cache:
            return self._cache[model]

        # Fetch model info
        info = self.get_model_info(model)

        if info is None:
            result = ModelToolSupport(
                model=model,
                supports_tools=False,
                template_has_tools=False,
                error="Failed to fetch model info",
            )
            self._cache[model] = result
            return result

        # Detect tool support from template
        template = info.get("template", "")
        result = self.detect_tool_support(template)
        result.model = model

        # Add reliability rating
        result.tool_reliability = self._get_reliability_for_model(model)

        # Log warning for low-reliability models
        if result.supports_tools and result.tool_reliability == "low":
            logger.warning(
                f"Model '{model}' has tool support but LOW reliability. "
                f"Tool calls may fail or produce malformed output. "
                f"Consider using qwen3-coder-tools for reliable tool calling."
            )

        # Cache result
        self._cache[model] = result

        return result

    def get_tool_reliability(self, model: str) -> str:
        """Get the tool reliability rating for a model.

        Args:
            model: Model name

        Returns:
            "high", "medium", or "low"
        """
        return self.get_tool_support(model).tool_reliability

    def is_reliable_for_tools(self, model: str) -> bool:
        """Check if model is reliable for tool calling.

        Args:
            model: Model name

        Returns:
            True if model has high or medium reliability
        """
        reliability = self.get_tool_reliability(model)
        return reliability in ("high", "medium")

    def supports_tools(self, model: str) -> bool:
        """Quick check if model supports tools.

        Args:
            model: Model name

        Returns:
            True if model has native tool support
        """
        return self.get_tool_support(model).supports_tools

    def get_tool_format(self, model: str) -> str:
        """Get the tool response format for a model.

        Args:
            model: Model name

        Returns:
            "xml", "json", or "unknown"
        """
        return self.get_tool_support(model).tool_response_format

    def clear_cache(self):
        """Clear the capability cache."""
        self._cache.clear()


# Global detector instance for convenience
_global_detector: Optional[OllamaCapabilityDetector] = None


def get_global_detector(base_url: str = "http://localhost:11434") -> OllamaCapabilityDetector:
    """Get or create the global detector instance.

    Args:
        base_url: Ollama server URL (only used on first call)

    Returns:
        OllamaCapabilityDetector instance
    """
    global _global_detector
    if _global_detector is None:
        _global_detector = OllamaCapabilityDetector(base_url)
    return _global_detector


def get_model_tool_support(base_url: str, model: str) -> ModelToolSupport:
    """Check if a model supports native tool calling.

    Convenience function that creates a temporary detector.
    For repeated checks, use OllamaCapabilityDetector directly.

    Args:
        base_url: Ollama server URL
        model: Model name

    Returns:
        ModelToolSupport with detection results
    """
    detector = OllamaCapabilityDetector(base_url)
    return detector.get_tool_support(model)


def check_tool_support_sync(base_url: str, model: str) -> bool:
    """Quick synchronous check for tool support.

    Args:
        base_url: Ollama server URL
        model: Model name

    Returns:
        True if model has native tool support
    """
    return get_model_tool_support(base_url, model).supports_tools
