"""Provider protocol definitions.

These protocols define how verticals interact with LLM and other providers.
"""

from victor_sdk.providers.protocols.llm import LLMProvider

__all__ = [
    "LLMProvider",
]
