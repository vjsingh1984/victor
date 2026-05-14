"""Provider protocol definitions.

These protocols define how verticals interact with LLM and other providers.
"""

from victor_contracts.providers.protocols.llm import LLMProvider

__all__ = [
    "LLMProvider",
]
