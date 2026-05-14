"""LLM provider protocol definitions.

These protocols define how verticals interact with LLM providers.
"""

from __future__ import annotations

from typing import (
    Protocol,
    runtime_checkable,
    Dict,
    Any,
    List,
    Optional,
    Callable,
    Awaitable,
)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers.

    LLM providers manage interactions with language model APIs.
    """

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a completion.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text
        """
        ...

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Callable[[], Awaitable[str]]:
        """Generate a streaming completion.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters

        Returns:
            Async callable that yields text chunks
        """
        ...

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a chat completion.

        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated response
        """
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the model.

        Returns:
            Dictionary with model information
        """
        ...

    def supports_streaming(self) -> bool:
        """Check if provider supports streaming.

        Returns:
            True if streaming is supported
        """
        ...

    def supports_function_calling(self) -> bool:
        """Check if provider supports function calling.

        Returns:
            True if function calling is supported
        """
        ...
