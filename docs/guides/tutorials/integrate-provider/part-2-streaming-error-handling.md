# Tutorial: Integrating a New LLM Provider - Part 2

**Part 2 of 5:** Streaming Implementation and Error Handling

---

## Navigation

- [Part 1: Architecture & Steps 1-2](part-1-provider-architecture.md)
- **[Part 2: Streaming & Error Handling](#)** (Current)
- [Part 3: Tool Calling Adapter](part-3-tool-calling-adapter.md)
- [Part 4: Registration & Testing](part-4-registration-testing.md)
- [Part 5: Best Practices & Examples](part-5-best-practices-examples.md)
- [**Complete Guide**](integrate-provider.md)

---

## Step 3: Implement the stream() Method

The `stream()` method provides streaming responses:

```python
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
        """Stream chat completion response.

        Args:
            messages: Conversation messages
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools for function calling
            **kwargs: Additional provider-specific parameters

        Yields:
            StreamChunk objects with incremental content

        Raises:
            ProviderError: If the request fails
        """
        try:
            payload = self._build_request_payload(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                stream=True,
                **kwargs,
            )

            # Track accumulated tool calls across chunks
            accumulated_tool_calls: List[Dict[str, Any]] = []

            async with self.client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    # Handle Server-Sent Events format
                    if line.startswith("data: "):
                        data_str = line[6:]

                        # Check for stream end
                        if data_str.strip() == "[DONE]":
                            yield StreamChunk(
                                content="",
                                tool_calls=(
                                    accumulated_tool_calls
                                    if accumulated_tool_calls else None
                                ),
                                stop_reason="stop",
                                is_final=True,
                            )
                            break

                        try:
                            chunk_data = json.loads(data_str)
                            chunk = self._parse_stream_chunk(
                                chunk_data, accumulated_tool_calls
                            )
                            yield chunk
                        except json.JSONDecodeError:
                            logger.warning(
                                f"JSON decode error on line: {line[:100]}"
                            )

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"Stream timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e)
        except Exception as e:
            raise ProviderError(
                message=f"Stream error: {str(e)}",
                provider=self.name,
                raw_error=e,
            ) from e

    def _parse_stream_chunk(
        self,
        chunk_data: Dict[str, Any],
        accumulated_tool_calls: List[Dict[str, Any]],
    ) -> StreamChunk:
        """Parse a streaming chunk from the API.

        Args:
            chunk_data: Raw chunk data
            accumulated_tool_calls: List to accumulate tool call deltas

        Returns:
            Normalized StreamChunk
        """
        choices = chunk_data.get("choices", [])
        if not choices:
            return StreamChunk(content="", is_final=False)

        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content", "") or ""
        finish_reason = choice.get("finish_reason")

        # Handle tool call deltas (for streaming tool calls)
        tool_call_deltas = delta.get("tool_calls", [])
        for tc_delta in tool_call_deltas:
            idx = tc_delta.get("index", 0)

            # Ensure we have a slot for this tool call
            while len(accumulated_tool_calls) <= idx:
                accumulated_tool_calls.append({
                    "id": "",
                    "name": "",
                    "arguments": "",
                })

            # Accumulate tool call data
            if "id" in tc_delta:
                accumulated_tool_calls[idx]["id"] = tc_delta["id"]
            if "function" in tc_delta:
                func_delta = tc_delta["function"]
                if "name" in func_delta:
                    accumulated_tool_calls[idx]["name"] = func_delta["name"]
                if "arguments" in func_delta:
                    accumulated_tool_calls[idx]["arguments"] += (
                        func_delta["arguments"]
                    )

        # Finalize tool calls when stream ends
        final_tool_calls = None
        if finish_reason in ("tool_calls", "stop") and accumulated_tool_calls:
            final_tool_calls = []
            for tc in accumulated_tool_calls:
                if tc.get("name"):
                    args = tc.get("arguments", "{}")
                    try:
                        parsed_args = (
                            json.loads(args) if isinstance(args, str) else args
                        )
                    except json.JSONDecodeError:
                        parsed_args = {}
                    final_tool_calls.append({
                        "id": tc.get("id", ""),
                        "name": tc["name"],
                        "arguments": parsed_args,
                    })

        return StreamChunk(
            content=content,
            tool_calls=final_tool_calls,
            stop_reason=finish_reason,
            is_final=finish_reason is not None,
        )
```

### Step 4: Add Error Handling and close()

```python
    def _handle_http_error(self, error: httpx.HTTPStatusError) -> ProviderError:
        """Handle HTTP errors and convert to appropriate ProviderError.

        Args:
            error: The HTTP error

        Raises:
            ProviderAuthError: For authentication failures
            ProviderRateLimitError: For rate limiting
            ProviderError: For other errors
        """
        status_code = error.response.status_code
        error_body = ""
        try:
            error_body = error.response.text[:500]
        except Exception:
            pass

        if status_code == 401 or status_code == 403:
            raise ProviderAuthError(
                message=f"Authentication failed: {error_body}",
                provider=self.name,
                raw_error=error,
            )
        elif status_code == 429:
            raise ProviderRateLimitError(
                message=f"Rate limit exceeded: {error_body}",
                provider=self.name,
                status_code=429,
                raw_error=error,
            )
        else:
            raise ProviderError(
                message=f"HTTP error {status_code}: {error_body}",
                provider=self.name,
                status_code=status_code,
                raw_error=error,
            )

    async def close(self) -> None:
        """Close HTTP client and release resources."""
        await self.client.aclose()
```

---

**Continue to [Part 3: Tool Calling Adapter](part-3-tool-calling-adapter.md)**

---

**Last Updated:** February 01, 2026
**Part 2 of 5**
