# Integrating a New LLM Provider

Complete tutorial for creating a custom LLM provider for Victor.

---

## Tutorial Parts

1. **[Part 1: Provider Architecture & Steps 1-2](part-1-provider-architecture.md)**
   - The BaseProvider class
   - Required and optional methods
   - Core data types
   - Step 1: Create the provider file
   - Step 2: Implement the `chat()` method

2. **[Part 2: Streaming & Error Handling](part-2-streaming-error-handling.md)**
   - Step 3: Implement the `stream()` method
   - Step 4: Add error handling and `close()` method
   - HTTP error handling

3. **[Part 3: Tool Calling Adapter](part-3-tool-calling-adapter.md)**
   - When to create an adapter
   - Creating a custom tool calling adapter
   - Converting tools and parsing responses

4. **[Part 4: Registration & Testing](part-4-registration-testing.md)**
   - Model capabilities configuration
   - Provider registration methods
   - Unit and integration tests

5. **[Part 5: Best Practices & Examples](part-5-best-practices-examples.md)**
   - Error handling best practices
   - Circuit breaker and retry strategies
   - Complete provider example
   - Quick reference checklist

---

## What You Will Build

A complete LLM provider integration that:
- Connects to any LLM API (cloud or local)
- Supports both synchronous and streaming chat completions
- Handles tool/function calling
- Integrates with Victor's circuit breaker for resilience
- Is properly registered and testable

---

**Last Updated:** February 01, 2026
