Providers API Reference
========================

This section documents LLM provider integrations.

Supported Providers
-------------------

Victor supports 22+ LLM providers:

**OpenAI** - GPT-4, GPT-3.5 Turbo

**Anthropic** - Claude 3 Opus, Sonnet, Haiku

**Azure OpenAI** - Azure-hosted GPT models

**Google AI** - Gemini Pro, Ultra

**Cohere** - Command R, R+

**Ollama** - Local models (Llama 2, Mistral, etc.)

**And 16 more providers**

Provider Configuration
----------------------

Using Environment Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # OpenAI
   export OPENAI_API_KEY="sk-..."

   # Anthropic
   export ANTHROPIC_API_KEY="sk-ant-..."

   # Azure
   export AZURE_OPENAI_API_KEY="..."
   export AZURE_OPENAI_ENDPOINT="https://..."

Using Settings File
^^^^^^^^^^^^^^^^^^^^

Create ``~/.victor/settings.yaml``:

.. code-block:: yaml

   providers:
     openai:
       api_key: "sk-..."
       base_url: "https://api.openai.com/v1"

     anthropic:
       api_key: "sk-ant-..."
       max_tokens: 4096

     azure:
       api_key: "..."
       endpoint: "https://..."
       api_version: "2023-05-15"

Programmatic Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from victor import Agent

   # Use specific provider
   agent = Agent.create(
       provider="openai",
       model="gpt-4",
       temperature=0.7
   )

Provider Protocols
------------------

StreamingProvider
^^^^^^^^^^^^^^^^^

.. autoclass:: victor.providers.protocols.StreamingProvider
   :members:
   :undoc-members:

ToolCallingProvider
^^^^^^^^^^^^^^^^^^^

.. autoclass:: victor.providers.protocols.ToolCallingProvider
   :members:
   :undoc-members:

Base Provider
^^^^^^^^^^^^^

.. autoclass:: victor.providers.base.BaseProvider
   :members:
   :undoc-members:
   :show-inheritance:

Provider Examples
------------------

OpenAI Example
^^^^^^^^^^^^^^^

.. code-block:: python

   from victor import Agent

   # GPT-4
   agent = Agent.create(
       provider="openai",
       model="gpt-4"
   )

   # GPT-3.5 Turbo (faster, cheaper)
   agent = Agent.create(
       provider="openai",
       model="gpt-3.5-turbo"
   )

   # With custom parameters
   agent = Agent.create(
       provider="openai",
       model="gpt-4",
       temperature=0.7,
       max_tokens=2000
   )

Anthropic Example
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Claude 3 Opus (most capable)
   agent = Agent.create(
       provider="anthropic",
       model="claude-3-opus-20240229"
   )

   # Claude 3 Sonnet (balanced)
   agent = Agent.create(
       provider="anthropic",
       model="claude-3-sonnet-20240229"
   )

   # Claude 3 Haiku (fastest)
   agent = Agent.create(
       provider="anthropic",
       model="claude-3-haiku-20240307"
   )

Azure OpenAI Example
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   agent = Agent.create(
       provider="azure",
       model="gpt-4",
       endpoint="https://your-resource.openai.azure.com",
       api_version="2023-05-15"
   )

Google AI Example
^^^^^^^^^^^^^^^^^

.. code-block:: python

   agent = Agent.create(
       provider="google",
       model="gemini-pro"
   )

Ollama Example (Local)
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # No API key needed for local models
   agent = Agent.create(
       provider="ollama",
       model="llama2"
   )

   # Or Mistral
   agent = Agent.create(
       provider="ollama",
       model="mistral"
   )

Provider Selection Guide
------------------------

Use GPT-4 when:
- Complex reasoning required
- Code generation
- Detailed analysis
- Budget allows ($$$)

Use Claude when:
- Nuanced understanding needed
- Long context required
- Safety is critical
- Extensive outputs

Use GPT-3.5 when:
- Speed is important
- Cost is a concern
- Simple tasks
- High volume

Use Ollama when:
- Privacy required
- No internet access
- Cost must be $0
- Local inference

Provider Features
-----------------

.. list-table:: Provider Feature Comparison
   :widths: 25 25 25 25 25
   :header-rows: 1

   * - Feature
     - OpenAI
     - Anthropic
     - Azure
     - Ollama
   * - Streaming
     - ✓
     - ✓
     - ✓
     - ✓
   * - Tool Calling
     - ✓
     - ✓
     - ✓
     - ✗
   * - Function Calling
     - ✓
     - ✓
     - ✓
     - ✗
   * - Vision
     - ✓
     - ✓
     - ✓
     - ✓
   * - JSON Mode
     - ✓
     - ✗
     - ✓
     - ✗

Custom Providers
----------------

Creating a Custom Provider
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from victor.providers.base import BaseProvider
   from victor.providers.protocols import StreamingProvider, ToolCallingProvider

   class MyCustomProvider(BaseProvider, StreamingProvider, ToolCallingProvider):
       """Custom provider implementation."""

       def __init__(self, model: str, api_key: str):
           self.model = model
           self.api_key = api_key

       async def generate(self, messages: list, **kwargs) -> str:
           """Generate response."""
           # Implementation here
           pass

       async def stream_generate(self, messages: list, **kwargs):
           """Stream response."""
           # Implementation here
           pass

       def supports_tools(self) -> bool:
           """Check if tools supported."""
           return True

       async def call_tool(self, tool_name: str, tool_args: dict) -> str:
           """Call a tool."""
           # Implementation here
           pass

   # Register the provider
   from victor.providers.registry import ProviderRegistry
   ProviderRegistry.register("my_provider", MyCustomProvider)

Troubleshooting
---------------

API Key Errors
^^^^^^^^^^^^^^^

.. code-block:: python

   # Check if API key is set
   import os
   print(os.getenv("OPENAI_API_KEY"))

   # Or check in settings
   from victor.config import load_settings
   settings = load_settings()
   print(settings.providers.get("openai", {}))

Connection Errors
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Test provider connection
   import asyncio
   from victor import Agent

   async def test_connection():
       try:
           agent = Agent.create(provider="openai")
           result = await agent.run("Hello!")
           print("Connection successful!")
       except Exception as e:
           print(f"Connection failed: {e}")

   asyncio.run(test_connection())

Model Not Found
^^^^^^^^^^^^^^^

Check model name is correct:

.. code-block:: python

   # OpenAI models
   "gpt-4"
   "gpt-3.5-turbo"
   "gpt-4-turbo"

   # Anthropic models
   "claude-3-opus-20240229"
   "claude-3-sonnet-20240229"
   "claude-3-haiku-20240307"

Rate Limiting
^^^^^^^^^^^^^

Handle rate limits:

.. code-block:: python

   from tenacity import retry, stop_after_attempt, wait_exponential

   @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
   async def call_with_retry(agent, prompt):
       return await agent.run(prompt)
