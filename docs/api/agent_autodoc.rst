Agent API Reference
===================

This section documents the Victor Agent API.

.. autoclass:: victor.framework.agent.Agent
   :members:
   :undoc-members:
   :show-inheritance:

Creating Agents
---------------

Basic Agent
^^^^^^^^^^^

.. autoclass:: victor.framework.agent.Agent
   :members: create

.. code-block:: python

   from victor.framework import Agent

   # Create agent with defaults
   agent = await Agent.create()

   # Run a query
   result = await agent.run("Your question here")
   print(result.content)

Agent with Provider
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Use specific provider
   agent = await Agent.create(
       provider="anthropic",
       model="claude-sonnet-4-20250514"
   )

Agent with Tools
^^^^^^^^^^^^^^^^

.. code-block:: python

   from victor.framework import ToolSet

   # Agent with specific tools
   agent = await Agent.create(
       tools=ToolSet.default()
   )

Agent with Vertical
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Use domain-specific vertical
   agent = await Agent.create(
       vertical="coding"
   )

Agent Methods
-------------

run()
^^^^^

.. automethod:: victor.framework.agent.Agent.run

stream()
^^^^^^^

.. automethod:: victor.framework.agent.Agent.stream

chat()
^^^^^

.. automethod:: victor.framework.agent.Agent.chat

run_workflow()
^^^^^^^^^^^^^

.. automethod:: victor.framework.agent.Agent.run_workflow

stream_workflow()
^^^^^^^^^^^^^^^^^

.. automethod:: victor.framework.agent.Agent.stream_workflow

run_team()
^^^^^^^^^^

.. automethod:: victor.framework.agent.Agent.run_team

Agent Configuration
-------------------

Parameters
~~~~~~~~~~

provider
   LLM provider to use (e.g., "openai", "anthropic", "ollama")

model
   Model identifier (e.g., "gpt-4", "claude-sonnet-4-20250514")

temperature
   Sampling temperature (0.0 = focused, 1.0 = creative)

max_tokens
   Maximum tokens to generate

tools
   List of tool names or ToolSet preset

vertical
   Domain-specific vertical ("coding", "devops", "research", etc.)

system_prompt
   Custom system prompt to override default behavior

enable_observability
   Enable metrics collection and tracing

session_id
   Session identifier for tracking

Examples
--------

Example 1: Streaming Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   agent = await Agent.create()

   async for event in agent.stream("Tell me a story"):
       if event.type == EventType.CONTENT:
           print(event.content, end="", flush=True)

Example 2: Multi-turn Conversation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   agent = await Agent.create()

   response1 = await agent.chat("My name is Alice")
   response2 = await agent.chat("What's my name?")
   # Agent remembers "Alice"

Example 3: Code Review Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   agent = await Agent.create(
       vertical="coding",
       temperature=0.3
   )

   result = await agent.run(
       "Review the code in main.py and suggest improvements"
   )
