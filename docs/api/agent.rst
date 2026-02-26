Agent API Reference
===================

This section documents the Victor Agent API.

Agent Class
-----------

.. autoclass:: victor.framework.Agent
   :members:
   :undoc-members:
   :show-inheritance:

Creating Agents
---------------

Basic Agent
^^^^^^^^^^^

.. code-block:: python

   from victor import Agent

   # Create agent with defaults
   agent = Agent.create()

   # Run a query
   result = await agent.run("Your question here")
   print(result.content)

Agent with Provider
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Use specific provider
   agent = Agent.create(
       provider="anthropic",
       model="claude-3-opus-20240229"
   )

Agent with Tools
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Agent with specific tools
   agent = Agent.create(
       tools=["read", "write", "ls", "grep"]
   )

Agent with Vertical
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Use domain-specific vertical
   agent = Agent.create(
       vertical="coding",
       tools=["read", "write", "edit", "grep"]
   )

Agent Methods
-------------

run()
^^^^^

.. automethod:: victor.framework.Agent.run

stream()
^^^^^^^

.. automethod:: victor.framework.Agent.stream

chat()
^^^^^

.. automethod:: victor.framework.Agent.chat

run_workflow()
^^^^^^^^^^^^^

.. automethod:: victor.framework.Agent.run_workflow

stream_workflow()
^^^^^^^^^^^^^^^^^

.. automethod:: victor.framework.Agent.stream_workflow

run_team()
^^^^^^^^^^

.. automethod:: victor.framework.Agent.run_team

Agent Configuration
-------------------

Parameters
~~~~~~~~~~

provider
   LLM provider to use (e.g., "openai", "anthropic", "ollama")

model
   Model identifier (e.g., "gpt-4", "claude-3-opus-20240229")

temperature
   Sampling temperature (0.0 = focused, 1.0 = creative)

max_tokens
   Maximum tokens to generate

tools
   List of tool names or preset ("minimal", "default", "full", "airgapped")

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
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   agent = Agent.create()

   async for event in agent.stream("Tell me a story"):
       if event.type == "content":
           print(event.content, end="", flush=True)

Example 2: Multi-turn Conversation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   agent = Agent.create()

   response1 = await agent.chat("My name is Alice")
   response2 = await agent.chat("What's my name?")
   # Agent remembers "Alice"

Example 3: Code Review Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   agent = Agent.create(
       vertical="coding",
       tools=["read", "grep"],
       temperature=0.3,
       system_prompt="You are a senior code reviewer."
   )

   result = await agent.run(
       "Review the code in main.py and suggest improvements"
   )
