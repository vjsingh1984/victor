Workflows API Reference
========================

This section documents the workflow system and YAML workflow compiler.

WorkflowEngine
--------------

.. autoclass:: victor.framework.WorkflowEngine
   :members:
   :undoc-members:
   :show-inheritance:

YAML Workflows
--------------

Creating YAML Workflows
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   name: "My Workflow"
   description: "A simple workflow"

   nodes:
     - id: "step1"
       type: "agent"
       config:
         prompt: "Process: {{input}}"

   edges:
     - from: "start"
       to: "step1"
     - from: "step1"
       to: "complete"

Node Types
~~~~~~~~~~

**agent**: Run an LLM agent

.. code-block:: yaml

   - id: "my_agent"
     type: "agent"
     config:
       prompt: "Your prompt here"
       provider: "openai"
       model: "gpt-4"
       temperature: 0.7
       tools: ["read", "write"]

**handler**: Execute a tool or function

.. code-block:: yaml

   - id: "read_file"
     type: "handler"
     config:
       tool: "read"
       arguments:
         file_path: "{{file_path}}"

**human**: Human-in-the-loop approval

.. code-block:: yaml

   - id: "approval"
     type: "human"
     config:
       prompt: "Review and approve"
       instructions: "Please review the output"

**passthrough**: Pass data through unchanged

.. code-block:: yaml

   - id: "pass"
     type: "passthrough"
     config:
       output: "{{input}}"

**compute**: Execute a Python expression

.. code-block:: yaml

   - id: "calculate"
     type: "compute"
     config:
       expression: "input * 2"

Edge Types
~~~~~~~~~~

**Simple Edge**: Always follow this path

.. code-block:: yaml

   edges:
     - from: "node1"
       to: "node2"

**Conditional Edge**: Follow based on condition

.. code-block:: yaml

   edges:
     - from: "classifier"
       to: "option_a"
       condition: "{{classifier.output}} == 'a'"
     - from: "classifier"
       to: "option_b"
       condition: "{{classifier.output}} == 'b'"

WorkflowEngine Methods
----------------------

execute_yaml()
^^^^^^^^^^^^^^^

.. automethod:: victor.framework.WorkflowEngine.execute_yaml

stream_yaml()
^^^^^^^^^^^^^

.. automethod:: victor.framework.WorkflowEngine.stream_yaml

enable_caching()
^^^^^^^^^^^^^^^^

.. automethod:: victor.framework.WorkflowEngine.enable_caching

clear_cache()
^^^^^^^^^^^^

.. automethod:: victor.framework.WorkflowEngine.clear_cache

Examples
--------

Example 1: Content Analysis Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   name: "Content Analyzer"
   description: "Analyze and summarize content"

   nodes:
     - id: "analyze"
       type: "agent"
       config:
         prompt: |
           Analyze the following content:
           {{input}}

           Identify:
           1. Main topic
           2. Key points
           3. Sentiment

     - id: "summarize"
       type: "agent"
       config:
         prompt: |
           Summarize in 2-3 sentences:
           {{analyze.output}}

   edges:
     - from: "start"
       to: "analyze"
     - from: "analyze"
       to: "summarize"
     - from: "summarize"
       to: "complete"

Example 2: Parallel Processing Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   name: "Multi-Perspective Analysis"
   description: "Analyze from multiple angles"

   nodes:
     - id: "technical"
       type: "agent"
       config:
         prompt: "Technical analysis of: {{input}}"

     - id: "business"
       type: "agent"
       config:
         prompt: "Business analysis of: {{input}}"

     - id: "synthesize"
       type: "agent"
       config:
         prompt: |
           Combine these analyses:
           Technical: {{technical.output}}
           Business: {{business.output}}

   edges:
     - from: "start"
       to: "technical"
     - from: "start"
       to: "business"
     - from: "technical"
       to: "synthesize"
     - from: "business"
       to: "synthesize"
     - from: "synthesize"
       to: "complete"

Running Workflows
-----------------

From Agent
~~~~~~~~~~

.. code-block:: python

   from victor import Agent

   agent = Agent.create()

   # Run workflow
   result = await agent.run_workflow(
       "workflow.yaml",
       input={"topic": "AI"}
   )

   # Stream workflow
   async for node_id, state in agent.stream_workflow("workflow.yaml"):
       print(f"Completed: {node_id}")

From WorkflowEngine
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from victor.framework import WorkflowEngine

   engine = WorkflowEngine()

   # Execute workflow
   result = await engine.execute_yaml(
       "workflow.yaml",
       input={"data": "value"}
   )

   # Stream workflow
   async for node_id, state in engine.stream_yaml("workflow.yaml"):
       print(f"Node: {node_id}, State: {state}")

Workflow Best Practices
------------------------

1. **Keep nodes focused**: Each node should do one thing well
2. **Use descriptive names**: Make workflow structure clear
3. **Handle errors**: Add error handling where needed
4. **Test incrementally**: Build and test workflows step by step
5. **Document workflows**: Add descriptions and comments

Troubleshooting
---------------

Workflow Not Executing
~~~~~~~~~~~~~~~~~~~~~~

Check YAML syntax:

.. code-block:: bash

   python -c "import yaml; yaml.safe_load(open('workflow.yaml'))"

Nodes Not Connecting
~~~~~~~~~~~~~~~~~~~~

Verify node IDs match:

.. code-block:: yaml

   nodes:
     - id: "step1"  # Must match edge reference
   edges:
     - from: "start"
       to: "step1"  # Must match node ID

Conditional Edges Not Working
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check condition syntax:

.. code-block:: yaml

   condition: "{{previous_node.output}} == 'expected'"
