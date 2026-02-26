StateGraph API Reference
========================

This section documents the StateGraph workflow engine.

StateGraph Class
----------------

.. autoclass:: victor.framework.StateGraph
   :members:
   :undoc-members:
   :show-inheritance:

Creating StateGraphs
--------------------

Basic StateGraph
^^^^^^^^^^^^^^^^

.. code-block:: python

   from victor.framework import StateGraph

   workflow = StateGraph()

   # Add nodes
   workflow.add_node("step1", my_function)
   workflow.add_node("step2", another_function)

   # Add edges
   workflow.set_entry_point("step1")
   workflow.add_edge("step1", "step2")
   workflow.set_finish_point("step2")

   # Compile and run
   compiled = workflow.compile()
   result = await compiled.ainvoke({"input": "data"})

StateGraph with Conditional Edges
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def route_function(state):
       if state["category"] == "technical":
           return "technical_node"
       else:
           return "general_node"

   workflow = StateGraph()
   workflow.add_node("classify", classify_node)
   workflow.add_node("technical_node", technical_handler)
   workflow.add_node("general_node", general_handler)

   workflow.set_entry_point("classify")

   workflow.add_conditional_edges(
       "classify",
       route_function,
       {
           "technical": "technical_node",
           "general": "general_node"
       }
   )

StateGraph Methods
------------------

add_node()
^^^^^^^^^^

.. automethod:: victor.framework.StateGraph.add_node

add_edge()
^^^^^^^^^^

.. automethod:: victor.framework.StateGraph.add_edge

add_conditional_edges()
^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: victor.framework.StateGraph.add_conditional_edges

set_entry_point()
^^^^^^^^^^^^^^^^^

.. automethod:: victor.framework.StateGraph.set_entry_point

set_finish_point()
^^^^^^^^^^^^^^^^^^

.. automethod:: victor.framework.StateGraph.set_finish_point

compile()
^^^^^^^^^

.. automethod:: victor.framework.StateGraph.compile

StateGraph Features
-------------------

Checkpointing
~~~~~~~~~~~~~

.. code-block:: python

   workflow = StateGraph()

   # Enable checkpointing
   workflow.set_checkpointinator(checkpointinator)

   # Resume from checkpoint
   result = await compiled.ainvoke(
       {"input": "data"},
       config={"configurable": {"thread_id": "session-123"}}
   )

Copy-on-Write Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

StateGraph uses copy-on-write optimization for efficient state management:

.. code-block:: python

   # State is copied only when modified
   # Reduces memory usage for large workflows
   result = await compiled.ainvoke({"input": "data"})

Human-in-the-Loop
~~~~~~~~~~~~~~~~~

Add approval steps to workflows:

.. code-block:: python

   workflow = StateGraph()

   # Add human approval node
   workflow.add_node("human_approval", human_handler)

   # Only proceed if approved
   workflow.add_conditional_edges(
       "human_approval",
       lambda state: "proceed" if state["approved"] else "revise",
       {
           "proceed": "next_step",
           "revise": "revision_step"
       }
   )

Examples
--------

Example 1: Document Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   async def validate(state):
       # Validate input
       return {"valid": True}

   async def process(state):
       # Process document
       return {"result": "processed"}

   async def finalize(state):
       # Finalize output
       return {"output": "done"}

   workflow = StateGraph()
   workflow.add_node("validate", validate)
   workflow.add_node("process", process)
   workflow.add_node("finalize", finalize)

   workflow.set_entry_point("validate")
   workflow.add_edge("validate", "process")
   workflow.add_edge("process", "finalize")
   workflow.set_finish_point("finalize")

   compiled = workflow.compile()
   result = await compiled.ainvoke({"document": "content"})

Example 2: Parallel Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   async def analysis_a(state):
       return {"result_a": "analysis A"}

   async def analysis_b(state):
       return {"result_b": "analysis B"}

   async def combine(state):
       return {
           "combined": f"{state['result_a']} + {state['result_b']}"
       }

   workflow = StateGraph()
   workflow.add_node("analysis_a", analysis_a)
   workflow.add_node("analysis_b", analysis_b)
   workflow.add_node("combine", combine)

   workflow.set_entry_point("analysis_a")
   workflow.set_entry_point("analysis_b")

   workflow.add_edge("analysis_a", "combine")
   workflow.add_edge("analysis_b", "combine")
   workflow.set_finish_point("combine")

   compiled = workflow.compile()
   result = await compiled.ainvoke({"input": "data"})
