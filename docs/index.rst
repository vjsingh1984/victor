.. Victor documentation master file, created by sphinx-quickstart.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Victor AI Framework
==============================

Victor is an open-source agentic AI framework for building autonomous agents,
multi-step workflows, and domain-specific vertical applications.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   guides/quickstart
   guides/installation
   guides/first-agent
   guides/first-workflow
   api/agent
   api/stategraph
   api/workflows
   api/tools
   api/providers

Quick Links
-----------

* `Quick Start Guide <guides/quickstart.html>`_
* `Installation Guide <guides/installation.html>`_
* `API Reference <api/index.html>`_
* `Examples <examples/README.html>`_

Key Features
-----------

* **Multi-Provider Support**: 22 LLM provider integrations
* **Tool System**: 33+ built-in tools across 9 categories
* **Workflows**: YAML and Python-based workflow orchestration
* **StateGraph**: LangGraph-inspired workflow engine with checkpoints
* **Verticals**: Domain-specific applications (coding, devops, research, RAG)
* **Observability**: Built-in metrics, tracing, and event system

Getting Started
---------------

.. code-block:: python

   import asyncio
   from victor import Agent

   async def main():
       # Create an agent
       agent = Agent.create()

       # Run a query
       result = await agent.run("What is the capital of France?")
       print(result.content)

   asyncio.run(main())

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
