Tools API Reference
===================

This section documents the Victor tool system.

Built-in Tools
--------------

Victor includes 33+ tools organized into categories:

Filesystem Tools
^^^^^^^^^^^^^^^^

**read** - Read file contents

.. code-block:: python

   result = await tool_registry.execute("read", {
       "file_path": "path/to/file.txt"
   })

**write** - Write content to file

.. code-block:: python

   result = await tool_registry.execute("write", {
       "file_path": "path/to/file.txt",
       "content": "File content"
   })

**edit** - Edit file with search/replace

.. code-block:: python

   result = await tool_registry.execute("edit", {
       "file_path": "path/to/file.txt",
       "old_string": "old text",
       "new_string": "new text"
   })

**ls** - List directory contents

.. code-block:: python

   result = await tool_registry.execute("ls", {
       "path": "path/to/directory"
   })

**grep** - Search files for patterns

.. code-block:: python

   result = await tool_registry.execute("grep", {
       "pattern": "search_term",
       "path": "path/to/search"
   })

Git Tools
^^^^^^^^^

**git_status** - Show git repository status

**git_commit** - Commit changes

**git_push** - Push to remote

**git_pull** - Pull from remote

**git_diff** - Show differences

Web Tools
^^^^^^^^^

**web_search** - Search the web

.. code-block:: python

   result = await tool_registry.execute("web_search", {
       "query": "search query"
   })

**web_fetch** - Fetch URL content

.. code-block:: python

   result = await tool_registry.execute("web_fetch", {
       "url": "https://example.com"
   })

Execution Tools
^^^^^^^^^^^^^^^

**shell** - Execute shell commands

.. code-block:: python

   result = await tool_registry.execute("shell", {
       "command": "ls -la"
   })

**python** - Execute Python code

.. code-block:: python

   result = await tool_registry.execute("python", {
       "code": "print('Hello, World!')"
   })

Docker Tools
^^^^^^^^^^^^

**docker_build** - Build Docker image

**docker_run** - Run Docker container

**docker_exec** - Execute in container

**docker_stop** - Stop container

Analysis Tools
^^^^^^^^^^^^^^

**overview** - Codebase overview

**code_search** - Semantic code search

**graph** - Code graph analysis

Tool Categories
---------------

.. autoclass:: victor.framework.ToolCategory
   :members:
   :undoc-members:

Tool Presets
------------

**minimal** - No filesystem access

.. code-block:: python

   agent = Agent.create(tools="minimal")

**default** - Safe filesystem operations

.. code-block:: python

   agent = Agent.create(tools="default")

**full** - All available tools

.. code-block:: python

   agent = Agent.create(tools="full")

**airgapped** - No external API access

.. code-block:: python

   agent = Agent.create(tools="airgapped")

Custom Tool Selection
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   agent = Agent.create(tools=[
       "read", "write", "ls", "grep", "shell"
   ])

Creating Custom Tools
---------------------

Base Tool Class
^^^^^^^^^^^^^^^

.. autoclass:: victor.tools.base.BaseTool
   :members:
   :undoc-members:
   :show-inheritance:

Tool Metadata
^^^^^^^^^^^^^

.. autoclass:: victor.tools.base.ToolMetadata
   :members:
   :undoc-members:

Example: Custom Tool
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from victor.tools.base import BaseTool, ToolMetadata, ToolCategory

   class MyCustomTool(BaseTool):
       """A custom tool example."""

       @property
       def metadata(self) -> ToolMetadata:
           return ToolMetadata(
               name="my_tool",
               category=ToolCategory.CUSTOM,
               description="Description of my tool",
               cost_tier=CostTier.LOW,
               idempotent=True,
               keywords=["custom", "example"]
           )

       async def execute(self, **kwargs) -> str:
           """Execute the tool."""
           # Tool implementation here
           return "Result from my tool"

   # Register the tool
   from victor.framework import SharedToolRegistry
   SharedToolRegistry.get_instance().register(MyCustomTool())

Tool Decorator (Coming Soon)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from victor.framework.tools import tool

   @tool(
       category=ToolCategory.FILESYSTEM,
       cost_tier=CostTier.LOW,
       idempotent=True
   )
   async def my_tool(file_path: str) -> str:
       """Tool description."""
       return Path(file_path).read_text()

Tool Safety
-----------

Some tools have safety restrictions:

**Destructive operations**: May require confirmation

**System commands**: May be restricted in airgapped mode

**File operations**: May have path restrictions

Configure safety rules:

.. code-block:: python

   from victor.framework.middleware import GitSafetyMiddleware

   # Add git safety middleware
   agent = Agent.create(
       middleware=[GitSafetyMiddleware()]
   )

Examples
--------

Example 1: File Analysis Tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from victor.tools.base import BaseTool, ToolMetadata, ToolCategory
   from pathlib import Path

   class FileAnalyzer(BaseTool):
       """Analyze file statistics."""

       @property
       def metadata(self) -> ToolMetadata:
           return ToolMetadata(
               name="analyze_file",
               category=ToolCategory.FILESYSTEM,
               description="Analyze file statistics",
               cost_tier=CostTier.LOW,
               idempotent=True
           )

       async def execute(self, file_path: str) -> dict:
           """Analyze file."""
           path = Path(file_path)

           return {
               "size": path.stat().st_size,
               "lines": len(path.read_text().splitlines()),
               "extension": path.suffix
           }

Example 2: API Call Tool
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import aiohttp

   class APICall(BaseTool):
       """Make API calls."""

       @property
       def metadata(self) -> ToolMetadata:
           return ToolMetadata(
               name="api_call",
               category=ToolCategory.WEB,
               description="Make HTTP API calls",
               cost_tier=CostTier.MEDIUM,
               idempotent=False
           )

       async def execute(self, url: str, method: str = "GET") -> dict:
           """Execute API call."""
           async with aiohttp.ClientSession() as session:
               async with session.request(method, url) as response:
                   return await response.json()

Tool Registry
-------------

.. autoclass:: victor.framework.SharedToolRegistry
   :members:
   :undoc-members:

Accessing Tools
^^^^^^^^^^^^^^^

.. code-block:: python

   from victor.framework import SharedToolRegistry

   registry = SharedToolRegistry.get_instance()

   # Get tool by name
   tool = registry.get_tool("read")

   # List all tools
   all_tools = registry.list_tools()

   # Get tools by category
   fs_tools = registry.get_tools_by_category(ToolCategory.FILESYSTEM)
