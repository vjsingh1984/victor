# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example: Creating a custom workflow compiler plugin.

This demonstrates how to create a custom plugin that compiles workflows
from a custom source (in this case, embedded Python dictionaries).
"""

import asyncio
from typing import Any, Dict, Optional

from victor import Agent
from victor.workflows.plugins.compiler_plugin import WorkflowCompilerPlugin
from victor.workflows.create import create_compiler


class DictCompilerPlugin(WorkflowCompilerPlugin):
    """Plugin that loads workflows from Python dictionaries.

    This is useful for:
    - Testing workflows without files
    - Dynamically generated workflows
    - Workflow templates
    """

    def __init__(self, workflow_dict: Dict[str, Any]):
        """Initialize with workflow dictionary.

        Args:
            workflow_dict: Dictionary containing workflow definition
        """
        self.workflow_dict = workflow_dict

    def compile(
        self,
        source: str,
        *,
        workflow_name: Optional[str] = None,
        validate: bool = True,
    ) -> Any:
        """Compile workflow from dictionary.

        Args:
            source: Key to lookup in workflow_dict
            workflow_name: Name of workflow (ignored, uses source)
            validate: Whether to validate before compilation

        Returns:
            CompiledGraphProtocol instance
        """
        # Look up workflow in dictionary
        if source not in self.workflow_dict:
            raise ValueError(
                f"Workflow '{source}' not found. "
                f"Available workflows: {list(self.workflow_dict.keys())}"
            )

        workflow_def = self.workflow_dict[source]

        # Validate structure
        if validate:
            self._validate_workflow(workflow_def)

        # Convert to YAML and delegate to YAML compiler
        import yaml

        yaml_str = yaml.dump(workflow_def)

        from victor.workflows.create import create_compiler

        yaml_compiler = create_compiler("yaml://")
        return yaml_compiler.compile(yaml_str, validate=validate)

    def _validate_workflow(self, workflow: Dict[str, Any]) -> None:
        """Validate workflow structure.

        Args:
            workflow: Workflow definition

        Raises:
            ValueError: If workflow is invalid
        """
        # Check for required keys
        if "workflows" not in workflow:
            raise ValueError("Workflow must contain 'workflows' key")

        if not isinstance(workflow["workflows"], dict):
            raise ValueError("'workflows' must be a dictionary")

        # Validate each workflow
        for name, wf_def in workflow["workflows"].items():
            if "nodes" not in wf_def:
                raise ValueError(f"Workflow '{name}' must contain 'nodes'")

    def validate_source(self, source: str) -> bool:
        """Check if source exists in dictionary.

        Args:
            source: Workflow key

        Returns:
            True if source exists, False otherwise
        """
        return source in self.workflow_dict

    def get_cache_key(self, source: str) -> str:
        """Generate cache key.

        Args:
            source: Workflow key

        Returns:
            Cache key string
        """
        import hashlib

        # Convert workflow to string for hashing
        workflow_str = str(self.workflow_dict.get(source, {}))
        return hashlib.md5(workflow_str.encode()).hexdigest()


# Example workflow definition
EXAMPLE_WORKFLOW = {
    "workflows": {
        "simple_task": {
            "metadata": {
                "name": "Simple Task",
                "description": "A simple one-step workflow",
                "version": "0.5.0",
            },
            "nodes": [
                {
                    "id": "start",
                    "type": "agent",
                    "role": "assistant",
                    "goal": "Answer the user's question helpfully",
                    "tool_budget": 5,
                    "next": [],
                }
            ],
        }
    }
}


async def main():
    """Demonstrate custom plugin usage."""
    print("🔌 Custom Workflow Compiler Plugin Demo\n")
    print("=" * 60)

    # Step 1: Create the plugin
    print("\n📦 Step 1: Creating custom plugin")
    print("-" * 60)

    plugin = DictCompilerPlugin(EXAMPLE_WORKFLOW)
    print(f"✅ Plugin created with {len(EXAMPLE_WORKFLOW['workflows'])} workflow(s)")

    # Step 2: Register and create compiler
    print("\n⚙️  Step 2: Registering plugin and creating compiler")
    print("-" * 60)

    # Register plugin for 'dict://' scheme
    compiler = create_compiler("dict://simple_task", plugin_class=DictCompilerPlugin)
    print("✅ Compiler registered for 'dict://' scheme")

    # Step 3: Compile workflow
    print("\n🔨 Step 3: Compiling workflow from dictionary")
    print("-" * 60)

    try:
        compiled_workflow = compiler.compile("simple_task", validate=True)
        print("✅ Workflow compiled successfully!")
        print(f"   Workflow: {EXAMPLE_WORKFLOW['workflows']['simple_task']['metadata']['name']}")
        print(f"   Description: {EXAMPLE_WORKFLOW['workflows']['simple_task']['metadata']['description']}")
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        return

    # Step 4: Execute workflow
    print("\n🚀 Step 4: Executing workflow")
    print("-" * 60)

    try:
        # Create agent for workflow execution
        agent = await Agent.create()

        # Execute workflow
        result = await compiled_workflow.invoke({"query": "What is Python?"})
        print("✅ Workflow executed successfully!")
        print(f"   Result: {result}")
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        print("   (This is expected if no API keys are configured)")

    # Step 5: Demonstrate validation
    print("\n✓ Step 5: Demonstrating validation")
    print("-" * 60)

    # Valid source
    is_valid = plugin.validate_source("simple_task")
    print(f"   'simple_task' is valid: {is_valid}")

    # Invalid source
    is_valid = plugin.validate_source("nonexistent")
    print(f"   'nonexistent' is valid: {is_valid}")

    # Step 6: Demonstrate cache keys
    print("\n🔑 Step 6: Demonstrating cache keys")
    print("-" * 60)

    cache_key = plugin.get_cache_key("simple_task")
    print(f"   Cache key for 'simple_task': {cache_key}")

    # Summary
    print("\n📊 Summary")
    print("-" * 60)
    print("✅ Created custom DictCompilerPlugin")
    print("✅ Registered plugin for 'dict://' scheme")
    print("✅ Compiled workflow from Python dictionary")
    print("✅ Executed workflow (if API keys configured)")
    print("✅ Demonstrated validation and caching")

    print("\n💡 Key Takeaways:")
    print("- Plugins enable loading workflows from any source")
    print("- Implement WorkflowCompilerPlugin protocol")
    print("- Register with create_compiler() for custom URI schemes")
    print("- Useful for testing, dynamic workflows, templates")

    print("\n🔗 Next Steps:")
    print("- See examples/plugins/json_compiler_plugin.py for JSON plugin")
    print("- See examples/plugins/s3_compiler_plugin.py for S3 plugin")
    print("- Read docs/features/plugin_development.md for full guide")


async def advanced_example():
    """Advanced example with multiple workflows."""
    print("\n\n🎯 Advanced Example: Multiple Workflows\n")
    print("=" * 60)

    # Define multiple workflows
    workflows = {
        "research": {
            "workflows": {
                "deep_research": {
                    "metadata": {"name": "Deep Research", "version": "1.0"},
                    "nodes": [
                        {
                            "id": "search",
                            "type": "agent",
                            "role": "researcher",
                            "goal": "Search for information",
                            "tool_budget": 10,
                            "next": ["analyze"],
                        },
                        {
                            "id": "analyze",
                            "type": "agent",
                            "role": "analyst",
                            "goal": "Analyze findings",
                            "tool_budget": 5,
                            "next": [],
                        },
                    ],
                }
            }
        },
        "coding": {
            "workflows": {
                "code_review": {
                    "metadata": {"name": "Code Review", "version": "1.0"},
                    "nodes": [
                        {
                            "id": "review",
                            "type": "agent",
                            "role": "reviewer",
                            "goal": "Review code for quality",
                            "tool_budget": 15,
                            "next": [],
                        }
                    ],
                }
            }
        },
    }

    # Create plugin with multiple workflows
    plugin = DictCompilerPlugin(workflows)

    print(f"✅ Plugin created with {len(workflows)} workflow categories")

    # List all available workflows
    print("\n📋 Available workflows:")
    for category, workflow_dict in workflows.items():
        for name in workflow_dict["workflows"].keys():
            print(f"   - {category}:{name}")

    # Compile and execute specific workflow
    print("\n🔨 Compiling 'research:deep_research'...")
    compiler = create_compiler(
        "dict://deep_research", plugin_class=DictCompilerPlugin, workflow_dict=workflows
    )

    try:
        compiled = compiler.compile("deep_research", validate=True)
        print("✅ Workflow compiled successfully!")
    except Exception as e:
        print(f"❌ Compilation failed: {e}")


if __name__ == "__main__":
    print("Custom Plugin Example")
    print("=" * 60)
    print("This example demonstrates creating a custom workflow")
    print("compiler plugin that loads workflows from Python dictionaries.")
    print("\n")

    # Run basic example
    asyncio.run(main())

    # Run advanced example
    asyncio.run(advanced_example())

    print("\n✅ Examples completed!")
