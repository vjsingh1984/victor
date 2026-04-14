#!/usr/bin/env python3
"""ToolFactory plugin example for lazy tool creation.

This example demonstrates the ToolFactory protocol for creating
tools on-demand. Use this pattern for:

- Expensive-to-initialize tools (e.g., loading large models)
- Tools requiring runtime configuration
- Tools with heavy resource requirements
- Tools that should only be created when actually used

Key concepts:
- ToolFactory protocol
- ToolPluginHelper.from_factories()
- Lazy initialization
"""

from typing import Any, Dict

from victor_sdk.core.plugins import VictorPlugin, PluginContext
from victor_sdk.verticals.protocols import ToolFactory, ToolPluginHelper
from victor.tools.base import BaseTool, ToolMetadata, ToolResult
from victor.tools.enums import AccessMode, CostTier, DangerLevel


# =============================================================================
# Expensive Tool (should be created lazily)
# =============================================================================

class ExpensiveAnalysisTool(BaseTool):
    """An expensive tool that should be created lazily.

    This tool simulates expensive initialization like:
    - Loading large ML models
    - Connecting to databases
    - Allocating significant resources
    """

    def __init__(self, model_path: str, cache_size: int = 1000):
        """Initialize the expensive tool.

        Args:
            model_path: Path to the model file
            cache_size: Size of cache in MB
        """
        super().__init__(
            name="expensive_analysis",
            description="Deep code analysis with expensive initialization",
            metadata=ToolMetadata(
                name="expensive_analysis",
                description="Deep code analysis with expensive initialization",
                cost_tier=CostTier.HIGH,
                danger_level=DangerLevel.LOW,
                access_mode=AccessMode.READ,
            ),
        )

        self._model_path = model_path
        self._cache_size = cache_size

        # Simulate expensive initialization
        print(f"  [ExpensiveAnalysisTool] Loading model from {model_path}...")
        print(f"  [ExpensiveAnalysisTool] Allocating {cache_size}MB cache...")
        print(f"  [ExpensiveAnalysisTool] Initialization complete!")

    @property
    def name(self) -> str:
        return "expensive_analysis"

    @property
    def description(self) -> str:
        return "Deep code analysis with expensive initialization"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Code to analyze",
                },
            },
            "required": ["code"],
        }

    async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
        """Execute the analysis."""
        code = kwargs.get("code", "")

        # Simulate analysis
        analysis = {
            "complexity": len(code) // 10,
            "functions": code.count("def "),
            "classes": code.count("class "),
        }

        return ToolResult(
            success=True,
            output=analysis,
            error=None,
        )


# =============================================================================
# ToolFactory Implementation
# =============================================================================

class AnalysisToolFactory(ToolFactory):
    """Factory for creating ExpensiveAnalysisTool instances.

    This factory implements the ToolFactory protocol:
    - __call__(): Create and return tool instance
    - name property: Return tool name
    """

    def __init__(self, model_path: str, cache_size: int = 1000):
        """Initialize the factory.

        Args:
            model_path: Path to model file (for tool initialization)
            cache_size: Cache size in MB (for tool initialization)
        """
        self._model_path = model_path
        self._cache_size = cache_size
        self._call_count = 0

    def __call__(self) -> ExpensiveAnalysisTool:
        """
        Create tool instance on demand.

        This method is called only when the tool is actually needed.
        The tool is created lazily - not during plugin registration.

        Returns:
            ExpensiveAnalysisTool instance
        """
        self._call_count += 1
        print(f"  [AnalysisToolFactory] Creating tool instance (call #{self._call_count})...")

        return ExpensiveAnalysisTool(
            model_path=self._model_path,
            cache_size=self._cache_size,
        )

    @property
    def name(self) -> str:
        """Return tool name."""
        return "expensive_analysis"


# =============================================================================
# Plugin Implementation
# =============================================================================

class FactoryPlugin(VictorPlugin):
    """Plugin demonstrating ToolFactory usage.

    This plugin shows how to:
    1. Create ToolFactory instances
    2. Use ToolPluginHelper.from_factories()
    3. Enable lazy tool creation
    """

    @property
    def name(self) -> str:
        """Return plugin identifier."""
        return "factory_plugin"

    def register(self, context: PluginContext) -> None:
        """Register tools using factories."""
        # Create factories for different configurations
        factories = {
            # Fast analysis (smaller cache)
            "fast_analysis": AnalysisToolFactory(
                model_path="/models/fast_analysis.pkl",
                cache_size=500,
            ),
            # Deep analysis (larger cache)
            "deep_analysis": AnalysisToolFactory(
                model_path="/models/deep_analysis.pkl",
                cache_size=2000,
            ),
        }

        # Use ToolPluginHelper to register factories
        # Tools will only be created when actually used
        helper = ToolPluginHelper.from_factories(factories)
        helper.register(context)

        print(f"[{self.name}] Registered 2 tool factories (lazy creation)")
        print(f"[{self.name}] Tools will be created on first use")


# =============================================================================
# Entry Point
# =============================================================================

def plugin() -> VictorPlugin:
    """Entry point for plugin discovery."""
    return FactoryPlugin()


# =============================================================================
# Demo / Testing
# =============================================================================

async def demo():
    """Demonstrate factory plugin with lazy creation."""
    from victor.tools.registry import ToolRegistry

    print("=" * 60)
    print("ToolFactory Plugin Demo - Lazy Creation")
    print("=" * 60)

    # Create registry and register plugin
    registry = ToolRegistry()
    plugin_instance = plugin()

    print(f"\nPlugin Name: {plugin_instance.name}")
    print("\nRegistering plugin with factories...")
    registry.register_plugin(plugin_instance)

    print("\nAvailable Tools (not yet created):")
    for tool_name in registry.list_tools():
        print(f"  - {tool_name}")

    print("\n" + "-" * 60)
    print("First access to fast_analysis (triggers creation):")
    print("-" * 60)

    # First access - tool is created now
    tool1 = registry.get("fast_analysis")
    if tool1:
        result = await tool1.execute({}, code="def hello(): pass")
        print(f"Result: {result.output}")

    print("\n" + "-" * 60)
    print("Second access to fast_analysis (uses cached instance):")
    print("-" * 60)

    # Second access - tool already exists
    tool2 = registry.get("fast_analysis")
    if tool2:
        result = await tool2.execute({}, code="class Foo: pass")
        print(f"Result: {result.output}")

    print("\n" + "-" * 60)
    print("Access to deep_analysis (triggers creation):")
    print("-" * 60)

    # Access to different tool
    tool3 = registry.get("deep_analysis")
    if tool3:
        result = await tool3.execute({}, code="def bar(): pass")
        print(f"Result: {result.output}")

    print("\n" + "=" * 60)
    print("Notice: Tools created only when first accessed!")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio

    print("""
╔══════════════════════════════════════════════════════════════╗
║          Victor SDK Plugin - ToolFactory Example               ║
╠══════════════════════════════════════════════════════════════╣
║  This example demonstrates:                                  ║
║  1. ToolFactory protocol for lazy creation                    ║
║  2. ToolPluginHelper.from_factories()                         ║
║  3. Expensive tool initialization                             ║
║  4. On-demand tool creation                                  ║
╚══════════════════════════════════════════════════════════════╝
""")

    asyncio.run(demo())
