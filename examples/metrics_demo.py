"""Demo of Victor's Code Metrics Tool.

Usage:
    python examples/metrics_demo.py
"""

import asyncio
import tempfile
from pathlib import Path
from victor.tools.metrics_tool import MetricsTool


async def main():
    """Run metrics demos."""
    print("🎯 Victor Code Metrics Tool Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create demo file with varying complexity
        demo_code = '''
def simple_function(x):
    return x * 2

def complex_function(data, threshold):
    # TODO: Optimize this function
    result = []
    for item in data:
        if item > threshold:
            if item % 2 == 0:
                for i in range(item):
                    if i % 3 == 0:
                        result.append(i)
                    elif i % 5 == 0:
                        result.append(i * 2)
                    else:
                        result.append(i + 1)
    # FIXME: Handle edge cases
    return result

def moderate_function(values):
    """Process values."""
    if not values:
        return []
    return [v for v in values if v > 0]
'''
        
        file_path = Path(temp_dir) / "code.py"
        file_path.write_text(demo_code)

        tool = MetricsTool()

        print("\n📊 Complexity Analysis")
        print("=" * 70)
        result = await tool.execute(operation="complexity", file=str(file_path))
        print(result.output)

        print("\n\n📈 Maintainability Index")
        print("=" * 70)
        result = await tool.execute(operation="maintainability", file=str(file_path))
        print(result.output)

        print("\n\n💰 Technical Debt")
        print("=" * 70)
        result = await tool.execute(operation="debt", file=str(file_path))
        print(result.output)

        print("\n\n✨ Demo Complete!")
        print("\nVictor's Metrics Tool provides:")
        print("  • Cyclomatic complexity analysis")
        print("  • Maintainability index calculation")
        print("  • Technical debt estimation")
        print("  • Code quality insights")

if __name__ == "__main__":
    asyncio.run(main())
