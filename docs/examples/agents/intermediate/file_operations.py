"""
File Operations Example

Agent performs various file system operations.
"""

import asyncio
from victor import Agent


async def search_codebase(query: str, path: str = "."):
    """Search codebase for specific patterns."""
    agent = Agent.create(
        tools=["grep", "read"],
        temperature=0.2
    )

    result = await agent.run(
        f"""Search the codebase in {path} for: {query}

        Find:
        1. Files containing the query
        2. Function definitions
        3. Class definitions
        4. Usage examples

        Provide file paths and line numbers."""
    )

    return result.content


async def analyze_file_structure(directory: str):
    """Analyze directory structure and organization."""
    agent = Agent.create(
        tools=["ls", "read"],
        temperature=0.3
    )

    result = await agent.run(
        f"""Analyze the file structure in {directory}.

        Report on:
        1. Directory organization
        2. File types present
        3. Potential issues (deep nesting, scattered files)
        4. Suggestions for improvement"""
    )

    return result.content


async def find_unused_imports(file_path: str):
    """Find potentially unused imports in a Python file."""
    agent = Agent.create(
        tools=["read", "grep"],
        vertical="coding",
        temperature=0.2
    )

    result = await agent.run(
        f"""Analyze {file_path} for unused imports.

        Check:
        1. Import statements
        2. Usage of imported modules/functions
        3. Suggest which imports can be removed"""
    )

    return result.content


async def generate_file_documentation(file_path: str):
    """Generate documentation for a code file."""
    agent = Agent.create(
        tools=["read"],
        vertical="coding",
        temperature=0.4
    )

    result = await agent.run(
        f"""Generate comprehensive documentation for {file_path}.

        Include:
        1. Module docstring
        2. Function/class documentation
        3. Usage examples
        4. Type information"""
    )

    return result.content


async def main():
    """Run file operation examples."""
    print("=== File Operations ===\n")

    # Search codebase
    print("Searching codebase...")
    search_results = await search_codebase("async def", "src/")
    print(search_results)

    print("\n" + "="*60 + "\n")

    # Analyze file structure
    print("Analyzing file structure...")
    structure = await analyze_file_structure("src/")
    print(structure)


if __name__ == "__main__":
    asyncio.run(main())
