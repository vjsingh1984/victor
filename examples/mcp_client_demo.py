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

"""Demo of Victor's MCP Client.

This demonstrates how Victor can connect to external MCP servers to access
their tools and resources, extending Victor's capabilities.

Usage:
    # First, start an MCP server in another terminal:
    python examples/mcp_server_demo.py --stdio

    # Then run this client:
    python examples/mcp_client_demo.py
"""

import asyncio
import sys
from victor.integrations.mcp import MCPClient


async def demo_client():
    """Demo MCP client capabilities."""
    print("🎯 Victor MCP Client Demo")
    print("=" * 70)
    print("\nThis client connects to external MCP servers to use their tools.\n")

    # Create MCP client
    print("1️⃣ Creating MCP Client...")
    print("-" * 70)
    client = MCPClient(name="Victor MCP Client", version="1.0.0")
    print(f"✓ Client created: {client.name} v{client.version}")

    # Connect to server
    print("\n2️⃣ Connecting to MCP Server...")
    print("-" * 70)
    print("Starting Victor's MCP server...")

    # Command to start the server
    server_command = [
        sys.executable,  # Use same Python interpreter
        "examples/mcp_server_demo.py",
        "--stdio",
    ]

    success = await client.connect(server_command)

    if not success:
        print("✗ Failed to connect to server")
        print("\nMake sure the server demo exists:")
        print("  python examples/mcp_server_demo.py --stdio")
        return

    print(f"✓ Connected to server")
    if client.server_info:
        print(f"  Server: {client.server_info.name} v{client.server_info.version}")
        print(f"  Capabilities:")
        print(f"    - Tools: {client.server_info.capabilities.tools}")
        print(f"    - Resources: {client.server_info.capabilities.resources}")

    # List available tools
    print("\n3️⃣ Discovering Tools...")
    print("-" * 70)
    tools = await client.refresh_tools()
    print(f"✓ Found {len(tools)} tools:")
    for tool in tools:
        print(f"\n  📦 {tool.name}")
        print(f"     {tool.description[:80]}...")
        if tool.parameters:
            print(f"     Parameters: {len(tool.parameters)}")
            for param in tool.parameters[:2]:  # Show first 2
                req = "required" if param.required else "optional"
                print(f"       - {param.name} ({param.type}, {req})")

    # List available resources
    print("\n4️⃣ Discovering Resources...")
    print("-" * 70)
    resources = await client.refresh_resources()
    print(f"✓ Found {len(resources)} resources:")
    for resource in resources:
        print(f"\n  📄 {resource.name}")
        print(f"     URI: {resource.uri}")
        print(f"     Type: {resource.mime_type or 'unknown'}")
        print(f"     Description: {resource.description}")

    # Test ping
    print("\n5️⃣ Testing Connection...")
    print("-" * 70)
    ping_ok = await client.ping()
    print(f"✓ Ping: {'Success' if ping_ok else 'Failed'}")

    # Call a tool
    print("\n6️⃣ Calling Tools...")
    print("-" * 70)

    # Example 1: List directory
    print("\nExample 1: List directory")
    result = await client.call_tool("ls", path=".")
    if result.success:
        print(f"✓ Tool call succeeded")
        output = str(result.result)[:200]
        print(f"  Output (preview): {output}...")
    else:
        print(f"✗ Tool call failed: {result.error}")

    # Example 2: Read file (if available)
    if client.get_tool_by_name("read"):
        print("\nExample 2: Read file")
        result = await client.call_tool("read", path="README.md")
        if result.success:
            print(f"✓ Tool call succeeded")
            output = str(result.result)[:200]
            print(f"  Content (preview): {output}...")
        else:
            print(f"✗ Tool call failed: {result.error}")

    # Read a resource
    print("\n7️⃣ Reading Resources...")
    print("-" * 70)
    if resources:
        first_resource = resources[0]
        print(f"\nReading: {first_resource.name}")
        content = await client.read_resource(first_resource.uri)
        if content:
            print(f"✓ Resource read successfully")
            preview = content[:200]
            print(f"  Content (preview): {preview}...")
        else:
            print(f"✗ Failed to read resource")

    # Show client status
    print("\n8️⃣ Client Status")
    print("-" * 70)
    status = client.get_status()
    print(f"Connected: {status['connected']}")
    print(f"Tools available: {status['tools_count']}")
    print(f"Resources available: {status['resources_count']}")
    if status["server"]:
        print(f"Server: {status['server']['name']} v{status['server']['version']}")

    # Disconnect
    print("\n9️⃣ Disconnecting...")
    print("-" * 70)
    client.disconnect()
    print("✓ Disconnected from server")

    print("\n\n✨ Demo Complete!")
    print("\nMCP Client Features:")
    print("  ✓ Connects to external MCP servers")
    print("  ✓ Discovers available tools and resources")
    print("  ✓ Calls tools with type-safe parameters")
    print("  ✓ Reads resources (files, data)")
    print("  ✓ Standard JSON-RPC 2.0 protocol")
    print("  ✓ Stdio transport support")

    print("\n\n📚 Use Cases:")
    print(
        """
1. Extend Victor with external tools:
   - Database query tools
   - API integration tools
   - Custom business logic

2. Access external data sources:
   - Company documentation
   - Internal wikis
   - Database schemas

3. Integrate with other services:
   - Slack notifications
   - JIRA ticket creation
   - Deployment tools

4. Chain multiple MCP servers:
   - Victor + Database MCP + API MCP
   - Compose powerful workflows
"""
    )

    print("\n\n🔌 Integration Example:")
    print(
        """
# In Victor's agent orchestrator:
from victor.integrations.mcp import MCPClient

# Connect to external MCP server
async with MCPClient() as client:
    await client.connect(["python", "database_server.py", "--stdio"])

    # Now Victor can use database tools
    result = await client.call_tool(
        "query_database",
        sql="SELECT * FROM users LIMIT 10"
    )

    # Use results in agent response
    print(result.result)
"""
    )


if __name__ == "__main__":
    asyncio.run(demo_client())
