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

"""Demo of Victor's Advanced Tools.

This demonstrates the additional tools for database operations,
Docker management, and HTTP/API testing.

Usage:
    python examples/advanced_tools_demo.py
"""

import asyncio
import tempfile
from pathlib import Path

from victor.tools.database_tool import DatabaseTool
from victor.tools.docker_tool import DockerTool
from victor.tools.http_tool import http_request, http_test


async def demo_database_tool():
    """Demo database tool capabilities."""
    print("\n🗄️  Database Tool Demo")
    print("=" * 70)

    db_tool = DatabaseTool(allow_modifications=True)

    # Create temp SQLite database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # Connect
        print("\n1️⃣ Connecting to SQLite database...")
        result = await db_tool.execute(operation="connect", db_type="sqlite", database=db_path)
        print(result.output if result.success else f"Error: {result.error}")

        if not result.success:
            return

        # Extract connection ID
        conn_id = result.output.split("Connection ID: ")[1].strip()

        # Create table
        print("\n2️⃣ Creating table...")
        result = await db_tool.execute(
            operation="query",
            connection_id=conn_id,
            sql="CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)",
        )
        print(result.output if result.success else f"Error: {result.error}")

        # Insert data
        print("\n3️⃣ Inserting data...")
        result = await db_tool.execute(
            operation="query",
            connection_id=conn_id,
            sql="INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com'), ('Bob', 'bob@example.com')",
        )
        print(result.output if result.success else f"Error: {result.error}")

        # Query data
        print("\n4️⃣ Querying data...")
        result = await db_tool.execute(
            operation="query", connection_id=conn_id, sql="SELECT * FROM users"
        )
        print(result.output if result.success else f"Error: {result.error}")

        # List tables
        print("\n5️⃣ Listing tables...")
        result = await db_tool.execute(operation="tables", connection_id=conn_id)
        print(result.output if result.success else f"Error: {result.error}")

        # Describe table
        print("\n6️⃣ Describing table structure...")
        result = await db_tool.execute(operation="describe", connection_id=conn_id, table="users")
        print(result.output if result.success else f"Error: {result.error}")

        # Disconnect
        print("\n7️⃣ Disconnecting...")
        result = await db_tool.execute(operation="disconnect", connection_id=conn_id)
        print(result.output if result.success else f"Error: {result.error}")

    finally:
        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    print("\n✅ Database Tool Features:")
    print("  ✓ Multiple database support (SQLite, PostgreSQL, MySQL, SQL Server)")
    print("  ✓ Safe query execution with validation")
    print("  ✓ Schema inspection")
    print("  ✓ Read-only by default")
    print("  ✓ Connection management")


async def demo_docker_tool():
    """Demo Docker tool capabilities."""
    print("\n\n🐳 Docker Tool Demo")
    print("=" * 70)

    docker_tool = DockerTool()

    # List images
    print("\n1️⃣ Listing Docker images...")
    result = await docker_tool.execute(operation="images")
    if result.success:
        print(result.output[:500] + "..." if len(result.output) > 500 else result.output)
    else:
        print(f"Error: {result.error}")
        if "not found" in result.error:
            print("\nNote: Docker CLI not installed or not in PATH")
            print("Install Docker to use this tool")
            return

    # List containers
    print("\n2️⃣ Listing running containers...")
    result = await docker_tool.execute(operation="ps")
    if result.success:
        print(result.output[:500] + "..." if len(result.output) > 500 else result.output)
    else:
        print(f"Error: {result.error}")

    # List all containers (including stopped)
    print("\n3️⃣ Listing all containers...")
    result = await docker_tool.execute(operation="ps", all=True)
    if result.success:
        print(result.output[:500] + "..." if len(result.output) > 500 else result.output)
    else:
        print(f"Error: {result.error}")

    # List networks
    print("\n4️⃣ Listing Docker networks...")
    result = await docker_tool.execute(operation="networks")
    if result.success:
        print(result.output[:500] + "..." if len(result.output) > 500 else result.output)
    else:
        print(f"Error: {result.error}")

    print("\n✅ Docker Tool Features:")
    print("  ✓ Container management (list, start, stop, remove)")
    print("  ✓ Image operations (list, pull, remove)")
    print("  ✓ Logs and stats")
    print("  ✓ Network and volume inspection")
    print("  ✓ Command execution in containers")


async def demo_http_tool():
    """Demo HTTP tool capabilities."""
    print("\n\n🌐 HTTP Tool Demo")
    print("=" * 70)

    # Simple GET request
    print("\n1️⃣ GET request to GitHub API...")
    result = await http_request(
        method="GET", url="https://api.github.com/users/octocat", timeout=15
    )
    if result.get("success"):
        print(f"Status: {result['status_code']}")
        print(f"Duration: {result['duration_ms']}ms")
        print(f"User: {result['body'].get('name', 'N/A')}")
    else:
        print(f"Error: {result.get('error')}")

    # GET with query parameters
    print("\n2️⃣ GET with query parameters...")
    result = await http_request(
        method="GET",
        url="https://api.github.com/search/repositories",
        params={"q": "language:python", "sort": "stars", "per_page": 3},
        timeout=15,
    )
    if result.get("success"):
        print(f"Status: {result['status_code']}")
        print(f"Duration: {result['duration_ms']}ms")
        if isinstance(result["body"], dict):
            print(f"Total count: {result['body'].get('total_count', 'N/A')}")
    else:
        print(f"Error: {result.get('error')}")

    # API testing with validation
    print("\n3️⃣ API testing with validation...")
    result = await http_test(
        method="GET", url="https://api.github.com", expected_status=200, timeout=15
    )
    if result.get("success"):
        print(f"Test passed: {result['all_passed']}")
        print(f"Status: {result['status_code']}")
        print(f"Duration: {result['duration_ms']}ms")
    else:
        print(f"Error: {result.get('error')}")

    # POST request (httpbin echo)
    print("\n4️⃣ POST request with JSON body...")
    result = await http_request(
        method="POST",
        url="https://httpbin.org/post",
        headers={"Content-Type": "application/json"},
        json={"key": "value", "test": True},
        timeout=15,
    )
    if result.get("success"):
        print(f"Status: {result['status_code']}")
        print(f"Duration: {result['duration_ms']}ms")
    else:
        print(f"Error: {result.get('error')}")

    print("\n✅ HTTP Tool Features:")
    print("  ✓ All HTTP methods (GET, POST, PUT, PATCH, DELETE)")
    print("  ✓ Headers and authentication")
    print("  ✓ JSON and form data")
    print("  ✓ Response validation")
    print("  ✓ Performance metrics")


async def main():
    """Run all demos."""
    print("🎯 Victor Advanced Tools Demo")
    print("=" * 70)
    print("\nDemonstrating Database, Docker, and HTTP tools\n")

    # Database demo
    await demo_database_tool()

    # Docker demo
    await demo_docker_tool()

    # HTTP demo
    await demo_http_tool()

    print("\n\n✨ Demo Complete!")
    print("\nVictor's Advanced Tools extend capabilities for:")
    print("  • Database operations and queries")
    print("  • Docker container management")
    print("  • HTTP/API testing and automation")
    print("\nAll tools are production-ready and agent-integrated!")


if __name__ == "__main__":
    asyncio.run(main())
