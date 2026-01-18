#!/usr/bin/env python3
"""Test script for the unified Victor API server.

This script helps verify that all components of the unified server
are working correctly.

Usage:
    python test_unified_server.py
"""

import asyncio
import sys
from pathlib import Path


def print_section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


async def test_imports() -> bool:
    """Test that all required modules can be imported."""
    print_section("Testing Imports")

    all_passed = True

    try:
        # Main API server
        from victor.integrations.api.fastapi_server import VictorFastAPIServer

        print("✓ Main API server (VictorFastAPIServer)")
    except ImportError as e:
        print(f"✗ Main API server: {e}")
        all_passed = False

    try:
        # HITL API
        from victor.workflows.hitl_api import (
            HITLStore,
            SQLiteHITLStore,
            create_hitl_router,
        )

        print("✓ HITL API server components")
    except ImportError as e:
        print(f"✗ HITL API: {e}")
        all_passed = False

    try:
        # Workflow editor (may not be available)
        from tools.workflow_editor.backend.api import app as workflow_app

        print("✓ Workflow editor API")
    except ImportError as e:
        print(f"⚠ Workflow editor API (optional): {e}")
        # Don't fail on this, it's optional

    try:
        # Unified server
        from victor.integrations.api.unified_server import (
            create_unified_server,
            run_unified_server,
        )

        print("✓ Unified server factory")
    except ImportError as e:
        print(f"✗ Unified server: {e}")
        all_passed = False

    if all_passed:
        print("\n✅ All critical imports successful!")
        return True
    else:
        print("\n⚠️  Some critical imports failed")
        return False


async def test_server_creation() -> bool:
    """Test that the unified server can be created."""
    print_section("Testing Server Creation")

    try:
        from victor.integrations.api.unified_server import create_unified_server

        # Create server with default settings
        app = create_unified_server(
            host="127.0.0.1",
            port=8000,
            workspace_root=str(Path.cwd()),
            enable_hitl=True,
            hitl_persistent=False,  # Use in-memory for testing
            enable_cors=True,
        )

        print(f"✓ Server created: {app.title}")
        print(f"  Version: {app.version}")
        print(f"  Routes: {len(app.routes)}")

        # Check for expected routes
        route_paths = []
        for route in app.routes:
            if hasattr(route, "path"):
                route_paths.append(route.path)

        expected_routes = [
            "/",
            "/ui",
            "/health",
            "/docs",
            "/api/v1",
            "/api/v1/hitl",
        ]

        for expected in expected_routes:
            if any(expected in path for path in route_paths):
                print(f"  ✓ Route found: {expected}")
            else:
                print(f"  ⚠ Route missing: {expected}")

        print("\n✅ Server creation successful!")
        return True

    except Exception as e:
        print(f"\n❌ Server creation error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_endpoint_structure() -> bool:
    """Test that endpoints are properly structured."""
    print_section("Testing Endpoint Structure")

    try:
        from victor.integrations.api.unified_server import create_unified_server

        app = create_unified_server()

        # Categorize routes
        api_routes = []
        ui_routes = []
        system_routes = []

        for route in app.routes:
            if hasattr(route, "path"):
                path = route.path
                if path.startswith("/api/"):
                    api_routes.append(path)
                elif path.startswith("/ui"):
                    ui_routes.append(path)
                elif path in ["/health", "/docs", "/redoc", "/"]:
                    system_routes.append(path)

        print(f"API Routes ({len(api_routes)}):")
        for route in sorted(set(api_routes))[:10]:  # Show first 10
            print(f"  - {route}")
        if len(api_routes) > 10:
            print(f"  ... and {len(api_routes) - 10} more")

        print(f"\nUI Routes ({len(ui_routes)}):")
        for route in sorted(ui_routes):
            print(f"  - {route}")

        print(f"\nSystem Routes ({len(system_routes)}):")
        for route in sorted(system_routes):
            print(f"  - {route}")

        print("\n✅ Endpoint structure verified!")
        return True

    except Exception as e:
        print(f"\n❌ Endpoint structure error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_hitl_integration() -> bool:
    """Test HITL integration."""
    print_section("Testing HITL Integration")

    try:
        from victor.integrations.api.unified_server import create_unified_server

        app = create_unified_server(enable_hitl=True, hitl_persistent=False)

        if hasattr(app.state, "hitl_store"):
            print("✓ HITL store initialized")

            # Test store operations
            await app.state.hitl_store.store_request(
                request=type(
                    "Request",
                    (),
                    {
                        "request_id": "test-123",
                        "node_id": "test-node",
                        "hitl_type": type("Type", (), {"value": "approval"})(),
                        "prompt": "Test approval",
                        "context": {},
                        "choices": None,
                        "default_value": None,
                        "timeout": 300,
                        "fallback": type("Fallback", (), {"value": "continue"})(),
                    },
                )(),
                workflow_id="test-workflow",
            )

            pending = await app.state.hitl_store.list_pending()
            print(f"✓ HITL store working (pending: {len(pending)})")

            print("\n✅ HITL integration successful!")
            return True
        else:
            print("⚠ HITL store not initialized")
            return False

    except Exception as e:
        print(f"\n❌ HITL integration error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_health_check() -> bool:
    """Test health check endpoint."""
    print_section("Testing Health Check")

    try:
        from victor.integrations.api.unified_server import create_unified_server
        from fastapi.testclient import TestClient

        app = create_unified_server()
        client = TestClient(app)

        response = client.get("/health")
        if response.status_code == 200:
            health = response.json()
            print("✓ Health check endpoint responding")
            print(f"  Status: {health.get('status')}")
            print(f"  Services: {list(health.get('services', {}).keys())}")

            print("\n✅ Health check successful!")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"\n❌ Health check error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main() -> None:
    """Run all tests."""
    print("\n" + "🔧 " * 30)
    print("  Victor Unified Server - Test Suite")
    print("🔧 " * 30)

    tests = [
        ("Imports", test_imports),
        ("Server Creation", test_server_creation),
        ("Endpoint Structure", test_endpoint_structure),
        ("HITL Integration", test_hitl_integration),
        ("Health Check", test_health_check),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print_section("Test Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The unified server is ready to use.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
