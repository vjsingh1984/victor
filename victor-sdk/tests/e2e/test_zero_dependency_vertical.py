"""E2E test for zero-dependency vertical.

This test verifies that a vertical can be defined using only victor-sdk
without any victor-ai runtime dependencies.
"""

import subprocess
import sys
from typing import List


def run_python_script(script: str) -> tuple[bool, str, str]:
    """Run a Python script and return success status and output.

    Args:
        script: Python script to run

    Returns:
        Tuple of (success, stdout, stderr)
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Script timed out"
    except Exception as e:
        return False, "", str(e)


def test_zero_dependency_import():
    """Test that victor-sdk can be imported without victor-ai."""
    script = """
# This should work without victor-ai installed
from victor_sdk.verticals.protocols.base import VerticalBase
from victor_sdk.core.types import VerticalConfig, StageDefinition
from victor_sdk.discovery import ProtocolRegistry

print("✓ All imports successful")
"""
    success, stdout, stderr = run_python_script(script)
    assert success, f"Import failed: {stderr}"
    assert "✓ All imports successful" in stdout


def test_zero_dependency_vertical_definition():
    """Test that a vertical can be defined using only victor-sdk."""
    script = """
from victor_sdk.verticals.protocols.base import VerticalBase

class TestVertical(VerticalBase):
    @classmethod
    def get_name(cls) -> str:
        return "test"

    @classmethod
    def get_description(cls) -> str:
        return "Test vertical"

    @classmethod
    def get_tools(cls) -> list[str]:
        return ["read", "write"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are a test assistant"

# Test all methods work
assert TestVertical.get_name() == "test"
assert TestVertical.get_description() == "Test vertical"
assert TestVertical.get_tools() == ["read", "write"]
assert TestVertical.get_system_prompt() == "You are a test assistant"

# Test config generation
config = TestVertical.get_config()
assert config.name == "test"
assert config.get_tool_names() == ["read", "write"]

print("✓ Zero-dependency vertical definition works")
"""
    success, stdout, stderr = run_python_script(script)
    assert success, f"Vertical definition failed: {stderr}"
    assert "✓ Zero-dependency vertical definition works" in stdout


def test_protocol_implementation():
    """Test that protocols can be implemented without victor-ai."""
    script = """
from victor_sdk.verticals.protocols import ToolProvider

class MyToolProvider(ToolProvider):
    def get_tools(self) -> list[str]:
        return ["read", "write", "search"]

provider = MyToolProvider()
tools = provider.get_tools()
assert tools == ["read", "write", "search"]

print("✓ Protocol implementation works")
"""
    success, stdout, stderr = run_python_script(script)
    assert success, f"Protocol implementation failed: {stderr}"
    assert "✓ Protocol implementation works" in stdout


def test_discovery_system():
    """Test that discovery system works without victor-ai."""
    script = """
from victor_sdk.discovery import ProtocolRegistry, DiscoveryStats

registry = ProtocolRegistry()
stats = registry.get_discovery_stats()

assert stats.total_verticals == 0
assert stats.total_protocols == 0
assert isinstance(stats, DiscoveryStats)

print("✓ Discovery system works")
"""
    success, stdout, stderr = run_python_script(script)
    assert success, f"Discovery system failed: {stderr}"
    assert "✓ Discovery system works" in stdout


def test_core_types():
    """Test that core types work without victor-ai."""
    script = """
from victor_sdk.core.types import VerticalConfig, StageDefinition, TieredToolConfig, ToolSet, Tier

# Test StageDefinition
stage = StageDefinition(
    name="test",
    description="Test stage",
    required_tools=["read"],
)
assert stage.name == "test"

# Test TieredToolConfig
tiered = TieredToolConfig(
    basic_tools=["read"],
    standard_tools=["write"],  # Note: standard_tools are ADDITIONAL tools
)
assert tiered.get_tools_for_tier(Tier.BASIC) == ["read"]
# STANDARD tier includes BASIC + STANDARD tools
assert set(tiered.get_tools_for_tier(Tier.STANDARD)) == {"read", "write"}

# Test ToolSet
toolset = ToolSet(names=["read", "write"], description="Test tools")
assert "read" in toolset
assert "write" in toolset
assert len(toolset) == 2

# Test VerticalConfig
config = VerticalConfig(
    name="test",
    description="Test",
    tools=["read"],
    system_prompt="Test prompt",
)
assert config.name == "test"
assert config.get_tool_names() == ["read"]

print("✓ All core types work")
"""
    success, stdout, stderr = run_python_script(script)
    assert success, f"Core types failed: {stderr}"
    assert "✓ All core types work" in stdout


def test_protocol_compatibility():
    """Test that protocols are runtime_checkable."""
    script = """
from victor_sdk.verticals.protocols import ToolProvider

class MyToolProvider(ToolProvider):
    def get_tools(self) -> list[str]:
        return ["read", "write"]

provider = MyToolProvider()

# Test isinstance works (requires @runtime_checkable)
assert isinstance(provider, ToolProvider)

print("✓ Protocol compatibility works")
"""
    success, stdout, stderr = run_python_script(script)
    assert success, f"Protocol compatibility failed: {stderr}"
    assert "✓ Protocol compatibility works" in stdout


def test_exceptions():
    """Test that exceptions work without victor-ai."""
    script = """
from victor_sdk.core.exceptions import VerticalException, VerticalConfigurationError

# Test exception creation
exc1 = VerticalException("Test error", vertical_name="test")
assert "Test error" in str(exc1)
assert "test" in str(exc1)

exc2 = VerticalConfigurationError("Config error")
assert isinstance(exc2, VerticalException)

print("✓ Exceptions work")
"""
    success, stdout, stderr = run_python_script(script)
    assert success, f"Exceptions failed: {stderr}"
    assert "✓ Exceptions work" in stdout


def test_complete_vertical_workflow():
    """Test complete vertical workflow with all features."""
    script = """
from victor_sdk.verticals.protocols.base import VerticalBase
from victor_sdk.core.types import StageDefinition, Tier

class CompleteVertical(VerticalBase):
    @classmethod
    def get_name(cls) -> str:
        return "complete"

    @classmethod
    def get_description(cls) -> str:
        return "Complete test vertical"

    @classmethod
    def get_tools(cls) -> list[str]:
        return ["read", "write", "search", "git"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are a complete test assistant"

    @classmethod
    def get_stages(cls):
        return {
            "planning": StageDefinition(
                name="planning",
                description="Plan the work",
                required_tools=["search"],
            ),
            "execution": StageDefinition(
                name="execution",
                description="Execute the plan",
                required_tools=["read", "write"],
            ),
        }

# Test all features
vertical = CompleteVertical

# Basic methods
assert vertical.get_name() == "complete"
assert vertical.get_tools() == ["read", "write", "search", "git"]

# Config generation
config = vertical.get_config()
assert config.name == "complete"
assert len(config.get_tool_names()) == 4

# Stages
stages = vertical.get_stages()
assert "planning" in stages
assert "execution" in stages
assert stages["planning"].required_tools == ["search"]

print("✓ Complete vertical workflow works")
"""
    success, stdout, stderr = run_python_script(script)
    assert success, f"Complete workflow failed: {stderr}"
    assert "✓ Complete vertical workflow works" in stdout


def test_no_victor_ai_required():
    """Verify that victor-ai is NOT imported or required."""
    script = """
import sys

# Try to import victor-ai (should fail in this test environment)
try:
    import victor.core.verticals.base
    print("WARNING: victor-ai is installed, skipping test")
    sys.exit(0)  # Skip test if victor-ai is available
except ImportError:
    pass  # Expected - victor-ai should not be available

# Now verify victor-sdk works independently
from victor_sdk.verticals.protocols.base import VerticalBase

class TestVertical(VerticalBase):
    @classmethod
    def get_name(cls) -> str:
        return "test"

    @classmethod
    def get_description(cls) -> str:
        return "Test"

    @classmethod
    def get_tools(cls) -> list[str]:
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Test"

# This should work without victor-ai
config = TestVertical.get_config()
assert config.name == "test"

print("✓ victor-sdk works independently of victor-ai")
"""
    success, stdout, stderr = run_python_script(script)
    # This test might be skipped if victor-ai is installed
    if "WARNING" in stdout:
        print("Note: victor-ai is installed, test was skipped")
    else:
        assert success, f"Independence test failed: {stderr}"
        assert "✓ victor-sdk works independently of victor-ai" in stdout


def run_all_e2e_tests():
    """Run all E2E tests."""
    print("=" * 60)
    print("E2E Tests: Zero-Dependency Vertical")
    print("=" * 60)

    tests = [
        ("Zero-Dependency Import", test_zero_dependency_import),
        (
            "Zero-Dependency Vertical Definition",
            test_zero_dependency_vertical_definition,
        ),
        ("Protocol Implementation", test_protocol_implementation),
        ("Discovery System", test_discovery_system),
        ("Core Types", test_core_types),
        ("Protocol Compatibility", test_protocol_compatibility),
        ("Exceptions", test_exceptions),
        ("Complete Vertical Workflow", test_complete_vertical_workflow),
        ("No victor-ai Required", test_no_victor_ai_required),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n{name}...", end=" ")
        try:
            test_func()
            print("✓ PASS")
            passed += 1
        except AssertionError as e:
            print(f"✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_e2e_tests()
    sys.exit(0 if success else 1)
