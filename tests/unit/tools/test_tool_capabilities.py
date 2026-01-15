# Test file for tool capabilities - simplified version

import pytest
from victor.tools.capabilities.system import (
    ToolCapability,
    CapabilityDefinition,
    CapabilityRegistry,
    CapabilitySelector,
)
from victor.tools.capabilities.definitions import BUILTIN_CAPABILITIES


class TestToolCapabilityEnum:
    """Test ToolCapability enum has required capabilities."""
    
    def test_file_read_exists(self):
        assert hasattr(ToolCapability, 'FILE_READ')
        assert ToolCapability.FILE_READ.value == "file_read"
    
    def test_file_write_exists(self):
        assert hasattr(ToolCapability, 'FILE_WRITE')
        assert ToolCapability.FILE_WRITE.value == "file_write"
    
    def test_has_at_least_30_capabilities(self):
        """Test that ToolCapability enum has 30+ capabilities."""
        capabilities = [cap.value for cap in ToolCapability]
        assert len(set(capabilities)) >= 30


class TestCapabilityDefinition:
    """Test CapabilityDefinition dataclass."""
    
    def test_create_definition(self):
        definition = CapabilityDefinition(
            name=ToolCapability.FILE_READ,
            description="Read files",
            tools=["read", "ls"],
            dependencies=[],
            conflicts=[],
        )
        assert definition.name == ToolCapability.FILE_READ
        assert definition.tools == ["read", "ls"]


class TestCapabilityRegistry:
    """Test CapabilityRegistry."""
    
    def test_register_and_get_tools(self):
        registry = CapabilityRegistry()
        definition = CapabilityDefinition(
            name=ToolCapability.FILE_READ,
            description="Read files",
            tools=["read"],
            dependencies=[],
            conflicts=[],
        )
        registry.register_capability(definition)
        tools = registry.get_tools_for_capability(ToolCapability.FILE_READ)
        assert "read" in tools
    
    def test_duplicate_raises_error(self):
        registry = CapabilityRegistry()
        definition = CapabilityDefinition(
            name=ToolCapability.FILE_READ,
            description="Read files",
            tools=["read"],
            dependencies=[],
            conflicts=[],
        )
        registry.register_capability(definition)
        with pytest.raises(ValueError):
            registry.register_capability(definition)
    
    def test_resolve_dependencies(self):
        registry = CapabilityRegistry()
        
        registry.register_capability(CapabilityDefinition(
            name=ToolCapability.FILE_READ,
            description="Read files",
            tools=["read"],
            dependencies=[],
            conflicts=[],
        ))
        
        registry.register_capability(CapabilityDefinition(
            name=ToolCapability.FILE_WRITE,
            description="Write files",
            tools=["write"],
            dependencies=[ToolCapability.FILE_READ],
            conflicts=[],
        ))
        
        resolved = registry.resolve_dependencies([ToolCapability.FILE_WRITE])
        assert ToolCapability.FILE_WRITE in resolved
        assert ToolCapability.FILE_READ in resolved


class TestCapabilitySelector:
    """Test CapabilitySelector."""
    
    def test_select_tools_simple(self):
        registry = CapabilityRegistry()
        registry.register_capability(CapabilityDefinition(
            name=ToolCapability.FILE_READ,
            description="Read files",
            tools=["read", "ls"],
            dependencies=[],
            conflicts=[],
        ))
        
        selector = CapabilitySelector(registry)
        tools = selector.select_tools(
            required_capabilities=[ToolCapability.FILE_READ],
            excluded_tools=None,
        )
        assert "read" in tools
        assert "ls" in tools
    
    def test_select_tools_with_exclusions(self):
        registry = CapabilityRegistry()
        registry.register_capability(CapabilityDefinition(
            name=ToolCapability.FILE_READ,
            description="Read files",
            tools=["read", "ls"],
            dependencies=[],
            conflicts=[],
        ))
        
        selector = CapabilitySelector(registry)
        tools = selector.select_tools(
            required_capabilities=[ToolCapability.FILE_READ],
            excluded_tools={"ls"},
        )
        assert "read" in tools
        assert "ls" not in tools


class TestBuiltinCapabilities:
    """Test built-in capability definitions."""
    
    def test_builtin_is_list(self):
        assert isinstance(BUILTIN_CAPABILITIES, list)
    
    def test_builtin_not_empty(self):
        assert len(BUILTIN_CAPABILITIES) > 0
    
    def test_all_are_definitions(self):
        for cap in BUILTIN_CAPABILITIES:
            assert isinstance(cap, CapabilityDefinition)
    
    def test_file_read_exists(self):
        names = [cap.name for cap in BUILTIN_CAPABILITIES]
        assert ToolCapability.FILE_READ in names
    
    def test_file_write_depends_on_read(self):
        file_write = None
        for cap in BUILTIN_CAPABILITIES:
            if cap.name == ToolCapability.FILE_WRITE:
                file_write = cap
                break
        assert file_write is not None
        assert ToolCapability.FILE_READ in file_write.dependencies


class TestExtendedCapabilities:
    """Test new extended capabilities."""

    def test_code_refactoring_exists(self):
        """Test CODE_REFACTORING capability exists."""
        assert hasattr(ToolCapability, 'CODE_REFACTORING')
        assert ToolCapability.CODE_REFACTORING.value == "code_refactoring"

    def test_semantic_search_exists(self):
        """Test SEMANTIC_SEARCH capability exists."""
        assert hasattr(ToolCapability, 'SEMANTIC_SEARCH')
        assert ToolCapability.SEMANTIC_SEARCH.value == "semantic_search"

    def test_knowledge_base_exists(self):
        """Test KNOWLEDGE_BASE capability exists."""
        assert hasattr(ToolCapability, 'KNOWLEDGE_BASE')
        assert ToolCapability.KNOWLEDGE_BASE.value == "knowledge_base"

    def test_change_management_exists(self):
        """Test CHANGE_MANAGEMENT capability exists."""
        assert hasattr(ToolCapability, 'CHANGE_MANAGEMENT')
        assert ToolCapability.CHANGE_MANAGEMENT.value == "change_management"

    def test_containerization_exists(self):
        """Test CONTAINERIZATION capability exists."""
        assert hasattr(ToolCapability, 'CONTAINERIZATION')
        assert ToolCapability.CONTAINERIZATION.value == "containerization"

    def test_cloud_infra_exists(self):
        """Test CLOUD_INFRA capability exists."""
        assert hasattr(ToolCapability, 'CLOUD_INFRA')
        assert ToolCapability.CLOUD_INFRA.value == "cloud_infra"

    def test_monitoring_exists(self):
        """Test MONITORING capability exists."""
        assert hasattr(ToolCapability, 'MONITORING')
        assert ToolCapability.MONITORING.value == "monitoring"

    def test_scaffolding_exists(self):
        """Test SCAFFOLDING capability exists."""
        assert hasattr(ToolCapability, 'SCAFFOLDING')
        assert ToolCapability.SCAFFOLDING.value == "scaffolding"

    def test_code_execution_exists(self):
        """Test CODE_EXECUTION capability exists."""
        assert hasattr(ToolCapability, 'CODE_EXECUTION')
        assert ToolCapability.CODE_EXECUTION.value == "code_execution"

    def test_browser_automation_exists(self):
        """Test BROWSER_AUTOMATION capability exists."""
        assert hasattr(ToolCapability, 'BROWSER_AUTOMATION')
        assert ToolCapability.BROWSER_AUTOMATION.value == "browser_automation"

    def test_messaging_exists(self):
        """Test MESSAGING capability exists."""
        assert hasattr(ToolCapability, 'MESSAGING')
        assert ToolCapability.MESSAGING.value == "messaging"

    def test_issue_tracking_exists(self):
        """Test ISSUE_TRACKING capability exists."""
        assert hasattr(ToolCapability, 'ISSUE_TRACKING')
        assert ToolCapability.ISSUE_TRACKING.value == "issue_tracking"

    def test_notification_exists(self):
        """Test NOTIFICATION capability exists."""
        assert hasattr(ToolCapability, 'NOTIFICATION')
        assert ToolCapability.NOTIFICATION.value == "notification"

    def test_workflow_orchestration_exists(self):
        """Test WORKFLOW_ORCHESTRATION capability exists."""
        assert hasattr(ToolCapability, 'WORKFLOW_ORCHESTRATION')
        assert ToolCapability.WORKFLOW_ORCHESTRATION.value == "workflow_orchestration"

    def test_lsp_integration_exists(self):
        """Test LSP_INTEGRATION capability exists."""
        assert hasattr(ToolCapability, 'LSP_INTEGRATION')
        assert ToolCapability.LSP_INTEGRATION.value == "lsp_integration"

    def test_ai_assistance_exists(self):
        """Test AI_ASSISTANCE capability exists."""
        assert hasattr(ToolCapability, 'AI_ASSISTANCE')
        assert ToolCapability.AI_ASSISTANCE.value == "ai_assistance"

    def test_security_scanning_exists(self):
        """Test SECURITY_SCANNING capability exists."""
        assert hasattr(ToolCapability, 'SECURITY_SCANNING')
        assert ToolCapability.SECURITY_SCANNING.value == "security_scanning"

    def test_compliance_audit_exists(self):
        """Test COMPLIANCE_AUDIT capability exists."""
        assert hasattr(ToolCapability, 'COMPLIANCE_AUDIT')
        assert ToolCapability.COMPLIANCE_AUDIT.value == "compliance_audit"

    def test_api_integration_exists(self):
        """Test API_INTEGRATION capability exists."""
        assert hasattr(ToolCapability, 'API_INTEGRATION')
        assert ToolCapability.API_INTEGRATION.value == "api_integration"

    def test_graph_analysis_exists(self):
        """Test GRAPH_ANALYSIS capability exists."""
        assert hasattr(ToolCapability, 'GRAPH_ANALYSIS')
        assert ToolCapability.GRAPH_ANALYSIS.value == "graph_analysis"


class TestCapabilityRegistryExtended:
    """Test registry with extended capabilities."""

    def test_all_capabilities_registered(self):
        """Test that all enum capabilities have definitions."""
        registry = CapabilityRegistry()
        for definition in BUILTIN_CAPABILITIES:
            registry.register_capability(definition)

        # Check that all capabilities are registered
        enum_capabilities = set(ToolCapability)
        registered_capabilities = set(registry._capabilities.keys())

        assert enum_capabilities == registered_capabilities

    def test_capability_lookup(self):
        """Test looking up capabilities by name."""
        registry = CapabilityRegistry()
        for definition in BUILTIN_CAPABILITIES:
            registry.register_capability(definition)

        # Test specific capability lookups
        code_refactor_def = registry._capabilities.get(ToolCapability.CODE_REFACTORING)
        assert code_refactor_def is not None
        assert "refactor" in code_refactor_def.description.lower()

        semantic_search_def = registry._capabilities.get(ToolCapability.SEMANTIC_SEARCH)
        assert semantic_search_def is not None
        assert "semantic" in semantic_search_def.description.lower()

    def test_dependency_resolution_extended(self):
        """Test dependency resolution for new capabilities."""
        registry = CapabilityRegistry()
        for definition in BUILTIN_CAPABILITIES:
            registry.register_capability(definition)

        # Test CODE_REFACTORING depends on FILE_READ, FILE_WRITE, CODE_ANALYSIS
        resolved = registry.resolve_dependencies([ToolCapability.CODE_REFACTORING])
        assert ToolCapability.CODE_REFACTORING in resolved
        assert ToolCapability.FILE_READ in resolved
        assert ToolCapability.FILE_WRITE in resolved
        assert ToolCapability.CODE_ANALYSIS in resolved

    def test_get_tools_for_capability_extended(self):
        """Test getting tools for extended capabilities."""
        registry = CapabilityRegistry()
        for definition in BUILTIN_CAPABILITIES:
            registry.register_capability(definition)

        # Test CONTAINERIZATION returns docker_tool
        tools = registry.get_tools_for_capability(ToolCapability.CONTAINERIZATION)
        assert "docker_tool" in tools

        # Test MONITORING returns metrics_tool
        tools = registry.get_tools_for_capability(ToolCapability.MONITORING)
        assert "metrics_tool" in tools

        # Test API_INTEGRATION returns http
        tools = registry.get_tools_for_capability(ToolCapability.API_INTEGRATION)
        assert "http" in tools


class TestBackwardCompatibility:
    """Test backward compatibility with existing 18 capabilities."""

    def test_existing_capabilities_unchanged(self):
        """Test that original 18 capabilities remain unchanged."""
        # Original capabilities from the first implementation
        original_capabilities = [
            "FILE_READ", "FILE_WRITE", "FILE_MANAGEMENT",
            "CODE_ANALYSIS", "CODE_SEARCH", "CODE_REVIEW", "CODE_INTELLIGENCE",
            "WEB_SEARCH",
            "VERSION_CONTROL", "DATABASE", "DOCKER", "CI_CD",
            "TESTING", "DOCUMENTATION", "DEPENDENCY",
            "BASH", "BROWSER",
            "CACHE", "BATCH", "AUDIT"
        ]

        for cap_name in original_capabilities:
            assert hasattr(ToolCapability, cap_name), f"Missing {cap_name}"

    def test_no_duplicate_capability_values(self):
        """Test that all capability values are unique."""
        values = [cap.value for cap in ToolCapability]
        assert len(values) == len(set(values)), "Duplicate capability values found"

    def test_capability_count(self):
        """Test exact count of capabilities."""
        count = len(list(ToolCapability))
        assert count >= 30, f"Expected at least 30 capabilities, got {count}"


# ===== NEW TESTS FOR AUTO-DISCOVERY =====


class MockTool:
    """Mock tool for testing auto-discovery."""

    def __init__(
        self,
        name: str,
        description: str,
        metadata=None,
        cost_tier=None,
    ):
        from victor.tools.enums import CostTier
        self._name = name
        self._description = description
        self._metadata = metadata
        self._cost_tier = cost_tier or CostTier.FREE

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    @property
    def metadata(self):
        return self._metadata

    def get_metadata(self):
        from victor.tools.metadata import ToolMetadata
        if self._metadata:
            return self._metadata
        return ToolMetadata.generate_from_tool(
            self.name, self.description, self.parameters, self._cost_tier
        )

    @property
    def cost_tier(self):
        return self._cost_tier


class TestCapabilityRegistryAutoDiscovery:
    """Test auto-discovery of capabilities from tool metadata."""

    def test_auto_discover_from_explicit_capability(self):
        """Test auto-discovery when tools have explicit capability in metadata."""
        from victor.tools.metadata import ToolMetadata

        registry = CapabilityRegistry()

        # Create mock tools - auto-discovery uses category/keywords mapping
        read_tool = MockTool(
            name="file_read",
            description="Read files from filesystem",
            metadata=ToolMetadata(
                category="filesystem",
                keywords=["read", "file"]
            )
        )

        # Auto-discover should register the capability
        import asyncio
        asyncio.run(registry.auto_discover_capabilities([read_tool]))

        # Verify capability was registered
        tools = registry.get_tools_for_capability(ToolCapability.FILE_READ)
        assert "file_read" in tools

    def test_auto_discover_from_category_mapping(self):
        """Test auto-discovery maps tool categories to capabilities."""
        registry = CapabilityRegistry()

        # Create mock tools with specific categories
        git_tool = MockTool(
            name="git",
            description="Git version control",
            metadata=None  # Auto-generated metadata will have category="git"
        )

        docker_tool = MockTool(
            name="docker_tool",
            description="Docker container management",
            metadata=None  # Auto-generated metadata will have category="docker"
        )

        # Auto-discover should map categories to capabilities
        import asyncio
        asyncio.run(registry.auto_discover_capabilities([git_tool, docker_tool]))

        # Verify git -> VERSION_CONTROL
        git_tools = registry.get_tools_for_capability(ToolCapability.VERSION_CONTROL)
        assert "git" in git_tools

        # Verify docker -> CONTAINERIZATION
        docker_tools = registry.get_tools_for_capability(ToolCapability.CONTAINERIZATION)
        assert "docker_tool" in docker_tools

    def test_auto_discover_logs_warning_for_unmapped_tools(self):
        """Test that tools without clear capability mapping generate warnings."""
        import logging
        from unittest.mock import patch

        registry = CapabilityRegistry()

        # Create a tool with unclear category
        unclear_tool = MockTool(
            name="mystery_tool",
            description="Does something unclear",
            metadata=None  # Will generate category="mystery"
        )

        # Mock logger to capture warnings
        with patch('victor.tools.capabilities.system.logger') as mock_logger:
            import asyncio
            asyncio.run(registry.auto_discover_capabilities([unclear_tool]))

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            assert "mystery_tool" in call_args
            assert "Could not map" in call_args

    def test_manual_registration_overrides_auto_discovery(self):
        """Test that manually registered capabilities take precedence."""
        from victor.tools.metadata import ToolMetadata

        registry = CapabilityRegistry()

        # First, manually register FILE_READ with specific tools
        manual_definition = CapabilityDefinition(
            name=ToolCapability.FILE_READ,
            description="Manual registration",
            tools=["manual_read_tool"],
            dependencies=[],
            conflicts=[],
        )
        registry.register_capability(manual_definition)

        # Then auto-discover with different tools
        auto_tool = MockTool(
            name="auto_read_tool",
            description="Auto-discovered read tool",
            metadata=ToolMetadata(
                category="filesystem",
                keywords=["read"]
            )
        )

        # Auto-discovery should not override manual registration
        import asyncio
        asyncio.run(registry.auto_discover_capabilities([auto_tool]))

        # Verify manual registration is preserved
        tools = registry.get_tools_for_capability(ToolCapability.FILE_READ)
        assert "manual_read_tool" in tools
        # Auto-discovered tool should be added, not replace
        assert "auto_read_tool" in tools


class TestCapabilityRegistryGetCapabilityForTool:
    """Test get_capability_for_tool method."""

    def test_get_capability_for_registered_tool(self):
        """Test finding capability for a registered tool."""
        registry = CapabilityRegistry()
        definition = CapabilityDefinition(
            name=ToolCapability.FILE_READ,
            description="Read files",
            tools=["read_tool", "cat"],
            dependencies=[],
            conflicts=[],
        )
        registry.register_capability(definition)

        # Find capability for read_tool
        capability = registry.get_capability_for_tool("read_tool")
        assert capability == ToolCapability.FILE_READ

    def test_get_capability_for_unknown_tool(self):
        """Test finding capability for an unknown tool returns None."""
        registry = CapabilityRegistry()
        definition = CapabilityDefinition(
            name=ToolCapability.FILE_READ,
            description="Read files",
            tools=["read_tool"],
            dependencies=[],
            conflicts=[],
        )
        registry.register_capability(definition)

        # Unknown tool should return None
        capability = registry.get_capability_for_tool("unknown_tool")
        assert capability is None

    def test_get_capability_for_tool_in_multiple_capabilities(self):
        """Test tool that appears in multiple capabilities returns first match."""
        registry = CapabilityRegistry()

        # Register FILE_READ with read_tool
        registry.register_capability(CapabilityDefinition(
            name=ToolCapability.FILE_READ,
            description="Read files",
            tools=["read_tool"],
            dependencies=[],
            conflicts=[],
        ))

        # Register FILE_MANAGEMENT also with read_tool
        registry.register_capability(CapabilityDefinition(
            name=ToolCapability.FILE_MANAGEMENT,
            description="Manage files",
            tools=["read_tool", "copy_tool"],
            dependencies=[],
            conflicts=[],
        ))

        # Should return one of them (first found)
        capability = registry.get_capability_for_tool("read_tool")
        assert capability in [ToolCapability.FILE_READ, ToolCapability.FILE_MANAGEMENT]


class TestCapabilityRegistryGetAllTools:
    """Test get_all_tools method."""

    def test_get_all_tools_returns_unique_tools(self):
        """Test that get_all_tools returns unique tool names."""
        registry = CapabilityRegistry()

        # Register FILE_READ with read_tool
        registry.register_capability(CapabilityDefinition(
            name=ToolCapability.FILE_READ,
            description="Read files",
            tools=["read_tool", "cat"],
            dependencies=[],
            conflicts=[],
        ))

        # Register FILE_WRITE with write_tool and cat (overlap)
        registry.register_capability(CapabilityDefinition(
            name=ToolCapability.FILE_WRITE,
            description="Write files",
            tools=["write_tool", "cat"],
            dependencies=[],
            conflicts=[],
        ))

        # Get all tools
        all_tools = registry.get_all_tools()

        # Should have unique tools
        assert len(all_tools) == len(set(all_tools))
        assert "read_tool" in all_tools
        assert "write_tool" in all_tools
        assert "cat" in all_tools
        assert len(all_tools) == 3  # read_tool, write_tool, cat

    def test_get_all_tools_from_empty_registry(self):
        """Test get_all_tools from empty registry returns empty set."""
        registry = CapabilityRegistry()
        all_tools = registry.get_all_tools()
        assert len(all_tools) == 0
        assert isinstance(all_tools, set)


class TestCapabilitySelectorRecommendCapabilities:
    """Test recommend_capabilities method enhancements."""

    def test_recommend_for_code_edit_task(self):
        """Test recommending capabilities for code editing task."""
        selector = CapabilitySelector()
        capabilities = selector.recommend_capabilities("Edit the Python function to fix bugs")

        # Should recommend code editing capabilities
        assert ToolCapability.CODE_ANALYSIS in capabilities
        assert ToolCapability.FILE_WRITE in capabilities

    def test_recommend_for_docker_task(self):
        """Test recommending capabilities for Docker task."""
        selector = CapabilitySelector()
        capabilities = selector.recommend_capabilities("Build and deploy Docker container")

        # Should recommend containerization capability
        assert ToolCapability.CONTAINERIZATION in capabilities

    def test_recommend_for_testing_task(self):
        """Test recommending capabilities for testing task."""
        selector = CapabilitySelector()
        capabilities = selector.recommend_capabilities("Run unit tests and generate coverage report")

        # Should recommend testing capability
        assert ToolCapability.TESTING in capabilities

    def test_recommend_for_web_research_task(self):
        """Test recommending capabilities for web research task."""
        selector = CapabilitySelector()
        capabilities = selector.recommend_capabilities("Search the web for latest AI trends")

        # Should recommend web search capability
        assert ToolCapability.WEB_SEARCH in capabilities

    def test_recommend_for_vague_task_returns_empty(self):
        """Test that vague task descriptions return minimal recommendations."""
        selector = CapabilitySelector()
        capabilities = selector.recommend_capabilities("do something")

        # Should return empty or minimal recommendations
        assert len(capabilities) == 0


class TestCapabilitySelectorGetCapabilitySummary:
    """Test get_capability_summary method."""

    def test_get_capability_summary(self):
        """Test getting summary of all capabilities and their tools."""
        registry = CapabilityRegistry()

        # Register capabilities
        registry.register_capability(CapabilityDefinition(
            name=ToolCapability.FILE_READ,
            description="Read files",
            tools=["read", "cat"],
            dependencies=[],
            conflicts=[],
        ))

        registry.register_capability(CapabilityDefinition(
            name=ToolCapability.FILE_WRITE,
            description="Write files",
            tools=["write", "edit"],
            dependencies=[],
            conflicts=[],
        ))

        selector = CapabilitySelector(registry)
        summary = selector.get_capability_summary()

        # Verify structure
        assert isinstance(summary, dict)
        assert "file_read" in summary
        assert "file_write" in summary

        # Verify tools
        assert "read" in summary["file_read"]
        assert "cat" in summary["file_read"]
        assert "write" in summary["file_write"]
        assert "edit" in summary["file_write"]

    def test_get_capability_summary_empty_registry(self):
        """Test get_capability_summary with empty registry."""
        selector = CapabilitySelector()
        summary = selector.get_capability_summary()
        assert isinstance(summary, dict)
        assert len(summary) == 0


class TestAutoDiscoveryIntegration:
    """Integration tests for auto-discovery functionality."""

    def test_full_auto_discovery_workflow(self):
        """Test complete workflow: auto-discover, query, summarize."""
        from victor.tools.metadata import ToolMetadata

        registry = CapabilityRegistry()

        # Create mock tools
        tools = [
            MockTool(
                name="git",
                description="Version control",
                metadata=ToolMetadata(
                    category="git",
                    keywords=["commit", "branch"]
                )
            ),
            MockTool(
                name="read",
                description="Read files",
                metadata=ToolMetadata(
                    category="filesystem",
                    keywords=["read", "file"]
                )
            ),
            MockTool(
                name="docker_tool",
                description="Docker operations",
                metadata=ToolMetadata(
                    category="docker",
                    keywords=["container", "docker"]
                )
            ),
        ]

        # Auto-discover
        import asyncio
        asyncio.run(registry.auto_discover_capabilities(tools))

        # Query tools by capability
        git_tools = registry.get_tools_for_capability(ToolCapability.VERSION_CONTROL)
        assert "git" in git_tools

        file_tools = registry.get_tools_for_capability(ToolCapability.FILE_READ)
        assert "read" in file_tools

        docker_tools = registry.get_tools_for_capability(ToolCapability.CONTAINERIZATION)
        assert "docker_tool" in docker_tools

        # Get all tools
        all_tools = registry.get_all_tools()
        assert "git" in all_tools
        assert "read" in all_tools
        assert "docker_tool" in all_tools

        # Find capability for specific tool
        git_cap = registry.get_capability_for_tool("git")
        assert git_cap == ToolCapability.VERSION_CONTROL

    def test_auto_discovery_with_mixed_metadata(self):
        """Test auto-discovery with mix of explicit and auto-generated metadata."""
        from victor.tools.metadata import ToolMetadata

        registry = CapabilityRegistry()

        tools = [
            # Tool with explicit metadata (custom category)
            MockTool(
                name="cache_tool",
                description="Caching tool",
                metadata=ToolMetadata(
                    category="cache",
                    keywords=["cache", "memoize"]
                )
            ),
            # Tool with auto-generated metadata
            MockTool(
                name="git",
                description="Version control",
                metadata=None
            ),
        ]

        # Auto-discover
        import asyncio
        asyncio.run(registry.auto_discover_capabilities(tools))

        # Both should be registered
        cache_cap = registry.get_capability_for_tool("cache_tool")
        assert cache_cap == ToolCapability.CACHE

        git_cap = registry.get_capability_for_tool("git")
        assert git_cap == ToolCapability.VERSION_CONTROL
