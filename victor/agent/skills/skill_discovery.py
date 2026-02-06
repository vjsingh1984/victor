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

"""Dynamic skill discovery system for runtime tool composition.

This module provides the SkillDiscoveryEngine which enables:
- Runtime discovery of available tools (including MCP tools)
- Task-to-tool matching using semantic similarity
- Skill composition from multiple tools
- Dynamic skill registration and management

Key Features:
- Discover tools from ToolRegistry and MCP servers
- Match tools to tasks using semantic similarity
- Compose skills from multiple tools
- Register and manage custom skills

Architecture:
    SkillDiscoveryEngine
    ├── discover_tools()      # Find available tools
    ├── discover_mcp_tools()  # Find MCP tools
    ├── match_tools_to_task() # Semantic matching
    ├── compose_skill()       # Create skill from tools
    └── register_skill()      # Register custom skill

Design Principles:
    - SRP: Single responsibility for tool discovery and skill composition
    - OCP: Open for extension (new skills), closed for modification
    - DIP: Depend on ToolRegistryProtocol, not concrete implementations
    - ISP: Focused interfaces for discovery, matching, composition

Usage:
    from victor.agent.skills.skill_discovery import SkillDiscoveryEngine

    engine = SkillDiscoveryEngine(tool_registry=registry, event_bus=event_bus)

    # Discover available tools
    tools = await engine.discover_tools(context={"category": "coding"})

    # Match tools to task
    matched_tools = await engine.match_tools_to_task("Analyze Python code", tools)

    # Compose a skill
    skill = await engine.compose_skill(
        name="code_analyzer",
        tools=matched_tools,
        description="Analyzes Python code for quality and security"
    )

    # Register skill
    await engine.register_skill(skill)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from victor.core.events import UnifiedEventType
from victor.protocols.tool_selector import (
    IToolSelector,
    ToolSelectionContext,
)
from victor.tools.base import BaseTool
from victor.tools.enums import CostTier

logger = logging.getLogger(__name__)


@dataclass
class AvailableTool:
    """Representation of a discovered tool.

    Attributes:
        name: Tool name/identifier
        description: Tool description
        parameters: JSON schema of tool parameters
        cost_tier: Cost tier (FREE, LOW, MEDIUM, HIGH)
        category: Tool category (coding, devops, rag, etc.)
        source: Source of tool (registry, mcp, custom)
        metadata: Additional tool metadata
        enabled: Whether tool is currently enabled
    """

    name: str
    description: str
    parameters: dict[str, Any]
    cost_tier: CostTier
    category: str = "general"
    source: str = "registry"
    metadata: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "cost_tier": self.cost_tier.value,
            "category": self.category,
            "source": self.source,
            "metadata": self.metadata,
            "enabled": self.enabled,
        }

    @classmethod
    def from_base_tool(cls, tool: BaseTool, category: str = "general") -> "AvailableTool":
        """Create AvailableTool from BaseTool instance.

        Args:
            tool: BaseTool instance
            category: Tool category

        Returns:
            AvailableTool instance
        """
        return cls(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            cost_tier=tool.cost_tier,
            category=category,
            source="registry",
            metadata={
                "version": getattr(tool, "version", None),
                "author": getattr(tool, "author", None),
            },
        )


@dataclass
class ToolSignature:
    """Function signature for tool matching.

    Attributes:
        tool_name: Name of the tool
        input_types: Expected input types
        output_type: Expected output type
        semantic_tags: Semantic tags for matching
        complexity: Tool complexity (1-10)
    """

    tool_name: str
    input_types: dict[str, str]
    output_type: str
    semantic_tags: list[str] = field(default_factory=list)
    complexity: int = 5

    def matches_signature(self, other: "ToolSignature") -> float:
        """Calculate signature match score.

        Args:
            other: Another ToolSignature to compare against

        Returns:
            Match score from 0.0 (no match) to 1.0 (perfect match)
        """
        # Match input types
        input_match = len(set(self.input_types.items()) & set(other.input_types.items()))
        input_score = input_match / max(len(self.input_types), len(other.input_types), 1)

        # Match output types
        output_score = 1.0 if self.output_type == other.output_type else 0.0

        # Match semantic tags
        tag_intersection = set(self.semantic_tags) & set(other.semantic_tags)
        tag_union = set(self.semantic_tags) | set(other.semantic_tags)
        tag_score = len(tag_intersection) / len(tag_union) if tag_union else 0.0

        # Weighted average
        return (input_score * 0.4) + (output_score * 0.3) + (tag_score * 0.3)


@dataclass
class Skill:
    """Composed abstraction over multiple tools.

    Attributes:
        id: Unique skill identifier
        name: Skill name
        description: Skill description
        tools: List of tools in this skill
        dependencies: List of skill IDs this skill depends on
        tags: Semantic tags for discovery
        version: Skill version
        created_at: Creation timestamp
        updated_at: Last update timestamp
        metadata: Additional metadata
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    tools: list[AvailableTool] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    version: str = "0.5.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_tool(self, tool: AvailableTool) -> None:
        """Add a tool to the skill.

        Args:
            tool: Tool to add
        """
        if tool not in self.tools:
            self.tools.append(tool)
            self.updated_at = datetime.utcnow()

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from the skill.

        Args:
            tool_name: Name of tool to remove

        Returns:
            True if tool was removed, False if not found
        """
        original_length = len(self.tools)
        self.tools = [t for t in self.tools if t.name != tool_name]
        if len(self.tools) < original_length:
            self.updated_at = datetime.utcnow()
            return True
        return False

    def get_tool_names(self) -> list[str]:
        """Get list of tool names in this skill.

        Returns:
            List of tool names
        """
        return [t.name for t in self.tools]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tools": [t.to_dict() for t in self.tools],
            "dependencies": self.dependencies,
            "tags": self.tags,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SkillCapabilities:
    """Capabilities analysis for a tool.

    Provides detailed analysis of what a tool can do, including
    input/output types, complexity level, and reliability metrics.

    Attributes:
        input_types: Expected input parameter types
        output_types: Expected output type(s)
        complexity: Complexity score (1-10, higher is more complex)
        reliability: Reliability score (0.0-1.0, higher is more reliable)
        performance: Expected performance tier (fast, medium, slow)
        side_effects: Whether tool has side effects (modifications)
        idempotent: Whether tool is idempotent (same input = same output)
    """

    input_types: dict[str, str]
    output_types: list[str]
    complexity: int = 5
    reliability: float = 0.9
    performance: str = "medium"
    side_effects: bool = False
    idempotent: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of capabilities
        """
        return {
            "input_types": self.input_types,
            "output_types": self.output_types,
            "complexity": self.complexity,
            "reliability": self.reliability,
            "performance": self.performance,
            "side_effects": self.side_effects,
            "idempotent": self.idempotent,
        }

    @classmethod
    def from_tool(cls, tool: BaseTool) -> "SkillCapabilities":
        """Analyze capabilities from a BaseTool.

        Args:
            tool: BaseTool instance

        Returns:
            SkillCapabilities for the tool
        """
        # Extract input types
        input_types = {}
        if "properties" in tool.parameters:
            for param_name, param_def in tool.parameters["properties"].items():
                param_type = param_def.get("type", "string")
                input_types[param_name] = param_type

        # Infer output types from description
        output_types = ["any"]
        desc_lower = tool.description.lower()

        if "file" in desc_lower or "code" in desc_lower:
            output_types = ["file", "code"]
        elif "list" in desc_lower or "search" in desc_lower:
            output_types = ["list", "array"]
        elif "analyz" in desc_lower or "check" in desc_lower:
            output_types = ["analysis", "report"]
        elif "test" in desc_lower:
            output_types = ["test_result", "boolean"]

        # Infer complexity
        complexity = 5
        num_params = len(input_types)
        if num_params <= 2:
            complexity = 3
        elif num_params <= 5:
            complexity = 5
        else:
            complexity = 7

        # Check for side effects based on cost tier
        side_effects = tool.cost_tier in [CostTier.MEDIUM, CostTier.HIGH]

        # Check idempotency
        idempotent = getattr(tool, "is_idempotent", False)
        if not idempotent:
            # Infer from description
            idempotent_keywords = ["read", "get", "fetch", "list", "search", "find", "check"]
            idempotent = any(kw in desc_lower for kw in idempotent_keywords)

        return cls(
            input_types=input_types,
            output_types=output_types,
            complexity=complexity,
            side_effects=side_effects,
            idempotent=idempotent,
        )


@dataclass
class MCPTool:
    """Representation of an MCP-discovered tool.

    Attributes:
        name: Tool name
        description: Tool description
        server_name: MCP server name
        server_url: MCP server URL
        parameters: Tool parameters schema
        enabled: Whether tool is enabled
    """

    name: str
    description: str
    server_name: str
    server_url: str
    parameters: dict[str, Any]
    enabled: bool = True

    def to_available_tool(self) -> AvailableTool:
        """Convert to AvailableTool.

        Returns:
            AvailableTool instance
        """
        return AvailableTool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            cost_tier=CostTier.LOW,  # MCP tools default to LOW cost
            category="mcp",
            source="mcp",
            metadata={"server_name": self.server_name, "server_url": self.server_url},
        )


class SkillDiscoveryEngine:
    """Engine for dynamic skill discovery and composition.

    This class provides:
    - Tool discovery from ToolRegistry and MCP servers
    - Semantic tool-to-task matching
    - Skill composition from tools
    - Skill registration and management

    Design Principles:
        - SRP: Single responsibility for discovery and composition
        - OCP: Open for extension (new tools), closed for modification
        - DIP: Depend on protocols (IToolSelector), not concrete classes
    """

    def __init__(
        self,
        tool_registry: Any,
        tool_selector: Optional[IToolSelector] = None,
        event_bus: Optional[Any] = None,
    ):
        """Initialize SkillDiscoveryEngine.

        Args:
            tool_registry: Tool registry instance
            tool_selector: Optional tool selector for semantic matching
            event_bus: Optional event bus for publishing events
        """
        self._tool_registry = tool_registry
        self._tool_selector = tool_selector
        self._event_bus = event_bus
        self._registered_skills: dict[str, Skill] = {}
        self._mcp_tools_cache: dict[str, list[MCPTool]] = {}

    async def discover_tools(
        self,
        context: Optional[dict[str, Any]] = None,
        categories: Optional[list[str]] = None,
        include_disabled: bool = False,
    ) -> list[AvailableTool]:
        """Discover available tools from registry.

        Args:
            context: Optional context for discovery
            categories: Filter by tool categories
            include_disabled: Include disabled tools

        Returns:
            List of discovered tools
        """
        logger.info("Discovering tools from registry")
        tools = []

        # Get all tools from registry
        if hasattr(self._tool_registry, "list_tools"):
            tool_names = self._tool_registry.list_tools()
        elif hasattr(self._tool_registry, "tools"):
            tool_names = list(self._tool_registry.tools.keys())
        else:
            logger.warning("Tool registry does not expose list_tools() or tools")
            return []

        for tool_name in tool_names:
            try:
                tool = self._tool_registry.get_tool(tool_name)
                if tool is None:
                    continue

                # Check if enabled
                if not include_disabled and not getattr(tool, "enabled", True):
                    continue

                # Determine category
                category = (
                    getattr(tool, "category", None) or context.get("category", "general")
                    if context
                    else "general"
                )

                # Filter by categories
                if categories and category not in categories:
                    continue

                available_tool = AvailableTool.from_base_tool(tool, category)
                tools.append(available_tool)

            except Exception as e:
                logger.warning(f"Error discovering tool {tool_name}: {e}")

        logger.info(f"Discovered {len(tools)} tools")

        # Publish event
        await self._publish_event(
            UnifiedEventType.TOOL_RESULT,
            {"tools_discovered": len(tools), "categories": categories, "event_type": "discovery"},
        )

        return tools

    async def discover_mcp_tools(
        self,
        server_url: Optional[str] = None,
        refresh_cache: bool = False,
    ) -> list[MCPTool]:
        """Discover tools from MCP servers.

        Args:
            server_url: Specific MCP server URL (None = all servers)
            refresh_cache: Force refresh of cached tools

        Returns:
            List of discovered MCP tools
        """
        logger.info(f"Discovering MCP tools from {server_url or 'all servers'}")

        cache_key = server_url or "all"

        # Return cached tools if available
        if not refresh_cache and cache_key in self._mcp_tools_cache:
            logger.info(
                f"Returning cached MCP tools: {len(self._mcp_tools_cache[cache_key])} tools"
            )
            return self._mcp_tools_cache[cache_key]

        mcp_tools = []

        # Try to get MCP connector from registry or container
        mcp_connector = None
        if hasattr(self._tool_registry, "_mcp_connector"):
            mcp_connector = self._tool_registry._mcp_connector

        if mcp_connector and hasattr(mcp_connector, "get_servers"):
            try:
                servers = mcp_connector.get_servers()

                for server_info in servers:
                    # Filter by server_url if specified
                    if server_url:
                        server_url_attr = getattr(server_info, "url", None)
                        if server_url_attr != server_url:
                            continue

                    # Get tools from server
                    if hasattr(server_info, "tools"):
                        for tool_info in server_info.tools:
                            mcp_tool = MCPTool(
                                name=tool_info.get("name", "unknown"),
                                description=tool_info.get("description", ""),
                                server_name=server_info.name,
                                server_url=getattr(server_info, "url", ""),
                                parameters=tool_info.get("inputSchema", {}),
                            )
                            mcp_tools.append(mcp_tool)

            except Exception as e:
                logger.warning(f"Error discovering MCP tools: {e}")

        # Cache results
        self._mcp_tools_cache[cache_key] = mcp_tools

        logger.info(f"Discovered {len(mcp_tools)} MCP tools")

        # Publish event
        await self._publish_event(
            UnifiedEventType.TOOL_RESULT,
            {
                "mcp_tools_discovered": len(mcp_tools),
                "server_url": server_url,
                "event_type": "mcp_discovery",
            },
        )

        return mcp_tools

    def get_tool_signatures(self, tools: list[AvailableTool]) -> list[ToolSignature]:
        """Extract tool signatures for matching.

        Args:
            tools: List of available tools

        Returns:
            List of tool signatures
        """
        signatures = []

        for tool in tools:
            # Extract input types from parameters
            input_types = {}
            if "properties" in tool.parameters:
                for param_name, param_def in tool.parameters["properties"].items():
                    param_type = param_def.get("type", "string")
                    input_types[param_name] = param_type

            # Infer output type from description
            output_type = "any"
            desc_lower = tool.description.lower()
            if "file" in desc_lower or "code" in desc_lower:
                output_type = "file"
            elif "list" in desc_lower or "search" in desc_lower:
                output_type = "list"
            elif "analyz" in desc_lower or "check" in desc_lower:
                output_type = "analysis"

            # Extract semantic tags from description
            tags = []
            desc_words = set(tool.description.lower().split())
            keyword_sets = [
                {"read", "file", "load", "open"},
                {"write", "save", "create", "file"},
                {"search", "find", "locate"},
                {"test", "check", "verify"},
                {"git", "commit", "push"},
                {"docker", "container", "image"},
            ]
            for keyword_set in keyword_sets:
                if desc_words & keyword_set:
                    tags.extend(list(keyword_set))

            signature = ToolSignature(
                tool_name=tool.name,
                input_types=input_types,
                output_type=output_type,
                semantic_tags=tags,
                complexity=tool.metadata.get("complexity", 5),
            )
            signatures.append(signature)

        return signatures

    async def match_tools_to_task(
        self,
        task: str,
        available_tools: list[AvailableTool],
        limit: int = 10,
        min_score: float = 0.3,
    ) -> list[AvailableTool]:
        """Match tools to task using semantic similarity.

        Args:
            task: Task description
            available_tools: List of available tools
            limit: Maximum number of tools to return
            min_score: Minimum relevance score threshold

        Returns:
            List of matched tools ranked by relevance
        """
        logger.info(f"Matching tools to task: {task[:50]}...")

        # Use tool selector if available
        if self._tool_selector:
            try:
                # Don't pass available_tools in metadata - semantic selector needs ToolRegistry, not list
                # The semantic selector will use its stored _tools_registry from initialize_tool_embeddings
                selection_context = ToolSelectionContext(
                    task_description=task,
                )

                result = await self._tool_selector.select_tools(
                    task,
                    limit=limit,
                    min_score=min_score,
                    context=selection_context,
                )

                # Handle both ToolSelectionResult and List[ToolDefinition] return types
                if isinstance(result, list):
                    # result is List[ToolDefinition]
                    from victor.tools.tool import ToolDefinition  

                    tool_map = {t.name: t for t in available_tools}
                    matched_tools = [
                        tool_map[td.name]
                        for td in result
                        if isinstance(td, ToolDefinition) and td.name in tool_map
                    ]
                else:
                    # result is ToolSelectionResult
                    tool_map = {t.name: t for t in available_tools}
                    matched_tools = [
                        tool_map[name] for name in result.selected_tool_names if name in tool_map
                    ]

                logger.info(f"Matched {len(matched_tools)} tools using selector")
                return matched_tools

            except Exception as e:
                logger.warning(f"Tool selector failed, falling back to basic matching: {e}")

        # Fallback: Basic keyword matching
        task_lower = task.lower()
        task_words = set(task_lower.split())

        scored_tools = []
        for tool in available_tools:
            desc_lower = tool.description.lower()
            desc_words = set(desc_lower.split())

            # Calculate overlap score
            overlap = task_words & desc_words
            score = len(overlap) / max(len(task_words), 1)

            if score >= min_score:
                scored_tools.append((tool, score))

        # Sort by score and return top N
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        matched_tools = [t for t, s in scored_tools[:limit]]

        logger.info(f"Matched {len(matched_tools)} tools using basic matching")

        return matched_tools

    async def compose_skill(
        self,
        name: str,
        tools: list[AvailableTool],
        description: str,
        dependencies: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Skill:
        """Compose a skill from multiple tools.

        Args:
            name: Skill name
            tools: List of tools to include
            description: Skill description
            dependencies: Optional list of skill dependencies
            tags: Optional semantic tags
            metadata: Optional metadata

        Returns:
            Composed skill
        """
        logger.info(f"Composing skill '{name}' with {len(tools)} tools")

        # Validate tools
        if not tools:
            raise ValueError("Cannot compose skill with no tools")

        # Extract tags from tools if not provided
        if tags is None:
            all_tags = set()
            for tool in tools:
                desc_words = set(tool.description.lower().split())
                all_tags.update(desc_words)
            tags = list(all_tags)[:10]  # Limit to 10 tags

        skill = Skill(
            name=name,
            description=description,
            tools=tools.copy(),
            dependencies=dependencies or [],
            tags=tags,
            metadata=metadata or {},
        )

        logger.info(f"Composed skill '{name}' with {len(tools)} tools")

        # Publish event
        await self._publish_event(
            UnifiedEventType.TOOL_RESULT,
            {
                "skill_name": name,
                "tool_count": len(tools),
                "tool_names": [t.name for t in tools],
                "event_type": "composition",
            },
        )

        return skill

    async def register_skill(self, skill: Skill) -> bool:
        """Register a skill for later use.

        Args:
            skill: Skill to register

        Returns:
            True if registered successfully, False otherwise
        """
        if not skill.name:
            logger.warning("Cannot register skill with no name")
            return False

        # Check if skill already exists
        if skill.name in self._registered_skills:
            logger.warning(f"Skill '{skill.name}' already exists, updating")

        self._registered_skills[skill.name] = skill

        logger.info(f"Registered skill '{skill.name}' with {len(skill.tools)} tools")

        # Publish event
        await self._publish_event(
            UnifiedEventType.TOOL_RESULT,
            {
                "skill_name": skill.name,
                "tool_count": len(skill.tools),
                "version": skill.version,
                "event_type": "registration",
            },
        )

        return True

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a registered skill by name.

        Args:
            name: Skill name

        Returns:
            Skill if found, None otherwise
        """
        return self._registered_skills.get(name)

    def list_skills(self, tag: Optional[str] = None) -> list[Skill]:
        """List registered skills.

        Args:
            tag: Optional tag filter

        Returns:
            List of skills
        """
        skills = list(self._registered_skills.values())

        if tag:
            skills = [s for s in skills if tag in s.tags]

        return skills

    def unregister_skill(self, name: str) -> bool:
        """Unregister a skill.

        Args:
            name: Skill name

        Returns:
            True if unregistered, False if not found
        """
        if name in self._registered_skills:
            del self._registered_skills[name]
            logger.info(f"Unregistered skill '{name}'")
            return True

        logger.warning(f"Skill '{name}' not found for unregistration")
        return False

    def analyze_tool_capabilities(self, tool: BaseTool) -> SkillCapabilities:
        """Analyze capabilities of a tool.

        Examines a tool's parameters, description, and cost tier to determine
        its capabilities including input/output types, complexity, reliability,
        and operational characteristics.

        Args:
            tool: BaseTool instance to analyze

        Returns:
            SkillCapabilities analysis

        Example:
            >>> capabilities = engine.analyze_tool_capabilities(my_tool)
            >>> print(f"Complexity: {capabilities.complexity}/10")
            >>> print(f"Reliability: {capabilities.reliability * 100}%")
        """
        logger.debug(f"Analyzing capabilities for tool: {tool.name}")

        capabilities = SkillCapabilities.from_tool(tool)

        logger.debug(
            f"Tool '{tool.name}' capabilities: "
            f"complexity={capabilities.complexity}, "
            f"reliability={capabilities.reliability}, "
            f"side_effects={capabilities.side_effects}"
        )

        return capabilities

    def rank_skills_by_relevance(
        self,
        query: str,
        skills: list[Skill],
        top_k: Optional[int] = None,
    ) -> list[Skill]:
        """Rank skills by semantic relevance to a query.

        Uses semantic matching over skill names, descriptions, tags, and tools
        to rank skills by their relevance to the given query.

        Args:
            query: Search query describing the needed capability
            skills: List of skills to rank
            top_k: Optional limit on number of results

        Returns:
            List of skills ranked by relevance (highest first)

        Example:
            >>> skills = engine.list_skills()
            >>> ranked = engine.rank_skills_by_relevance("code analysis", skills)
            >>> top_skill = ranked[0]
        """
        logger.info(f"Ranking {len(skills)} skills by relevance to query: {query[:50]}...")

        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Calculate relevance scores for each skill
        scored_skills = []
        for skill in skills:
            score = 0.0

            # Match in skill name (highest weight)
            if query_lower in skill.name.lower():
                score += 0.4

            # Match in description (high weight)
            desc_words = set(skill.description.lower().split())
            desc_overlap = query_words & desc_words
            score += (len(desc_overlap) / max(len(query_words), 1)) * 0.3

            # Match in tags (medium weight)
            tag_matches = sum(1 for tag in skill.tags if query_lower in tag.lower())
            score += (tag_matches / max(len(skill.tags), 1)) * 0.2

            # Match in tool names/descriptions (lower weight)
            tool_text = " ".join(f"{t.name} {t.description}" for t in skill.tools).lower()
            tool_words = set(tool_text.split())
            tool_overlap = query_words & tool_words
            score += (len(tool_overlap) / max(len(query_words), 1)) * 0.1

            scored_skills.append((skill, score))

        # Sort by score descending
        scored_skills.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k limit if specified
        if top_k is not None:
            scored_skills = scored_skills[:top_k]

        ranked_skills = [skill for skill, score in scored_skills]

        logger.info(
            f"Ranked {len(ranked_skills)} skills, "
            f"top score: {scored_skills[0][1] if scored_skills else 0:.2f}"
        )

        return ranked_skills

    async def _publish_event(self, event_type: UnifiedEventType, data: dict[str, Any]) -> None:
        """Publish event to event bus.

        Args:
            event_type: Event type
            data: Event data
        """
        if self._event_bus:
            try:
                if hasattr(self._event_bus, "publish"):
                    await self._event_bus.publish(event_type, data)
                elif hasattr(self._event_bus, "emit"):
                    await self._event_bus.emit(event_type.value, data)
            except Exception as e:
                logger.warning(f"Failed to publish event {event_type}: {e}")


__all__ = [
    "AvailableTool",
    "ToolSignature",
    "MCPTool",
    "Skill",
    "SkillCapabilities",
    "SkillDiscoveryEngine",
]
