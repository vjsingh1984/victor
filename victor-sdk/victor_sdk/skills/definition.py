"""Skill definition contract for Victor SDK.

A SkillDefinition is a composable unit of agent expertise that sits
between verticals (coarse-grained domain apps) and tools (atomic operations).
Skills bind a focused prompt fragment with a tool subset and constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Set


@dataclass(frozen=True)
class SkillDefinition:
    """A composable unit of agent expertise.

    Skills are the middle layer in Victor's extensibility stack:
        Vertical (domain app) -> Skills -> Tools

    Each skill carries a focused prompt fragment that gets injected into
    the agent's system prompt when the skill is active, along with the
    tool subset needed to execute the skill.

    Attributes:
        name: Unique skill identifier (e.g., "debug_test_failure")
        description: Human-readable description for selection/routing
        category: Skill category (e.g., "coding", "devops", "analysis")
        prompt_fragment: Instructions injected into agent prompt when active
        required_tools: Tools that MUST be available for this skill
        optional_tools: Tools that MAY be used if available
        tags: Searchable tags for discovery
        max_tool_calls: Maximum tool calls when this skill is active
        version: Skill definition version
    """

    name: str
    description: str
    category: str
    prompt_fragment: str
    required_tools: List[str]
    optional_tools: List[str] = field(default_factory=list)
    tags: FrozenSet[str] = field(default_factory=frozenset)
    max_tool_calls: int = 20
    version: str = "1.0.0"
    phase: str = "action"  # "diagnostic", "action", "verification", "documentation"

    @property
    def all_tools(self) -> Set[str]:
        """Return the union of required and optional tools."""
        return set(self.required_tools) | set(self.optional_tools)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "prompt_fragment": self.prompt_fragment,
            "required_tools": list(self.required_tools),
            "optional_tools": list(self.optional_tools),
            "tags": sorted(self.tags),
            "max_tool_calls": self.max_tool_calls,
            "version": self.version,
            "phase": self.phase,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SkillDefinition:
        """Deserialize from dictionary."""
        tags_raw = data.get("tags", [])
        tags = frozenset(tags_raw) if not isinstance(tags_raw, frozenset) else tags_raw
        return cls(
            name=data["name"],
            description=data["description"],
            category=data["category"],
            prompt_fragment=data["prompt_fragment"],
            required_tools=list(data["required_tools"]),
            optional_tools=list(data.get("optional_tools", [])),
            tags=tags,
            max_tool_calls=data.get("max_tool_calls", 20),
            version=data.get("version", "1.0.0"),
            phase=data.get("phase", "action"),
        )
