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

"""Team template system for pre-configured multi-agent teams.

This module provides a comprehensive template system for creating multi-agent
teams with pre-defined configurations. Templates encapsulate best practices
for different use cases, verticals, and team formations.

Example:
    # Get a template
    template = TeamTemplateRegistry.get_template("code_review_parallel")

    # Apply to workflow
    team_config = template.to_team_config(
        goal="Review PR #123",
        context={"pr_number": 123}
    )

    # Or create from YAML
    template = TeamTemplate.from_yaml("my_template.yaml")
    TeamTemplateRegistry.register(template)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

import yaml

from victor.core.errors import ConfigurationError
from victor.teams.types import TeamConfig, TeamFormation, TeamMember
from victor.agent.subagents.base import SubAgentRole

if TYPE_CHECKING:
    from victor.workflows.definition import TeamNodeWorkflow


logger = logging.getLogger(__name__)


class TaskComplexity(str, Enum):
    """Task complexity levels for template selection."""

    TRIVIAL = "trivial"  # < 5 minutes, 1-2 agents
    QUICK = "quick"  # 5-10 minutes, 2-3 agents
    STANDARD = "standard"  # 15-30 minutes, 4-6 agents
    COMPLEX = "complex"  # 30-60 minutes, 7-10 agents
    CRITICAL = "critical"  # 60+ minutes, consensus-based, verification-heavy


class VerticalType(str, Enum):
    """Vertical types for template categorization."""

    CODING = "coding"
    RESEARCH = "research"
    DEVOPS = "devops"
    DATA_ANALYSIS = "dataanalysis"
    RAG = "rag"
    BENCHMARK = "benchmark"
    GENERAL = "general"


@dataclass
class TeamMemberSpec:
    """Specification for a team member in a template.

    Attributes:
        id: Unique member identifier
        role: SubAgentRole specialization
        name: Human-readable name
        goal: Member's objective
        backstory: Rich persona description
        expertise: Domain expertise areas
        personality: Communication style
        tool_budget: Tool call budget
        allowed_tools: Specific tools allowed
        can_delegate: Delegation capability
        max_delegation_depth: Maximum delegation levels
        memory: Enable persistent memory
        cache: Enable tool result caching
        max_iterations: Iteration limit
    """

    id: str
    role: str
    name: str
    goal: str
    backstory: str = ""
    expertise: List[str] = field(default_factory=list)
    personality: str = ""
    tool_budget: int = 15
    allowed_tools: Optional[List[str]] = None
    can_delegate: bool = False
    max_delegation_depth: int = 0
    memory: bool = False
    cache: bool = True
    max_iterations: Optional[int] = None

    def to_member(self) -> TeamMember:
        """Convert to TeamMember instance.

        Returns:
            TeamMember configured from this spec
        """
        return TeamMember(
            id=self.id,
            role=SubAgentRole(self.role),
            name=self.name,
            goal=self.goal,
            backstory=self.backstory,
            expertise=self.expertise,
            personality=self.personality,
            tool_budget=self.tool_budget,
            allowed_tools=self.allowed_tools,
            can_delegate=self.can_delegate,
            max_delegation_depth=self.max_delegation_depth,
            memory=self.memory,
            cache=self.cache,
            max_iterations=self.max_iterations,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeamMemberSpec":
        """Create from dictionary.

        Args:
            data: Dictionary with member spec fields

        Returns:
            TeamMemberSpec instance
        """
        return cls(
            id=data["id"],
            role=data.get("role", "executor"),
            name=data.get("name", data["id"]),
            goal=data.get("goal", ""),
            backstory=data.get("backstory", ""),
            expertise=data.get("expertise", []),
            personality=data.get("personality", ""),
            tool_budget=data.get("tool_budget", 15),
            allowed_tools=data.get("allowed_tools"),
            can_delegate=data.get("can_delegate", False),
            max_delegation_depth=data.get("max_delegation_depth", 0),
            memory=data.get("memory", False),
            cache=data.get("cache", True),
            max_iterations=data.get("max_iterations"),
        )


@dataclass
class TeamTemplate:
    """Template for pre-configured multi-agent teams.

    Templates encapsulate best practices for team composition, formation,
    and configuration. They can be instantiated and customized for specific tasks.

    Attributes:
        name: Unique template identifier
        display_name: Human-readable name
        description: What this template does
        long_description: Detailed explanation
        version: Template version
        author: Template author
        vertical: Primary vertical (coding, research, etc.)
        formation: Team formation pattern
        members: Member specifications
        use_cases: List of applicable use cases
        tags: Discoverability tags
        complexity: Typical task complexity
        max_iterations: Team iteration limit
        total_tool_budget: Total tool budget
        timeout_seconds: Execution timeout
        config: Additional configuration
        metadata: Template metadata
        examples: Usage examples
    """

    name: str
    display_name: str
    description: str
    formation: str
    members: List[TeamMemberSpec]
    version: str = "0.5.0"
    author: str = "Victor AI"
    vertical: str = "general"
    long_description: str = ""
    use_cases: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    complexity: str = "standard"
    max_iterations: int = 50
    total_tool_budget: int = 100
    timeout_seconds: int = 600
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate template configuration."""
        if not self.members:
            raise ValueError(f"Template '{self.name}' must have at least one member")

        # Validate formation
        valid_formations = [f.value for f in TeamFormation]
        if self.formation not in valid_formations:
            raise ValueError(f"Invalid formation '{self.formation}'. " f"Valid: {valid_formations}")

        # Validate hierarchical has exactly one manager
        if self.formation == "hierarchical":
            managers = [m for m in self.members if m.can_delegate]
            if len(managers) != 1:
                raise ValueError(
                    f"Hierarchical template '{self.name}' must have exactly one member "
                    f"with can_delegate=True (manager)"
                )

    def to_team_config(
        self,
        goal: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **overrides: Any,
    ) -> TeamConfig:
        """Convert to executable TeamConfig.

        Args:
            goal: Override team goal
            context: Initial shared context
            **overrides: Additional overrides (max_iterations, timeout, etc.)

        Returns:
            TeamConfig ready for execution
        """
        # Convert members
        members = [spec.to_member() for spec in self.members]

        # Use custom goal or template description
        team_goal = goal or self.description

        # Apply overrides
        max_iterations = overrides.get("max_iterations", self.max_iterations)
        total_tool_budget = overrides.get("total_tool_budget", self.total_tool_budget)
        timeout_seconds = overrides.get("timeout_seconds", self.timeout_seconds)

        return TeamConfig(
            name=self.display_name,
            goal=team_goal,
            members=members,
            formation=TeamFormation(self.formation),
            max_iterations=max_iterations,
            total_tool_budget=total_tool_budget,
            shared_context=context or {},
            timeout_seconds=timeout_seconds,
        )

    def to_team_node(
        self,
        node_id: str,
        goal: Optional[str] = None,
        output_key: Optional[str] = None,
        **overrides: Any,
    ) -> "TeamNodeWorkflow":
        """Convert to TeamNodeWorkflow for use in workflows.

        Args:
            node_id: Node identifier
            goal: Override team goal
            output_key: Output key for results
            **overrides: Additional overrides

        Returns:
            TeamNodeWorkflow instance
        """
        from victor.workflows.definition import TeamNodeWorkflow

        return TeamNodeWorkflow(
            id=node_id,
            name=self.display_name,
            goal=goal or self.description,
            team_formation=self.formation,
            members=[m.to_member().to_dict() for m in self.members],
            timeout_seconds=overrides.get("timeout_seconds", self.timeout_seconds),
            total_tool_budget=overrides.get("total_tool_budget", self.total_tool_budget),
            max_iterations=overrides.get("max_iterations", self.max_iterations),
            output_key=output_key or f"{node_id}_result",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "long_description": self.long_description,
            "version": self.version,
            "author": self.author,
            "vertical": self.vertical,
            "formation": self.formation,
            "members": [
                {
                    "id": m.id,
                    "role": m.role,
                    "name": m.name,
                    "goal": m.goal,
                    "backstory": m.backstory,
                    "expertise": m.expertise,
                    "personality": m.personality,
                    "tool_budget": m.tool_budget,
                    "allowed_tools": m.allowed_tools,
                    "can_delegate": m.can_delegate,
                    "max_delegation_depth": m.max_delegation_depth,
                    "memory": m.memory,
                    "cache": m.cache,
                    "max_iterations": m.max_iterations,
                }
                for m in self.members
            ],
            "use_cases": self.use_cases,
            "tags": self.tags,
            "complexity": self.complexity,
            "max_iterations": self.max_iterations,
            "total_tool_budget": self.total_tool_budget,
            "timeout_seconds": self.timeout_seconds,
            "config": self.config,
            "metadata": self.metadata,
            "examples": self.examples,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeamTemplate":
        """Create from dictionary.

        Args:
            data: Dictionary with template fields

        Returns:
            TeamTemplate instance
        """
        members = [TeamMemberSpec.from_dict(m_data) for m_data in data.get("members", [])]

        return cls(
            name=data["name"],
            display_name=data.get("display_name", data["name"]),
            description=data.get("description", ""),
            long_description=data.get("long_description", ""),
            version=data.get("version", "0.5.0"),
            author=data.get("author", "Victor AI"),
            vertical=data.get("vertical", "general"),
            formation=data.get("formation", "sequential"),
            members=members,
            use_cases=data.get("use_cases", []),
            tags=data.get("tags", []),
            complexity=data.get("complexity", "standard"),
            max_iterations=data.get("max_iterations", 50),
            total_tool_budget=data.get("total_tool_budget", 100),
            timeout_seconds=data.get("timeout_seconds", 600),
            config=data.get("config", {}),
            metadata=data.get("metadata", {}),
            examples=data.get("examples", []),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "TeamTemplate":
        """Load template from YAML file.

        Args:
            yaml_path: Path to YAML file

        Returns:
            TeamTemplate instance

        Raises:
            ConfigurationError: If file is invalid
        """
        path = Path(yaml_path)
        if not path.exists():
            raise ConfigurationError(
                f"Template file not found: {path}",
                config_key=f"template.{path.name}",
                recovery_hint="Check the file path and try again.",
            )

        try:
            content = path.read_text()
            data = yaml.safe_load(content)
            return cls.from_dict(data)
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in template file {path}: {e}",
                config_key=f"template.{path.name}",
                recovery_hint="Fix YAML syntax errors.",
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load template from {path}: {e}",
                config_key=f"template.{path.name}",
            )

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save template to YAML file.

        Args:
            yaml_path: Destination path
        """
        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved template '{self.name}' to {path}")


class TeamTemplateRegistry:
    """Registry for team templates.

    Provides centralized template management with search and filtering.

    Attributes:
        _templates: Registered templates
        _template_dir: Directory for YAML templates
    """

    _instance: Optional["TeamTemplateRegistry"] = None

    def __init__(self, template_dir: Optional[Path] = None):
        """Initialize registry.

        Args:
            template_dir: Directory containing YAML templates
        """
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        self._template_dir = Path(template_dir)
        self._templates: Dict[str, TeamTemplate] = {}
        self._manually_registered: Set[str] = set()
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "TeamTemplateRegistry":
        """Get singleton registry instance.

        Returns:
            TeamTemplateRegistry instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_templates(self, force_reload: bool = False) -> None:
        """Load all templates from YAML files.

        Args:
            force_reload: Force reload even if already loaded
        """
        if self._loaded and not force_reload:
            return

        # Preserve manually registered templates
        manual_templates = {
            name: template
            for name, template in self._templates.items()
            if name in self._manually_registered
        }

        self._templates.clear()

        if not self._template_dir.exists():
            logger.warning(f"Template directory not found: {self._template_dir}")
            # Restore manually registered templates
            self._templates.update(manual_templates)
            self._loaded = True
            return

        # Load all YAML files
        for yaml_file in self._template_dir.rglob("*.yaml"):
            try:
                template = TeamTemplate.from_yaml(yaml_file)
                self.register(template, silent=True)
            except Exception as e:
                logger.warning(f"Failed to load template from {yaml_file}: {e}")

        # Restore manually registered templates (they override YAML templates)
        self._templates.update(manual_templates)

        self._loaded = True
        logger.info(f"Loaded {len(self._templates)} templates from {self._template_dir}")

    def register(self, template: TeamTemplate, silent: bool = False) -> None:
        """Register a template.

        Args:
            template: Template to register
            silent: Skip logging
        """
        if template.name in self._templates:
            if not silent:
                logger.warning(f"Template '{template.name}' already registered, overwriting")
        self._templates[template.name] = template
        # Track manually registered templates (not from YAML files)
        if not silent:
            self._manually_registered.add(template.name)
        if not silent:
            logger.info(f"Registered template '{template.name}'")

    def get_template(self, name: str) -> Optional[TeamTemplate]:
        """Get template by name.

        Args:
            name: Template name

        Returns:
            TeamTemplate or None if not found
        """
        self.load_templates()
        return self._templates.get(name)

    def list_templates(
        self,
        vertical: Optional[str] = None,
        formation: Optional[str] = None,
        complexity: Optional[str] = None,
        tags: Optional[Set[str]] = None,
    ) -> List[str]:
        """List templates with optional filters.

        Args:
            vertical: Filter by vertical
            formation: Filter by formation
            complexity: Filter by complexity
            tags: Filter by tags (must have all)

        Returns:
            List of template names
        """
        self.load_templates()

        templates = list(self._templates.values())

        # Apply filters
        if vertical:
            templates = [t for t in templates if t.vertical == vertical]

        if formation:
            templates = [t for t in templates if t.formation == formation]

        if complexity:
            templates = [t for t in templates if t.complexity == complexity]

        if tags:
            templates = [t for t in templates if tags.issubset(set(t.tags))]

        return [t.name for t in templates]

    def search(
        self,
        query: str,
        vertical: Optional[str] = None,
        formation: Optional[str] = None,
    ) -> List[TeamTemplate]:
        """Search templates by query string.

        Args:
            query: Search query
            vertical: Optional vertical filter
            formation: Optional formation filter

        Returns:
            List of matching templates
        """
        self.load_templates()

        query_lower = query.lower()
        templates = list(self._templates.values())

        # Apply filters
        results = []
        for template in templates:
            # Check vertical filter
            if vertical and template.vertical != vertical:
                continue

            # Check formation filter
            if formation and template.formation != formation:
                continue

            # Search in name, description, tags, use_cases
            searchable_text = " ".join(
                [
                    template.name,
                    template.display_name,
                    template.description,
                    " ".join(template.tags),
                    " ".join(template.use_cases),
                ]
            ).lower()

            if query_lower in searchable_text:
                results.append(template)

        return results

    def get_by_vertical(self, vertical: str) -> List[TeamTemplate]:
        """Get all templates for a vertical.

        Args:
            vertical: Vertical name

        Returns:
            List of templates
        """
        self.load_templates()
        return [t for t in self._templates.values() if t.vertical == vertical]

    def get_by_formation(self, formation: str) -> List[TeamTemplate]:
        """Get all templates with a formation.

        Args:
            formation: Formation type

        Returns:
            List of templates
        """
        self.load_templates()
        return [t for t in self._templates.values() if t.formation == formation]

    def suggest_template(
        self,
        task_description: str,
        vertical: Optional[str] = None,
        complexity: Optional[str] = None,
    ) -> Optional[TeamTemplate]:
        """Suggest best template for a task.

        Args:
            task_description: Task description
            vertical: Optional vertical hint
            complexity: Optional complexity hint

        Returns:
            Best matching template or None
        """
        self.load_templates()

        # Search by task description
        results = self.search(task_description, vertical=vertical)

        # Filter by complexity if specified
        if complexity:
            results = [t for t in results if t.complexity == complexity]

        # Return first match or None
        return results[0] if results else None

    def validate_template(self, template: TeamTemplate) -> List[str]:
        """Validate template configuration.

        Args:
            template: Template to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required fields
        if not template.name:
            errors.append("Template name is required")

        if not template.display_name:
            errors.append("Template display_name is required")

        if not template.description:
            errors.append("Template description is required")

        if not template.members:
            errors.append("Template must have at least one member")

        # Validate formation
        valid_formations = [f.value for f in TeamFormation]
        if template.formation not in valid_formations:
            errors.append(f"Invalid formation '{template.formation}'. Valid: {valid_formations}")

        # Validate members
        member_ids = set()
        for i, member in enumerate(template.members):
            if not member.id:
                errors.append(f"Member {i}: id is required")

            if member.id in member_ids:
                errors.append(f"Duplicate member id: {member.id}")
            member_ids.add(member.id)

            # Validate role
            valid_roles = [r.value for r in SubAgentRole]
            if member.role not in valid_roles:
                errors.append(
                    f"Member {member.id}: invalid role '{member.role}'. " f"Valid: {valid_roles}"
                )

        # Validate hierarchical formation
        if template.formation == "hierarchical":
            managers = [m for m in template.members if m.can_delegate]
            if len(managers) != 1:
                errors.append(
                    f"Hierarchical template must have exactly one member with "
                    f"can_delegate=True, found {len(managers)}"
                )

        return errors

    def invalidate_cache(self) -> None:
        """Invalidate template cache.

        Forces reload on next access.
        """
        self._loaded = False
        self._templates.clear()
        self._manually_registered.clear()


# Global registry instance
_global_registry: Optional[TeamTemplateRegistry] = None


def get_registry() -> TeamTemplateRegistry:
    """Get global template registry.

    Returns:
        TeamTemplateRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = TeamTemplateRegistry.get_instance()
    return _global_registry


def register_template(template: TeamTemplate) -> None:
    """Register template in global registry.

    Args:
        template: Template to register
    """
    registry = get_registry()
    registry.register(template)


def get_template(name: str) -> Optional[TeamTemplate]:
    """Get template from global registry.

    Args:
        name: Template name

    Returns:
        TeamTemplate or None
    """
    registry = get_registry()
    return registry.get_template(name)


def list_templates(**filters: Any) -> List[str]:
    """List templates with optional filters.

    Args:
        **filters: Filter arguments (vertical, formation, complexity, tags)

    Returns:
        List of template names
    """
    registry = get_registry()
    return registry.list_templates(**filters)


def search_templates(
    query: str,
    vertical: Optional[str] = None,
    formation: Optional[str] = None,
) -> List[TeamTemplate]:
    """Search templates.

    Args:
        query: Search query
        vertical: Optional vertical filter
        formation: Optional formation filter

    Returns:
        List of matching templates
    """
    registry = get_registry()
    return registry.search(query, vertical=vertical, formation=formation)


__all__ = [
    # Enums
    "TaskComplexity",
    "VerticalType",
    # Classes
    "TeamMemberSpec",
    "TeamTemplate",
    "TeamTemplateRegistry",
    # Convenience functions
    "get_registry",
    "register_template",
    "get_template",
    "list_templates",
    "search_templates",
]
