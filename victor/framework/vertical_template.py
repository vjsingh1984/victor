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

"""Vertical Template dataclass for declarative vertical definitions.

This module provides the VerticalTemplate dataclass that defines the structure
of a vertical template. Templates are used to generate new verticals through
the scaffolding system, reducing code duplication by 65-70%.

Templates can be defined in YAML or created programmatically, and contain all
the information needed to generate a complete vertical implementation including:
- Metadata (name, description, version)
- Core configuration (tools, system prompt, stages)
- Extension specifications (middleware, safety, prompts)
- Workflow definitions
- Team formations
- Capability configurations

Example:
    template = VerticalTemplate(
        metadata=VerticalMetadata(
            name="security",
            description="Security analysis and vulnerability detection",
            version="1.0.0",
        ),
        tools=["read", "grep", "security_scan"],
        system_prompt="You are a security expert...",
        stages={...},
    )

    # Generate vertical from template
    generator = VerticalGenerator(template)
    generator.generate(output_dir="victor/security")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from victor.core.vertical_types import StageDefinition

logger = logging.getLogger(__name__)


# =============================================================================
# Metadata Components
# =============================================================================


@dataclass
class VerticalMetadata:
    """Metadata for a vertical template.

    Attributes:
        name: Vertical identifier (e.g., "coding", "research")
        description: Human-readable description
        version: Semantic version (e.g., "1.0.0")
        author: Optional author name
        license: License name (default: "Apache-2.0")
        category: Vertical category (coding, devops, rag, etc.)
        tags: List of tags for discovery
        provider_hints: Hints for LLM provider selection
        evaluation_criteria: Criteria for evaluating agent performance
    """

    name: str
    description: str
    version: str = "1.0.0"
    author: Optional[str] = None
    license: str = "Apache-2.0"
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    provider_hints: Dict[str, Any] = field(default_factory=dict)
    evaluation_criteria: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "license": self.license,
            "category": self.category,
            "tags": self.tags,
            "provider_hints": self.provider_hints,
            "evaluation_criteria": self.evaluation_criteria,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerticalMetadata":
        """Create from dictionary (YAML deserialization)."""
        return cls(
            name=data["name"],
            description=data["description"],
            version=data.get("version", "1.0.0"),
            author=data.get("author"),
            license=data.get("license", "Apache-2.0"),
            category=data.get("category", "general"),
            tags=data.get("tags", []),
            provider_hints=data.get("provider_hints", {}),
            evaluation_criteria=data.get("evaluation_criteria", []),
        )


# =============================================================================
# Extension Specifications
# =============================================================================


@dataclass
class MiddlewareSpec:
    """Specification for a middleware component.

    Attributes:
        name: Middleware identifier
        class_name: Python class name
        module: Import path (e.g., "victor.coding.middleware")
        enabled: Whether middleware is active
        config: Middleware configuration parameters
    """

    name: str
    class_name: str
    module: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "class_name": self.class_name,
            "module": self.module,
            "enabled": self.enabled,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MiddlewareSpec":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            class_name=data["class_name"],
            module=data["module"],
            enabled=data.get("enabled", True),
            config=data.get("config", {}),
        )


@dataclass
class SafetyPatternSpec:
    """Specification for a safety pattern.

    Attributes:
        name: Pattern identifier
        pattern: Regex pattern to match
        description: What this pattern detects
        severity: Severity level (low, medium, high, critical)
        category: Pattern category (git, files, commands, etc.)
    """

    name: str
    pattern: str
    description: str
    severity: str = "medium"
    category: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "pattern": self.pattern,
            "description": self.description,
            "severity": self.severity,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyPatternSpec":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            pattern=data["pattern"],
            description=data["description"],
            severity=data.get("severity", "medium"),
            category=data.get("category", "general"),
        )


@dataclass
class PromptHintSpec:
    """Specification for a task type prompt hint.

    Attributes:
        task_type: Task type identifier (e.g., "create", "edit", "debug")
        hint: Prompt hint text
        tool_budget: Suggested tool budget
        priority_tools: List of priority tool names
    """

    task_type: str
    hint: str
    tool_budget: int = 10
    priority_tools: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_type": self.task_type,
            "hint": self.hint,
            "tool_budget": self.tool_budget,
            "priority_tools": self.priority_tools,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptHintSpec":
        """Create from dictionary."""
        return cls(
            task_type=data["task_type"],
            hint=data["hint"],
            tool_budget=data.get("tool_budget", 10),
            priority_tools=data.get("priority_tools", []),
        )


@dataclass
class ExtensionSpecs:
    """Collection of extension specifications.

    Attributes:
        middleware: List of middleware specifications
        safety_patterns: List of safety pattern specifications
        prompt_hints: List of prompt hint specifications
        handlers: Dict of workflow handler specifications
        personas: Dict of persona definitions
        composed_chains: Dict of LCEL chain definitions
    """

    middleware: List[MiddlewareSpec] = field(default_factory=list)
    safety_patterns: List[SafetyPatternSpec] = field(default_factory=list)
    prompt_hints: List[PromptHintSpec] = field(default_factory=list)
    handlers: Dict[str, Any] = field(default_factory=dict)
    personas: Dict[str, Any] = field(default_factory=dict)
    composed_chains: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "middleware": [m.to_dict() for m in self.middleware],
            "safety_patterns": [p.to_dict() for p in self.safety_patterns],
            "prompt_hints": [h.to_dict() for h in self.prompt_hints],
            "handlers": self.handlers,
            "personas": self.personas,
            "composed_chains": self.composed_chains,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtensionSpecs":
        """Create from dictionary."""
        return cls(
            middleware=[MiddlewareSpec.from_dict(m) for m in data.get("middleware", [])],
            safety_patterns=[
                SafetyPatternSpec.from_dict(p) for p in data.get("safety_patterns", [])
            ],
            prompt_hints=[PromptHintSpec.from_dict(h) for h in data.get("prompt_hints", [])],
            handlers=data.get("handlers", {}),
            personas=data.get("personas", {}),
            composed_chains=data.get("composed_chains", {}),
        )


# =============================================================================
# Workflow and Team Specifications
# =============================================================================


@dataclass
class WorkflowSpec:
    """Specification for a workflow.

    Attributes:
        name: Workflow identifier
        description: Workflow description
        yaml_path: Path to YAML workflow definition
        handler_module: Module containing workflow handlers
    """

    name: str
    description: str
    yaml_path: Optional[str] = None
    handler_module: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "yaml_path": self.yaml_path,
            "handler_module": self.handler_module,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowSpec":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            yaml_path=data.get("yaml_path"),
            handler_module=data.get("handler_module"),
        )


@dataclass
class TeamRoleSpec:
    """Specification for a team role.

    Attributes:
        name: Role identifier
        display_name: Human-readable name
        description: Role description
        persona: Persona text for this role
        tool_categories: Categories of tools this role uses
        capabilities: Capabilities required for this role
    """

    name: str
    display_name: str
    description: str
    persona: str
    tool_categories: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "persona": self.persona,
            "tool_categories": self.tool_categories,
            "capabilities": self.capabilities,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeamRoleSpec":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            display_name=data["display_name"],
            description=data["description"],
            persona=data["persona"],
            tool_categories=data.get("tool_categories", []),
            capabilities=data.get("capabilities", []),
        )


@dataclass
class TeamSpec:
    """Specification for a team formation.

    Attributes:
        name: Team identifier
        display_name: Human-readable name
        description: Team description
        formation: Formation type (pipeline, parallel, sequential, etc.)
        communication_style: Communication style (structured, casual, etc.)
        max_iterations: Maximum iterations for team coordination
        roles: List of role specifications
    """

    name: str
    display_name: str
    description: str
    formation: str = "parallel"
    communication_style: str = "structured"
    max_iterations: int = 5
    roles: List[TeamRoleSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "formation": self.formation,
            "communication_style": self.communication_style,
            "max_iterations": self.max_iterations,
            "roles": [r.to_dict() for r in self.roles],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeamSpec":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            display_name=data["display_name"],
            description=data["description"],
            formation=data.get("formation", "parallel"),
            communication_style=data.get("communication_style", "structured"),
            max_iterations=data.get("max_iterations", 5),
            roles=[TeamRoleSpec.from_dict(r) for r in data.get("roles", [])],
        )


# =============================================================================
# Capability Specifications
# =============================================================================


@dataclass
class CapabilitySpec:
    """Specification for a capability.

    Attributes:
        name: Capability identifier
        type: Capability type (tool, workflow, middleware, validator, observer)
        description: Capability description
        enabled: Whether capability is active
        handler: Import path to handler (e.g., "victor.coding.review:QualityChecker")
        config: Capability configuration
    """

    name: str
    type: str
    description: str
    enabled: bool = True
    handler: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "enabled": self.enabled,
            "handler": self.handler,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilitySpec":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            type=data["type"],
            description=data["description"],
            enabled=data.get("enabled", True),
            handler=data.get("handler"),
            config=data.get("config", {}),
        )


# =============================================================================
# Main Template Class
# =============================================================================


@dataclass
class VerticalTemplate:
    """Template for declarative vertical definitions.

    A VerticalTemplate contains all the information needed to generate a complete
    vertical implementation, including metadata, tools, stages, extensions,
    workflows, teams, and capabilities.

    Templates can be:
    1. Defined in YAML and loaded via VerticalTemplateLoader
    2. Created programmatically using this dataclass
    3. Generated from existing verticals using VerticalExtractor

    Templates support inheritance through the parent_template field, allowing
    child templates to extend and override parent templates.

    Attributes:
        metadata: Vertical metadata (name, description, version, etc.)
        tools: List of tool names for this vertical
        system_prompt: System prompt text
        stages: Stage definitions for workflow
        extensions: Extension specifications (middleware, safety, prompts, etc.)
        workflows: Workflow specifications
        teams: Team formation specifications
        capabilities: Capability specifications
        custom_config: Custom vertical-specific configuration
        file_templates: Custom file templates (e.g., custom __init__.py)
        parent_template: Optional parent template name for inheritance
    """

    metadata: VerticalMetadata
    tools: List[str]
    system_prompt: str
    stages: Dict[str, StageDefinition]
    extensions: ExtensionSpecs = field(default_factory=ExtensionSpecs)
    workflows: List[WorkflowSpec] = field(default_factory=list)
    teams: List[TeamSpec] = field(default_factory=list)
    capabilities: List[CapabilitySpec] = field(default_factory=list)
    custom_config: Dict[str, Any] = field(default_factory=dict)
    file_templates: Dict[str, str] = field(default_factory=dict)
    parent_template: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary for YAML serialization.

        Returns:
            Dictionary representation of the template
        """
        result = {
            "metadata": self.metadata.to_dict(),
            "tools": self.tools,
            "system_prompt": self.system_prompt,
            "stages": {
                name: {
                    "name": stage.name,
                    "description": stage.description,
                    "tools": list(stage.tools) if stage.tools else [],
                    "keywords": list(stage.keywords) if stage.keywords else [],
                    "next_stages": list(stage.next_stages) if stage.next_stages else [],
                }
                for name, stage in self.stages.items()
            },
            "extensions": self.extensions.to_dict(),
            "workflows": [w.to_dict() for w in self.workflows],
            "teams": [t.to_dict() for t in self.teams],
            "capabilities": [c.to_dict() for c in self.capabilities],
            "custom_config": self.custom_config,
            "file_templates": self.file_templates,
        }

        # Add parent_template if set
        if self.parent_template:
            result["parent_template"] = self.parent_template

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerticalTemplate":
        """Create template from dictionary (YAML deserialization).

        Args:
            data: Dictionary containing template data

        Returns:
            VerticalTemplate instance
        """
        metadata = VerticalMetadata.from_dict(data["metadata"])

        # Parse stages
        stages_data = data.get("stages", {})
        stages = {}
        for stage_name, stage_dict in stages_data.items():
            stages[stage_name] = StageDefinition(
                name=stage_dict["name"],
                description=stage_dict["description"],
                tools=set(stage_dict.get("tools", [])),
                keywords=stage_dict.get("keywords", []),
                next_stages=set(stage_dict.get("next_stages", [])),
            )

        return cls(
            metadata=metadata,
            tools=data["tools"],
            system_prompt=data["system_prompt"],
            stages=stages,
            extensions=ExtensionSpecs.from_dict(data.get("extensions", {})),
            workflows=[WorkflowSpec.from_dict(w) for w in data.get("workflows", [])],
            teams=[TeamSpec.from_dict(t) for t in data.get("teams", [])],
            capabilities=[CapabilitySpec.from_dict(c) for c in data.get("capabilities", [])],
            custom_config=data.get("custom_config", {}),
            file_templates=data.get("file_templates", {}),
            parent_template=data.get("parent_template"),
        )

    def validate(self) -> List[str]:
        """Validate template completeness.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Validate metadata
        if not self.metadata.name:
            errors.append("Template metadata.name is required")
        if not self.metadata.description:
            errors.append("Template metadata.description is required")

        # Validate tools
        if not self.tools:
            errors.append("Template must specify at least one tool")
        elif not isinstance(self.tools, list):
            errors.append("Template tools must be a list")

        # Validate system prompt
        if not self.system_prompt:
            errors.append("Template must specify a system prompt")

        # Validate stages
        if not self.stages:
            errors.append("Template must specify at least one stage")
        else:
            required_stages = ["INITIAL", "COMPLETION"]
            for req_stage in required_stages:
                if req_stage not in self.stages:
                    errors.append(f"Template must define {req_stage} stage")

            # Validate stage structure
            for stage_name, stage in self.stages.items():
                if not stage.name:
                    errors.append(f"Stage {stage_name} must have a name")
                if not stage.description:
                    errors.append(f"Stage {stage_name} must have a description")

        # Validate extensions
        for i, middleware in enumerate(self.extensions.middleware):
            if not middleware.name:
                errors.append(f"Middleware {i} must have a name")
            if not middleware.class_name:
                errors.append(f"Middleware {i} must have a class_name")
            if not middleware.module:
                errors.append(f"Middleware {i} must have a module")

        # Validate workflows
        for workflow in self.workflows:
            if not workflow.name:
                errors.append("Workflow must have a name")
            if not workflow.description:
                errors.append("Workflow must have a description")

        # Validate teams
        for team in self.teams:
            if not team.name:
                errors.append("Team must have a name")
            if not team.formation:
                errors.append("Team must have a formation type")
            if team.formation not in [
                "pipeline",
                "parallel",
                "sequential",
                "hierarchical",
                "consensus",
            ]:
                errors.append(f"Team {team.name} has invalid formation: {team.formation}")

        # Validate capabilities
        for capability in self.capabilities:
            if not capability.name:
                errors.append("Capability must have a name")
            if not capability.type:
                errors.append("Capability must have a type")
            if capability.type not in ["tool", "workflow", "middleware", "validator", "observer"]:
                errors.append(f"Capability {capability.name} has invalid type: {capability.type}")

        return errors

    def is_valid(self) -> bool:
        """Check if template is valid.

        Returns:
            True if template passes validation
        """
        return len(self.validate()) == 0

    def merge_with_parent(self, parent: "VerticalTemplate") -> "VerticalTemplate":
        """Merge this template with parent template (inheritance).

        Child template values override or extend parent values:
        - Metadata: Child values override parent
        - Tools: Child tools added to parent tools
        - System prompt: Child prompt overrides parent
        - Stages: Child stages override parent stages with same name
        - Extensions: Child extensions extend parent extensions
        - Workflows/Teams/Capabilities: Child items added to parent

        Args:
            parent: Parent template to merge with

        Returns:
            New merged VerticalTemplate

        Example:
            base_template = VerticalTemplate(...)
            custom_template = VerticalTemplate(...)
            merged = custom_template.merge_with_parent(base_template)
        """
        from dataclasses import replace

        # Merge metadata (child overrides parent)
        merged_metadata_dict = parent.metadata.to_dict()
        child_metadata_dict = self.metadata.to_dict()

        # Override parent metadata with child metadata
        for key, value in child_metadata_dict.items():
            if value or (isinstance(value, list) and len(value) > 0):
                merged_metadata_dict[key] = value

        merged_metadata = VerticalMetadata.from_dict(merged_metadata_dict)

        # Merge tools (child adds to parent, deduplicated)
        merged_tools = list(set(parent.tools + self.tools))

        # Use child's system prompt if provided, otherwise parent's
        merged_system_prompt = self.system_prompt if self.system_prompt.strip() else parent.system_prompt

        # Merge stages (child overrides parent stages with same name)
        merged_stages = {**parent.stages}
        for name, stage in self.stages.items():
            merged_stages[name] = stage

        # Merge extensions
        merged_extensions_dict = parent.extensions.to_dict()
        child_extensions_dict = self.extensions.to_dict()

        # Extend middleware
        if "middleware" in child_extensions_dict and child_extensions_dict["middleware"]:
            merged_extensions_dict.setdefault("middleware", []).extend(child_extensions_dict["middleware"])

        # Extend safety patterns
        if "safety_patterns" in child_extensions_dict and child_extensions_dict["safety_patterns"]:
            merged_extensions_dict.setdefault("safety_patterns", []).extend(
                child_extensions_dict["safety_patterns"]
            )

        # Extend prompt hints
        if "prompt_hints" in child_extensions_dict and child_extensions_dict["prompt_hints"]:
            merged_extensions_dict.setdefault("prompt_hints", []).extend(child_extensions_dict["prompt_hints"])

        # Update handlers, personas, composed_chains (child overrides parent)
        for key in ["handlers", "personas", "composed_chains"]:
            if key in child_extensions_dict and child_extensions_dict[key]:
                merged_extensions_dict[key] = {**merged_extensions_dict.get(key, {}), **child_extensions_dict[key]}

        merged_extensions = ExtensionSpecs.from_dict(merged_extensions_dict)

        # Merge workflows, teams, capabilities (child adds to parent)
        merged_workflows = parent.workflows + self.workflows
        merged_teams = parent.teams + self.teams
        merged_capabilities = parent.capabilities + self.capabilities

        # Merge custom config (child overrides parent)
        merged_custom_config = {**parent.custom_config, **self.custom_config}

        # Merge file templates (child overrides parent)
        merged_file_templates = {**parent.file_templates, **self.file_templates}

        return VerticalTemplate(
            metadata=merged_metadata,
            tools=merged_tools,
            system_prompt=merged_system_prompt,
            stages=merged_stages,
            extensions=merged_extensions,
            workflows=merged_workflows,
            teams=merged_teams,
            capabilities=merged_capabilities,
            custom_config=merged_custom_config,
            file_templates=merged_file_templates,
        )


__all__ = [
    # Metadata
    "VerticalMetadata",
    # Extensions
    "MiddlewareSpec",
    "SafetyPatternSpec",
    "PromptHintSpec",
    "ExtensionSpecs",
    # Workflows and Teams
    "WorkflowSpec",
    "TeamRoleSpec",
    "TeamSpec",
    # Capabilities
    "CapabilitySpec",
    # Main Template
    "VerticalTemplate",
]
