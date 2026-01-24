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

"""Feature flag definitions for Victor 0.5.0 capabilities.

This module defines all feature flags for gradual rollout of new capabilities.
Each flag includes metadata for documentation, validation, and rollout strategy.

Flag Metadata:
- description: Human-readable description
- default: Default value (False for gradual rollout, True for immediate)
- rollout_strategy: immediate | gradual | staged | percentage
- dependencies: List of flags that must be enabled first
- category: Category for grouping (planning, memory, skills, multimodal, personas, performance)
- since: Version when feature was introduced
- stable: Whether feature is considered stable (safe for production)
"""

from __future__ import annotations

from typing import Dict, List, Any


# Feature flag definitions
FEATURE_FLAGS: Dict[str, Dict[str, Any]] = {
    # ==========================================================================
    # Planning Capabilities
    # ==========================================================================
    "hierarchical_planning_enabled": {
        "description": "Enable hierarchical task decomposition and planning engine",
        "default": False,
        "rollout_strategy": "gradual",
        "dependencies": [],
        "category": "planning",
        "since": "0.5.0",
        "stable": False,
        "tags": ["planning", "decomposition", "agent"],
        "metadata": {
            "max_depth": 5,
            "min_subtasks": 2,
            "max_subtasks": 10,
        },
    },
    # ==========================================================================
    # Memory Systems
    # ==========================================================================
    "enhanced_memory_enabled": {
        "description": "Enable episodic and semantic memory with consolidation",
        "default": False,
        "rollout_strategy": "gradual",
        "dependencies": [],
        "category": "memory",
        "since": "0.5.0",
        "stable": False,
        "tags": ["memory", "episodic", "semantic", "consolidation"],
        "metadata": {
            "episodic_max_episodes": 1000,
            "episodic_recall_threshold": 0.3,
            "semantic_max_facts": 5000,
            "semantic_query_threshold": 0.25,
        },
    },
    # ==========================================================================
    # Dynamic Skills
    # ==========================================================================
    "dynamic_skills_enabled": {
        "description": "Enable runtime tool discovery and skill composition",
        "default": False,
        "rollout_strategy": "gradual",
        "dependencies": [],
        "category": "skills",
        "since": "0.5.0",
        "stable": False,
        "tags": ["skills", "discovery", "composition", "tools"],
        "metadata": {
            "max_tools": 20,
            "min_compatibility": 0.5,
            "auto_composition": True,
        },
    },
    "self_improvement_enabled": {
        "description": "Enable proficiency tracking and self-improvement loops",
        "default": False,
        "rollout_strategy": "gradual",
        "dependencies": ["dynamic_skills_enabled"],
        "category": "skills",
        "since": "0.5.0",
        "stable": False,
        "tags": ["self-improvement", "proficiency", "learning", "rl"],
        "metadata": {
            "window_size": 100,
            "decay_rate": 0.95,
            "min_samples": 5,
        },
    },
    # ==========================================================================
    # Multimodal Capabilities
    # ==========================================================================
    "multimodal_vision_enabled": {
        "description": "Enable vision/image processing capabilities",
        "default": False,
        "rollout_strategy": "gradual",
        "dependencies": [],
        "category": "multimodal",
        "since": "0.5.0",
        "stable": False,
        "tags": ["multimodal", "vision", "image", "analysis"],
        "metadata": {
            "supported_formats": ["png", "jpg", "jpeg", "gif", "webp"],
            "max_image_size_mb": 10,
        },
    },
    "multimodal_audio_enabled": {
        "description": "Enable audio/speech processing capabilities",
        "default": False,
        "rollout_strategy": "gradual",
        "dependencies": [],
        "category": "multimodal",
        "since": "0.5.0",
        "stable": False,
        "tags": ["multimodal", "audio", "speech", "transcription"],
        "metadata": {
            "supported_formats": ["wav", "mp3", "ogg", "flac"],
            "max_audio_size_mb": 25,
        },
    },
    # ==========================================================================
    # Dynamic Personas
    # ==========================================================================
    "dynamic_personas_enabled": {
        "description": "Enable dynamic persona management and adaptation",
        "default": False,
        "rollout_strategy": "gradual",
        "dependencies": [],
        "category": "personas",
        "since": "0.5.0",
        "stable": False,
        "tags": ["personas", "adaptation", "context", "behavior"],
        "metadata": {
            "max_personas": 10,
            "adaptation_threshold": 0.7,
            "switching_cooldown_seconds": 300,
        },
    },
    # ==========================================================================
    # Performance Optimizations
    # ==========================================================================
    "lazy_loading_enabled": {
        "description": "Enable lazy component loading for faster initialization",
        "default": True,
        "rollout_strategy": "immediate",
        "dependencies": [],
        "category": "performance",
        "since": "0.5.0",
        "stable": True,
        "tags": ["performance", "lazy", "initialization", "startup"],
        "metadata": {
            "load_on_access": True,
            "preload_critical": True,
        },
    },
    "parallel_execution_enabled": {
        "description": "Enable parallel tool and workflow execution",
        "default": True,
        "rollout_strategy": "immediate",
        "dependencies": [],
        "category": "performance",
        "since": "0.5.0",
        "stable": True,
        "tags": ["performance", "parallel", "concurrency", "speedup"],
        "metadata": {
            "max_workers": 10,
            "timeout_seconds": 300,
            "chunk_size": 5,
        },
    },
}


def get_flag_metadata(flag_name: str) -> Dict[str, Any]:
    """Get metadata for a specific feature flag.

    Args:
        flag_name: Name of the flag

    Returns:
        Metadata dictionary or empty dict if flag not found

    Example:
        metadata = get_flag_metadata("hierarchical_planning_enabled")
        print(metadata["description"])
    """
    return FEATURE_FLAGS.get(flag_name, {}).copy()


def validate_flag_dependencies(flag_name: str, enabled_flags: Dict[str, bool]) -> bool:
    """Validate that all dependencies for a flag are enabled.

    Args:
        flag_name: Name of the flag to validate
        enabled_flags: Dictionary of all currently enabled flags

    Returns:
        True if all dependencies are satisfied

    Raises:
        ValueError: If flag has circular dependencies

    Example:
        enabled = {"dynamic_skills_enabled": True}
        is_valid = validate_flag_dependencies("self_improvement_enabled", enabled)
    """
    flag_def = FEATURE_FLAGS.get(flag_name)
    if not flag_def:
        return False

    dependencies = flag_def.get("dependencies", [])

    # Check for circular dependencies
    if flag_name in dependencies:
        raise ValueError(f"Flag '{flag_name}' has circular dependency on itself")

    # Validate all dependencies are enabled
    for dep in dependencies:
        if not enabled_flags.get(dep, False):
            return False

    return True


def get_flags_by_category(category: str) -> Dict[str, Dict[str, Any]]:
    """Get all flags in a specific category.

    Args:
        category: Category name (planning, memory, skills, multimodal, personas, performance)

    Returns:
        Dictionary of flag names to metadata

    Example:
        planning_flags = get_flags_by_category("planning")
        for flag_name, metadata in planning_flags.items():
            print(f"{flag_name}: {metadata['description']}")
    """
    return {
        flag_name: flag_def
        for flag_name, flag_def in FEATURE_FLAGS.items()
        if flag_def.get("category") == category
    }


def get_stable_flags() -> Dict[str, Dict[str, Any]]:
    """Get all flags marked as stable (safe for production).

    Returns:
        Dictionary of stable flag names to metadata

    Example:
        stable = get_stable_flags()
        print(f"Stable flags: {list(stable.keys())}")
    """
    return {
        flag_name: flag_def
        for flag_name, flag_def in FEATURE_FLAGS.items()
        if flag_def.get("stable", False)
    }


def get_experimental_flags() -> Dict[str, Dict[str, Any]]:
    """Get all flags marked as experimental (not stable).

    Returns:
        Dictionary of experimental flag names to metadata

    Example:
        experimental = get_experimental_flags()
        print(f"Experimental flags: {list(experimental.keys())}")
    """
    return {
        flag_name: flag_def
        for flag_name, flag_def in FEATURE_FLAGS.items()
        if not flag_def.get("stable", False)
    }


def get_all_flag_names() -> List[str]:
    """Get list of all feature flag names.

    Returns:
        List of flag names

    Example:
        flags = get_all_flag_names()
        for flag in flags:
            print(flag)
    """
    return list(FEATURE_FLAGS.keys())


def get_flag_categories() -> List[str]:
    """Get list of all flag categories.

    Returns:
        List of unique category names

    Example:
        categories = get_flag_categories()
        print(categories)  # ['planning', 'memory', 'skills', ...]
    """
    categories = {flag_def.get("category") for flag_def in FEATURE_FLAGS.values()}
    return sorted(categories)
