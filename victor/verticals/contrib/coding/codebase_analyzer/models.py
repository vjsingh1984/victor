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

"""Data models for codebase analysis.

Contains the core data structures used by the scanner, metrics,
and generator modules.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ClassInfo:
    """Information about a discovered class."""

    name: str
    file_path: str
    line_number: int
    base_classes: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    is_abstract: bool = False
    category: Optional[str] = None  # e.g., "provider", "tool", "manager"


@dataclass
class ModuleInfo:
    """Information about a Python module."""

    name: str
    path: str
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    description: Optional[str] = None


@dataclass
class CodebaseAnalysis:
    """Complete analysis of a codebase."""

    project_name: str
    root_path: Path
    main_package: Optional[str] = None
    deprecated_paths: List[str] = field(default_factory=list)
    packages: Dict[str, List[ModuleInfo]] = field(default_factory=dict)
    key_components: List[ClassInfo] = field(default_factory=list)
    entry_points: Dict[str, str] = field(default_factory=dict)
    cli_commands: List[str] = field(default_factory=list)
    architecture_patterns: List[str] = field(default_factory=list)
    config_files: List[Tuple[str, str]] = field(default_factory=list)  # (path, description)
    # Enhanced fields
    dependencies: Dict[str, List[str]] = field(default_factory=dict)  # {category: [dep1, dep2]}
    test_coverage: Optional[float] = None  # Coverage percentage if available
    loc_stats: Dict[str, int] = field(
        default_factory=dict
    )  # {total_lines, total_files, largest_file}
    top_imports: List[Tuple[str, int]] = field(default_factory=list)  # [(module, import_count)]
    method_count: int = 0
    protocol_count: int = 0  # Python Protocol/ABC count
