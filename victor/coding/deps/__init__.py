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

"""Smart dependency management module.

This module has been promoted to victor-core.
Please use `from victor.deps import ...` instead.

This module re-exports from victor.deps for backward compatibility.
"""

# Re-export from victor-core for backward compatibility
from victor.deps import (
    # Protocol types
    Dependency,
    DependencyAnalysis,
    DependencyConflict,
    DependencyGraph,
    DependencyType,
    DependencyUpdate,
    DependencyVulnerability,
    DepsConfig,
    LockFile,
    PackageManager,
    Version,
    VersionConstraint,
    # Parsers
    BaseDependencyParser,
    CargoTomlParser,
    GoModParser,
    PackageJsonParser,
    PyprojectParser,
    RequirementsTxtParser,
    detect_package_manager,
    get_parser,
    # Manager
    DepsManager,
    get_deps_manager,
    reset_deps_manager,
)

__all__ = [
    # Protocol types
    "Dependency",
    "DependencyAnalysis",
    "DependencyConflict",
    "DependencyGraph",
    "DependencyType",
    "DependencyUpdate",
    "DependencyVulnerability",
    "DepsConfig",
    "LockFile",
    "PackageManager",
    "Version",
    "VersionConstraint",
    # Parsers
    "BaseDependencyParser",
    "CargoTomlParser",
    "GoModParser",
    "PackageJsonParser",
    "PyprojectParser",
    "RequirementsTxtParser",
    "detect_package_manager",
    "get_parser",
    # Manager
    "DepsManager",
    "get_deps_manager",
    "reset_deps_manager",
]
