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

"""Framework safety pattern system (Phase 6.1).

This module provides a centralized safety pattern registry:
- SafetyPattern: Declarative pattern definitions
- SafetyPatternRegistry: Thread-safe singleton for pattern management
- Built-in scanners: SecretScanner, CommandScanner, FilePathScanner
- YAML-based pattern loading

Example:
    from victor.framework.safety import (
        SafetyPatternRegistry,
        SafetyPattern,
        Severity,
        SecretScanner,
    )

    # Get registry singleton
    registry = SafetyPatternRegistry.get_instance()

    # Register pattern
    pattern = SafetyPattern(
        name="api_key",
        pattern=r"AKIA[0-9A-Z]{16}",
        severity=Severity.HIGH,
        message="AWS API key detected",
    )
    registry.register_pattern(pattern)

    # Or use built-in scanner
    scanner = SecretScanner()
    registry.register_scanner("secrets", scanner)

    # Scan content
    violations = registry.scan(content, domain="coding")
"""

from victor.framework.safety.types import (
    SafetyPattern,
    SafetyViolation,
    Severity,
    Action,
)
from victor.framework.safety.registry import (
    SafetyPatternRegistry,
    validate_pattern_yaml,
    ScannerProtocol,
)
from victor.framework.safety.scanners import (
    BaseScanner,
    SecretScanner,
    CommandScanner,
    FilePathScanner,
)
from victor.framework.safety.yaml_loader import (
    get_vertical_pattern_path,
    register_vertical_patterns,
    register_all_vertical_patterns,
    register_builtin_scanners,
)

__all__ = [
    # Types
    "SafetyPattern",
    "SafetyViolation",
    "Severity",
    "Action",
    # Registry
    "SafetyPatternRegistry",
    "validate_pattern_yaml",
    "ScannerProtocol",
    # Scanners
    "BaseScanner",
    "SecretScanner",
    "CommandScanner",
    "FilePathScanner",
    # YAML Loader (Phase 6.2)
    "get_vertical_pattern_path",
    "register_vertical_patterns",
    "register_all_vertical_patterns",
    "register_builtin_scanners",
]
