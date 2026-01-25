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

"""Built-in safety scanners (Phase 6.1).

This module provides pre-configured scanners for common safety concerns:
- SecretScanner: API keys, passwords, tokens
- CommandScanner: Dangerous shell commands
- FilePathScanner: Sensitive file paths

These scanners are compatible with the existing ISafetyScanner protocol
and can be registered with SafetyPatternRegistry.

Example:
    from victor.framework.safety.scanners import SecretScanner

    scanner = SecretScanner()
    violations = scanner.scan(code_content)
    for v in violations:
        print(f"{v.severity}: {v.message}")
"""

from __future__ import annotations

import logging
from typing import List, Protocol

from victor.framework.safety.types import (
    SafetyPattern,
    SafetyViolation,
    Severity,
    Action,
)

logger = logging.getLogger(__name__)


class ScannerProtocol(Protocol):
    """Protocol for safety scanners.

    Compatible with existing ISafetyScanner protocol.
    """

    def scan(self, content: str) -> List[SafetyViolation]:
        """Scan content for safety violations.

        Args:
            content: Text content to scan

        Returns:
            List of SafetyViolation instances
        """
        ...


class BaseScanner:
    """Base class for safety scanners.

    Provides common functionality for pattern-based scanning.
    """

    def __init__(self, patterns: List[SafetyPattern] | None = None):
        """Initialize scanner with patterns.

        Args:
            patterns: List of SafetyPattern instances to use
        """
        self._patterns = patterns or self._get_default_patterns()

    def _get_default_patterns(self) -> List[SafetyPattern]:
        """Get default patterns for this scanner.

        Override in subclasses to provide built-in patterns.

        Returns:
            List of SafetyPattern instances
        """
        return []

    def scan(self, content: str) -> List[SafetyViolation]:
        """Scan content for safety violations.

        Args:
            content: Text content to scan

        Returns:
            List of SafetyViolation instances
        """
        violations: List[SafetyViolation] = []

        for pattern in self._patterns:
            if pattern.matches(content):
                matches = pattern.find_matches(content)
                for matched_text in matches:
                    violations.append(
                        SafetyViolation(
                            pattern_name=pattern.name,
                            severity=pattern.severity,
                            message=pattern.message,
                            matched_text=matched_text,
                            action=pattern.action,
                        )
                    )

        return violations


class SecretScanner(BaseScanner):
    """Scanner for detecting secrets and credentials.

    Detects common patterns for:
    - AWS credentials (access keys, secret keys)
    - GitHub tokens
    - Generic API keys
    - Private keys
    - Passwords in code
    """

    def _get_default_patterns(self) -> List[SafetyPattern]:
        """Get default secret detection patterns."""
        return [
            SafetyPattern(
                name="aws_access_key",
                pattern=r"AKIA[0-9A-Z]{16}",
                severity=Severity.CRITICAL,
                message="AWS Access Key ID detected",
                action=Action.BLOCK,
            ),
            SafetyPattern(
                name="aws_secret_key",
                pattern=r"(?i)aws[_\-]?secret[_\-]?(?:access[_\-]?)?key['\"\s:=]+['\"]?[A-Za-z0-9/+=]{40}",
                severity=Severity.CRITICAL,
                message="AWS Secret Access Key detected",
                action=Action.BLOCK,
            ),
            SafetyPattern(
                name="github_token",
                pattern=r"gh[pousr]_[A-Za-z0-9_]{36,251}",
                severity=Severity.CRITICAL,
                message="GitHub token detected",
                action=Action.BLOCK,
            ),
            SafetyPattern(
                name="github_classic_token",
                pattern=r"ghp_[A-Za-z0-9]{36}",
                severity=Severity.CRITICAL,
                message="GitHub personal access token detected",
                action=Action.BLOCK,
            ),
            SafetyPattern(
                name="generic_api_key",
                pattern=r"(?i)api[_\-]?key['\"\s:=]+['\"]?[A-Za-z0-9\-_]{20,}['\"]?",
                severity=Severity.HIGH,
                message="Generic API key pattern detected",
                action=Action.WARN,
            ),
            SafetyPattern(
                name="private_key",
                pattern=r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
                severity=Severity.CRITICAL,
                message="Private key detected",
                action=Action.BLOCK,
            ),
            SafetyPattern(
                name="slack_token",
                pattern=r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*",
                severity=Severity.CRITICAL,
                message="Slack token detected",
                action=Action.BLOCK,
            ),
            SafetyPattern(
                name="password_in_code",
                pattern=r"(?i)password['\"\s:=]+['\"][^'\"]{8,}['\"]",
                severity=Severity.HIGH,
                message="Password literal detected in code",
                action=Action.WARN,
            ),
            SafetyPattern(
                name="openai_api_key",
                pattern=r"sk-[A-Za-z0-9]{20,}T3BlbkFJ[A-Za-z0-9]{20,}",
                severity=Severity.CRITICAL,
                message="OpenAI API key detected",
                action=Action.BLOCK,
            ),
            SafetyPattern(
                name="anthropic_api_key",
                pattern=r"sk-ant-[A-Za-z0-9\-_]{80,}",
                severity=Severity.CRITICAL,
                message="Anthropic API key detected",
                action=Action.BLOCK,
            ),
        ]


class CommandScanner(BaseScanner):
    """Scanner for detecting dangerous shell commands.

    Detects patterns for:
    - Destructive file operations (rm -rf)
    - Dangerous permission changes
    - Force push operations
    - Sudo abuse
    """

    def _get_default_patterns(self) -> List[SafetyPattern]:
        """Get default dangerous command patterns."""
        return [
            SafetyPattern(
                name="rm_rf_root",
                pattern=r"rm\s+(-[a-z]*r[a-z]*f[a-z]*|-[a-z]*f[a-z]*r[a-z]*)\s+(/|/\*)",
                severity=Severity.CRITICAL,
                message="Destructive rm -rf on root filesystem detected",
                action=Action.BLOCK,
            ),
            SafetyPattern(
                name="rm_rf_recursive",
                pattern=r"rm\s+(-[a-z]*r[a-z]*f[a-z]*|-[a-z]*f[a-z]*r[a-z]*)\s+",
                severity=Severity.HIGH,
                message="Recursive forced deletion detected",
                action=Action.WARN,
            ),
            SafetyPattern(
                name="chmod_777",
                pattern=r"chmod\s+777\s+",
                severity=Severity.HIGH,
                message="Overly permissive chmod 777 detected",
                action=Action.WARN,
            ),
            SafetyPattern(
                name="chmod_world_writable",
                pattern=r"chmod\s+[0-7]{0,2}[2367]\s+",
                severity=Severity.MEDIUM,
                message="World-writable permission detected",
                action=Action.WARN,
            ),
            SafetyPattern(
                name="git_force_push",
                pattern=r"git\s+push\s+.*--force",
                severity=Severity.CRITICAL,
                message="Git force push detected",
                action=Action.BLOCK,
            ),
            SafetyPattern(
                name="git_force_push_short",
                pattern=r"git\s+push\s+-f\b",
                severity=Severity.CRITICAL,
                message="Git force push (-f) detected",
                action=Action.BLOCK,
            ),
            SafetyPattern(
                name="sudo_rm",
                pattern=r"sudo\s+rm\s+",
                severity=Severity.HIGH,
                message="Sudo rm command detected",
                action=Action.WARN,
            ),
            SafetyPattern(
                name="dd_if_dev",
                pattern=r"dd\s+if=/dev/(?:zero|random|urandom)\s+of=/dev/",
                severity=Severity.CRITICAL,
                message="Potentially destructive dd command detected",
                action=Action.BLOCK,
            ),
            SafetyPattern(
                name="mkfs_format",
                pattern=r"mkfs\.\w+\s+/dev/",
                severity=Severity.CRITICAL,
                message="Filesystem format command detected",
                action=Action.BLOCK,
            ),
            SafetyPattern(
                name="curl_pipe_bash",
                pattern=r"curl\s+.*\|\s*(?:sudo\s+)?(?:ba)?sh",
                severity=Severity.HIGH,
                message="Piping curl to shell detected",
                action=Action.WARN,
            ),
            SafetyPattern(
                name="wget_pipe_bash",
                pattern=r"wget\s+.*\|\s*(?:sudo\s+)?(?:ba)?sh",
                severity=Severity.HIGH,
                message="Piping wget to shell detected",
                action=Action.WARN,
            ),
        ]


class FilePathScanner(BaseScanner):
    """Scanner for detecting sensitive file path access.

    Detects patterns for:
    - System credential files (/etc/shadow, /etc/passwd)
    - SSH keys
    - AWS credentials
    - Environment files
    """

    def _get_default_patterns(self) -> List[SafetyPattern]:
        """Get default sensitive file path patterns."""
        return [
            SafetyPattern(
                name="etc_shadow",
                pattern=r"/etc/shadow",
                severity=Severity.CRITICAL,
                message="Access to /etc/shadow detected",
                action=Action.BLOCK,
            ),
            SafetyPattern(
                name="etc_passwd",
                pattern=r"/etc/passwd(?!\w)",
                severity=Severity.HIGH,
                message="Access to /etc/passwd detected",
                action=Action.WARN,
            ),
            SafetyPattern(
                name="ssh_private_key",
                pattern=r"~?/\.ssh/id_(?:rsa|dsa|ecdsa|ed25519)(?!\\.pub)",
                severity=Severity.CRITICAL,
                message="SSH private key access detected",
                action=Action.BLOCK,
            ),
            SafetyPattern(
                name="ssh_dir",
                pattern=r"~?/\.ssh/",
                severity=Severity.HIGH,
                message="SSH directory access detected",
                action=Action.WARN,
            ),
            SafetyPattern(
                name="aws_credentials",
                pattern=r"~?/\.aws/credentials",
                severity=Severity.CRITICAL,
                message="AWS credentials file access detected",
                action=Action.BLOCK,
            ),
            SafetyPattern(
                name="env_file",
                pattern=r"\.env(?:\.local|\.production|\.staging)?(?!\w)",
                severity=Severity.HIGH,
                message="Environment file access detected",
                action=Action.WARN,
            ),
            SafetyPattern(
                name="kubernetes_config",
                pattern=r"~?/\.kube/config",
                severity=Severity.HIGH,
                message="Kubernetes config access detected",
                action=Action.WARN,
            ),
            SafetyPattern(
                name="docker_config",
                pattern=r"~?/\.docker/config\.json",
                severity=Severity.HIGH,
                message="Docker config access detected",
                action=Action.WARN,
            ),
            SafetyPattern(
                name="gnupg_dir",
                pattern=r"~?/\.gnupg/",
                severity=Severity.HIGH,
                message="GPG directory access detected",
                action=Action.WARN,
            ),
        ]


__all__ = [
    "ScannerProtocol",
    "BaseScanner",
    "SecretScanner",
    "CommandScanner",
    "FilePathScanner",
]
