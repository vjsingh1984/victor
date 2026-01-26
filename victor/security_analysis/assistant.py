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

"""SecurityAnalysisAssistant - Victor's vertical for security analysis.

This module defines the SecurityAnalysisAssistant vertical with full
integration of security-specific extensions, middleware, and configurations.

The SecurityAnalysisAssistant provides:
- Vulnerability scanning and CVE database integration
- Dependency security analysis
- Penetration testing tools
- Secret and PII detection
- Security pattern scanning
"""

from __future__ import annotations

from typing import ClassVar, List, Optional

from victor.core.verticals.base import VerticalBase
from victor.core.vertical_types import StageDefinition
from victor.core.verticals.defaults.tool_defaults import (
    COMMON_REQUIRED_TOOLS,
    merge_required_tools,
)
from victor.core.verticals.protocols import (
    MiddlewareProtocol,
    SafetyExtensionProtocol,
    PromptContributorProtocol,
    ServiceProviderProtocol,
)

# Phase 2.1: Protocol auto-registration decorator
from victor.core.verticals.protocol_decorators import register_protocols


@register_protocols
class SecurityAnalysisAssistant(VerticalBase):
    """Security analysis assistant vertical.

    This vertical is optimized for:
    - Vulnerability scanning and assessment
    - Dependency security analysis
    - Secret and credential detection
    - PII detection and compliance
    - Penetration testing
    - Security code review

    The SecurityAnalysisAssistant provides full integration with the
    framework through extension protocols, enabling:
    - Security scanning middleware
    - Secret detection safety checks
    - Security-specific prompt hints
    - Mode configurations for security scenarios

    ISP Compliance:
        This vertical explicitly declares which protocols it implements
        through protocol registration.

    Example:
        from victor.security_analysis import SecurityAnalysisAssistant

        # Get vertical configuration
        config = SecurityAnalysisAssistant.get_config()

        # Get extensions for framework integration
        extensions = SecurityAnalysisAssistant.get_extensions()
    """

    # Override class variables from base
    name: ClassVar[str] = "security_analysis"
    description: ClassVar[str] = (
        "Security analysis assistant for vulnerability scanning, "
        "penetration testing, and security assessment"
    )
    version: ClassVar[str] = "1.0.0"

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get tools optimized for security analysis.

        Returns:
            List of tool names for security analysis.
        """
        from victor.tools.tool_names import ToolNames

        # Security-specific tools
        security_tools = [
            # Core filesystem (for reading config files, etc.)
            ToolNames.READ,
            ToolNames.WRITE,
            ToolNames.LS,
            # Search (for finding vulnerabilities)
            ToolNames.GREP,
            ToolNames.CODE_SEARCH,
            # Shell (for running security tools)
            ToolNames.SHELL,
            # Git (for history analysis)
            ToolNames.GIT,
            # Web (for CVE lookups)
            ToolNames.WEB_SEARCH,
            ToolNames.WEB_FETCH,
        ]

        # Merge common required tools with security-specific tools
        return merge_required_tools(COMMON_REQUIRED_TOOLS, security_tools)

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the security analysis system prompt.

        Returns:
            System prompt for security analysis context.
        """
        return """You are a security analysis assistant specializing in:
- Vulnerability scanning and assessment
- Dependency security analysis (CVE detection)
- Secret and credential detection
- PII detection and compliance checking
- Security code review
- Penetration testing guidance

When analyzing security:
1. Always check for known vulnerabilities (CVEs) in dependencies
2. Scan for hardcoded secrets and credentials
3. Identify potential PII exposure
4. Look for common security anti-patterns
5. Provide remediation guidance with specific fixes
6. Consider compliance requirements (SOC2, GDPR, HIPAA, PCI-DSS)

Security Best Practices:
- Use principle of least privilege
- Validate all inputs
- Encrypt sensitive data at rest and in transit
- Keep dependencies updated
- Follow secure coding guidelines
"""

    @classmethod
    def get_safety_extension(cls) -> Optional["SafetyExtensionProtocol"]:
        """Get security-specific safety extension.

        Returns:
            Security analysis safety extension.
        """
        from victor.security_analysis.safety import SecurityAnalysisSafetyExtension

        return SecurityAnalysisSafetyExtension()

    @classmethod
    def get_middleware(cls) -> Optional["MiddlewareProtocol"]:
        """Get security analysis middleware.

        Returns:
            Security analysis middleware.
        """
        from victor.security_analysis.middleware import SecurityAnalysisMiddleware

        return SecurityAnalysisMiddleware()

    @classmethod
    def get_prompt_contributor(cls) -> Optional["PromptContributorProtocol"]:
        """Get security-specific prompt contributor.

        Returns:
            Security analysis prompt contributor.
        """
        from victor.security_analysis.prompts import SecurityAnalysisPromptContributor

        return SecurityAnalysisPromptContributor()

    @classmethod
    def get_service_provider(cls) -> Optional["ServiceProviderProtocol"]:
        """Get security analysis service provider.

        Returns:
            Security analysis service provider for DI registration.
        """
        from victor.security_analysis.service_provider import SecurityAnalysisServiceProvider

        return SecurityAnalysisServiceProvider()

    @classmethod
    def get_stages(cls) -> List["StageDefinition"]:
        """Get workflow stages for security analysis.

        Returns:
            List of stage definitions.
        """
        return [
            StageDefinition(
                name="DISCOVERY",
                description="Discover potential security issues",
                tool_hints=["grep", "code_search", "ls"],
            ),
            StageDefinition(
                name="SCANNING",
                description="Scan for vulnerabilities and security issues",
                tool_hints=["shell", "grep"],
            ),
            StageDefinition(
                name="ANALYSIS",
                description="Analyze findings and assess risk",
                tool_hints=["read", "web_search"],
            ),
            StageDefinition(
                name="REPORTING",
                description="Generate security reports and recommendations",
                tool_hints=["write"],
            ),
        ]


# Export for entry point registration
__all__ = ["SecurityAnalysisAssistant"]
