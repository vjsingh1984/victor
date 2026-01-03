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

"""Security Safety Extension - Patterns for safe security analysis operations.

This module demonstrates how to implement SafetyExtensionProtocol for an external
vertical. Safety extensions protect against dangerous operations by:

1. Defining bash command patterns that should trigger warnings or blocks
2. Defining file operation patterns that need caution
3. Providing tool-specific argument restrictions

For security analysis, we need to be especially careful about:
- Commands that could exploit vulnerabilities instead of just detecting them
- Network operations that could attack external systems
- File modifications that could introduce vulnerabilities
- Credential exfiltration or abuse

Pattern Categories:
- CRITICAL: Block entirely - could cause severe damage
- HIGH: Require explicit confirmation
- MEDIUM: Warning with context
- LOW: Informational notice
"""

from typing import Dict, List

from victor.core.verticals.protocols import SafetyExtensionProtocol, SafetyPattern


# Risk level constants for clarity
CRITICAL = "CRITICAL"
HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"


class SecuritySafetyExtension(SafetyExtensionProtocol):
    """Safety extension for security analysis operations.

    Implements SafetyExtensionProtocol to provide:
    - Bash patterns for dangerous security-related commands
    - File patterns for sensitive file operations
    - Tool restrictions for security tools

    This extension is particularly important for security verticals because:
    1. Security tools can be misused for attacks
    2. Vulnerability scanning can trigger IDS/IPS alerts
    3. Some operations could accidentally exploit vulnerabilities
    4. Credential detection should not lead to credential abuse

    Example:
        extension = SecuritySafetyExtension()

        # Check if a command is dangerous
        patterns = extension.get_bash_patterns()
        for pattern in patterns:
            if re.search(pattern.pattern, user_command):
                print(f"Warning: {pattern.description}")
    """

    def get_bash_patterns(self) -> List[SafetyPattern]:
        r"""Return security-specific bash command patterns.

        These patterns detect dangerous commands that could:
        - Exploit vulnerabilities instead of detecting them
        - Attack external systems
        - Exfiltrate discovered credentials
        - Modify security-sensitive configurations
        - Execute malicious payloads

        Returns:
            List of SafetyPattern for dangerous bash commands.
        """
        return [
            # =========================================================
            # CRITICAL: Exploitation and Attack Commands
            # =========================================================
            SafetyPattern(
                pattern=r"(?i)metasploit|msfconsole|msfvenom",
                description="Metasploit exploitation framework - potential attack tool",
                risk_level=CRITICAL,
                category="security",
            ),
            SafetyPattern(
                pattern=r"(?i)sqlmap.*--dump|sqlmap.*--os-shell",
                description="SQLMap exploitation mode - data exfiltration risk",
                risk_level=CRITICAL,
                category="security",
            ),
            SafetyPattern(
                pattern=r"(?i)hydra|medusa|john\s+--",
                description="Password cracking tool - potential unauthorized access",
                risk_level=CRITICAL,
                category="security",
            ),
            SafetyPattern(
                pattern=r"(?i)nc\s+-e|ncat\s+-e|netcat.*-e",
                description="Netcat with execute flag - reverse shell potential",
                risk_level=CRITICAL,
                category="security",
            ),
            SafetyPattern(
                pattern=r"(?i)curl.*\|\s*(sh|bash)|wget.*\|\s*(sh|bash)",
                description="Remote script execution - potential malware download",
                risk_level=CRITICAL,
                category="security",
            ),
            # =========================================================
            # HIGH: Dangerous Scanning and Enumeration
            # =========================================================
            SafetyPattern(
                pattern=r"(?i)nmap.*(-sS|-sX|-sN|--script)",
                description="Aggressive nmap scan - could trigger security alerts",
                risk_level=HIGH,
                category="security",
            ),
            SafetyPattern(
                pattern=r"(?i)nikto|dirb|gobuster|ffuf",
                description="Web vulnerability scanner - aggressive external scanning",
                risk_level=HIGH,
                category="security",
            ),
            SafetyPattern(
                pattern=r"(?i)burp|zaproxy|zap-cli",
                description="Web proxy tool - potential for active attacks",
                risk_level=HIGH,
                category="security",
            ),
            SafetyPattern(
                pattern=r"(?i)masscan|unicornscan",
                description="High-speed port scanner - network disruption risk",
                risk_level=HIGH,
                category="security",
            ),
            SafetyPattern(
                pattern=r"(?i)wpscan|joomscan|drupalgeddon",
                description="CMS vulnerability scanner - aggressive scanning",
                risk_level=HIGH,
                category="security",
            ),
            # =========================================================
            # MEDIUM: Credential and Secret Handling
            # =========================================================
            SafetyPattern(
                pattern=r"(?i)(curl|wget|http).*Authorization:|Bearer\s+[A-Za-z0-9\-_]+",
                description="HTTP request with credentials - potential credential leak",
                risk_level=MEDIUM,
                category="security",
            ),
            SafetyPattern(
                pattern=r"(?i)echo.*password|echo.*api[_-]?key|echo.*secret",
                description="Echoing credentials - potential exposure",
                risk_level=MEDIUM,
                category="security",
            ),
            SafetyPattern(
                pattern=r"(?i)gitleaks.*--no-git|trufflehog.*--json.*>",
                description="Secret detection with output - handle findings carefully",
                risk_level=MEDIUM,
                category="security",
            ),
            SafetyPattern(
                pattern=r"(?i)aws\s+configure|gcloud\s+auth|az\s+login",
                description="Cloud credential configuration - sensitive operation",
                risk_level=MEDIUM,
                category="security",
            ),
            # =========================================================
            # MEDIUM: Network and Remote Operations
            # =========================================================
            SafetyPattern(
                pattern=r"(?i)ssh\s+-o\s+StrictHostKeyChecking=no",
                description="SSH with disabled host key checking - MITM vulnerability",
                risk_level=MEDIUM,
                category="security",
            ),
            SafetyPattern(
                pattern=r"(?i)curl\s+(-k|--insecure)",
                description="HTTP request ignoring SSL - MITM vulnerability",
                risk_level=MEDIUM,
                category="security",
            ),
            SafetyPattern(
                pattern=r"(?i)ping\s+-f|ping\s+.*-s\s+\d{5,}",
                description="Aggressive ping - potential denial of service",
                risk_level=MEDIUM,
                category="security",
            ),
            # =========================================================
            # LOW: Safe but Notable Security Operations
            # =========================================================
            SafetyPattern(
                pattern=r"(?i)bandit|safety\s+check|snyk\s+test|trivy\s+fs",
                description="Security scanner execution - review findings carefully",
                risk_level=LOW,
                category="security",
            ),
            SafetyPattern(
                pattern=r"(?i)npm\s+audit|yarn\s+audit|pip-audit|bundler-audit",
                description="Dependency vulnerability check - review CVE findings",
                risk_level=LOW,
                category="security",
            ),
            SafetyPattern(
                pattern=r"(?i)semgrep|codeql|sonarqube",
                description="Static analysis tool - review security findings",
                risk_level=LOW,
                category="security",
            ),
        ]

    def get_file_patterns(self) -> List[SafetyPattern]:
        """Return patterns for sensitive file operations.

        Security analysis may encounter sensitive files that should
        be handled with extra care.

        Returns:
            List of SafetyPattern for file operations.
        """
        return [
            # Credential files
            SafetyPattern(
                pattern=r"(?i)\.env|\.env\.(local|prod|production)|\.envrc",
                description="Environment file with potential secrets",
                risk_level=HIGH,
                category="security",
            ),
            SafetyPattern(
                pattern=r"(?i)(id_rsa|id_ed25519|id_ecdsa|\.pem|\.key)$",
                description="Private key file - handle with extreme care",
                risk_level=CRITICAL,
                category="security",
            ),
            SafetyPattern(
                pattern=r"(?i)(credentials|secrets|passwords)\.(json|yaml|yml|xml)$",
                description="Credential storage file - sensitive content",
                risk_level=HIGH,
                category="security",
            ),
            SafetyPattern(
                pattern=r"(?i)\.htpasswd|\.htaccess|shadow|passwd$",
                description="Authentication file - critical security file",
                risk_level=HIGH,
                category="security",
            ),
            # Configuration files
            SafetyPattern(
                pattern=r"(?i)(nginx|apache|httpd)\.conf$",
                description="Web server configuration - security implications",
                risk_level=MEDIUM,
                category="security",
            ),
            SafetyPattern(
                pattern=r"(?i)(docker-compose|kubernetes|k8s).*\.(yaml|yml)$",
                description="Container orchestration config - check for secrets",
                risk_level=MEDIUM,
                category="security",
            ),
            SafetyPattern(
                pattern=r"(?i)terraform.*\.(tf|tfvars)$",
                description="Infrastructure as code - may contain sensitive values",
                risk_level=MEDIUM,
                category="security",
            ),
        ]

    def get_tool_restrictions(self) -> Dict[str, List[str]]:
        """Get tool-specific argument restrictions.

        Returns restrictions on specific tool arguments that could
        be dangerous in a security analysis context.

        Returns:
            Dict mapping tool names to restricted argument patterns.
        """
        return {
            "bash": [
                # Prevent execution of discovered malicious code
                r"eval\s+",
                r"exec\s+",
                # Prevent network attacks
                r"nc\s+-l",
                r"python.*-m\s+http\.server",
                # Prevent credential abuse
                r"curl.*-d.*password",
                r"wget.*--post-data.*password",
            ],
            "write": [
                # Prevent writing to sensitive locations
                r"/etc/",
                r"/root/",
                r"\.ssh/",
                r"\.aws/",
                # Prevent creating executable scripts
                r"\.sh$",
                r"\.bash$",
            ],
        }

    def get_category(self) -> str:
        """Get the category name for these patterns.

        Returns:
            Category identifier for the security vertical.
        """
        return "security"

    def get_blocked_operations(self) -> List[str]:
        """Return operations that should be completely blocked.

        These operations are never appropriate in a security analysis context.

        Returns:
            List of operation identifiers to block.
        """
        return [
            "execute_exploit",  # Never run actual exploits
            "exfiltrate_data",  # Never extract discovered secrets
            "modify_production",  # Never change production systems
            "delete_logs",  # Never tamper with audit trails
        ]

    def get_safety_reminders(self) -> List[str]:
        """Return safety reminders for security operations.

        Returns:
            List of safety reminders to include in prompts.
        """
        return [
            "Only analyze systems you have authorization to test",
            "Report vulnerabilities responsibly - do not exploit",
            "Handle discovered credentials with extreme care - never exfiltrate",
            "Be aware that aggressive scanning may trigger security alerts",
            "Document all security findings with proper context",
            "Never execute code from untrusted sources",
        ]
