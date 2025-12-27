"""DevOps Safety Extension - Security patterns for infrastructure work."""

from typing import Dict, List, Tuple

from victor.verticals.protocols import SafetyExtensionProtocol, SafetyPattern


# Risk levels
HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"

# DevOps-specific danger patterns as tuples
_DEVOPS_DANGER_TUPLES: List[Tuple[str, str, str]] = [
    # High-risk patterns - destructive operations
    (r"rm\s+-rf\s+/(?!tmp)", "Destructive filesystem deletion", HIGH),
    (r"kubectl\s+delete\s+(?:namespace|ns)\s+(?!test|dev|local)", "Namespace deletion", HIGH),
    (r"terraform\s+destroy(?!\s+--target)", "Full infrastructure destruction", HIGH),
    (r"docker\s+system\s+prune\s+-a", "Docker full system prune", HIGH),
    (r"DROP\s+DATABASE|DROP\s+TABLE", "Database destruction", HIGH),
    (r"--force\s+push|push\s+--force|-f\s+push", "Force push to remote", HIGH),
    # Secrets exposure
    (r"password\s*[=:]\s*['\"][^'\"]+['\"]", "Hardcoded password", HIGH),
    (r"api[_-]?key\s*[=:]\s*['\"][^'\"]+['\"]", "Hardcoded API key", HIGH),
    (r"AWS_SECRET_ACCESS_KEY\s*=\s*['\"][^'\"]+['\"]", "AWS secret in code", HIGH),
    (r"PRIVATE[_-]?KEY\s*[=:]\s*['\"]", "Private key in code", HIGH),
    # Medium-risk patterns - need review
    (r"privileged:\s*true", "Privileged container mode", MEDIUM),
    (r"hostNetwork:\s*true", "Host network access", MEDIUM),
    (r"hostPID:\s*true", "Host PID namespace", MEDIUM),
    (r"capabilities:.*SYS_ADMIN", "SYS_ADMIN capability", MEDIUM),
    (r"allowPrivilegeEscalation:\s*true", "Privilege escalation enabled", MEDIUM),
    (r"runAsUser:\s*0", "Running as root", MEDIUM),
    (r"--net=host", "Docker host networking", MEDIUM),
    # Low-risk patterns - style/best practices
    (r"FROM\s+\S+:latest", "Using latest tag", LOW),
    (r"resources:\s*\{\s*\}", "Missing resource limits", LOW),
    (r"replicas:\s*1(?:\s|$)", "Single replica deployment", LOW),
]

# Credential patterns to detect
CREDENTIAL_PATTERNS: Dict[str, str] = {
    "aws_access_key": r"AKIA[0-9A-Z]{16}",
    "aws_secret_key": r"[0-9a-zA-Z/+]{40}",
    "github_token": r"gh[ps]_[0-9a-zA-Z]{36}",
    "generic_secret": r"(?i)(password|secret|token|key)\s*[=:]\s*['\"][^'\"]{8,}['\"]",
}


class DevOpsSafetyExtension(SafetyExtensionProtocol):
    """Safety extension for DevOps tasks."""

    def get_bash_patterns(self) -> List[SafetyPattern]:
        """Return DevOps-specific bash patterns.

        Returns:
            List of SafetyPattern for dangerous bash commands.
        """
        return [
            SafetyPattern(
                pattern=p,
                description=d,
                risk_level=r,
                category="devops",
            )
            for p, d, r in _DEVOPS_DANGER_TUPLES
        ]

    def get_danger_patterns(self) -> List[Tuple[str, str, str]]:
        """Return DevOps-specific danger patterns (legacy format).

        Returns:
            List of (regex_pattern, description, risk_level) tuples.
        """
        return _DEVOPS_DANGER_TUPLES

    def get_blocked_operations(self) -> List[str]:
        """Return operations that should be blocked in DevOps context."""
        return [
            "delete_production_database",
            "destroy_production_infrastructure",
            "expose_secrets_to_logs",
            "disable_security_features",
            "create_public_s3_bucket",
        ]

    def get_credential_patterns(self) -> Dict[str, str]:
        """Return patterns for detecting credentials.

        Returns:
            Dict of credential_type -> regex_pattern.
        """
        return CREDENTIAL_PATTERNS

    def validate_dockerfile(self, content: str) -> List[str]:
        """Validate Dockerfile security best practices.

        Returns:
            List of security warnings found.
        """
        warnings = []
        import re

        # Check for latest tag
        if re.search(r"FROM\s+\S+:latest", content):
            warnings.append("Using ':latest' tag - pin to specific version")

        # Check for root user
        if not re.search(r"USER\s+(?!root)\w+", content):
            warnings.append("No non-root USER specified")

        # Check for COPY with --chown
        if re.search(r"COPY\s+(?!--chown)", content) and "USER" in content:
            warnings.append("Consider using COPY --chown for file ownership")

        # Check for health check
        if "HEALTHCHECK" not in content:
            warnings.append("No HEALTHCHECK instruction")

        return warnings

    def validate_kubernetes_manifest(self, content: str) -> List[str]:
        """Validate Kubernetes manifest security.

        Returns:
            List of security warnings found.
        """
        warnings = []
        import re

        # Check for privileged containers
        if re.search(r"privileged:\s*true", content):
            warnings.append("Privileged container detected")

        # Check for resource limits
        if "limits:" not in content:
            warnings.append("No resource limits defined")

        # Check for probes
        if "livenessProbe:" not in content:
            warnings.append("No liveness probe defined")
        if "readinessProbe:" not in content:
            warnings.append("No readiness probe defined")

        return warnings

    def get_safety_reminders(self) -> List[str]:
        """Return safety reminders for DevOps output."""
        return [
            "Never commit secrets to version control",
            "Use environment variables or secrets managers for credentials",
            "Test infrastructure changes in non-production first",
            "Enable audit logging for compliance",
            "Use least-privilege permissions",
            "Pin all dependency versions",
        ]
