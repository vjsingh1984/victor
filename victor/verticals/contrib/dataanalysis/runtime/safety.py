"""Data Analysis Safety Extension - Privacy and data protection patterns."""

from typing import Dict, List, Tuple

from victor.core.verticals.protocols import SafetyExtensionProtocol, SafetyPattern

# Import PII detection from core safety module
from victor.security.safety.pii import (
    PIIScanner,
    PIIType,
    detect_pii_columns as core_detect_pii_columns,
    get_anonymization_suggestion as core_get_anonymization_suggestion,
    get_safety_reminders as core_get_safety_reminders,
)

# Risk levels
HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"

# Data analysis-specific safety patterns as tuples
# Note: PII detection patterns are now in victor.security.safety.pii
_DATA_ANALYSIS_SAFETY_TUPLES: List[Tuple[str, str, str]] = [
    # High-risk patterns - PII exposure (kept for SafetyPattern interface)
    (r"(?i)(social[_\s-]?security|ssn)[^\w]", "Social Security Number exposure", HIGH),
    (r"(?i)(credit[_\s-]?card|card[_\s-]?number)", "Credit card data exposure", HIGH),
    (r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b", "SSN pattern detected", HIGH),
    (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "Credit card pattern detected", HIGH),
    (r"(?i)password|passwd|pwd", "Password column exposure", HIGH),
    (r"(?i)medical|diagnosis|health[_\s-]?record", "Medical data exposure", HIGH),
    # Medium-risk patterns - semi-sensitive
    (r"(?i)(email|e-mail)[^\w]", "Email addresses in output", MEDIUM),
    (r"(?i)(phone|mobile|cell)[^\w]", "Phone numbers in output", MEDIUM),
    (r"(?i)(address|street|zip[_\s-]?code)", "Physical address exposure", MEDIUM),
    (r"(?i)(date[_\s-]?of[_\s-]?birth|dob|birth[_\s-]?date)", "Date of birth exposure", MEDIUM),
    (r"(?i)(salary|income|wage)", "Financial data exposure", MEDIUM),
    # Low-risk patterns - best practices
    (r"(?i)print\(.*df\)", "Full dataframe print", LOW),
    (r"\.to_csv\([^)]*index\s*=\s*True", "Index in CSV output", LOW),
    (r"(?i)random_state\s*=\s*None", "Non-reproducible random state", LOW),
]


class DataAnalysisSafetyExtension(SafetyExtensionProtocol):
    """Safety extension for data analysis tasks."""

    def get_bash_patterns(self) -> List[SafetyPattern]:
        """Return data analysis-specific bash patterns.

        Returns:
            List of SafetyPattern for dangerous bash commands.
        """
        return [
            SafetyPattern(
                pattern=p,
                description=d,
                risk_level=r,
                category="data_analysis",
            )
            for p, d, r in _DATA_ANALYSIS_SAFETY_TUPLES
        ]

    def get_danger_patterns(self) -> List[Tuple[str, str, str]]:
        """Return data analysis-specific danger patterns (legacy format).

        Returns:
            List of (regex_pattern, description, risk_level) tuples.
        """
        return _DATA_ANALYSIS_SAFETY_TUPLES

    def get_blocked_operations(self) -> List[str]:
        """Return operations that should be blocked in data analysis."""
        return [
            "export_pii_unencrypted",
            "upload_data_externally",
            "share_credentials",
            "access_production_database_directly",
        ]

    def get_pii_patterns(self) -> Dict[str, str]:
        """Return patterns for detecting PII columns.

        Uses patterns from victor.security.safety.pii for comprehensive detection.

        Returns:
            Dict of pii_type -> regex_pattern for column names.
        """
        from victor.security.safety.pii import PII_COLUMN_PATTERNS as CORE_PII_PATTERNS

        # Return simplified dict format for backward compatibility
        return {pii_type.value: pattern for pii_type, pattern in CORE_PII_PATTERNS.items()}

    def detect_pii_columns(self, columns: List[str]) -> List[Tuple[str, str]]:
        """Detect potential PII columns in a dataframe.

        Uses victor.security.safety.pii.detect_pii_columns for detection.

        Args:
            columns: List of column names.

        Returns:
            List of (column_name, pii_type) tuples.
        """
        results = core_detect_pii_columns(columns)
        # Convert PIIType enum to string for backward compatibility
        return [(col, pii_type.value) for col, pii_type in results]

    def get_anonymization_suggestions(self, pii_type: str) -> str:
        """Get suggestions for anonymizing a PII type.

        Uses victor.security.safety.pii.get_anonymization_suggestion.

        Args:
            pii_type: Type of PII detected (string).

        Returns:
            Suggestion string for anonymization.
        """
        # Convert string to PIIType enum
        try:
            pii_enum = PIIType(pii_type)
            return core_get_anonymization_suggestion(pii_enum)
        except ValueError:
            return "Consider removing or hashing"

    def get_safety_reminders(self) -> List[str]:
        """Return safety reminders for data analysis.

        Uses victor.security.safety.pii.get_safety_reminders.
        """
        return core_get_safety_reminders()

    def scan_for_pii(self, content: str) -> List[Dict]:
        """Scan content for PII using the core PIIScanner.

        Args:
            content: Text content to scan

        Returns:
            List of PII match dictionaries
        """
        scanner = PIIScanner()
        matches = scanner.scan_content(content)
        return [
            {
                "type": m.pii_type.value,
                "severity": m.severity.value,
                "source": m.source,
                "suggestion": m.suggestion,
            }
            for m in matches
        ]


__all__ = [
    "DataAnalysisSafetyExtension",
    "HIGH",
    "MEDIUM",
    "LOW",
    # New framework-based safety rules
    "create_dataanalysis_pii_safety_rules",
    "create_dataanalysis_export_safety_rules",
    "create_all_dataanalysis_safety_rules",
]


# =============================================================================
# Framework-Based Safety Rules (New)
# =============================================================================

"""Framework-based safety rules for data analysis operations.

This section provides factory functions that register safety rules
with the framework-level SafetyEnforcer. This is the new recommended
approach for safety enforcement in data analysis workflows.

Example:
    from victor.framework.config import SafetyEnforcer, SafetyConfig, SafetyLevel
    from victor.verticals.contrib.dataanalysis.safety import create_all_dataanalysis_safety_rules

    enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
    create_all_dataanalysis_safety_rules(enforcer)

    # Check operations
    allowed, reason = enforcer.check_operation("export data with SSN")
    if not allowed:
        print(f"Blocked: {reason}")
"""

from victor.framework.config import SafetyEnforcer, SafetyRule, SafetyLevel


def create_dataanalysis_pii_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_pii_exports: bool = True,
    warn_on_pii_columns: bool = True,
    require_anonymization: bool = False,
) -> None:
    """Register data analysis PII-specific safety rules.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_pii_exports: Block exporting data containing PII
        warn_on_pii_columns: Warn when PII columns are detected
        require_anonymization: Require anonymization for PII data

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_dataanalysis_pii_safety_rules(
            enforcer,
            block_pii_exports=True,
            warn_on_pii_columns=True
        )
    """
    if block_pii_exports:
        enforcer.add_rule(
            SafetyRule(
                name="dataanalysis_block_pii_export",
                description="Block exporting data containing PII",
                check_fn=lambda op: any(
                    phrase in op.lower()
                    for phrase in [
                        "export",
                        "to_csv",
                        "to_excel",
                        "to_json",
                        "upload",
                    ]
                )
                and any(
                    pii in op.lower()
                    for pii in [
                        "ssn",
                        "social security",
                        "credit card",
                        "password",
                        "medical",
                        "health",
                    ]
                ),
                level=SafetyLevel.HIGH,
                allow_override=False,  # PII exports should NEVER be allowed
            )
        )

    if warn_on_pii_columns:
        enforcer.add_rule(
            SafetyRule(
                name="dataanalysis_warn_pii_columns",
                description="Warn when PII columns are detected in operations",
                check_fn=lambda op: any(
                    pii in op.lower()
                    for pii in [
                        "ssn",
                        "social security",
                        "credit card",
                        "email",
                        "phone",
                        "address",
                        "date of birth",
                        "salary",
                        "income",
                    ]
                )
                and ("df" in op.lower() or "dataframe" in op.lower() or "column" in op.lower()),
                level=SafetyLevel.LOW,  # Warn only
                allow_override=True,
            )
        )

    if require_anonymization:
        enforcer.add_rule(
            SafetyRule(
                name="dataanalysis_require_anonymization",
                description="Require anonymization for PII data before export",
                check_fn=lambda op: any(
                    phrase in op.lower()
                    for phrase in [
                        "export",
                        "to_csv",
                        "to_excel",
                        "upload",
                        "share",
                    ]
                )
                and any(
                    pii in op.lower()
                    for pii in ["ssn", "social security", "credit card", "password"]
                )
                and "anonymize" not in op.lower()
                and "hash" not in op.lower()
                and "mask" not in op.lower(),
                level=SafetyLevel.MEDIUM,
                allow_override=True,
            )
        )


def create_dataanalysis_export_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_external_uploads: bool = True,
    block_production_db_access: bool = True,
    require_encryption: bool = True,
) -> None:
    """Register data analysis export-specific safety rules.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_external_uploads: Block uploading data externally
        block_production_db_access: Block direct production database access
        require_encryption: Require encryption for data exports

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_dataanalysis_export_safety_rules(
            enforcer,
            block_external_uploads=True,
            require_encryption=True
        )
    """
    if block_external_uploads:
        enforcer.add_rule(
            SafetyRule(
                name="dataanalysis_block_external_uploads",
                description="Block uploading data to external services",
                check_fn=lambda op: any(
                    phrase in op.lower()
                    for phrase in [
                        "upload to",
                        "send to",
                        "transfer to",
                    ]
                )
                and any(
                    external in op.lower()
                    for external in [
                        "s3",
                        "gcs",
                        "azure",
                        "dropbox",
                        "google drive",
                        "external",
                    ]
                ),
                level=SafetyLevel.HIGH,
                allow_override=True,
            )
        )

    if block_production_db_access:
        enforcer.add_rule(
            SafetyRule(
                name="dataanalysis_block_production_db",
                description="Block direct access to production databases",
                check_fn=lambda op: any(
                    phrase in op.lower()
                    for phrase in [
                        "production",
                        "prod",
                    ]
                )
                and any(
                    db in op.lower()
                    for db in [
                        "database",
                        "db.",
                        "sql",
                        "query",
                        "connect",
                    ]
                ),
                level=SafetyLevel.HIGH,
                allow_override=True,
            )
        )

    if require_encryption:
        enforcer.add_rule(
            SafetyRule(
                name="dataanalysis_require_encryption",
                description="Require encryption for sensitive data exports",
                check_fn=lambda op: any(
                    phrase in op.lower()
                    for phrase in [
                        "export",
                        "to_csv",
                        "to_excel",
                        "to_json",
                        "save",
                    ]
                )
                and any(
                    sensitive in op.lower()
                    for sensitive in [
                        "ssn",
                        "credit card",
                        "password",
                        "medical",
                        "personal",
                    ]
                )
                and "encrypt" not in op.lower()
                and "secure" not in op.lower(),
                level=SafetyLevel.MEDIUM,
                allow_override=True,
            )
        )


def create_all_dataanalysis_safety_rules(
    enforcer: SafetyEnforcer,
) -> None:
    """Register all data analysis safety rules at once.

    This is a convenience function that registers all data analysis-specific
    safety rules with appropriate defaults.

    Args:
        enforcer: SafetyEnforcer to register rules with

    Example:
        from victor.framework.config import SafetyEnforcer, SafetyConfig, SafetyLevel

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_all_dataanalysis_safety_rules(enforcer)

        # Now all operations are checked
        allowed, reason = enforcer.check_operation("export data with SSN")
    """
    create_dataanalysis_pii_safety_rules(enforcer)
    create_dataanalysis_export_safety_rules(enforcer)
