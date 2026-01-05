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
