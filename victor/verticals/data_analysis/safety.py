"""Data Analysis Safety Extension - Privacy and data protection patterns."""

from typing import Dict, List, Tuple

from victor.verticals.protocols import SafetyExtensionProtocol, SafetyPattern


# Risk levels
HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"

# Data analysis-specific safety patterns as tuples
_DATA_ANALYSIS_SAFETY_TUPLES: List[Tuple[str, str, str]] = [
    # High-risk patterns - PII exposure
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

# PII column patterns
PII_COLUMN_PATTERNS: Dict[str, str] = {
    "name": r"(?i)^(first|last|full)?[_\s-]?name$",
    "email": r"(?i)e?mail",
    "phone": r"(?i)(phone|mobile|cell)",
    "address": r"(?i)(address|street|city|state|zip)",
    "ssn": r"(?i)(ssn|social[_\s-]?security)",
    "dob": r"(?i)(dob|birth|bday)",
    "id": r"(?i)(user[_\s-]?id|customer[_\s-]?id|account)",
}


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

        Returns:
            Dict of pii_type -> regex_pattern for column names.
        """
        return PII_COLUMN_PATTERNS

    def detect_pii_columns(self, columns: List[str]) -> List[Tuple[str, str]]:
        """Detect potential PII columns in a dataframe.

        Args:
            columns: List of column names.

        Returns:
            List of (column_name, pii_type) tuples.
        """
        import re
        detected = []
        for col in columns:
            for pii_type, pattern in PII_COLUMN_PATTERNS.items():
                if re.search(pattern, col):
                    detected.append((col, pii_type))
                    break
        return detected

    def get_anonymization_suggestions(self, pii_type: str) -> str:
        """Get suggestions for anonymizing a PII type.

        Args:
            pii_type: Type of PII detected.

        Returns:
            Suggestion string for anonymization.
        """
        suggestions = {
            "name": "Replace with fake names (faker library) or hash",
            "email": "Replace domain, hash local part, or remove entirely",
            "phone": "Remove or replace with random numbers",
            "address": "Generalize to region/zip prefix only",
            "ssn": "Remove entirely - never store in analysis",
            "dob": "Convert to age ranges or year only",
            "id": "Hash with salt or replace with sequential IDs",
        }
        return suggestions.get(pii_type, "Consider removing or hashing")

    def get_safety_reminders(self) -> List[str]:
        """Return safety reminders for data analysis."""
        return [
            "Check for PII before sharing results",
            "Use sample data for development, not full production data",
            "Anonymize sensitive columns before analysis",
            "Document data sources and permissions",
            "Set random_state for reproducibility",
            "Never commit data files to version control",
        ]
