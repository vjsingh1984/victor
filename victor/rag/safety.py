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

"""RAG Safety Extension - Security patterns for RAG operations.

This module provides RAG-specific safety patterns for:
- Bulk deletion warnings
- Untrusted source ingestion detection
- PII detection in documents
- Data loss prevention

Example:
    from victor.rag.safety import RAGSafetyExtension

    safety = RAGSafetyExtension()
    patterns = safety.get_bash_patterns()

    # Check for dangerous operations
    for pattern in patterns:
        if pattern.matches(command):
            print(f"Warning: {pattern.description}")
"""

import re
from typing import Dict, List, Tuple

from victor.core.verticals.protocols import SafetyExtensionProtocol, SafetyPattern


# Risk levels
HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"


# RAG-specific dangerous patterns
RAG_DANGER_PATTERNS: List[SafetyPattern] = [
    # Bulk deletion patterns
    SafetyPattern(
        pattern=r"rag_delete\s+.*\*",
        description="Bulk deletion with wildcard - may delete all documents",
        risk_level=HIGH,
        category="rag",
    ),
    SafetyPattern(
        pattern=r"rag_delete\s+--all",
        description="Delete all documents from knowledge base",
        risk_level=HIGH,
        category="rag",
    ),
    SafetyPattern(
        pattern=r"DELETE\s+FROM\s+.*WHERE\s+1\s*=\s*1",
        description="SQL pattern that deletes all records",
        risk_level=HIGH,
        category="rag",
    ),
    # Untrusted source patterns
    SafetyPattern(
        pattern=r"rag_ingest.*http://(?!localhost)",
        description="Ingesting from non-HTTPS URL - potential untrusted source",
        risk_level=MEDIUM,
        category="rag",
    ),
    SafetyPattern(
        pattern=r"rag_ingest.*(?:pastebin|gist\.github|hastebin)",
        description="Ingesting from paste service - potentially untrusted content",
        risk_level=MEDIUM,
        category="rag",
    ),
    # Large batch operations
    SafetyPattern(
        pattern=r"rag_ingest.*--batch\s+\d{4,}",
        description="Large batch ingestion (1000+ files) - verify this is intentional",
        risk_level=LOW,
        category="rag",
    ),
]

# PII detection patterns for document content
PII_PATTERNS: Dict[str, Tuple[str, str, str]] = {
    "ssn": (
        r"\b\d{3}-\d{2}-\d{4}\b",
        "Social Security Number detected",
        HIGH,
    ),
    "credit_card": (
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b",
        "Credit card number detected",
        HIGH,
    ),
    "email": (
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "Email address detected",
        LOW,
    ),
    "phone": (
        r"\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b",
        "Phone number detected",
        LOW,
    ),
    "ip_address": (
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "IP address detected",
        LOW,
    ),
    "aws_key": (
        r"(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}",
        "AWS access key detected",
        HIGH,
    ),
    "api_key": (
        r"(?i)(?:api[_-]?key|apikey)['\"]?\s*[:=]\s*['\"]?[a-zA-Z0-9_-]{20,}",
        "API key pattern detected",
        HIGH,
    ),
    "private_key": (
        r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----",
        "Private key detected",
        HIGH,
    ),
}

# Document ingestion safety patterns
INGESTION_SAFETY_PATTERNS: List[SafetyPattern] = [
    SafetyPattern(
        pattern=r"\.(?:exe|dll|bat|cmd|sh|ps1)$",
        description="Executable file type - should not be ingested as document",
        risk_level=MEDIUM,
        category="rag",
    ),
    SafetyPattern(
        pattern=r"\.(?:zip|tar|gz|7z|rar)$",
        description="Archive file - consider extracting before ingestion",
        risk_level=LOW,
        category="rag",
    ),
    SafetyPattern(
        pattern=r"(?:/etc/passwd|/etc/shadow|\.ssh/)",
        description="System file path - may contain sensitive data",
        risk_level=HIGH,
        category="rag",
    ),
]


class RAGSafetyExtension(SafetyExtensionProtocol):
    """Safety extension for RAG tasks.

    Provides RAG-specific dangerous operation patterns including:
    - Bulk deletion warnings
    - Untrusted source detection
    - PII detection in documents
    - Data loss prevention

    Example:
        safety = RAGSafetyExtension()

        # Get all dangerous patterns
        patterns = safety.get_bash_patterns()

        # Scan document content for PII
        pii_matches = safety.scan_for_pii(document_content)
    """

    def __init__(
        self,
        include_pii_detection: bool = True,
        include_ingestion_safety: bool = True,
    ):
        """Initialize the safety extension.

        Args:
            include_pii_detection: Include PII detection patterns
            include_ingestion_safety: Include ingestion safety patterns
        """
        self._include_pii_detection = include_pii_detection
        self._include_ingestion_safety = include_ingestion_safety

    def get_bash_patterns(self) -> List[SafetyPattern]:
        """Return RAG-specific bash patterns.

        Returns:
            List of SafetyPattern for dangerous RAG commands.
        """
        patterns = list(RAG_DANGER_PATTERNS)
        if self._include_ingestion_safety:
            patterns.extend(INGESTION_SAFETY_PATTERNS)
        return patterns

    def get_danger_patterns(self) -> List[Tuple[str, str, str]]:
        """Return RAG-specific danger patterns (legacy format).

        Returns:
            List of (regex_pattern, description, risk_level) tuples.
        """
        return [(p.pattern, p.description, p.risk_level) for p in self.get_bash_patterns()]

    def get_file_patterns(self) -> List[SafetyPattern]:
        """Return file operation patterns for RAG.

        Returns:
            List of safety patterns for dangerous file operations.
        """
        return INGESTION_SAFETY_PATTERNS if self._include_ingestion_safety else []

    def get_blocked_operations(self) -> List[str]:
        """Return operations that should be blocked in RAG context.

        Returns:
            List of blocked operation descriptions.
        """
        return [
            "bulk_delete_all_documents",
            "ingest_executable_files",
            "expose_pii_to_logs",
            "ingest_system_files",
        ]

    def get_pii_patterns(self) -> Dict[str, Tuple[str, str, str]]:
        """Return patterns for detecting PII in documents.

        Returns:
            Dict of pii_type -> (regex_pattern, description, risk_level).
        """
        return PII_PATTERNS.copy()

    def scan_for_pii(self, content: str) -> List[Dict]:
        """Scan content for PII.

        Args:
            content: Text content to scan

        Returns:
            List of PII match dictionaries with type, severity, and location
        """
        if not self._include_pii_detection:
            return []

        matches = []
        lines = content.split("\n")

        for pii_type, (pattern, description, risk_level) in PII_PATTERNS.items():
            compiled = re.compile(pattern)
            for line_num, line in enumerate(lines, 1):
                for match in compiled.finditer(line):
                    matches.append(
                        {
                            "type": pii_type,
                            "description": description,
                            "severity": risk_level,
                            "line": line_num,
                            "column": match.start(),
                            # Mask the actual value
                            "masked_value": self._mask_value(match.group(), pii_type),
                        }
                    )

        return matches

    def _mask_value(self, value: str, pii_type: str) -> str:
        """Mask a detected PII value.

        Args:
            value: The detected value
            pii_type: Type of PII

        Returns:
            Masked version of the value
        """
        if len(value) <= 4:
            return "*" * len(value)
        return value[:2] + "*" * (len(value) - 4) + value[-2:]

    def validate_ingestion_source(self, source: str) -> List[str]:
        """Validate an ingestion source for safety issues.

        Args:
            source: File path or URL to validate

        Returns:
            List of warning messages (empty if safe)
        """
        warnings = []

        for pattern in INGESTION_SAFETY_PATTERNS:
            if re.search(pattern.pattern, source, re.IGNORECASE):
                warnings.append(f"{pattern.risk_level}: {pattern.description}")

        # Check for untrusted URLs
        for pattern in RAG_DANGER_PATTERNS:
            if "ingest" in pattern.pattern.lower():
                if re.search(pattern.pattern, f"rag_ingest {source}", re.IGNORECASE):
                    warnings.append(f"{pattern.risk_level}: {pattern.description}")

        return warnings

    def get_safety_reminders(self) -> List[str]:
        """Return safety reminders for RAG output.

        Returns:
            List of safety reminder strings.
        """
        return [
            "Verify source authenticity before ingesting external content",
            "Review documents for PII before adding to knowledge base",
            "Use --dry-run for bulk operations when available",
            "Backup knowledge base before major deletions",
            "Consider content licensing and copyright restrictions",
        ]

    def get_category(self) -> str:
        """Get the category name for these patterns.

        Returns:
            Category identifier
        """
        return "rag"

    def get_tool_restrictions(self) -> Dict[str, List[str]]:
        """Get tool-specific argument restrictions.

        Returns:
            Dict mapping tool names to list of restricted argument patterns.
        """
        return {
            "rag_delete": [
                r"--all",
                r"\*",
                r"--force\s+--all",
            ],
            "rag_ingest": [
                r"/etc/",
                r"\.ssh/",
                r"credentials",
                r"secrets",
            ],
        }


__all__ = [
    "RAGSafetyExtension",
    "RAG_DANGER_PATTERNS",
    "INGESTION_SAFETY_PATTERNS",
    "PII_PATTERNS",
    "HIGH",
    "MEDIUM",
    "LOW",
]
