"""Tests for SDK-owned PII safety helpers."""

from victor_contracts.safety import (
    PII_COLUMN_PATTERNS,
    PIIScanner,
    PIISeverity,
    PIIType,
    detect_pii_columns,
    detect_pii_in_content,
    get_anonymization_suggestion,
    has_pii,
)


def test_sdk_pii_helpers_detect_columns_and_content() -> None:
    columns = ["email_address", "age", "notes"]

    detected_columns = detect_pii_columns(columns)
    detected_content = detect_pii_in_content("Contact user@example.com or 555-123-4567")

    assert ("email_address", PIIType.EMAIL) in detected_columns
    assert ("age", PIIType.AGE) in detected_columns
    assert {match.pii_type for match in detected_content} == {PIIType.EMAIL, PIIType.PHONE}
    assert all("***" in match.matched_text for match in detected_content)
    assert has_pii("SSN 123-45-6789") is True


def test_sdk_pii_scanner_summary_and_metadata_exports() -> None:
    scanner = PIIScanner()

    matches = scanner.scan_content("Email a@example.com and passport AB1234567")
    summary = scanner.get_summary(matches)

    assert summary[PIISeverity.MEDIUM.value] == 1
    assert summary[PIISeverity.CRITICAL.value] == 1
    assert PII_COLUMN_PATTERNS[PIIType.EMAIL]
    assert "example.com" in get_anonymization_suggestion(PIIType.EMAIL)
