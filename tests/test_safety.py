# Tests for victor-rag safety rules
# Migrated from victor/tests/unit/framework/test_config.py

import pytest

from victor.framework.config import SafetyConfig, SafetyEnforcer, SafetyLevel


class TestRAGDeletionSafety:
    """Tests for RAG deletion safety rules."""

    def test_rag_safety_deletion_rules(self):
        """RAG deletion safety rules should block bulk deletions."""
        from victor_rag.safety import create_rag_deletion_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_rag_deletion_safety_rules(enforcer, block_bulk_delete=True, block_delete_all=True)

        # Test bulk delete blocking
        allowed, reason = enforcer.check_operation("rag_delete *")
        assert allowed is False
        assert "bulk" in reason.lower() or "delete" in reason.lower()

        # Test delete all blocking
        allowed, reason = enforcer.check_operation("rag_delete --all")
        assert allowed is False
        assert "all" in reason.lower() or "delete" in reason.lower()


class TestRAGIngestionSafety:
    """Tests for RAG ingestion safety rules."""

    def test_rag_safety_ingestion_rules(self):
        """RAG ingestion safety rules should block unsafe ingestion."""
        from victor_rag.safety import create_rag_ingestion_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_rag_ingestion_safety_rules(
            enforcer,
            block_executable_files=True,
            block_system_files=True,
            require_https=True,
        )

        # Test executable file blocking
        allowed, reason = enforcer.check_operation("rag_ingest document.exe")
        assert allowed is False
        assert "executable" in reason.lower()

        # Test system file blocking
        allowed, reason = enforcer.check_operation("rag_ingest /etc/passwd")
        assert allowed is False
        assert "system" in reason.lower()

        # Test HTTPS requirement
        allowed, reason = enforcer.check_operation("rag_ingest http://example.com/data.json")
        assert allowed is False
        assert "https" in reason.lower()


class TestRAGAllSafetyRules:
    """Tests for all RAG safety rules combined."""

    def test_create_all_rag_safety_rules(self):
        """create_all_rag_safety_rules should register all RAG rules."""
        from victor_rag.safety import create_all_rag_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_all_rag_safety_rules(enforcer)

        # Should have rules from deletion and ingestion categories
        assert len(enforcer.rules) > 0

        # Verify bulk delete is blocked
        allowed, _ = enforcer.check_operation("rag_delete *")
        assert allowed is False

        # Verify executable ingestion is blocked
        allowed, _ = enforcer.check_operation("rag_ingest malware.exe")
        assert allowed is False
