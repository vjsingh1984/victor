"""Integration tests for vertical safety rules with actual workflows.

These tests verify that the framework-based safety rules work correctly
when integrated with actual workflows and operations from each vertical.
"""

import pytest
from victor.framework.config import SafetyEnforcer, SafetyConfig, SafetyLevel
from victor.coding.safety import create_all_coding_safety_rules
from victor.devops.safety import create_all_devops_safety_rules
from victor.rag.safety import create_all_rag_safety_rules
from victor.research.safety import create_all_research_safety_rules
from victor.dataanalysis.safety import create_all_dataanalysis_safety_rules
from victor.benchmark.safety import create_all_benchmark_safety_rules


class TestSafetyIntegration:
    """Integration tests for safety rules with real-world scenarios."""

    def test_rag_workflow_with_safety(self):
        """RAG workflow should block dangerous deletion and ingestion operations."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_all_rag_safety_rules(enforcer)

        # Simulate RAG workflow operations
        operations = [
            # Safe operations - should be allowed
            ("rag_ingest document.pdf", True),
            ("rag_search query text", True),
            ("rag_add_document doc1", True),
            
            # Dangerous operations - should be blocked
            ("rag_delete --all", False),
            ("rag_delete *", False),
            ("rag_ingest malware.exe", False),
            ("rag_ingest /etc/passwd", False),
            ("rag_ingest http://example.com/data.json", False),
        ]

        for operation, should_be_allowed in operations:
            allowed, reason = enforcer.check_operation(operation)
            
            if should_be_allowed:
                assert allowed is True, f"Safe operation should be allowed: {operation}. Reason: {reason}"
            else:
                assert allowed is False, f"Dangerous operation should be blocked: {operation}"

    def test_research_workflow_with_safety(self):
        """Research workflow should block low-credibility sources and fabricated content."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_all_research_safety_rules(enforcer)

        # Simulate research workflow operations
        operations = [
            # Safe operations - should be allowed
            ("cite arxiv.org paper", True),
            ("search .gov sources", True),
            ("analyze research findings", True),
            
            # Dangerous operations - should be blocked
            ("cite fake-blog.blogspot.com", False),
            ("invent citation for claim", False),
            ("fabricate source data", False),
            ("cite tumblr.com source", False),
        ]

        for operation, should_be_allowed in operations:
            allowed, reason = enforcer.check_operation(operation)
            
            if should_be_allowed:
                assert allowed is True, f"Safe operation should be allowed: {operation}. Reason: {reason}"
            else:
                assert allowed is False, f"Dangerous operation should be blocked: {operation}"

    def test_dataanalysis_workflow_with_safety(self):
        """DataAnalysis workflow should block PII exports and external uploads."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_all_dataanalysis_safety_rules(enforcer)

        # Simulate data analysis workflow operations
        operations = [
            # Safe operations - should be allowed
            ("analyze dataset.csv", True),
            ("create visualization", True),
            ("export anonymized data", True),
            
            # Dangerous operations - should be blocked
            ("export data with SSN to CSV", False),
            ("to_csv credit card data", False),
            ("upload to s3 bucket", True),  # Actually should be blocked
            ("query production database", False),
            ("export password column", False),
        ]

        for operation, should_be_allowed in operations:
            allowed, reason = enforcer.check_operation(operation)
            
            # Note: "upload to s3 bucket" should actually be blocked
            if operation == "upload to s3 bucket":
                should_be_allowed = False
            
            if should_be_allowed:
                assert allowed is True, f"Safe operation should be allowed: {operation}. Reason: {reason}"
            else:
                assert allowed is False, f"Dangerous operation should be blocked: {operation}. Reason: {reason}"

    def test_all_verticals_safety_integration(self):
        """All vertical safety rules should work together without conflicts."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))

        # Register all vertical safety rules
        create_all_coding_safety_rules(enforcer)
        create_all_devops_safety_rules(enforcer)
        create_all_rag_safety_rules(enforcer)
        create_all_research_safety_rules(enforcer)
        create_all_dataanalysis_safety_rules(enforcer)

        # Should have rules from all verticals
        assert len(enforcer.rules) > 20

        # Test operations from different verticals
        # Format: (operation, should_be_allowed)
        test_cases = [
            ("git push --force origin main", False),  # Coding - should be blocked
            ("kubectl delete deployment -n production app", False),  # DevOps - should be blocked
            ("rag_delete --all", False),  # RAG - should be blocked
            ("cite fake-blog.blogspot.com", False),  # Research - should be blocked
            ("export data with SSN", False),  # DataAnalysis - should be blocked
        ]

        for operation, should_be_allowed in test_cases:
            allowed, reason = enforcer.check_operation(operation)
            if should_be_allowed:
                assert allowed is True, f"Operation should be allowed: {operation}. Reason: {reason}"
            else:
                assert allowed is False, f"Operation should be blocked: {operation}. Reason: {reason}"

    def test_safety_levels_enforcement(self):
        """Safety levels should be enforced correctly across all verticals."""
        # HIGH level - block all dangerous operations
        enforcer_high = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_all_rag_safety_rules(enforcer_high)
        
        allowed, _ = enforcer_high.check_operation("rag_delete --all")
        assert allowed is False, "HIGH level should block dangerous operations"

        # LOW level - warn but allow most operations
        enforcer_low = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.LOW))
        create_all_rag_safety_rules(enforcer_low)
        
        # At LOW level, LOW priority rules should only warn (not block)
        # But HIGH priority rules should still block
        allowed, _ = enforcer_low.check_operation("rag_delete --all")
        assert allowed is False, "HIGH priority rules should still block at LOW level"

    def test_custom_safety_rules_with_verticals(self):
        """Custom safety rules should work alongside vertical rules."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        
        # Register vertical rules
        create_all_coding_safety_rules(enforcer)
        
        # Add custom rule
        from victor.framework.config import SafetyRule
        enforcer.add_rule(
            SafetyRule(
                name="custom_block_specific_file",
                description="Block editing specific file",
                check_fn=lambda op: "important_config.yaml" in op,
                level=SafetyLevel.HIGH,
            )
        )

        # Vertical rule should work
        allowed, _ = enforcer.check_operation("git push --force origin main")
        assert allowed is False

        # Custom rule should work
        allowed, _ = enforcer.check_operation("edit important_config.yaml")
        assert allowed is False

        # Other operations should be allowed
        allowed, _ = enforcer.check_operation("edit other_file.py")
        assert allowed is True

    @pytest.mark.parametrize("safety_level", [SafetyLevel.LOW, SafetyLevel.MEDIUM, SafetyLevel.HIGH])
    def test_safety_level_consistency_across_verticals(self, safety_level):
        """Safety levels should work consistently for all verticals."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=safety_level))

        # Register all vertical rules
        create_all_coding_safety_rules(enforcer)
        create_all_devops_safety_rules(enforcer)
        create_all_rag_safety_rules(enforcer)
        create_all_research_safety_rules(enforcer)
        create_all_dataanalysis_safety_rules(enforcer)

        # Test that HIGH priority rules are always blocked
        # (regardless of config level)
        high_priority_ops = [
            ("rag_delete --all", "RAG bulk delete"),
            ("export data with SSN", "PII export"),
        ]

        for operation, description in high_priority_ops:
            allowed, _ = enforcer.check_operation(operation)
            assert allowed is False, f"{description} should be blocked at {safety_level} level"

    def test_benchmark_workflow_with_safety(self):
        """Benchmark workflow should block production repo access and data leaks."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_all_benchmark_safety_rules(
            enforcer,
            protected_repositories=["/production", "/prod", "release"],
            max_timeout_seconds=300,
        )

        # Simulate benchmark workflow operations
        operations = [
            # Safe operations - should be allowed
            ("run benchmark on test repo", True),
            ("execute swe-bench task", True),
            ("test solution locally", True),
            ("timeout=240 seconds", True),

            # Dangerous operations - should be blocked
            ("modify code in /production directory", False),
            ("git push to release branch", False),
            ("tool_budget=-1 unlimited", False),
            ("run tests on production environment", False),
            ("drop table in test", False),
            ("upload benchmark data to s3", False),
            ("export benchmark task description", False),
            ("share humaneval solution on api", False),
            ("timeout=900 seconds", False),
        ]

        for operation, should_be_allowed in operations:
            allowed, reason = enforcer.check_operation(operation)

            if should_be_allowed:
                assert allowed is True, f"Safe operation should be allowed: {operation}. Reason: {reason}"
            else:
                assert allowed is False, f"Dangerous operation should be blocked: {operation}. Reason: {reason}"

    def test_all_six_verticals_safety_integration(self):
        """All 6 vertical safety rules should work together without conflicts."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))

        # Register all 6 vertical safety rules
        create_all_coding_safety_rules(enforcer)
        create_all_devops_safety_rules(enforcer)
        create_all_rag_safety_rules(enforcer)
        create_all_research_safety_rules(enforcer)
        create_all_dataanalysis_safety_rules(enforcer)
        create_all_benchmark_safety_rules(enforcer)

        # Should have rules from all 6 verticals
        assert len(enforcer.rules) > 30

        # Test operations from all 6 verticals
        # Format: (operation, should_be_allowed)
        test_cases = [
            # Coding vertical
            ("git push --force origin main", False),
            ("rm -rf /important/data", False),

            # DevOps vertical
            ("kubectl delete deployment -n production app", False),
            ("docker run --privileged container", False),

            # RAG vertical
            ("rag_delete --all", False),
            ("rag_ingest malware.exe", False),

            # Research vertical
            ("cite fake-blog.blogspot.com", False),
            ("fabricate source data", False),

            # DataAnalysis vertical
            ("export data with SSN", False),
            ("upload to s3 bucket", False),

            # Benchmark vertical
            ("modify /production directory", False),
            ("tool_budget=-1", False),
            ("upload benchmark data to s3", False),
            ("git push to release", False),
        ]

        for operation, should_be_allowed in test_cases:
            allowed, reason = enforcer.check_operation(operation)
            if should_be_allowed:
                assert allowed is True, f"Operation should be allowed: {operation}. Reason: {reason}"
            else:
                assert allowed is False, f"Operation should be blocked: {operation}. Reason: {reason}"

    def test_benchmark_data_privacy_enforcement(self):
        """Benchmark data privacy rules should be enforced at all safety levels."""
        # Test at different safety levels
        for safety_level in [SafetyLevel.LOW, SafetyLevel.MEDIUM, SafetyLevel.HIGH]:
            enforcer = SafetyEnforcer(config=SafetyConfig(level=safety_level))
            create_all_benchmark_safety_rules(enforcer)

            # These HIGH priority rules should ALWAYS block (allow_override=False)
            dangerous_ops = [
                "upload benchmark data to s3",
                "export benchmark task description",
                "share humaneval solution externally",
            ]

            for operation in dangerous_ops:
                allowed, reason = enforcer.check_operation(operation)
                assert allowed is False, f"Data privacy rule should block at {safety_level} level: {operation}. Reason: {reason}"

    def test_benchmark_repository_isolation(self):
        """Benchmark repository isolation should prevent production access."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_all_benchmark_safety_rules(
            enforcer,
            protected_repositories=["/production", "/prod", "release"],
        )

        # Test various operations on protected repositories
        protected_operations = [
            "write code to /production",
            "edit file in /prod directory",
            "delete from release",
            "git push to release branch",
            "git commit to /prod",
        ]

        for operation in protected_operations:
            allowed, reason = enforcer.check_operation(operation)
            assert allowed is False, f"Should block operation on protected repo: {operation}. Reason: {reason}"

        # Test that safe operations are allowed
        safe_operations = [
            "write code to /benchmark/workspace",
            "edit file in test branch",
            "run tests in /tmp",
        ]

        for operation in safe_operations:
            allowed, reason = enforcer.check_operation(operation)
            assert allowed is True, f"Should allow safe operation: {operation}. Reason: {reason}"
