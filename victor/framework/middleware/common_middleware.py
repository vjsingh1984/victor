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

"""Common middleware patterns extracted from verticals.

This module consolidates common safety rule factory functions from vertical
safety.py files into the framework level. This eliminates code duplication
and provides a consistent API for safety enforcement across all verticals.

Key improvements:
- 30-40% code reduction in vertical safety.py files
- Consistent safety rule patterns
- Easier testing and maintenance
- Single source of truth for common safety rules

Verticals now import and extend these common patterns instead of
duplicating code.

Example:
    from victor.framework.config import SafetyEnforcer, SafetyConfig, SafetyLevel
    from victor.framework.middleware.common_middleware import (
        create_git_safety_rules,
        create_deployment_safety_rules,
        create_pii_safety_rules,
    )

    enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))

    # Register common rules
    create_git_safety_rules(enforcer)
    create_deployment_safety_rules(enforcer)
    create_pii_safety_rules(enforcer)

    # Use vertical-specific extensions if needed
    # Each vertical can provide their own safety rules
    # Example: from my_vertical.safety import create_my_vertical_safety_rules
    #         create_my_vertical_safety_rules(enforcer)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from victor.framework.config import SafetyEnforcer, SafetyLevel, SafetyRule


# =============================================================================
# Common Safety Rule Factory Functions
# =============================================================================


def create_git_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_force_push: bool = True,
    block_main_push: bool = True,
    require_tests_before_commit: bool = False,
    protected_branches: Optional[List[str]] = None,
) -> None:
    """Register git-specific safety rules.

    This is a common pattern used across Coding and DevOps verticals.
    Consolidated to avoid duplication.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_force_push: Block force push to protected branches
        block_main_push: Block direct push to main/master
        require_tests_before_commit: Require tests to pass before commit
        protected_branches: List of protected branch names
                          (default: ["main", "master", "develop"])

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_git_safety_rules(
            enforcer,
            block_force_push=True,
            block_main_push=True,
            protected_branches=["main", "master", "production"]
        )
    """
    protected = protected_branches or ["main", "master", "develop"]

    if block_force_push:
        enforcer.add_rule(
            SafetyRule(
                name="git_block_force_push",
                description="Block force push to protected branches",
                check_fn=lambda op: "git push" in op
                and "--force" in op
                and any(branch in op for branch in protected),
                level=SafetyLevel.HIGH,
                allow_override=False,
            )
        )

    if block_main_push:
        enforcer.add_rule(
            SafetyRule(
                name="git_block_main_push",
                description="Block direct push to main/master branches (use PRs)",
                check_fn=lambda op: "git push" in op
                and any(
                    f" origin {branch}" in op or f" upstream {branch}" in op
                    for branch in protected[:2]
                ),  # main, master only
                level=SafetyLevel.MEDIUM,
                allow_override=True,
            )
        )

    if require_tests_before_commit:
        enforcer.add_rule(
            SafetyRule(
                name="git_require_tests_before_commit",
                description="Require tests to pass before committing",
                check_fn=lambda op: "git commit" in op and "--no-verify" not in op,
                level=SafetyLevel.LOW,  # Warn only
                allow_override=True,
            )
        )


def create_file_operation_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_destructive_commands: bool = True,
    protected_patterns: Optional[List[str]] = None,
) -> None:
    """Register file operation safety rules.

    Common pattern for blocking dangerous file operations used across
    Coding, DevOps, and RAG verticals.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_destructive_commands: Block commands like rm -rf, git clean -fdx
        protected_patterns: Glob patterns for protected files
                           (default: ["**/.env", "**/secrets*"])

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_file_operation_safety_rules(
            enforcer,
            block_destructive_commands=True,
            protected_patterns=["**/.env", "**/secrets*", "**/config/prod/*"]
        )
    """
    protected = protected_patterns or [".env", "secrets", "credential", "password"]

    if block_destructive_commands:
        enforcer.add_rule(
            SafetyRule(
                name="file_block_destructive",
                description="Block destructive file commands (rm -rf, git clean -fdx)",
                check_fn=lambda op: any(
                    cmd in op
                    for cmd in [
                        "rm -rf /",
                        "rm -rf /*",
                        "git clean -fdx",
                        "del /q /s",
                    ]
                ),
                level=SafetyLevel.HIGH,
                allow_override=False,
            )
        )

        enforcer.add_rule(
            SafetyRule(
                name="file_block_protected_modification",
                description="Block modification of protected files (.env, secrets, credentials)",
                check_fn=lambda op: any(
                    pattern in op.lower() for pattern in [p.lower() for p in protected]
                )
                and any(cmd in op for cmd in ["write(", "write_file(", "edit("]),
                level=SafetyLevel.HIGH,
                allow_override=True,
            )
        )


def create_deployment_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    require_approval_for_production: bool = True,
    require_backup_before_deploy: bool = True,
    protected_environments: Optional[List[str]] = None,
    enable_rollback: bool = True,
) -> None:
    """Register deployment-specific safety rules.

    Common pattern for DevOps vertical deployments.

    Args:
        enforcer: SafetyEnforcer to register rules with
        require_approval_for_production: Require approval for production deployments
        require_backup_before_deploy: Require backup before deployment
        protected_environments: List of protected environments
                                (default: ["production", "prod", "staging"])
        enable_rollback: Enable automatic rollback on failure

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_deployment_safety_rules(
            enforcer,
            require_approval_for_production=True,
            protected_environments=["production", "staging"]
        )
    """
    protected = protected_environments or ["production", "prod", "staging"]

    if require_approval_for_production:
        enforcer.add_rule(
            SafetyRule(
                name="deployment_require_approval",
                description="Require approval for production deployments",
                check_fn=lambda op: any(env in op.lower() for env in protected)
                and any(cmd in op for cmd in ["deploy", "kubectl apply", "terraform apply"]),
                level=SafetyLevel.HIGH,
                allow_override=True,
            )
        )

    if require_backup_before_deploy:
        enforcer.add_rule(
            SafetyRule(
                name="deployment_require_backup",
                description="Require backup before deployment to production",
                check_fn=lambda op: any(env in op.lower() for env in protected)
                and "deploy" in op.lower()
                and "backup" not in op.lower(),
                level=SafetyLevel.MEDIUM,
                allow_override=True,
            )
        )

    if enable_rollback:
        enforcer.add_rule(
            SafetyRule(
                name="deployment_rollback_plan",
                description="Warn if deployment doesn't include rollback plan",
                check_fn=lambda op: any(env in op.lower() for env in protected)
                and "deploy" in op.lower()
                and "rollback" not in op.lower(),
                level=SafetyLevel.LOW,  # Warn only
                allow_override=True,
            )
        )


def create_container_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_privileged_containers: bool = True,
    block_root_user: bool = True,
    require_health_checks: bool = False,
) -> None:
    """Register container-specific safety rules.

    Common pattern for Docker/Kubernetes container operations.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_privileged_containers: Block privileged container creation
        block_root_user: Block containers running as root user
        require_health_checks: Require health checks in container definitions

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_container_safety_rules(
            enforcer,
            block_privileged_containers=True,
            block_root_user=True
        )
    """
    if block_privileged_containers:
        enforcer.add_rule(
            SafetyRule(
                name="container_block_privileged",
                description="Block privileged container creation",
                check_fn=lambda op: ("--privileged" in op or "privileged: true" in op.lower()),
                level=SafetyLevel.HIGH,
                allow_override=False,
            )
        )

    if block_root_user:
        enforcer.add_rule(
            SafetyRule(
                name="container_block_root",
                description="Block containers running as root user",
                check_fn=lambda op: ("user: 0" in op or "USER root" in op or "--user 0" in op),
                level=SafetyLevel.HIGH,
                allow_override=True,
            )
        )

    if require_health_checks:
        enforcer.add_rule(
            SafetyRule(
                name="container_require_healthcheck",
                description="Require health checks in container definitions",
                check_fn=lambda op: ("docker" in op.lower() or "kubernetes" in op.lower())
                and "healthcheck" not in op.lower()
                and "readinessprobe" not in op.lower()
                and "livenessprobe" not in op.lower(),
                level=SafetyLevel.LOW,  # Warn only
                allow_override=True,
            )
        )


def create_infrastructure_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_destructive_commands: bool = True,
    require_state_backup: bool = True,
    protected_resources: Optional[List[str]] = None,
) -> None:
    """Register infrastructure-specific safety rules.

    Common pattern for Terraform/Kubernetes infrastructure operations.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_destructive_commands: Block destructive infrastructure commands
        require_state_backup: Require Terraform state backup before modification
        protected_resources: List of protected resources
                             (default: ["database", "storage", "vpc"])

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_infrastructure_safety_rules(
            enforcer,
            block_destructive_commands=True,
            protected_resources=["database", "storage", "vpc", "load-balancer"]
        )
    """
    protected = protected_resources or ["database", "storage", "vpc"]

    if block_destructive_commands:
        enforcer.add_rule(
            SafetyRule(
                name="infra_block_destructive",
                description="Block destructive infrastructure commands",
                check_fn=lambda op: any(
                    cmd in op
                    for cmd in [
                        "terraform destroy",
                        "kubectl delete",
                        "helm uninstall",
                        "aws cloudformation delete-stack",
                        "gcloud deployment-manager delete",
                    ]
                ),
                level=SafetyLevel.HIGH,
                allow_override=False,
            )
        )

        enforcer.add_rule(
            SafetyRule(
                name="infra_block_protected_resource_deletion",
                description=f"Block deletion of protected resources: {', '.join(protected)}",
                check_fn=lambda op: ("delete" in op.lower() or "destroy" in op.lower())
                and any(resource in op.lower() for resource in protected),
                level=SafetyLevel.HIGH,
                allow_override=True,
            )
        )

    if require_state_backup:
        enforcer.add_rule(
            SafetyRule(
                name="infra_require_state_backup",
                description="Require Terraform state backup before modification",
                check_fn=lambda op: "terraform" in op.lower()
                and any(cmd in op for cmd in ["apply", "destroy"])
                and "backup" not in op.lower(),
                level=SafetyLevel.MEDIUM,
                allow_override=True,
            )
        )


def create_pii_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_pii_exports: bool = True,
    warn_on_pii_columns: bool = True,
    require_anonymization: bool = False,
) -> None:
    """Register PII (Personally Identifiable Information) safety rules.

    Common pattern for DataAnalysis and RAG verticals.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_pii_exports: Block exporting data containing PII
        warn_on_pii_columns: Warn when PII columns are detected
        require_anonymization: Require anonymization for PII data

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_pii_safety_rules(
            enforcer,
            block_pii_exports=True,
            warn_on_pii_columns=True
        )
    """
    if block_pii_exports:
        enforcer.add_rule(
            SafetyRule(
                name="pii_block_export",
                description="Block exporting data containing PII (SSN, credit card, password, medical)",
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
                name="pii_warn_columns",
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
                name="pii_require_anonymization",
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


def create_source_credibility_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_low_credibility_sources: bool = True,
    require_source_verification: bool = False,
    blocked_domains: Optional[List[str]] = None,
) -> None:
    """Register research source credibility safety rules.

    Common pattern for Research vertical.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_low_credibility_sources: Block sources from low-credibility domains
        require_source_verification: Require verification for non-.edu/.gov sources
        blocked_domains: List of specific domains to block

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.MEDIUM))
        create_source_credibility_safety_rules(
            enforcer,
            block_low_credibility_sources=True,
            blocked_domains=["fake-news-site.com"]
        )
    """
    if block_low_credibility_sources:
        enforcer.add_rule(
            SafetyRule(
                name="source_block_low_credibility",
                description="Block low-credibility sources (.blogspot, tumblr, etc.)",
                check_fn=lambda op: any(
                    domain in op.lower()
                    for domain in [
                        ".blogspot.",
                        "blogspot.com",
                        "wordpress.com/",
                        "tumblr.com",
                        "fake-news",
                        "conspiracy",
                    ]
                ),
                level=SafetyLevel.MEDIUM,
                allow_override=True,
            )
        )

    if require_source_verification:
        enforcer.add_rule(
            SafetyRule(
                name="source_require_verification",
                description="Require verification for non-.edu/.gov sources",
                check_fn=lambda op: ("cite" in op.lower() or "source" in op.lower())
                and not any(
                    domain in op.lower() for domain in [".edu", ".gov", "arxiv.org", "doi.org"]
                ),
                level=SafetyLevel.LOW,  # Warn only
                allow_override=True,
            )
        )

    if blocked_domains:
        for domain in blocked_domains:
            enforcer.add_rule(
                SafetyRule(
                    name=f"source_block_{domain.replace('.', '_')}",
                    description=f"Block sources from {domain}",
                    check_fn=lambda op, d=domain: d in op.lower(),
                    level=SafetyLevel.HIGH,
                    allow_override=False,
                )
            )


def create_content_quality_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_fabricated_content: bool = True,
    warn_absolute_claims: bool = True,
    block_plagiarism_risk: bool = True,
) -> None:
    """Register research content quality safety rules.

    Common pattern for Research vertical.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_fabricated_content: Block fabricating sources/citations
        warn_absolute_claims: Warn about absolute claims (always, never, proven)
        block_plagiarism_risk: Block plagiarism risks

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_content_quality_safety_rules(
            enforcer,
            block_fabricated_content=True,
            warn_absolute_claims=True
        )
    """
    if block_fabricated_content:
        enforcer.add_rule(
            SafetyRule(
                name="content_block_fabricated",
                description="Block fabricating sources or citations",
                check_fn=lambda op: any(
                    phrase in op.lower()
                    for phrase in [
                        "fake source",
                        "fabricate",
                        "invent citation",
                        "fake citation",
                    ]
                ),
                level=SafetyLevel.HIGH,
                allow_override=False,
            )
        )

    if warn_absolute_claims:
        enforcer.add_rule(
            SafetyRule(
                name="content_warn_absolute_claims",
                description="Warn about absolute claims (always, never, proven)",
                check_fn=lambda op: any(
                    phrase in op.lower()
                    for phrase in [
                        "always ",
                        "never ",
                        "guaranteed",
                        "proven fact",
                        "everyone knows",
                    ]
                ),
                level=SafetyLevel.LOW,  # Warn only
                allow_override=True,
            )
        )

    if block_plagiarism_risk:
        enforcer.add_rule(
            SafetyRule(
                name="content_warn_plagiarism",
                description="Warn about plagiarism risk",
                check_fn=lambda op: any(
                    phrase in op.lower()
                    for phrase in [
                        "plagiarism",
                        "plagiarize",
                        "copy without attribution",
                        "unattributed",
                    ]
                ),
                level=SafetyLevel.MEDIUM,
                allow_override=True,
            )
        )


def create_bulk_operation_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_bulk_delete: bool = True,
    block_delete_all: bool = True,
    protected_collections: Optional[List[str]] = None,
) -> None:
    """Register bulk operation safety rules.

    Common pattern for RAG vertical (bulk document operations).

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_bulk_delete: Block bulk deletion operations with wildcards
        block_delete_all: Block deleting all documents
        protected_collections: List of protected collection names

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_bulk_operation_safety_rules(
            enforcer,
            block_bulk_delete=True,
            protected_collections=["production", "main"]
        )
    """
    if block_bulk_delete:
        enforcer.add_rule(
            SafetyRule(
                name="bulk_block_delete_wildcard",
                description="Block bulk deletion with wildcards (*, --all)",
                check_fn=lambda op: "delete" in op
                and ("*" in op or "--all" in op or "WHERE 1=1" in op),
                level=SafetyLevel.HIGH,
                allow_override=False,
            )
        )

    if block_delete_all:
        enforcer.add_rule(
            SafetyRule(
                name="bulk_block_delete_all",
                description="Block deleting all documents/records",
                check_fn=lambda op: "delete" in op and "--all" in op,
                level=SafetyLevel.HIGH,
                allow_override=True,
            )
        )


def create_ingestion_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_executable_files: bool = True,
    block_system_files: bool = True,
    require_https: bool = True,
    warn_large_batches: bool = True,
) -> None:
    """Register ingestion safety rules.

    Common pattern for RAG vertical (document ingestion).

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_executable_files: Block ingestion of executable files (.exe, .dll, .sh)
        block_system_files: Block ingestion of system files (/etc/, ~/.ssh/)
        require_https: Require HTTPS URLs for ingestion
        warn_large_batches: Warn for large batch ingestion (1000+ files)

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_ingestion_safety_rules(
            enforcer,
            block_executable_files=True,
            require_https=True
        )
    """
    if block_executable_files:
        enforcer.add_rule(
            SafetyRule(
                name="ingestion_block_executable",
                description="Block ingestion of executable files",
                check_fn=lambda op: "ingest" in op.lower()
                and any(
                    ext in op.lower() for ext in [".exe", ".dll", ".bat", ".cmd", ".sh", ".ps1"]
                ),
                level=SafetyLevel.HIGH,
                allow_override=False,
            )
        )

    if block_system_files:
        enforcer.add_rule(
            SafetyRule(
                name="ingestion_block_system_files",
                description="Block ingestion of system files (/etc/, ~/.ssh/)",
                check_fn=lambda op: "ingest" in op.lower()
                and any(path in op for path in ["/etc/", "/.ssh/", "~/.ssh/", "passwd", "shadow"]),
                level=SafetyLevel.HIGH,
                allow_override=True,
            )
        )

    if require_https:
        enforcer.add_rule(
            SafetyRule(
                name="ingestion_require_https",
                description="Require HTTPS for remote ingestion (block http://)",
                check_fn=lambda op: "ingest" in op.lower()
                and "http://" in op
                and "https://" not in op
                and "localhost" not in op,
                level=SafetyLevel.MEDIUM,
                allow_override=True,
            )
        )

    if warn_large_batches:
        enforcer.add_rule(
            SafetyRule(
                name="ingestion_warn_large_batches",
                description="Warn for large batch ingestion (1000+ files)",
                check_fn=lambda op: "ingest" in op.lower() and "--batch" in op,
                level=SafetyLevel.LOW,  # Warn only
                allow_override=True,
            )
        )


def create_data_export_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_external_uploads: bool = True,
    block_production_db_access: bool = True,
    require_encryption: bool = True,
) -> None:
    """Register data export safety rules.

    Common pattern for DataAnalysis vertical.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_external_uploads: Block uploading data externally
        block_production_db_access: Block direct production database access
        require_encryption: Require encryption for data exports

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_data_export_safety_rules(
            enforcer,
            block_external_uploads=True,
            require_encryption=True
        )
    """
    if block_external_uploads:
        enforcer.add_rule(
            SafetyRule(
                name="export_block_external_uploads",
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
                name="export_block_production_db",
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
                name="export_require_encryption",
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


# =============================================================================
# Convenience Functions
# =============================================================================


def create_all_common_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    include_git: bool = True,
    include_file_operations: bool = True,
    include_deployment: bool = False,
    include_containers: bool = False,
    include_infrastructure: bool = False,
    include_pii: bool = False,
    include_source_credibility: bool = False,
    include_content_quality: bool = False,
    include_bulk_operations: bool = False,
    include_ingestion: bool = False,
    include_data_export: bool = False,
    **kwargs,
) -> None:
    """Register all common safety rules at once.

    This is a convenience function that registers common safety rules
    across all verticals. Vertical-specific rules can be added separately.

    Args:
        enforcer: SafetyEnforcer to register rules with
        include_git: Include git safety rules
        include_file_operations: Include file operation safety rules
        include_deployment: Include deployment safety rules
        include_containers: Include container safety rules
        include_infrastructure: Include infrastructure safety rules
        include_pii: Include PII safety rules
        include_source_credibility: Include source credibility rules
        include_content_quality: Include content quality rules
        include_bulk_operations: Include bulk operation rules
        include_ingestion: Include ingestion safety rules
        include_data_export: Include data export safety rules
        **kwargs: Additional keyword arguments passed to specific rule creators

    Example:
        from victor.framework.config import SafetyEnforcer, SafetyConfig, SafetyLevel

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_all_common_safety_rules(
            enforcer,
            include_git=True,
            include_file_operations=True,
            include_pii=True,
            protected_branches=["main", "master"],
        )
    """
    if include_git:
        create_git_safety_rules(
            enforcer,
            protected_branches=kwargs.get("git_protected_branches"),
        )

    if include_file_operations:
        create_file_operation_safety_rules(
            enforcer,
            protected_patterns=kwargs.get("file_protected_patterns"),
        )

    if include_deployment:
        create_deployment_safety_rules(
            enforcer,
            protected_environments=kwargs.get("deployment_protected_environments"),
        )

    if include_containers:
        create_container_safety_rules(enforcer)

    if include_infrastructure:
        create_infrastructure_safety_rules(
            enforcer,
            protected_resources=kwargs.get("infrastructure_protected_resources"),
        )

    if include_pii:
        create_pii_safety_rules(enforcer)

    if include_source_credibility:
        create_source_credibility_safety_rules(
            enforcer,
            blocked_domains=kwargs.get("source_blocked_domains"),
        )

    if include_content_quality:
        create_content_quality_safety_rules(enforcer)

    if include_bulk_operations:
        create_bulk_operation_safety_rules(
            enforcer,
            protected_collections=kwargs.get("bulk_protected_collections"),
        )

    if include_ingestion:
        create_ingestion_safety_rules(enforcer)

    if include_data_export:
        create_data_export_safety_rules(enforcer)


__all__ = [
    # Git safety
    "create_git_safety_rules",
    # File operations
    "create_file_operation_safety_rules",
    # DevOps infrastructure
    "create_deployment_safety_rules",
    "create_container_safety_rules",
    "create_infrastructure_safety_rules",
    # Data privacy
    "create_pii_safety_rules",
    # Research
    "create_source_credibility_safety_rules",
    "create_content_quality_safety_rules",
    # RAG
    "create_bulk_operation_safety_rules",
    "create_ingestion_safety_rules",
    # Data analysis
    "create_data_export_safety_rules",
    # Convenience
    "create_all_common_safety_rules",
]
