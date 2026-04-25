# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for promoted config protocols — TDD."""

from __future__ import annotations


from victor_sdk.verticals.protocols.config import (
    ApiKeyProviderProtocol,
    ProjectPathsData,
    SettingsProviderProtocol,
)


class TestProjectPathsData:
    def test_basic_fields(self):
        p = ProjectPathsData(project_root="/home/user/project")
        assert p.project_root == "/home/user/project"

    def test_derived_victor_dir(self):
        p = ProjectPathsData(project_root="/home/user/project")
        assert p.victor_dir == "/home/user/project/.victor"

    def test_derived_logs_dir(self):
        p = ProjectPathsData(project_root="/home/user/project")
        assert p.logs_dir == "/home/user/project/.victor/logs"

    def test_custom_victor_dir(self):
        p = ProjectPathsData(
            project_root="/proj", victor_dir_name=".myvictor"
        )
        assert p.victor_dir == "/proj/.myvictor"

    def test_existing_properties(self):
        """All existing properties return correct paths."""
        paths = ProjectPathsData(project_root="/home/user/project")
        assert paths.embeddings_dir == "/home/user/project/.victor/embeddings"
        assert paths.graph_dir == "/home/user/project/.victor/graph"
        assert paths.sessions_dir == "/home/user/project/.victor/sessions"

    def test_backups_dir(self):
        paths = ProjectPathsData(project_root="/home/user/project")
        assert paths.backups_dir == "/home/user/project/.victor/backups"

    def test_changes_dir(self):
        paths = ProjectPathsData(project_root="/home/user/project")
        assert paths.changes_dir == "/home/user/project/.victor/changes"

    def test_conversation_db(self):
        paths = ProjectPathsData(project_root="/home/user/project")
        # Database consolidation: conversation_db is now an alias for project.db
        assert paths.conversation_db == "/home/user/project/.victor/project.db"

    def test_conversations_export_dir(self):
        paths = ProjectPathsData(project_root="/home/user/project")
        assert paths.conversations_export_dir == "/home/user/project/.victor/conversations"

    def test_index_metadata(self):
        paths = ProjectPathsData(project_root="/home/user/project")
        assert paths.index_metadata == "/home/user/project/.victor/index_metadata.json"

    def test_mcp_config(self):
        paths = ProjectPathsData(project_root="/home/user/project")
        assert paths.mcp_config == "/home/user/project/.victor/mcp.yaml"

    def test_project_context_file(self):
        paths = ProjectPathsData(project_root="/home/user/project")
        assert paths.project_context_file == "/home/user/project/.victor/init.md"

    def test_project_context_file_custom(self):
        paths = ProjectPathsData(
            project_root="/home/user/project",
            context_file_name="custom.md",
        )
        assert paths.project_context_file == "/home/user/project/.victor/custom.md"

    def test_custom_victor_dir_name_with_new_paths(self):
        paths = ProjectPathsData(
            project_root="/home/user/project",
            victor_dir_name=".custom_victor",
        )
        assert paths.backups_dir == "/home/user/project/.custom_victor/backups"
        # Database consolidation: conversation_db is now an alias for project.db
        assert paths.conversation_db == "/home/user/project/.custom_victor/project.db"


class TestSettingsProviderProtocol:
    def test_structural_check(self):
        class FakeSettings:
            def get_project_paths(self):
                return ProjectPathsData(project_root="/tmp")

            def get_setting(self, key, default=None):
                return default

        assert isinstance(FakeSettings(), SettingsProviderProtocol)


class TestApiKeyProviderProtocol:
    def test_structural_check(self):
        class FakeKeys:
            def get_service_key(self, service):
                return None

        assert isinstance(FakeKeys(), ApiKeyProviderProtocol)

    def test_rejects_incomplete(self):
        class NoMethods:
            pass

        assert not isinstance(NoMethods(), ApiKeyProviderProtocol)
