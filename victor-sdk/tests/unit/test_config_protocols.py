# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for promoted config protocols — TDD."""

from __future__ import annotations

import pytest

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
