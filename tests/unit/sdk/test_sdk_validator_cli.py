"""Tests for the victor-sdk validation CLI."""

from __future__ import annotations

from types import SimpleNamespace

from victor_sdk.cli import main
from victor_sdk.validation import validate_vertical_package
from victor_sdk.verticals.manifest import ExtensionManifest
from victor_sdk.verticals.protocols.base import VerticalBase


class _FakeEntryPoint:
    def __init__(self, name: str, obj):
        self.group = "victor.plugins"
        self.name = name
        self._obj = obj

    def load(self):
        return self._obj


class _ValidVertical(VerticalBase):
    name = "valid_vertical"
    description = "Valid vertical"
    version = "1.2.3"
    _victor_manifest = ExtensionManifest(
        name="valid_vertical",
        version="1.2.3",
        api_version=1,
        min_framework_version=">=0.1.0",
    )

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    @classmethod
    def get_description(cls) -> str:
        return cls.description

    @classmethod
    def get_tools(cls):
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "valid prompt"


class _TooNewVertical(VerticalBase):
    name = "too_new_vertical"
    description = "Too new for installed core"
    version = "1.2.3"
    _victor_manifest = ExtensionManifest(
        name="too_new_vertical",
        version="1.2.3",
        api_version=1,
        min_framework_version=">=99.0.0",
    )

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    @classmethod
    def get_description(cls) -> str:
        return cls.description

    @classmethod
    def get_tools(cls):
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "too new prompt"


def _patch_distribution(monkeypatch, entry_points):
    fake_dist = SimpleNamespace(entry_points=entry_points, metadata={"Name": "victor-fake"})
    monkeypatch.setattr(
        "victor_sdk.validation.distribution",
        lambda package_name: fake_dist,
    )
    monkeypatch.setattr(
        "victor_sdk.validation.installed_version",
        lambda name: "0.6.1",
    )


def test_validate_vertical_package_accepts_valid_sdk_vertical(monkeypatch):
    """A package with a valid registered vertical should pass validation."""

    _patch_distribution(monkeypatch, [_FakeEntryPoint("valid", _ValidVertical)])

    report = validate_vertical_package("victor-fake")

    assert report.ok is True
    assert report.package_name == "victor-fake"
    assert report.verticals == ["valid_vertical"]
    assert report.issues == []


def test_validate_vertical_package_reports_framework_version_skew(monkeypatch):
    """Validator should fail packages whose manifests require a newer core."""

    _patch_distribution(monkeypatch, [_FakeEntryPoint("too_new", _TooNewVertical)])

    report = validate_vertical_package("victor-fake")

    assert report.ok is False
    assert any(issue.code == "framework_version_incompatible" for issue in report.issues)


def test_cli_returns_nonzero_for_invalid_package(monkeypatch, capsys):
    """The CLI should exit nonzero and print findings for invalid packages."""

    _patch_distribution(monkeypatch, [_FakeEntryPoint("too_new", _TooNewVertical)])

    exit_code = main(["check", "victor-fake"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "framework_version_incompatible" in output
