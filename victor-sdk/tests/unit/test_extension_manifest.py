"""Tests for ExtensionManifest and ExtensionType."""

from victor_sdk.verticals.manifest import ExtensionManifest, ExtensionType


class TestExtensionType:
    def test_enum_values(self):
        assert ExtensionType.SAFETY == "safety"
        assert ExtensionType.TOOLS == "tool_dependencies"
        assert ExtensionType.MIDDLEWARE == "middleware"

    def test_all_types_are_strings(self):
        for member in ExtensionType:
            assert isinstance(member.value, str)


class TestExtensionManifest:
    def test_defaults(self):
        m = ExtensionManifest()
        assert m.api_version == 1
        assert m.name == ""
        assert m.version == "0.0.0"
        assert m.min_framework_version is None
        assert m.provides == set()
        assert m.requires == set()

    def test_is_provider(self):
        m = ExtensionManifest(provides={ExtensionType.SAFETY, ExtensionType.TOOLS})
        assert m.is_provider(ExtensionType.SAFETY)
        assert m.is_provider(ExtensionType.TOOLS)
        assert not m.is_provider(ExtensionType.WORKFLOWS)

    def test_has_requirement(self):
        m = ExtensionManifest(requires={ExtensionType.MIDDLEWARE})
        assert m.has_requirement(ExtensionType.MIDDLEWARE)
        assert not m.has_requirement(ExtensionType.SAFETY)

    def test_unmet_requirements_all_met(self):
        m = ExtensionManifest(requires={ExtensionType.SAFETY, ExtensionType.TOOLS})
        available = {ExtensionType.SAFETY, ExtensionType.TOOLS, ExtensionType.MIDDLEWARE}
        assert m.unmet_requirements(available) == set()

    def test_unmet_requirements_some_missing(self):
        m = ExtensionManifest(requires={ExtensionType.SAFETY, ExtensionType.WORKFLOWS})
        available = {ExtensionType.SAFETY}
        assert m.unmet_requirements(available) == {ExtensionType.WORKFLOWS}

    def test_custom_manifest(self):
        m = ExtensionManifest(
            api_version=2,
            name="coding",
            version="1.2.0",
            min_framework_version="0.5.0",
            provides={ExtensionType.SAFETY, ExtensionType.TOOLS},
            requires={ExtensionType.MIDDLEWARE},
        )
        assert m.api_version == 2
        assert m.name == "coding"
        assert m.version == "1.2.0"
        assert m.min_framework_version == "0.5.0"


class TestVerticalBaseGetManifest:
    """Test that VerticalBase.get_manifest() auto-builds from overridden methods."""

    def test_concrete_vertical_produces_valid_manifest(self):
        from victor_sdk.verticals.protocols.base import VerticalBase

        class MyVertical(VerticalBase):
            @classmethod
            def get_name(cls):
                return "my-vertical"

            @classmethod
            def get_description(cls):
                return "Test vertical"

            @classmethod
            def get_tools(cls):
                return ["read", "write"]

            @classmethod
            def get_system_prompt(cls):
                return "You are helpful."

        manifest = MyVertical.get_manifest()
        assert manifest.name == "my-vertical"
        assert manifest.api_version == 2
        assert ExtensionType.TOOLS in manifest.provides

    def test_manifest_detects_overridden_methods(self):
        from victor_sdk.verticals.protocols.base import VerticalBase

        class SafeVertical(VerticalBase):
            @classmethod
            def get_name(cls):
                return "safe"

            @classmethod
            def get_description(cls):
                return "Safe vertical"

            @classmethod
            def get_tools(cls):
                return ["read"]

            @classmethod
            def get_system_prompt(cls):
                return "prompt"

            @classmethod
            def get_capability_requirements(cls):
                return [{"capability_id": "code_execution"}]

        manifest = SafeVertical.get_manifest()
        assert ExtensionType.CAPABILITIES in manifest.provides
