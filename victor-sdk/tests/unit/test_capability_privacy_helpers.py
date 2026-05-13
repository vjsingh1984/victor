"""Tests for SDK privacy capability helpers."""

from victor_sdk.capabilities import configure_data_privacy, get_privacy_config


class Orchestrator:
    pass


def test_configure_data_privacy_stores_sdk_config_on_orchestrator() -> None:
    orchestrator = Orchestrator()

    configure_data_privacy(
        orchestrator,
        anonymize_pii=False,
        pii_columns=["email"],
        hash_identifiers=False,
        log_access=False,
        detect_secrets=False,
        secret_patterns=["secret"],
    )

    assert get_privacy_config(orchestrator) == {
        "anonymize_pii": False,
        "pii_columns": ["email"],
        "hash_identifiers": False,
        "log_access": False,
        "detect_secrets": False,
        "secret_patterns": ["secret"],
    }


def test_get_privacy_config_returns_defaults() -> None:
    config = get_privacy_config(Orchestrator())

    assert config["anonymize_pii"] is True
    assert config["pii_columns"] == []
    assert config["hash_identifiers"] is True
    assert config["log_access"] is True
    assert config["detect_secrets"] is True
    assert config["secret_patterns"]
