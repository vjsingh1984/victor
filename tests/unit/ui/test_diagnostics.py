"""Tests for victor.ui.diagnostics (faulthandler-based hang diagnostics)."""

from __future__ import annotations

import io
import subprocess
import sys
import textwrap

import pytest

from victor.ui import diagnostics


@pytest.fixture(autouse=True)
def _reset_installed_flag(monkeypatch):
    """Each test starts with diagnostics not-yet-installed."""
    monkeypatch.setattr(diagnostics, "_installed", False)
    # Default to enabled unless a test overrides it.
    monkeypatch.delenv("VICTOR_FAULTHANDLER", raising=False)
    monkeypatch.delenv("VICTOR_FAULTHANDLER_TIMEOUT", raising=False)
    yield


def test_install_returns_true_with_real_fd_stream(tmp_path):
    log = tmp_path / "fh.log"
    with log.open("w") as fh:
        assert diagnostics.install_fault_diagnostics(stream=fh) is True


def test_install_is_idempotent(tmp_path):
    log = tmp_path / "fh.log"
    with log.open("w") as fh:
        assert diagnostics.install_fault_diagnostics(stream=fh) is True
        # Second call is a no-op (already installed for this process).
        assert diagnostics.install_fault_diagnostics(stream=fh) is False


def test_disabled_via_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VICTOR_FAULTHANDLER", "0")
    log = tmp_path / "fh.log"
    with log.open("w") as fh:
        assert diagnostics.install_fault_diagnostics(stream=fh) is False


@pytest.mark.parametrize("value", ["false", "No", "OFF"])
def test_disabled_accepts_truthy_variants(monkeypatch, tmp_path, value):
    monkeypatch.setenv("VICTOR_FAULTHANDLER", value)
    with (tmp_path / "fh.log").open("w") as fh:
        assert diagnostics.install_fault_diagnostics(stream=fh) is False


def test_stream_without_fileno_is_skipped_gracefully():
    # io.StringIO has no fileno(); must not raise and must report not-installed.
    assert diagnostics.install_fault_diagnostics(stream=io.StringIO()) is False


@pytest.mark.skipif(not hasattr(__import__("signal"), "SIGUSR1"), reason="SIGUSR1 unavailable")
def test_sigusr1_dumps_stack_in_subprocess():
    """End-to-end: a registered SIGUSR1 actually dumps a Python traceback.

    Run in a subprocess so the signal handler / stack dump never disturbs the
    pytest process, and so we exercise the real faulthandler path.
    """
    script = textwrap.dedent("""
        import os, signal, sys, time, threading
        from victor.ui import diagnostics

        assert diagnostics.install_fault_diagnostics(stream=sys.stderr) is True

        def _spin_marker_frame():
            # A uniquely named frame we can assert appears in the dump.
            os.kill(os.getpid(), signal.SIGUSR1)
            time.sleep(0.05)

        _spin_marker_frame()
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # faulthandler writes the dump to stderr; the marker frame must be named.
    assert "_spin_marker_frame" in proc.stderr
    assert "Current thread" in proc.stderr or "Thread" in proc.stderr
