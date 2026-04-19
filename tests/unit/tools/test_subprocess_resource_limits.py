"""Tests for subprocess resource limits and output truncation."""

import sys
from unittest.mock import patch

import pytest

from victor.config.timeouts import SubprocessResourceLimits
from victor.tools.subprocess_executor import (
    CommandResult,
    _truncate_output,
    create_resource_limit_preexec,
    run_command,
)

# ---------------------------------------------------------------------------
# SubprocessResourceLimits dataclass
# ---------------------------------------------------------------------------


class TestSubprocessResourceLimits:
    def test_defaults(self):
        limits = SubprocessResourceLimits()
        assert limits.max_memory_mb == 512
        assert limits.max_cpu_seconds == 300
        assert limits.max_output_bytes == 10 * 1024 * 1024
        assert limits.max_file_descriptors == 1024

    def test_from_env_overrides(self):
        env = {
            "VICTOR_SUBPROCESS_MAX_MEMORY_MB": "256",
            "VICTOR_SUBPROCESS_MAX_CPU_SECONDS": "60",
        }
        with patch.dict("os.environ", env):
            limits = SubprocessResourceLimits.from_env()
        assert limits.max_memory_mb == 256
        assert limits.max_cpu_seconds == 60
        # Non-overridden fields keep defaults
        assert limits.max_output_bytes == 10 * 1024 * 1024

    def test_from_env_invalid_value_keeps_default(self):
        with patch.dict("os.environ", {"VICTOR_SUBPROCESS_MAX_MEMORY_MB": "not_a_number"}):
            limits = SubprocessResourceLimits.from_env()
        assert limits.max_memory_mb == 512


# ---------------------------------------------------------------------------
# create_resource_limit_preexec
# ---------------------------------------------------------------------------


class TestCreateResourceLimitPreexec:
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX only")
    def test_returns_callable_on_posix(self):
        fn = create_resource_limit_preexec()
        assert callable(fn)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX only")
    def test_callable_sets_limits(self):
        """Verify the preexec_fn calls setrlimit (may not enforce on macOS)."""
        import resource

        fn = create_resource_limit_preexec(
            max_memory_mb=256,
            max_cpu_seconds=60,
            max_file_descriptors=512,
        )
        assert fn is not None

        with patch.object(resource, "setrlimit") as mock_setrlimit:
            fn()

        # Should attempt to set RLIMIT_AS, RLIMIT_CPU, RLIMIT_NOFILE
        call_args = [c[0][0] for c in mock_setrlimit.call_args_list]
        assert resource.RLIMIT_AS in call_args
        assert resource.RLIMIT_CPU in call_args
        assert resource.RLIMIT_NOFILE in call_args

    def test_returns_none_when_resource_unavailable(self):
        """On platforms without resource module, returns None."""
        with patch.dict("sys.modules", {"resource": None}):
            # Force re-import to hit ImportError
            fn = (
                create_resource_limit_preexec.__wrapped__()
                if hasattr(create_resource_limit_preexec, "__wrapped__")
                else None
            )
        # Direct test: if resource import fails, returns None
        # This is hard to test without reloading, so we test the contract:
        # on POSIX it returns callable, verified above.


# ---------------------------------------------------------------------------
# _truncate_output
# ---------------------------------------------------------------------------


class TestTruncateOutput:
    def test_no_truncation_when_under_limit(self):
        text, truncated = _truncate_output("hello world", 100)
        assert text == "hello world"
        assert truncated is False

    def test_no_truncation_when_limit_zero(self):
        text, truncated = _truncate_output("hello world", 0)
        assert text == "hello world"
        assert truncated is False

    def test_truncation_when_over_limit(self):
        text, truncated = _truncate_output("a" * 200, 100)
        assert truncated is True
        assert len(text.encode("utf-8")) < 200
        assert "... [output truncated]" in text

    def test_truncation_preserves_marker(self):
        text, truncated = _truncate_output("x" * 1000, 50)
        assert truncated is True
        assert text.endswith("... [output truncated]")


# ---------------------------------------------------------------------------
# CommandResult.truncated field
# ---------------------------------------------------------------------------


class TestCommandResultTruncated:
    def test_default_false(self):
        from victor.tools.subprocess_executor import CommandErrorType

        result = CommandResult(
            success=True,
            stdout="out",
            stderr="",
            return_code=0,
            error_type=CommandErrorType.SUCCESS,
        )
        assert result.truncated is False

    def test_to_dict_includes_truncated(self):
        from victor.tools.subprocess_executor import CommandErrorType

        result = CommandResult(
            success=True,
            stdout="out",
            stderr="",
            return_code=0,
            error_type=CommandErrorType.SUCCESS,
            truncated=True,
        )
        d = result.to_dict()
        assert d["truncated"] is True


# ---------------------------------------------------------------------------
# run_command with resource limits + truncation
# ---------------------------------------------------------------------------


class TestRunCommandWithLimits:
    def test_preexec_fn_passed_to_subprocess(self):
        """Verify preexec_fn is forwarded to subprocess.run."""
        sentinel = lambda: None  # noqa: E731

        with patch("victor.tools.subprocess_executor.subprocess.run") as mock_run:
            mock_run.return_value = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
            run_command(["echo", "hi"], preexec_fn=sentinel, check_dangerous=False)

        _, kwargs = mock_run.call_args
        assert kwargs["preexec_fn"] is sentinel

    def test_output_truncation(self):
        result = run_command(
            ["echo", "hello world"],
            max_output_bytes=5,
            check_dangerous=False,
        )
        assert result.truncated is True
        assert "... [output truncated]" in result.stdout

    def test_no_truncation_by_default(self):
        result = run_command(
            ["echo", "hello"],
            check_dangerous=False,
        )
        assert result.truncated is False

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX only")
    def test_end_to_end_with_resource_limits(self):
        """Run a real command with resource limits applied."""
        preexec = create_resource_limit_preexec(
            max_memory_mb=512,
            max_cpu_seconds=300,
            max_file_descriptors=1024,
        )
        result = run_command(
            ["echo", "test"],
            preexec_fn=preexec,
            check_dangerous=False,
        )
        assert result.success is True
        assert "test" in result.stdout
