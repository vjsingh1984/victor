"""Tests for the UI-ephemeral tool progress sink."""

import victor.framework.tool_progress as tp


def teardown_function(_fn):
    tp.clear_progress_sink()


def test_emit_is_noop_without_sink():
    tp.clear_progress_sink()
    assert tp.has_progress_sink() is False
    # Should not raise and should do nothing.
    tp.emit_tool_progress(name="shell", stdout="hi")


def test_sink_receives_emissions():
    seen = []
    tp.set_progress_sink(lambda **kw: seen.append(kw))
    assert tp.has_progress_sink() is True

    tp.emit_tool_progress(name="shell", stdout="line\n", progress=0.5)
    assert seen == [
        {
            "name": "shell",
            "stdout": "line\n",
            "stderr": "",
            "progress": 0.5,
            "is_final": False,
        }
    ]


def test_clear_stops_delivery():
    seen = []
    tp.set_progress_sink(lambda **kw: seen.append(kw))
    tp.clear_progress_sink()
    tp.emit_tool_progress(name="shell", stdout="ignored")
    assert seen == []


def test_failing_sink_is_swallowed():
    def _boom(**_kw):
        raise RuntimeError("renderer blew up")

    tp.set_progress_sink(_boom)
    # Must not propagate — a UI failure can never break tool execution.
    tp.emit_tool_progress(name="shell", stdout="x")
