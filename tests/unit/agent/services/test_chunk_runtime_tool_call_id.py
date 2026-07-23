"""Regression test: the canonical ChunkGenerator facade must accept and forward
``tool_call_id`` — the streaming tool-execution path passes it (per-tool rows for
parallel batches), and dropping it crashed every tool call in interactive chat with
``TypeError: generate_tool_start_chunk() got an unexpected keyword argument``."""

from types import SimpleNamespace

from victor.agent.services.chunk_runtime import ChunkGenerator


class RecordingHandler:
    def __init__(self):
        self.calls = []

    def generate_tool_start_chunk(self, tool_name, tool_args, status_msg, tool_call_id=None):
        self.calls.append(
            {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "status_msg": status_msg,
                "tool_call_id": tool_call_id,
            }
        )
        return SimpleNamespace(metadata={"tool_start": {"name": tool_name}})


def make_generator(handler):
    generator = ChunkGenerator.__new__(ChunkGenerator)
    generator.streaming_handler = handler
    generator._event_bus = None
    return generator


def test_tool_call_id_is_accepted_and_forwarded():
    handler = RecordingHandler()
    generator = make_generator(handler)
    generator.generate_tool_start_chunk(
        "read_file", {"path": "x"}, "Reading…", tool_call_id="call_7"
    )
    assert handler.calls[0]["tool_call_id"] == "call_7"


def test_tool_call_id_defaults_to_none():
    handler = RecordingHandler()
    generator = make_generator(handler)
    generator.generate_tool_start_chunk("read_file", {"path": "x"}, "Reading…")
    assert handler.calls[0]["tool_call_id"] is None


def test_matches_streaming_caller_contract():
    """The exact call shape used by tool_execution._add_tool_start_chunks."""
    handler = RecordingHandler()
    generator = make_generator(handler)
    tool_call = {"name": "grep", "arguments": {"q": "x"}, "id": "call_9"}
    generator.generate_tool_start_chunk(
        tool_call.get("name", "tool"),
        tool_call.get("arguments", {}),
        "Searching…",
        tool_call_id=tool_call.get("id"),
    )
    assert handler.calls[0]["tool_call_id"] == "call_9"
