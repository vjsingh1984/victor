"""Tests for lossless prompt dictionary compression."""

from victor.framework.prompt_dictionary_compressor import compress_prompt_blocks


class TestPromptDictionaryCompressor:
    """Verify repeated boilerplate can be compressed losslessly."""

    def test_compresses_repeated_blocks_and_round_trips(self):
        repeated = (
            "Search first with the available code exploration tools before editing. "
            "Do not guess file paths, symbol locations, or repository structure."
        )
        blocks = [
            "You are a coding agent.",
            repeated,
            "Use evidence from tool output.",
            repeated,
        ]

        result = compress_prompt_blocks(blocks, min_block_chars=80, min_savings_chars=1)

        assert result.compressed is True
        assert result.saved_chars > 0
        assert repeated in result.compressed_prompt
        assert result.compressed_prompt.count(repeated) == 1
        assert "[[R1]]" in result.compressed_prompt
        assert result.expand() == "\n\n".join(blocks)

    def test_skips_compression_when_not_useful(self):
        blocks = [
            "Short note.",
            "Short note.",
            "Another short note.",
        ]

        result = compress_prompt_blocks(blocks, min_block_chars=40)

        assert result.compressed is False
        assert result.saved_chars == 0
        assert result.compressed_prompt == "\n\n".join(blocks)
        assert result.expand() == "\n\n".join(blocks)
