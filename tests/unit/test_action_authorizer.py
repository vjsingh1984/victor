# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the action_authorizer module.

Tests intent detection for action authorization:
- Display-only intent detection
- Write-allowed intent detection
- Read-only intent detection
- Ambiguous intent handling
- Custom signals
- Safe actions filtering
"""


from victor.agent.action_authorizer import (
    ActionIntent,
    ActionAuthorizer,
    IntentClassification,
    IntentDetector,
    detect_intent,
    is_write_authorized,
    get_prompt_guard,
    get_safe_tools,
    SAFE_ACTIONS,
    PROMPT_GUARDS,
)


class TestActionIntent:
    """Tests for ActionIntent enum."""

    def test_intent_values(self):
        """Test all intent enum values exist."""
        assert ActionIntent.DISPLAY_ONLY.value == "display_only"
        assert ActionIntent.WRITE_ALLOWED.value == "write_allowed"
        assert ActionIntent.AMBIGUOUS.value == "ambiguous"
        assert ActionIntent.READ_ONLY.value == "read_only"


class TestIntentClassification:
    """Tests for IntentClassification dataclass."""

    def test_classification_creation(self):
        """Test creating an IntentClassification."""
        classification = IntentClassification(
            intent=ActionIntent.WRITE_ALLOWED,
            confidence=0.9,
            matched_signals=["save_to_file", "create_file"],
            safe_actions={"write_file", "read_file"},
            prompt_guard="",
        )
        assert classification.intent == ActionIntent.WRITE_ALLOWED
        assert classification.confidence == 0.9
        assert len(classification.matched_signals) == 2
        assert "write_file" in classification.safe_actions


class TestIntentDetector:
    """Tests for the IntentDetector class."""

    def test_display_only_detection(self):
        """Test detection of display-only signals."""
        detector = IntentDetector()

        # "show me" patterns
        result = detector.detect("Show me a function that calculates factorial")
        assert result.intent == ActionIntent.DISPLAY_ONLY
        assert "show_me" in result.matched_signals

        # "just show" patterns
        result = detector.detect("just show me the code")
        assert result.intent == ActionIntent.DISPLAY_ONLY

        # "display only" patterns
        result = detector.detect("display only, don't save")
        assert result.intent == ActionIntent.DISPLAY_ONLY

        # "without saving" patterns
        result = detector.detect("show me code without saving to disk")
        assert result.intent == ActionIntent.DISPLAY_ONLY

    def test_write_allowed_detection(self):
        """Test detection of write-allowed signals."""
        detector = IntentDetector()

        # "save to file" patterns
        result = detector.detect("save this to main.py")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "save_to_file" in result.matched_signals

        # "create file" patterns
        result = detector.detect("create a new file called utils.py")
        assert result.intent == ActionIntent.WRITE_ALLOWED

        # "update file" patterns
        result = detector.detect("update the file with this code")
        assert result.intent == ActionIntent.WRITE_ALLOWED

        # "add to file" patterns
        result = detector.detect("add this function into helpers.py")
        assert result.intent == ActionIntent.WRITE_ALLOWED

    def test_read_only_detection(self):
        """Test detection of read-only signals."""
        detector = IntentDetector()

        # "list files" patterns
        result = detector.detect("list all files in the directory")
        assert result.intent == ActionIntent.READ_ONLY
        assert "list_files" in result.matched_signals

        # "explain" patterns
        result = detector.detect("explain the file contents")
        assert result.intent == ActionIntent.READ_ONLY

        # "git status" patterns
        result = detector.detect("git status")
        assert result.intent == ActionIntent.READ_ONLY
        assert "git_read" in result.matched_signals

        # "summarize" patterns
        result = detector.detect("summarize this module")
        assert result.intent == ActionIntent.READ_ONLY

    def test_ambiguous_detection(self):
        """Test ambiguous intent detection."""
        detector = IntentDetector()

        # No clear signals - should fall back to default (safe)
        result = detector.detect("do something with the code")
        # Should be ambiguous, write, or display (safe default)
        assert result.intent in (
            ActionIntent.AMBIGUOUS,
            ActionIntent.WRITE_ALLOWED,
            ActionIntent.DISPLAY_ONLY,
        )

    def test_default_intent_display_only(self):
        """Test that default intent is safe (display_only)."""
        detector = IntentDetector()

        # No signals match
        result = detector.detect("hello world")
        assert result.intent == ActionIntent.DISPLAY_ONLY
        assert result.confidence < 0.5

    def test_custom_default_intent(self):
        """Test setting custom default intent."""
        detector = IntentDetector(default_intent=ActionIntent.AMBIGUOUS)

        result = detector.detect("something random")
        assert result.intent == ActionIntent.AMBIGUOUS

    def test_write_overrides_display_when_strong(self):
        """Test that strong write signal results in write-allowed."""
        detector = IntentDetector()

        # Clear write signal - should be write allowed
        result = detector.detect("create a new file called config.py")
        assert result.intent == ActionIntent.WRITE_ALLOWED

    def test_confidence_scoring(self):
        """Test confidence scores are reasonable."""
        detector = IntentDetector()

        # Clear signal should have high confidence
        result = detector.detect("show me a function")
        assert result.confidence > 0.5

        # No signal should have low confidence
        result = detector.detect("hello")
        assert result.confidence < 0.5

    def test_is_write_authorized(self):
        """Test the is_write_authorized convenience method."""
        detector = IntentDetector()

        assert detector.is_write_authorized("save this to file.py") is True
        assert detector.is_write_authorized("show me a function") is False
        assert detector.is_write_authorized("list files") is False

    def test_custom_display_signals(self):
        """Test adding custom display signals."""
        custom_signals = [
            (r"\bpreview\b", 1.0, "preview"),
            (r"\bdemonstrate\b", 0.9, "demonstrate"),
        ]
        detector = IntentDetector(custom_display_signals=custom_signals)

        result = detector.detect("preview the code")
        assert result.intent == ActionIntent.DISPLAY_ONLY
        assert "preview" in result.matched_signals

    def test_custom_write_signals(self):
        """Test adding custom write signals."""
        custom_signals = [
            (r"\bpersist\b", 1.0, "persist"),
            (r"\bcommit\s+to\s+disk\b", 0.9, "commit_to_disk"),
        ]
        detector = IntentDetector(custom_write_signals=custom_signals)

        result = detector.detect("persist this configuration")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "persist" in result.matched_signals

    def test_custom_detector_function(self):
        """Test custom detector functions take precedence."""

        def custom_detector(message: str):
            if "magic" in message.lower():
                return IntentClassification(
                    intent=ActionIntent.WRITE_ALLOWED,
                    confidence=1.0,
                    matched_signals=["magic_word"],
                    safe_actions=set(),
                    prompt_guard="",
                )
            return None

        detector = IntentDetector(custom_detectors=[custom_detector])

        result = detector.detect("use the magic word")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "magic_word" in result.matched_signals

    def test_safe_actions_for_each_intent(self):
        """Test that safe actions are correctly assigned for each intent."""
        detector = IntentDetector()

        # Display only should have read + generate but not write
        result = detector.detect("show me a function")
        assert "read_file" in result.safe_actions
        assert "generate_response" in result.safe_actions
        assert "write_file" not in result.safe_actions

        # Write allowed should have all actions
        result = detector.detect("save this to file.py")
        assert "write_file" in result.safe_actions
        assert "edit_file" in result.safe_actions

    def test_prompt_guard_assignment(self):
        """Test that prompt guards are correctly assigned."""
        detector = IntentDetector()

        result = detector.detect("show me a function")
        assert "IMPORTANT" in result.prompt_guard
        assert "write_file" in result.prompt_guard or "NOT" in result.prompt_guard

        result = detector.detect("save this to file.py")
        assert result.prompt_guard == ""  # No guard for write

    def test_case_insensitive_matching(self):
        """Test that pattern matching is case insensitive."""
        detector = IntentDetector()

        result1 = detector.detect("SHOW ME a function")
        result2 = detector.detect("show me a function")

        assert result1.intent == result2.intent

    def test_invalid_regex_pattern_handling(self):
        """Test that invalid regex patterns are handled gracefully."""
        # Invalid regex should not crash, just log warning
        custom_signals = [
            (r"[invalid", 1.0, "invalid_pattern"),  # Invalid regex
        ]
        detector = IntentDetector(custom_display_signals=custom_signals)

        # Should still work with valid patterns
        result = detector.detect("show me a function")
        assert result.intent == ActionIntent.DISPLAY_ONLY


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_detect_intent(self):
        """Test the detect_intent convenience function."""
        result = detect_intent("show me a function")
        assert isinstance(result, IntentClassification)
        assert result.intent == ActionIntent.DISPLAY_ONLY

    def test_is_write_authorized_function(self):
        """Test the is_write_authorized convenience function."""
        assert is_write_authorized("save this to file.py") is True
        assert is_write_authorized("show me a function") is False

    def test_get_prompt_guard(self):
        """Test the get_prompt_guard convenience function."""
        guard = get_prompt_guard("show me a function")
        assert "IMPORTANT" in guard

        guard = get_prompt_guard("save this to file.py")
        assert guard == ""

    def test_get_safe_tools(self):
        """Test the get_safe_tools convenience function."""
        all_tools = {"read_file", "write_file", "list_directory", "execute_bash", "code_search"}

        # Display only should filter out write
        safe = get_safe_tools("show me a function", all_tools)
        assert "read_file" in safe
        assert "write_file" not in safe

        # Write allowed should include write
        safe = get_safe_tools("save this to file.py", all_tools)
        assert "write_file" in safe


class TestSafeActionsMapping:
    """Tests for the SAFE_ACTIONS constant."""

    def test_all_intents_have_safe_actions(self):
        """Test that all intents have defined safe actions."""
        for intent in ActionIntent:
            assert intent in SAFE_ACTIONS
            assert isinstance(SAFE_ACTIONS[intent], set)

    def test_read_only_actions(self):
        """Test read-only has minimal actions."""
        actions = SAFE_ACTIONS[ActionIntent.READ_ONLY]
        assert "read_file" in actions
        assert "write_file" not in actions
        assert "execute_bash" not in actions

    def test_write_allowed_actions(self):
        """Test write-allowed has all actions."""
        actions = SAFE_ACTIONS[ActionIntent.WRITE_ALLOWED]
        assert "read_file" in actions
        assert "write_file" in actions
        assert "edit_file" in actions
        assert "execute_bash" in actions


class TestPromptGuards:
    """Tests for the PROMPT_GUARDS constant."""

    def test_all_intents_have_prompt_guards(self):
        """Test that all intents have defined prompt guards."""
        for intent in ActionIntent:
            assert intent in PROMPT_GUARDS

    def test_write_allowed_has_empty_guard(self):
        """Test that write-allowed has no guard."""
        assert PROMPT_GUARDS[ActionIntent.WRITE_ALLOWED] == ""

    def test_display_only_has_guard(self):
        """Test that display-only has protective guard."""
        guard = PROMPT_GUARDS[ActionIntent.DISPLAY_ONLY]
        assert "NOT" in guard or "Do not" in guard.lower()


class TestAlias:
    """Tests for the ActionAuthorizer alias."""

    def test_alias_works(self):
        """Test that ActionAuthorizer is an alias for IntentDetector."""
        assert ActionAuthorizer is IntentDetector

        authorizer = ActionAuthorizer()
        assert isinstance(authorizer, IntentDetector)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_message(self):
        """Test handling of empty message."""
        detector = IntentDetector()
        result = detector.detect("")
        assert result.intent == ActionIntent.DISPLAY_ONLY
        assert result.confidence < 0.5

    def test_very_long_message(self):
        """Test handling of very long message."""
        detector = IntentDetector()
        long_message = "show me " * 1000
        result = detector.detect(long_message)
        assert result.intent == ActionIntent.DISPLAY_ONLY

    def test_special_characters(self):
        """Test handling of special characters."""
        detector = IntentDetector()
        result = detector.detect("show me a function! @#$%^&*()")
        assert result.intent == ActionIntent.DISPLAY_ONLY

    def test_unicode_message(self):
        """Test handling of unicode characters."""
        detector = IntentDetector()
        result = detector.detect("show me a function with unicode")
        assert result.intent == ActionIntent.DISPLAY_ONLY

    def test_multiple_signals_accumulate(self):
        """Test that multiple matching signals accumulate score."""
        detector = IntentDetector()

        # Multiple display signals
        result = detector.detect("just show me an example, display only please")
        assert result.confidence >= 1.0  # High confidence from multiple matches
