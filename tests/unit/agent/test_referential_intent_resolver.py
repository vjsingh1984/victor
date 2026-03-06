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

"""Tests for ReferentialIntentResolver."""

import re
import pytest

from victor.agent.referential_intent_resolver import ReferentialIntentResolver
from victor.agent.session_ledger import SessionLedger
from victor.config.orchestrator_constants import ReferentialIntentConfig


class TestReferentialIntentConfig:
    def test_defaults(self):
        config = ReferentialIntentConfig()
        assert config.enabled is True
        assert config.max_enrichment_chars == 1500


class TestIsReferential:
    def test_do_it_yes(self):
        resolver = ReferentialIntentResolver()
        assert resolver.is_referential("do it") is True

    def test_do_it_in_sentence(self):
        resolver = ReferentialIntentResolver()
        assert resolver.is_referential("yes, do it please") is True

    def test_update_them_yes(self):
        resolver = ReferentialIntentResolver()
        assert resolver.is_referential("update them") is True

    def test_apply_changes_yes(self):
        resolver = ReferentialIntentResolver()
        assert resolver.is_referential("apply the changes") is True

    def test_as_recommended(self):
        resolver = ReferentialIntentResolver()
        assert resolver.is_referential("update as recommended") is True

    def test_go_ahead(self):
        resolver = ReferentialIntentResolver()
        assert resolver.is_referential("go ahead") is True

    def test_proceed(self):
        resolver = ReferentialIntentResolver()
        assert resolver.is_referential("proceed") is True

    def test_hello_no(self):
        resolver = ReferentialIntentResolver()
        assert resolver.is_referential("hello") is False

    def test_long_message_no(self):
        resolver = ReferentialIntentResolver()
        # Long detailed messages are not referential
        assert resolver.is_referential("Please do it. " * 20) is False

    def test_disabled_config(self):
        resolver = ReferentialIntentResolver(config=ReferentialIntentConfig(enabled=False))
        assert resolver.is_referential("do it") is False


class TestEnrichWithLedger:
    def test_context_appended(self):
        ledger = SessionLedger()
        ledger.record_recommendation("Refactor the module into smaller classes", turn_index=1)
        ledger.record_decision("Use the strategy pattern", turn_index=2)
        resolver = ReferentialIntentResolver(session_ledger=ledger)
        result = resolver.enrich("do it")
        assert "[Context:" in result
        assert "recent items" in result

    def test_max_chars_respected(self):
        ledger = SessionLedger()
        for i in range(50):
            ledger.record_recommendation(
                f"Very long recommendation number {i} with lots of detail", turn_index=i
            )
        config = ReferentialIntentConfig(max_enrichment_chars=200)
        resolver = ReferentialIntentResolver(config=config, session_ledger=ledger)
        result = resolver.enrich("do it")
        # The enrichment portion should be limited
        enrichment = result[len("do it") :]
        assert len(enrichment) <= 210  # Allow small overshoot


class TestEnrichWithoutLedger:
    def test_returns_message_unchanged(self):
        resolver = ReferentialIntentResolver()
        assert resolver.enrich("do it") == "do it"


class TestEnrichNonReferential:
    def test_returns_message_unchanged(self):
        ledger = SessionLedger()
        ledger.record_recommendation("something important", turn_index=1)
        resolver = ReferentialIntentResolver(session_ledger=ledger)
        assert resolver.enrich("What is the weather?") == "What is the weather?"


class TestCustomPatterns:
    def test_config_override_works(self):
        custom_patterns = [re.compile(r"\bexecute\b", re.IGNORECASE)]
        resolver = ReferentialIntentResolver(patterns=custom_patterns)
        assert resolver.is_referential("execute") is True
        assert resolver.is_referential("do it") is False  # Not in custom patterns
