# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""FEP-0023 Phase 3: the shared referential-intent enrichment seam.

``resolve_referential_intent`` is the single transform both the streaming and
non-streaming user-message add paths delegate to, so enrichment cannot drift.
These tests pin its contract: no-op when unwired, enrich when wired + referential,
passthrough for non-referential or on error.
"""

from types import SimpleNamespace

from victor.agent.referential_intent_resolver import (
    ReferentialIntentResolver,
    resolve_referential_intent,
)
from victor.agent.session_ledger import SessionLedger


def _orch_with_resolver(resolver):
    return SimpleNamespace(_referential_intent_resolver=resolver)


def _ledger_with_recommendation():
    ledger = SessionLedger()
    ledger.record_recommendation("refactor the config module into smaller files", turn_index=1)
    return ledger


class TestReferentialIntentSeam:
    def test_no_resolver_wired_is_passthrough(self):
        """Flag OFF (no resolver on the orchestrator) → message unchanged."""
        orch = SimpleNamespace()  # no _referential_intent_resolver attribute
        assert resolve_referential_intent(orch, "do it") == "do it"

        orch_none = _orch_with_resolver(None)
        assert resolve_referential_intent(orch_none, "do it") == "do it"

    def test_referential_message_is_enriched(self):
        resolver = ReferentialIntentResolver(session_ledger=_ledger_with_recommendation())
        orch = _orch_with_resolver(resolver)

        out = resolve_referential_intent(orch, "do it")
        assert out != "do it"
        assert "[Context:" in out
        assert "refactor the config module" in out

    def test_non_referential_message_unchanged(self):
        resolver = ReferentialIntentResolver(session_ledger=_ledger_with_recommendation())
        orch = _orch_with_resolver(resolver)

        msg = "How do I configure structured logging in this project?"
        assert resolve_referential_intent(orch, msg) == msg

    def test_empty_ledger_leaves_message_unchanged(self):
        """Referential trigger but nothing to point at → no enrichment."""
        resolver = ReferentialIntentResolver(session_ledger=SessionLedger())
        orch = _orch_with_resolver(resolver)
        assert resolve_referential_intent(orch, "apply the changes") == "apply the changes"

    def test_resolver_error_degrades_to_passthrough(self):
        class _Boom:
            def enrich(self, _message):
                raise RuntimeError("boom")

        orch = _orch_with_resolver(_Boom())
        assert resolve_referential_intent(orch, "do it") == "do it"


class TestResolverFactoryWiring:
    def test_factory_wires_resolver_to_ledger(self):
        from victor.agent.factory.coordination_builders import CoordinationBuildersMixin

        class _Builder(CoordinationBuildersMixin):
            pass

        ledger = _ledger_with_recommendation()
        resolver = _Builder().create_referential_intent_resolver(ledger=ledger)

        # The resolver reads the ledger it was built with.
        out = resolver.enrich("do it")
        assert "refactor the config module" in out
