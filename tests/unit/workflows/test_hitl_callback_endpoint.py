"""End-to-end tests for the signed GET /hitl/respond callback link.

Exercises the full approve/reject round-trip via an in-process ASGI client
(single event loop, so the in-memory store's per-request asyncio primitives stay
loop-consistent). Covers the security properties: a tampered/missing token is
rejected, and replay of an already-decided request is rejected.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from victor.workflows.hitl import HITLNodeType, HITLRequest  # noqa: E402
from victor.workflows.hitl_api import HITLStore, create_hitl_router  # noqa: E402
from victor.workflows.hitl_signing import sign_action  # noqa: E402

SECRET = "endpoint-secret"


def _make_request(request_id: str) -> HITLRequest:
    return HITLRequest(
        request_id=request_id,
        node_id="n1",
        hitl_type=HITLNodeType.APPROVAL,
        prompt="Deploy?",
        timeout=300,
    )


def _app(store: HITLStore) -> FastAPI:
    app = FastAPI()
    app.include_router(create_hitl_router(store, require_auth=False), prefix="/hitl")
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_signed_approve_link_resolves_request(monkeypatch):
    monkeypatch.setenv("VICTOR_HITL_SIGNING_SECRET", SECRET)
    store = HITLStore()
    await store.store_request(_make_request("rid-approve"), workflow_id="wf")
    token = sign_action("rid-approve", "approve", secret=SECRET)

    async with _client(_app(store)) as ac:
        resp = await ac.get(f"/hitl/respond/rid-approve?action=approve&token={token}")

    assert resp.status_code == 200
    assert "approved" in resp.text.lower()
    stored = await store.get_request("rid-approve")
    assert stored.status != "pending"
    assert stored.response is not None and stored.response.approved is True


async def test_tampered_action_is_rejected(monkeypatch):
    monkeypatch.setenv("VICTOR_HITL_SIGNING_SECRET", SECRET)
    store = HITLStore()
    await store.store_request(_make_request("rid-tamper"), workflow_id="wf")
    token = sign_action("rid-tamper", "approve", secret=SECRET)  # approve-bound

    async with _client(_app(store)) as ac:
        resp = await ac.get(f"/hitl/respond/rid-tamper?action=reject&token={token}")

    assert resp.status_code == 403
    stored = await store.get_request("rid-tamper")
    assert stored.status == "pending"  # untouched


async def test_missing_token_rejected_when_secret_set(monkeypatch):
    monkeypatch.setenv("VICTOR_HITL_SIGNING_SECRET", SECRET)
    store = HITLStore()
    await store.store_request(_make_request("rid-x"), workflow_id="wf")

    async with _client(_app(store)) as ac:
        resp = await ac.get("/hitl/respond/rid-x?action=approve")

    assert resp.status_code == 403


async def test_replay_after_decision_is_rejected(monkeypatch):
    monkeypatch.setenv("VICTOR_HITL_SIGNING_SECRET", SECRET)
    store = HITLStore()
    await store.store_request(_make_request("rid-replay"), workflow_id="wf")
    token = sign_action("rid-replay", "approve", secret=SECRET)

    async with _client(_app(store)) as ac:
        first = await ac.get(f"/hitl/respond/rid-replay?action=approve&token={token}")
        second = await ac.get(f"/hitl/respond/rid-replay?action=approve&token={token}")

    assert first.status_code == 200
    assert second.status_code == 409  # already decided -> store rejects replay
