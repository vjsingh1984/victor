"""Unit tests for the SCM / ticketing / incident HITL transports.

Each transport is exercised against a fake aiohttp: ``send`` is checked for the
right request shape/reference, and ``poll`` for correct mapping of provider
responses to approve/reject. Network is never touched.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List

import pytest

from victor.workflows.hitl import HITLMode, HITLNodeType, HITLRequest, HITLStatus
from victor.workflows.hitl_transports import (
    GitHubConfig,
    GitHubDeploymentTransport,
    GitLabConfig,
    GitLabMRTransport,
    GitLabPipelineTransport,
    JiraConfig,
    JiraTransport,
    PagerDutyConfig,
    PagerDutyTransport,
    TerraformCloudConfig,
    TerraformCloudTransport,
    get_transport,
)


def _request(ctx=None):
    return HITLRequest(
        request_id="rid123456789",
        node_id="n1",
        hitl_type=HITLNodeType.APPROVAL,
        prompt="Deploy to prod?",
        context=ctx or {},
        timeout=300,
    )


def _install_fake_aiohttp(monkeypatch, handler):
    """handler(method, url) -> (status, payload). Records all calls."""
    calls: List[Dict[str, Any]] = []

    class _Resp:
        def __init__(self, status, payload):
            self.status, self._p = status, payload

        async def json(self):
            return self._p

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Session:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def post(self, url, **k):
            calls.append({"verb": "POST", "url": url, **k})
            return _Resp(*handler("POST", url))

        def get(self, url, **k):
            calls.append({"verb": "GET", "url": url, **k})
            return _Resp(*handler("GET", url))

    module = types.ModuleType("aiohttp")
    module.ClientSession = _Session
    module.BasicAuth = lambda *a, **k: ("basic", a)
    monkeypatch.setitem(sys.modules, "aiohttp", module)
    return calls


# --------------------------------------------------------------------------- #
def test_all_six_modes_now_have_transports():
    cases = [
        (HITLMode.GITHUB_DEPLOYMENT, GitHubDeploymentTransport, GitHubConfig()),
        (HITLMode.GITLAB_MR, GitLabMRTransport, GitLabConfig()),
        (HITLMode.GITLAB_PIPELINE, GitLabPipelineTransport, GitLabConfig()),
        (HITLMode.JIRA, JiraTransport, JiraConfig()),
        (HITLMode.PAGERDUTY, PagerDutyTransport, PagerDutyConfig()),
        (HITLMode.TERRAFORM_CLOUD, TerraformCloudTransport, TerraformCloudConfig()),
    ]
    for mode, cls, cfg in cases:
        assert isinstance(get_transport(mode, cfg), cls)


async def test_gitlab_mr_send_posts_note_and_poll_detects_approval(monkeypatch):
    calls = _install_fake_aiohttp(
        monkeypatch,
        lambda m, u: (201, {"id": 99}) if m == "POST" else (200, {"approved": True}),
    )
    t = GitLabMRTransport(GitLabConfig(token="tok", project_id="group/proj", mr_iid=7))
    ref = await t.send(_request(), "wf")
    note = next(c for c in calls if c["verb"] == "POST")
    assert "/merge_requests/7/notes" in note["url"]
    assert note["headers"]["PRIVATE-TOKEN"] == "tok"
    # callback links only appear when VICTOR_HITL_CALLBACK_URL is set; check static content
    assert "🔔 Workflow approval required" in note["json"]["body"]
    assert ref == "gitlab:mr:7:note:99"

    resp = await t.poll("rid", ref)
    assert resp.status == HITLStatus.APPROVED and resp.approved is True


async def test_gitlab_pipeline_poll_maps_job_status(monkeypatch):
    _install_fake_aiohttp(monkeypatch, lambda m, u: (200, {"status": "success"}))
    t = GitLabPipelineTransport(GitLabConfig(token="t", project_id="1"))
    ref = await t.send(_request({"job_id": 55}), "wf")
    assert ref == "gitlab:pipeline:job:55"
    assert (await t.poll("rid", ref)).approved is True

    _install_fake_aiohttp(monkeypatch, lambda m, u: (200, {"status": "canceled"}))
    assert (await t.poll("rid", "gitlab:pipeline:job:55")).status == HITLStatus.REJECTED


async def test_jira_send_creates_issue_and_poll_reads_status(monkeypatch):
    calls = _install_fake_aiohttp(
        monkeypatch,
        lambda m, u: (
            (201, {"key": "OPS-42"})
            if m == "POST"
            else (200, {"fields": {"status": {"name": "Approved"}}})
        ),
    )
    t = JiraTransport(
        JiraConfig(base_url="https://j.example", email="me@x", api_token="tok", project_key="OPS")
    )
    ref = await t.send(_request(), "wf")
    create = next(c for c in calls if c["verb"] == "POST")
    assert create["url"].endswith("/rest/api/2/issue")
    assert create["json"]["fields"]["project"]["key"] == "OPS"
    assert ref == "jira:issue:OPS-42"
    assert (await t.poll("rid", ref)).approved is True


async def test_pagerduty_send_triggers_incident_poll_is_callback_driven(monkeypatch):
    calls = _install_fake_aiohttp(monkeypatch, lambda m, u: (202, {"status": "success"}))
    t = PagerDutyTransport(PagerDutyConfig(routing_key="R1"))
    ref = await t.send(_request({"svc": "api"}), "wf")
    enqueue = calls[0]
    assert enqueue["url"] == "https://events.pagerduty.com/v2/enqueue"
    assert enqueue["json"]["routing_key"] == "R1"
    assert enqueue["json"]["event_action"] == "trigger"
    assert ref.startswith("pagerduty:incident:victor-hitl-")
    assert await t.poll("rid", ref) is None  # response via signed callback link


async def test_terraform_cloud_poll_maps_run_status(monkeypatch):
    _install_fake_aiohttp(
        monkeypatch, lambda m, u: (200, {"data": {"attributes": {"status": "applied"}}})
    )
    t = TerraformCloudTransport(TerraformCloudConfig(token="tk", organization="o", workspace="w"))
    ref = await t.send(_request({"run_id": "run-abc"}), "wf")
    assert ref == "tfc:run:run-abc"
    assert (await t.poll("rid", ref)).approved is True

    _install_fake_aiohttp(
        monkeypatch, lambda m, u: (200, {"data": {"attributes": {"status": "discarded"}}})
    )
    assert (await t.poll("rid", "tfc:run:run-abc")).status == HITLStatus.REJECTED


async def test_github_deployment_poll_resolves_when_not_pending(monkeypatch):
    # pending_deployments empty -> gate resolved; run not cancelled -> approved
    _install_fake_aiohttp(
        monkeypatch,
        lambda m, u: (200, []) if "pending_deployments" in u else (200, {"conclusion": "success"}),
    )
    t = GitHubDeploymentTransport(GitHubConfig(token="t", owner="o", repo="r", environment="prod"))
    ref = await t.send(_request({"run_id": 1234}), "wf")
    assert ref == "github:deployment:1234:prod"
    assert (await t.poll("rid", ref)).approved is True


async def test_missing_required_context_raises(monkeypatch):
    _install_fake_aiohttp(monkeypatch, lambda m, u: (200, {}))
    with pytest.raises(ValueError):
        await TerraformCloudTransport(TerraformCloudConfig(token="t")).send(_request(), "wf")
    with pytest.raises(ValueError):
        await GitHubDeploymentTransport(GitHubConfig(token="t")).send(_request(), "wf")
