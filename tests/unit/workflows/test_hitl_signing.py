"""Unit tests for signed HITL callback tokens."""

from __future__ import annotations

import time

from victor.workflows.hitl_signing import sign_action, verify_action

SECRET = "test-secret"


def test_valid_token_verifies():
    token = sign_action("req1", "approve", secret=SECRET)
    assert verify_action("req1", "approve", token, secret=SECRET) is True


def test_token_is_action_bound():
    # A token minted for "approve" must not validate "reject" (no flipping).
    token = sign_action("req1", "approve", secret=SECRET)
    assert verify_action("req1", "reject", token, secret=SECRET) is False


def test_token_is_request_bound():
    token = sign_action("req1", "approve", secret=SECRET)
    assert verify_action("req2", "approve", token, secret=SECRET) is False


def test_wrong_secret_fails():
    token = sign_action("req1", "approve", secret=SECRET)
    assert verify_action("req1", "approve", token, secret="other") is False


def test_expired_token_fails():
    token = sign_action("req1", "approve", secret=SECRET, ttl=-1)  # already expired
    assert verify_action("req1", "approve", token, secret=SECRET) is False


def test_future_expiry_within_ttl_passes():
    token = sign_action("req1", "approve", secret=SECRET, ttl=3600, now=time.time())
    assert verify_action("req1", "approve", token, secret=SECRET) is True


def test_malformed_or_empty_tokens_fail():
    for bad in (None, "", "garbage", "notanumber.deadbeef", "123"):
        assert verify_action("req1", "approve", bad, secret=SECRET) is False
