"""A transient 401 from the last provider standing gets one more attempt before it ends the turn.

An OpenAI-compatible aggregator was observed rejecting a valid key intermittently; once refresh,
rotation and the fallback chain have all declined, ``settle_unrecovered_error`` retries the call
once (guarded per ``TurnRetryState``) instead of dropping the turn. ``auth_permanent`` never does.
"""
import dataclasses
import types

import httpx
import openai
import pytest

from agent import conversation_loop
from agent import fallback_cooldown
from agent import turn_api_error
from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_retry_state import TurnRetryState

_TERMINAL = {"terminal": True}
_INTERRUPTED = {"interrupted": True}


def _auth_error():
    body = {"error": {"message": "Invalid 'Authorization' header or token.", "type": "authentication_error"}}
    response = httpx.Response(
        401, json=body, request=httpx.Request("POST", "https://llm.example.test/v1/chat/completions"),
    )
    return openai.AuthenticationError(body["error"]["message"], response=response, body=body)


@pytest.fixture
def harness(monkeypatch):
    waits = []
    sleep_result = {"value": None}

    def _sleep(agent, wait, _retry, **kwargs):
        waits.append(wait)
        return sleep_result["value"]

    monkeypatch.setattr(turn_api_error, "interruptible_backoff_sleep", _sleep)
    monkeypatch.setattr(turn_api_error, "settle_delivered_partial", lambda *a, **k: None)
    monkeypatch.setattr(turn_api_error, "nonretryable_client_error_result", lambda *a, **k: _TERMINAL)
    monkeypatch.setattr(conversation_loop, "_is_copilot_provider", lambda agent: False)
    monkeypatch.setattr(conversation_loop, "_is_stale_copilot_credential_error", lambda *a: False)
    monkeypatch.setattr(conversation_loop, "_arm_fallback_restart", lambda agent, msgs, prompt, r: prompt)
    monkeypatch.setattr(fallback_cooldown, "_mark_entitlement_rejected_model", lambda agent, err: None)

    statuses = []
    agent = types.SimpleNamespace(
        _has_pending_fallback=lambda: False,
        _try_activate_fallback=lambda **kwargs: False,
        _buffer_diagnostic_status=statuses.append,
        _buffer_vprint=lambda *a, **k: None,
        log_prefix="",
    )
    return types.SimpleNamespace(agent=agent, statuses=statuses, waits=waits, sleep_result=sleep_result)


def _settle(agent, api_error, classified, _retry):
    return turn_api_error.settle_unrecovered_error(
        agent, api_error=api_error, classified=classified, _retry=_retry, status_code=401,
        error_msg=str(api_error), is_context_length_error=False, is_rate_limited=False,
        _is_zai_coding_overload=False, _provider="custom", _base="https://llm.example.test/v1",
        _model="some-model", messages=[], api_messages=[], api_kwargs={}, active_system_prompt="sys",
        conversation_history=[], approx_tokens=0, retry_count=0, max_retries=3,
        compression_attempts=0, api_call_count=1,
    )


def _classified(api_error):
    return classify_api_error(api_error, provider="custom", model="some-model")


def test_first_transient_401_with_no_fallback_is_retried_not_dropped(harness):
    api_error = _auth_error()
    _retry = TurnRetryState()

    verdict = _settle(harness.agent, api_error, _classified(api_error), _retry)

    assert verdict.action == "fallthrough"
    assert verdict.result is None
    assert _retry.transient_auth_retry_attempted is True
    assert len(harness.waits) == 1
    assert any("retrying once" in s for s in harness.statuses)


def test_second_401_on_same_attempt_is_terminal(harness):
    api_error = _auth_error()
    classified = _classified(api_error)
    _retry = TurnRetryState()

    assert _settle(harness.agent, api_error, classified, _retry).action == "fallthrough"
    verdict = _settle(harness.agent, api_error, classified, _retry)

    assert verdict.action == "return"
    assert verdict.result is _TERMINAL
    assert len(harness.waits) == 1


def test_auth_permanent_is_terminal_without_retry(harness):
    api_error = _auth_error()
    classified = dataclasses.replace(_classified(api_error), reason=FailoverReason.auth_permanent)
    _retry = TurnRetryState()

    verdict = _settle(harness.agent, api_error, classified, _retry)

    assert verdict.action == "return"
    assert verdict.result is _TERMINAL
    assert harness.waits == []
    assert _retry.transient_auth_retry_attempted is False


def test_interrupt_during_retry_wait_ends_the_turn(harness):
    harness.sleep_result["value"] = _INTERRUPTED
    api_error = _auth_error()

    verdict = _settle(harness.agent, api_error, _classified(api_error), TurnRetryState())

    assert verdict.action == "return"
    assert verdict.result is _INTERRUPTED
