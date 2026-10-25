"""Streaming 401/403 retried ONCE non-streaming, so a queued delivery is not lost.

Regression shape (2026-10-25, OpenAI-compatible aggregator): a STREAMING request is
rejected with ``401 Invalid 'Authorization' header or token`` while the identical
non-streaming request with the SAME key succeeds seconds later. Observed as
same-second bursts, every one preceded by ``Streaming failed before delivery``, with
each key answering 200 on a 1-token non-streaming probe. Auth is classified
non-retryable — correct for a dead key — so without this the turn aborts and a queued
gateway delivery is lost.

Gating invariants (deliberately the same as the 5xx unmask probe):
- never after partial delivery, on a non-auth status, or on the anthropic wire
- one probe per window, timestamped on the AGENT (outer retries build fresh
  _StreamingCall instances, so an instance flag would re-probe every attempt)
- a probe that ALSO fails propagates the original auth error: a genuinely dead key
  must still fail fast and be classified as before
- a probe that succeeds delivers for the turn WITHOUT latching _disable_streaming
- the recovered delivery opens and closes its own stream pair
"""
from types import SimpleNamespace

from agent import chat_completion_helpers as h


def _make_call(api_kwargs, *, deltas_sent=False, api_mode="chat_completions", auth_probe_ts=None):
    call = h._StreamingCall.__new__(h._StreamingCall)
    call.agent = SimpleNamespace(
        provider="custom", model="deepseek/deepseek-v4.1-flash", api_mode=api_mode,
        _interrupt_requested=False,
        _is_provider_stream_parse_error=lambda e: False,
        _buffer_status=lambda text: call.buffered.append(text),
        _disable_streaming=False, _stream_5xx_probe_ts=None,
        _stream_auth_probe_ts=auth_probe_ts,
        _fire_stream_delta=lambda text: call.deltas.append(text),
        _reset_stream_delivery_tracking=lambda: None,
        _warning_presentation_enabled=lambda: False,
    )
    call.buffered = []
    call.deltas = []
    call.first_delta_fired = {"done": True}
    call.on_first_delta = None
    call._stream_stale_timeout = 180.0
    call.api_kwargs = api_kwargs
    call.result = {"response": None, "error": None, "partial_tool_names": []}
    call.deltas_were_sent = {"yes": deltas_sent}
    call.provider_tool_in_flight = {"yes": False}
    call._request_cancelled = {"value": False}
    return call


class _StreamErr(Exception):
    def __init__(self, status):
        super().__init__(f"Error code: {status} - Invalid 'Authorization' header or token.")
        self.status_code = status


def _run(call, e):
    return call._handle_stream_error(e, attempt=2, max_retries=2)


def _ok_response():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", reasoning_content=None))])


def test_stream_401_probe_success_delivers_without_dropping(monkeypatch):
    """The whole point: a streaming 401 that the non-streaming path accepts must deliver."""
    call = _make_call({"model": "m", "stream": True, "stream_options": {"x": 1}, "messages": []})
    seen = []

    def fake_probe(agent, kwargs):
        seen.append((kwargs, call._stream_stale_timeout))
        return _ok_response()

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe)

    handled = not _run(call, _StreamErr(401))

    assert handled
    assert call.result["response"] is not None
    assert call.result["error"] is None
    probe_kwargs, stale_during_probe = seen[0]
    assert "stream" not in probe_kwargs and "stream_options" not in probe_kwargs
    assert call.deltas == ["ok"]
    assert call.agent._disable_streaming is False  # one-turn recovery, never latched
    assert stale_during_probe == float("inf") and call._stream_stale_timeout == 180.0
    assert call.buffered and "non-streaming retry succeeded" in call.buffered[0]


def test_stream_403_is_retried_the_same_way(monkeypatch):
    call = _make_call({"model": "m", "messages": []})
    monkeypatch.setattr(h, "interruptible_api_call", lambda agent, kwargs: _ok_response())
    assert not _run(call, _StreamErr(403))
    assert call.result["response"] is not None


def test_probe_failing_again_propagates_the_ORIGINAL_auth_error(monkeypatch):
    """A dead key must still fail fast, with its own error — not the probe's."""
    call = _make_call({"model": "m", "messages": []})
    original = _StreamErr(401)

    def fake_probe(agent, kwargs):
        raise RuntimeError("probe also rejected")

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe)

    handled = _run(call, original)

    assert not handled
    assert call.result["error"] is original


def test_the_retry_refuses_after_partial_delivery(monkeypatch):
    """Calls the method directly: the handler resets the delta flag when nothing VISIBLE
    was delivered (so a retry there is correct), which means the method's own gate is the
    one to test here."""
    call = _make_call({"model": "m", "messages": []}, deltas_sent=True)
    called = []
    monkeypatch.setattr(h, "interruptible_api_call", lambda agent, kwargs: called.append(1) or _ok_response())

    assert call._retry_auth_error_with_nonstreaming(_StreamErr(401)) is False
    assert called == []


def test_no_probe_on_a_non_auth_status(monkeypatch):
    call = _make_call({"model": "m", "messages": []})
    called = []
    monkeypatch.setattr(h, "interruptible_api_call", lambda agent, kwargs: called.append(1) or _ok_response())
    _run(call, _StreamErr(400))
    assert called == []


def test_one_probe_per_window(monkeypatch):
    import time as _t
    call = _make_call({"model": "m", "messages": []}, auth_probe_ts=_t.monotonic())
    called = []
    monkeypatch.setattr(h, "interruptible_api_call", lambda agent, kwargs: called.append(1) or _ok_response())
    _run(call, _StreamErr(401))
    assert called == []  # inside the window: no second probe


def test_no_probe_on_the_anthropic_wire(monkeypatch):
    call = _make_call({"model": "m", "messages": []}, api_mode="anthropic_messages")
    called = []
    monkeypatch.setattr(h, "interruptible_api_call", lambda agent, kwargs: called.append(1) or _ok_response())
    _run(call, _StreamErr(401))
    assert called == []
