"""MCP client recovery from a superseded / closed server session.

A server that keys its live session on the credential (LatticeRun) refuses a
call at its session gate, before dispatching it, when another client on the
same token took the session over (``slot_superseded``) or it was hung up
(``session_closed``), and asks for a fresh ``initialize``. Before this, the
refusal matched no recovery classifier: the client kept the dead session id
and every later call failed the same way until the process restarted.
"""
import json
import threading
from unittest.mock import MagicMock

import pytest
from mcp.shared.exceptions import MCPError

from tools import mcp_tool_loop as _mcp_loop


def _refusal(reason: str, cause=None, message=None) -> MCPError:
    """The JSON-RPC error LatticeRun returns with its HTTP 409."""
    text = message or f"{reason}: \"Frontier\" lost its session to Frontier. Send initialize to take it back."
    return MCPError(code=-32003, message=text, data={"reason": reason, "cause": cause, "label": "Frontier"})


# ---------------------------------------------------------------------------
# _is_session_superseded_error — what counts as "re-initialize fixes it"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason, cause", [("slot_superseded", "replaced"), ("slot_superseded", None),
                                           ("session_closed", None)])
def test_superseded_or_closed_session_is_recoverable(reason, cause):
    from tools.mcp_tool_errors import _is_session_superseded_error
    assert _is_session_superseded_error(_refusal(reason, cause)) is True


def test_account_at_its_cap_is_not_recoverable_by_reinitialize():
    """``cause: cap`` displaced nobody; a new initialize is refused too, so retrying would only loop."""
    from tools.mcp_tool_errors import _is_session_superseded_error
    cap = _refusal("slot_superseded", "cap", "slot_superseded: the account is at its live-agent cap — all 5 slots "
                                             "are in use. Nothing was displaced. Free a session, then send initialize.")
    assert _is_session_superseded_error(cap) is False


def test_message_prefix_is_enough_when_a_proxy_drops_data():
    from tools.mcp_tool_errors import _is_session_superseded_error
    assert _is_session_superseded_error(RuntimeError("slot_superseded: lost its session. Send initialize.")) is True
    assert _is_session_superseded_error(
        RuntimeError("slot_superseded: account full. Nothing was displaced.")) is False


def test_unrelated_errors_are_not_superseded():
    from tools.mcp_tool_errors import _is_session_superseded_error
    assert _is_session_superseded_error(MCPError(code=-32602, message="Invalid params")) is False
    assert _is_session_superseded_error(RuntimeError("tool said slot_superseded somewhere")) is False


def test_found_through_a_wrapping_chain():
    from tools.mcp_tool_errors import _is_session_superseded_error
    try:
        try:
            raise _refusal("slot_superseded", "replaced")
        except MCPError as inner:
            raise RuntimeError("call failed") from inner
    except RuntimeError as outer:
        assert _is_session_superseded_error(outer) is True


# ---------------------------------------------------------------------------
# The tool handler re-initializes and retries — writes included
# ---------------------------------------------------------------------------


def _stub_server(name: str):
    """A server record whose reconnect swaps in a fresh session and reports ready."""
    _mcp_loop._ensure_mcp_loop()
    server = MagicMock()
    server.name = name
    ready = threading.Event()
    ready.set()
    server._ready = MagicMock(is_set=ready.is_set, clear=ready.clear, set=ready.set)
    reconnected = threading.Event()

    def _reconnect():
        reconnected.set()
        old = server.session
        server.session = MagicMock(call_tool=old.call_tool)
        ready.set()

    server._reconnect_event = MagicMock(set=_reconnect)
    server.session = MagicMock()
    return server, reconnected


@pytest.mark.parametrize("read_only", [False, True], ids=["write-capable", "read-only"])
def test_superseded_call_is_reinitialized_and_retried(monkeypatch, tmp_path, read_only):
    """Refused before dispatch, so even a write is safe to run once more on the new session."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool_handlers import _make_tool_handler

    server, reconnected = _stub_server("lr")
    calls = {"n": 0}

    async def _sequence(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _refusal("slot_superseded", "replaced")
        result = MagicMock()
        result.is_error = False
        result.content = [MagicMock(type="text", text="claimed")]
        result.structured_content = None
        return result

    server.session.call_tool = _sequence
    mcp_tool._servers["lr"] = server
    mcp_tool._server_error_counts.pop("lr", None)
    if read_only:
        mcp_tool._tool_read_only_hints.setdefault("lr", {})["start_task"] = True
    try:
        parsed = json.loads(_make_tool_handler("lr", "start_task", 10.0)({}))
        assert reconnected.is_set()
        assert parsed.get("result") == "claimed", parsed
        assert calls["n"] == 2
        assert mcp_tool._server_error_counts.get("lr", 0) == 0
    finally:
        mcp_tool._servers.pop("lr", None)
        mcp_tool._server_error_counts.pop("lr", None)
        mcp_tool._server_breaker_opened_at.pop("lr", None)
        mcp_tool._tool_read_only_hints.pop("lr", None)


def test_account_cap_refusal_is_not_retried(monkeypatch, tmp_path):
    """Control: a full account is reported, not re-initialized in a loop."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool_handlers import _make_tool_handler

    server, reconnected = _stub_server("lr-cap")
    calls = {"n": 0}

    async def _always_cap(*a, **kw):
        calls["n"] += 1
        raise _refusal("slot_superseded", "cap", "slot_superseded: the account is at its live-agent cap. "
                                                 "Nothing was displaced.")

    server.session.call_tool = _always_cap
    mcp_tool._servers["lr-cap"] = server
    mcp_tool._tool_read_only_hints.setdefault("lr-cap", {})["start_task"] = True
    try:
        parsed = json.loads(_make_tool_handler("lr-cap", "start_task", 10.0)({}))
        assert "error" in parsed, parsed
        assert not reconnected.is_set()
        assert calls["n"] == 1
    finally:
        mcp_tool._servers.pop("lr-cap", None)
        mcp_tool._server_error_counts.pop("lr-cap", None)
        mcp_tool._server_breaker_opened_at.pop("lr-cap", None)
        mcp_tool._tool_read_only_hints.pop("lr-cap", None)
