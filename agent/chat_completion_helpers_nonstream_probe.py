"""Non-streaming probes for a ``_StreamingCall`` whose stream failed before any delta.

``_handle_stream_error`` tries ``unmask_server_error`` (5xx) first, then ``retry_auth_error``
(401/403). Both take the ``_StreamingCall`` as ``call`` and return True = handled (caller must
not overwrite ``call.result``), False = propagate the original error. Facade names
(``interruptible_api_call``, ``_with_stream_emitters``, ...) are late-imported from
``agent.chat_completion_helpers``, the seam tests patch.
"""

import time

# One non-streaming re-issue per window. Covers the outer retry loop (up to ~3 attempts x
# backoff, well under 60s) so an outage doesn't double traffic every attempt, while later
# turns re-arm automatically.
_STREAM_5XX_PROBE_WINDOW_S = 60.0
_STREAM_AUTH_PROBE_WINDOW_S = 60.0


def _claim_probe(call, ts_attr: str, window_s: float) -> bool:
    """Shared gates; True = probe now (the window is claimed on the AGENT: each outer retry
    builds a fresh _StreamingCall, so an instance flag would re-probe every attempt)."""
    from agent import chat_completion_helpers as h

    agent = call.agent
    # A /stop that arrived before this point suppresses the probe: the loop's pre-retry
    # interrupt check owns that decision, so a pending stop must not buy one more request.
    if getattr(agent, "_interrupt_requested", False):
        return False
    if h._streams_llamacpp(agent):
        # Same error either way, and the router can't cancel this probe once its watchdog gives up.
        return False
    if call.deltas_were_sent["yes"]:
        return False
    if getattr(agent, "api_mode", "") not in ("", "chat_completions"):
        return False  # replay handles chat-completions shapes only
    now = time.monotonic()
    last_probe = getattr(agent, ts_attr, None)
    if last_probe is not None and now - last_probe < window_s:
        return False
    setattr(agent, ts_attr, now)
    return True


def _issue_probe(call):
    """Re-issue the request non-streaming. It runs on the worker thread while the stream
    monitor still polls, so the stale check is suspended (the probe has its own watchdog)."""
    from agent import chat_completion_helpers as h

    probe_kwargs = {k: v for k, v in call.api_kwargs.items() if k not in ("stream", "stream_options")}
    stale_timeout = call._stream_stale_timeout
    call._stream_stale_timeout = float("inf")  # no chunks arrive during the probe
    try:
        return h.interruptible_api_call(call.agent, probe_kwargs)
    finally:
        call._stream_stale_timeout = stale_timeout


def _deliver(call, probe, status_text: str, what: str) -> bool:
    """Replay a successful probe for this turn WITHOUT latching streaming off."""
    from agent import chat_completion_helpers as h
    logger = h.logger  # a bare name, so ruff's BLE001 sees the traceback logging

    call._quiet(call.agent._buffer_status, status_text)
    try:
        # The failed attempt already emitted its terminal on_stream_end(finished=False),
        # so the recovered delivery opens and closes its OWN stream pair — consumers must
        # never see deltas after that error event.
        replayed = h._with_stream_emitters(call.agent, lambda: call._replay_final_response(probe))
    except Exception as replay_err:
        # A response we cannot replay must not escape into _call()'s except block.
        logger.exception("%s response could not be replayed: %s", what, replay_err)
        return False
    call.result["response"] = replayed
    return True


def unmask_server_error(call, e: Exception) -> bool:
    """One non-streaming re-issue when a 5xx killed the stream before any delta.

    Some gateways validate the request only on their non-streaming path and crash
    opaquely ("500 something went wrong") when streaming — the real 4xx, with its
    actionable message, never reaches the user through stream retries. The probe
    surfaces it: on success the response is delivered WITHOUT latching non-streaming
    (a transient gateway 500 must not permanently disable streaming); on a probe 4xx
    that error REPLACES the opaque 5xx; any other probe failure keeps the original
    error, as does a response that cannot be replayed. Interrupts re-raise (the outer
    handler routes them).
    """
    from agent import chat_completion_helpers as h
    logger = h.logger  # a bare name, so ruff's BLE001 sees the traceback logging

    status = h._extract_status_code(e)
    if status is None or status < 500:
        return False
    if not _claim_probe(call, "_stream_5xx_probe_ts", _STREAM_5XX_PROBE_WINDOW_S):
        return False
    try:
        probe = _issue_probe(call)
    except (KeyboardInterrupt, InterruptedError):
        raise  # the outer handler routes user interrupts; never swallow them
    except Exception as probe_err:  # health: allow BLE001 -- pre-existing 5xx probe boundary moved verbatim from chat_completion_helpers
        probe_status = h._extract_status_code(probe_err)
        if probe_status is not None and probe_status < 500:
            # The provider's REAL validation error beats the opaque 5xx.
            logger.info("Non-streaming unmask probe surfaced the underlying error: %s", probe_err)
            call.result["error"] = probe_err
            return True
        logger.info("Non-streaming unmask probe failed: %s", probe_err)
        return False
    logger.info("Streaming 5xx re-issued non-streaming successfully for %s/%s "
                  "(not latched: the 5xx may be transient).",
                  call.agent.provider or "unknown", call.agent.model or "unknown")
    return _deliver(call, probe, "⚠  Streaming failed with a provider server error; the non-streaming retry succeeded.",
                    "Non-streaming unmask probe")


def retry_auth_error(call, e: Exception) -> bool:
    """One non-streaming retry when a 401/403 killed the stream before any delta.

    An OpenAI-compatible aggregator can reject a STREAMING request with
    ``401 Invalid 'Authorization' header or token`` while the identical
    non-streaming request with the same key succeeds seconds later: observed as
    same-second bursts, every one of them ``Streaming failed before delivery``,
    with each key answering 200 on a 1-token non-streaming probe. Auth is
    classified non-retryable (correct for a dead key), so without this the turn
    aborts and a queued gateway delivery is lost.

    Gated exactly like the 5xx unmask probe. A probe that fails propagates the
    ORIGINAL auth error, so a genuinely dead key still fails fast and is classified
    exactly as before; a probe that succeeds delivers WITHOUT latching streaming off.
    """
    from agent import chat_completion_helpers as h
    logger = h.logger  # a bare name, so ruff's BLE001 sees the traceback logging

    if h._extract_status_code(e) not in (401, 403):
        return False
    if not _claim_probe(call, "_stream_auth_probe_ts", _STREAM_AUTH_PROBE_WINDOW_S):
        return False
    try:
        probe = _issue_probe(call)
    except (KeyboardInterrupt, InterruptedError):
        raise  # the outer handler routes user interrupts; never swallow them
    except Exception as probe_err:
        logger.info("Non-streaming auth retry failed too (%s); keeping the original auth error",
                      h._extract_status_code(probe_err) or type(probe_err).__name__, exc_info=True)
        return False
    logger.info("Streaming auth failure re-issued non-streaming successfully for %s/%s "
                  "(not latched: the rejection may be transient).",
                  call.agent.provider or "unknown", call.agent.model or "unknown")
    return _deliver(call, probe, "⚠  Streaming was rejected by the provider; the non-streaming retry succeeded.",
                    "Non-streaming auth retry")
