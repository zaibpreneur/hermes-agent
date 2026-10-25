"""A malformed active-session entry fails closed only for decisions it concerns (HERMES 37.12).

The strict reader raised on ANY entry its schema rejects, so one such entry (a tracked claim with
an empty session id, another build's schema) refused every guard, claim and sweep in that home
until someone deleted it by hand, though it fenced no real session. The guard now fails closed
only for the target session's own entry, and the orphan sweep only for this process's own.
"""
import json
import os

import pytest

from hermes_cli import active_sessions


def _registry(home):
    return home / "runtime" / "active_sessions.json"


def _entries(home):
    return json.loads(_registry(home).read_text())["entries"]


def _plant_malformed(home, session_id, *, pid):
    path = _registry(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = _entries(home) if path.exists() else []
    entries.append({"lease_id": "f" * 32, "session_id": session_id, "surface": "cli", "pid": pid,
                    "process_start_time": "nan", "started_at": 1, "updated_at": 1})
    path.write_text(json.dumps({"entries": entries}))


def _claim(home, session_id, **kwargs):
    lease, refusal = active_sessions.try_acquire_active_session(
        session_id=session_id, surface="desktop", config={}, registry_home=home, **kwargs)
    assert lease is not None, refusal
    return lease


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_tracked_claim_without_a_session_id_writes_nothing(tmp_path, blank):
    draft = _claim(tmp_path, blank, metadata={"live_session_id": "draft"}, track_liveness=True)
    assert not _registry(tmp_path).exists()
    # Every strict reader still answers for real sessions.
    with active_sessions.active_session_liveness_guard("other", registry_home=tmp_path) as held:
        assert held is False
    real = _claim(tmp_path, "real", metadata={"live_session_id": "real"}, track_liveness=True)
    assert [e["session_id"] for e in active_sessions.active_session_registry_snapshot(tmp_path, strict=True)] == ["real"]
    # The draft's lease is still real: once its session is stored, that session is fenced.
    assert active_sessions.transfer_active_session(draft, session_id="saved", metadata={"live_session_id": "draft"})
    with active_sessions.active_session_liveness_guard("saved", registry_home=tmp_path) as held:
        assert held is True
    draft.release()
    real.release()
    assert _entries(tmp_path) == []


def test_the_guard_skips_a_malformed_entry_of_another_session_and_keeps_it(tmp_path):
    lease = _claim(tmp_path, "target", metadata={"live_session_id": "pane"}, track_liveness=True)
    _plant_malformed(tmp_path, "unrelated", pid=os.getpid())
    with active_sessions.active_session_liveness_guard("free", registry_home=tmp_path) as held:
        assert held is False
    with active_sessions.active_session_liveness_guard("target", registry_home=tmp_path) as held:
        assert held is True
    # The holder itself, for a refusal that names it.
    with active_sessions.active_session_holders_guard("target", registry_home=tmp_path) as holders:
        assert [(h["session_id"], h["surface"]) for h in holders] == [("target", "desktop")]
    assert {e["session_id"] for e in _entries(tmp_path)} == {"target", "unrelated"}
    assert lease.enabled and not lease.released


def test_the_targets_own_malformed_entry_still_fails_closed(tmp_path):
    _plant_malformed(tmp_path, "target", pid=os.getpid())
    with pytest.raises(active_sessions.ActiveSessionRegistryError):
        with active_sessions.active_session_liveness_guard("target", registry_home=tmp_path):
            pass
    assert [e["session_id"] for e in _entries(tmp_path)] == ["target"]


def test_the_orphan_sweep_drops_this_processs_orphan_beside_a_malformed_foreign_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(active_sessions, "_SELF_ORPHAN_GRACE_SECONDS", 0.0)
    orphan = _claim(tmp_path, "orphaned", metadata={"live_session_id": "gone"}, track_liveness=True)
    live = _claim(tmp_path, "live", metadata={"live_session_id": "here"}, track_liveness=True)
    _plant_malformed(tmp_path, "foreign", pid=os.getpid() + 1)

    assert active_sessions.release_orphaned_leases({live.lease_id}) == 1
    assert {e["session_id"] for e in _entries(tmp_path)} == {"live", "foreign"}
    assert orphan.lease_id not in {e["lease_id"] for e in _entries(tmp_path)}


def test_the_orphan_sweep_still_skips_a_home_with_its_own_malformed_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(active_sessions, "_SELF_ORPHAN_GRACE_SECONDS", 0.0)
    _claim(tmp_path, "orphaned", metadata={"live_session_id": "gone"}, track_liveness=True)
    _plant_malformed(tmp_path, "mine", pid=os.getpid())

    assert active_sessions.release_orphaned_leases(set()) == 0
    assert {e["session_id"] for e in _entries(tmp_path)} == {"orphaned", "mine"}
