"""Which active-session leases fence a Bot Chat DM, and what the refusal says (HERMES 37.12).

Exclusivity stays fail-closed: a live writer on a still-legacy (unbound) Bot Chat refuses the DM.
What it may not do is refuse for something unrelated to the target:

- a registry entry the strict schema rejects, for ANOTHER session, refused every DM until it was
  deleted by hand; the target's own unprovable entry still refuses, as "coordination unavailable",
  never as a held chat and never as an unknown outcome ("Do not resend");
- one leftover pre-authority mailbox ticket whose session is held refused every new DM; it now
  waits for a later pass while the new DM is admitted;
- the held refusal now names its holder (surface and age, never a pid or a path), and "safe to
  resend after ... closes" is said only of a named live session.

Stand-ins as in test_bot_dm_held_by_session_writer.py: the RPC hop, endpoint discovery and the LLM turn.
"""
import asyncio
import json
import os
import threading
import time
from types import SimpleNamespace

import pytest
import pytest_asyncio

from hermes_state_runtime import list_session_admissions
from tests.waiting import async_wait_until

CHAT = 'old-bot-chat'
LEGACY_KEY = 'a' * 64


@pytest_asyncio.fixture
async def owner(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from hermes_cli import gateway_runtime
    from hermes_cli.active_sessions import release_active_session
    from hermes_cli.gateway_client import rpc_error
    from hermes_state import SessionDB
    from tools import bot_live_delivery, bot_mode_dm

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    store._db = SessionDB(db_path=tmp_path / 'state.db')
    db = store._db
    # A Desktop Bot Chat from before the one-gateway runtime: adopted on its first delivery.
    db.create_session(CHAT, source='desktop', model='old', cwd=str(tmp_path))
    db.set_session_title(CHAT, 'Bot Chat')
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='fixture')
    connection = AuthorityConnection(authority, object(), {
        'user_id': 'uid:1000', 'provider': 'local', 'profile_id': 'default', 'instance_id': 'fixture',
        'capabilities': ['session:read', 'session:submit', 'session:create']})
    monkeypatch.setattr(connection, '_sibling_for', lambda profile: None)
    authority._native_legacy_transports = {connection.actor.transport_id: connection.actor}
    executed = []

    async def execute(authority, ref, row):
        executed.append(row['payload']['text'])
        return 'pong'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    loop = asyncio.get_running_loop()

    def rpc(method):
        def call(home, params):
            frame = asyncio.run_coroutine_threadsafe(connection.dispatch(
                {'id': 1, 'method': method, 'params': params}), loop).result()
            if 'error' in frame:
                raise rpc_error(frame['error'])
            return frame['result']
        return call

    monkeypatch.setattr(bot_live_delivery, 'authority_delivery', rpc('bot_relay.deliver'))
    monkeypatch.setattr(bot_live_delivery, 'authority_settle', rpc('bot_relay.settle'))
    monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint', lambda home, timeout: SimpleNamespace(
        state='ready', endpoint=SimpleNamespace(instance_id='fixture')))
    dm_dir = tmp_path / 'dm'
    dm_dir.mkdir(mode=0o700)
    monkeypatch.setattr(bot_mode_dm, '_dm_dir', lambda: dm_dir)
    monkeypatch.setattr(bot_mode_dm, '_spawn_delivery', lambda command, label, **kwargs: '{}')
    leases = []
    try:
        yield SimpleNamespace(authority=authority, db=db, home=tmp_path, executed=executed, leases=leases)
    finally:
        for lease in leases:
            if not lease.released:
                release_active_session(lease)
        db.close()


async def _send(owner, text):
    """``message_agent``'s local send: pin the intent, then call the owner's door."""
    from tools import bot_mode_dm
    raw = await asyncio.to_thread(bot_mode_dm._start_delivery, ['hermes', '-p', 'default'], text, '@default',
                                  stdin_file=False, task_id=None, agent=None, profile_home=owner.home)
    return json.loads(raw)


def _admissions(owner):
    return [row['request_id'] for row in list_session_admissions(owner.db, session_id=CHAT, pending_only=False)]


def _adopted(owner):
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_local_migration import LEGACY_PREFIX
    key = owner.db.get_session(CHAT).get('chat_id') or CHAT
    with owner.db._read_ctx() as conn:
        return conn.execute('SELECT 1 FROM state_meta WHERE key IN (?,?)',
                            (POLICY_PREFIX + key, LEGACY_PREFIX + key)).fetchone() is not None


def _registry(owner):
    return owner.home / 'runtime' / 'active_sessions.json'


def _plant_malformed_entry(owner, session_id):
    """One entry the strict schema rejects (``process_start_time`` is not finite)."""
    path = _registry(owner)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(path.read_text())['entries'] if path.exists() else []
    entries.append({'lease_id': 'f' * 32, 'session_id': session_id, 'surface': 'cli', 'pid': os.getpid(),
                    'process_start_time': 'nan', 'started_at': 1, 'updated_at': 1})
    path.write_text(json.dumps({'entries': entries}))


def _remove_malformed_entry(owner):
    path = _registry(owner)
    entries = [e for e in json.loads(path.read_text())['entries'] if e['lease_id'] != 'f' * 32]
    path.write_text(json.dumps({'entries': entries}))


def _legacy_ticket(owner):
    """A pre-authority mailbox ticket for the Bot Chat: an owner pin and no admission id."""
    root = owner.home / 'runtime' / 'bot_live_delivery'
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    record = dict(id=LEGACY_KEY, delivery_id=LEGACY_KEY, status='queued', created_at=1, sequence=1,
                  message='old queued DM', owner=dict(profile_home=str(owner.home.resolve()), session_id=CHAT,
                                                      lease_id='0' * 32, live_session_id='old-pane'))
    path = root / f'{LEGACY_KEY}.json'
    path.write_text(json.dumps(record))
    return path


def _lease(owner, surface, live_session_id, *, track_liveness=False):
    from hermes_cli.active_sessions import try_acquire_active_session
    lease, refusal = try_acquire_active_session(
        session_id=CHAT, surface=surface, config={}, registry_home=owner.home,
        metadata={'live_session_id': live_session_id}, track_liveness=track_liveness)
    assert lease is not None, refusal
    owner.leases.append(lease)
    return lease


def _assert_admitted(answer):
    assert answer.get('status') in {'queued', 'claimed', 'settled'}, answer


def _assert_coordination_unavailable(answer):
    from hermes_cli.active_sessions import SESSION_COORDINATION_UNAVAILABLE
    assert answer.get('reason') == SESSION_COORDINATION_UNAVAILABLE and 'status' not in answer, answer
    error = answer['error']
    assert 'NOT sent' in error and error.endswith('safe to resend later.'), error
    for wrong in ('held by another session writer', 'Do not resend', 'closes', 'active_sessions'):
        assert wrong not in error, error


@pytest.mark.asyncio
async def test_a_malformed_entry_for_another_session_does_not_refuse_a_bot_dm(owner):
    _plant_malformed_entry(owner, 'some-other-chat')

    sent = await _send(owner, 'ping')
    _assert_admitted(sent)
    assert _adopted(owner) and _admissions(owner) == ['bot:' + sent['delivery_id']]
    await async_wait_until(lambda: owner.executed == ['ping'], what='the adopted DM ran')
    # Skipped, not repaired: the entry is another writer's business.
    assert [e['session_id'] for e in json.loads(_registry(owner).read_text())['entries']] == ['some-other-chat']


@pytest.mark.asyncio
async def test_the_bot_chats_own_unprovable_entry_refuses_as_coordination_unavailable(owner):
    """On the adoption path and on the legacy-ticket path alike: still closed, never 'held'."""
    _plant_malformed_entry(owner, CHAT)
    on_adoption = await _send(owner, 'ping')
    _assert_coordination_unavailable(on_adoption)
    assert not _adopted(owner) and _admissions(owner) == []

    _remove_malformed_entry(owner)
    bound = await _send(owner, 'bind')
    _assert_admitted(bound)
    assert _adopted(owner)

    _legacy_ticket(owner)
    _plant_malformed_entry(owner, CHAT)
    on_legacy_ticket = await _send(owner, 'pong')
    _assert_coordination_unavailable(on_legacy_ticket)
    assert on_adoption['error'] == on_legacy_ticket['error']
    assert _admissions(owner) == ['bot:' + bound['delivery_id']]


@pytest.mark.asyncio
async def test_a_held_legacy_ticket_waits_without_refusing_a_new_dm(owner):
    """Bound chat, a live foreign lease on its id and one leftover legacy ticket: the new DM is
    admitted and runs; the ticket stays queued, neither admitted under the writer nor refused."""
    from tools.bot_live_delivery import read_delivery_result
    bound = await _send(owner, 'bind')
    _assert_admitted(bound)
    await async_wait_until(lambda: owner.executed == ['bind'], what='the binding DM ran')
    cli = _lease(owner, 'cli', CHAT)
    ticket = _legacy_ticket(owner)

    sent = await _send(owner, 'new DM')
    _assert_admitted(sent)
    await async_wait_until(lambda: owner.executed == ['bind', 'new DM'], what='the new DM ran')
    record = json.loads(ticket.read_text())
    assert record['status'] == 'queued' and not record.get('admission_id') and not record.get('refused'), record
    assert 'bot:' + LEGACY_KEY not in _admissions(owner)

    # Once the writer is gone the ticket is admitted on the next pass, after the DMs already queued.
    cli.release()
    later = await _send(owner, 'after the writer')
    _assert_admitted(later)
    await async_wait_until(lambda: owner.executed == ['bind', 'new DM', 'old queued DM', 'after the writer'],
                           what='the legacy ticket ran before the later DM')
    assert read_delivery_result(owner.home, LEGACY_KEY).get('admission_id')


@pytest.mark.asyncio
async def test_a_live_desktop_holder_is_named_in_the_refusal(owner):
    """An idle, attached Desktop pane on a still-legacy Bot Chat still fences it (two writers
    otherwise); the refusal says which session holds it, for how long, and that nothing was sent."""
    _lease(owner, 'desktop', 'pane-1', track_liveness=True)

    refused = await _send(owner, 'ping')
    assert refused.get('reason') == 'runtime_coordination_required' and 'status' not in refused, refused
    error = refused['error']
    assert 'an open Desktop chat, opened 0m ago' in error and 'NOT sent' in error, error
    assert error.endswith('safe to resend after that Desktop chat closes.'), error
    assert str(os.getpid()) not in error and str(owner.home) not in error, error
    assert not _adopted(owner) and _admissions(owner) == [] and owner.executed == []


class _Agent:
    def interrupt(self, *args, **kwargs):
        pass

    request_interrupt = interrupt

    def close(self):
        pass


class _Transport:
    _closed = False

    def write(self, *args, **kwargs):
        return True


@pytest.mark.asyncio
async def test_a_closed_desktop_pane_frees_the_bot_chat_for_the_next_dm(owner, monkeypatch):
    """Regression (finding D2): the pane's WS disconnects, the grace passes, the reaper ticks; the
    pane's lease is gone and the next DM adopts the legacy Bot Chat and runs."""
    from tui_gateway import server
    pane = {'agent': _Agent(), 'agent_ready': None, 'transport': _Transport(), 'running': False,
            'last_active': time.time(), 'created_at': time.time(), 'source': 'desktop', 'session_key': CHAT,
            'history': [], 'history_lock': threading.Lock(), 'active_session_lease': None,
            'profile_home': str(owner.home)}
    monkeypatch.setitem(server._sessions, 'pane-1', pane)
    assert server._ensure_active_session_slot('pane-1', pane) is None  # the pane's last turn claimed it
    lease = pane['active_session_lease']
    owner.leases.append(lease)
    refused = await _send(owner, 'while open')
    assert 'NOT sent' in refused.get('error', ''), refused

    monkeypatch.setattr(server, '_WS_ORPHAN_REAP_GRACE_S', 0.05)
    pane['transport']._closed = True
    server._close_sessions_for_transport(pane['transport'])
    await async_wait_until(lambda: lease.released, what='the reaped pane released its lease')
    server._reclaim_orphaned_leases()

    sent = await _send(owner, 'after close')
    _assert_admitted(sent)
    assert _adopted(owner)
    await async_wait_until(lambda: owner.executed == ['after close'], what='the adopted DM ran')
