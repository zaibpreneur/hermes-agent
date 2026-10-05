"""Tests for the two-phase ZIP replace and the shared venv-layout helpers.

``_atomic_replace_dir`` (#49145) made each *individual* directory swap safe,
but the ZIP update replaced ~70 top-level entries in a loop with no atomicity
across iterations. An interruption partway left some entries at the new
version and the rest at the old one -- every file valid Python, the
combination unbootable. That is the mechanism behind the ``ImportError`` in
#76091 and the field report in #63717.

Reference: issues #76104 (ZIP atomicity) and #76105 (venv-helper duplication).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import update_cmd, update_cmd_zip
from hermes_constants import venv_bin_dir, venv_python_path

# ---------------------------------------------------------------------------
# Two-phase replace
# ---------------------------------------------------------------------------

def _live_tree(root: Path, names: dict[str, str]) -> None:
    for name, marker in names.items():
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "version.txt").write_text(marker)

def _stage_all(root: Path, new: Path, names: list[str]) -> list[tuple[str, str]]:
    return [
        (
            update_cmd._stage_replacement(str(new / n), str(root / n)),
            str(root / n),
        )
        for n in names
    ]

def test_staging_touches_nothing_live(tmp_path):
    """Phase 1 must not modify the install -- a failure there is a no-op."""
    live, new = tmp_path / "live", tmp_path / "new"
    _live_tree(live, {"agent": "old", "tools": "old"})
    _live_tree(new, {"agent": "new", "tools": "new"})

    _stage_all(live, new, ["agent", "tools"])

    assert (live / "agent" / "version.txt").read_text() == "old"
    assert (live / "tools" / "version.txt").read_text() == "old"

def test_commit_swaps_every_entry(tmp_path):
    live, new = tmp_path / "live", tmp_path / "new"
    _live_tree(live, {"agent": "old", "tools": "old"})
    _live_tree(new, {"agent": "new", "tools": "new"})

    update_cmd._commit_staged_replacements(_stage_all(live, new, ["agent", "tools"]))

    assert (live / "agent" / "version.txt").read_text() == "new"
    assert (live / "tools" / "version.txt").read_text() == "new"
    # No staging/backup litter left behind.
    assert not [p for p in os.listdir(live) if "hermes-update" in p]

def test_failed_swap_rolls_back_every_earlier_swap(tmp_path, monkeypatch):
    """The regression: a mid-loop failure must not leave a mixed-version tree.

    Before the two-phase split this produced `agent/` new + `tools/` stale --
    the exact shape that yields
    `ImportError: cannot import name 'TODO_INJECTION_HEADER'`.
    """
    live, new = tmp_path / "live", tmp_path / "new"
    _live_tree(live, {"agent": "old", "tools": "old"})
    _live_tree(new, {"agent": "new", "tools": "new"})
    staged = _stage_all(live, new, ["agent", "tools"])

    real_rename = os.rename
    calls = {"n": 0}

    def flaky_rename(src, dst):
        calls["n"] += 1
        # Let the first entry swap fully (2 renames), then break the second.
        if calls["n"] == 4:
            raise OSError("simulated AV interference")
        return real_rename(src, dst)

    monkeypatch.setattr(update_cmd.os, "rename", flaky_rename)

    with pytest.raises(OSError):
        update_cmd._commit_staged_replacements(staged)

    monkeypatch.undo()
    # Both entries must be back at the OLD version -- not one new, one old.
    versions = {
        n: (live / n / "version.txt").read_text() for n in ("agent", "tools")
    }
    assert versions == {"agent": "old", "tools": "old"}, (
        f"mixed-version tree after rollback: {versions}"
    )

def test_commit_handles_entries_absent_from_the_install(tmp_path):
    """A brand-new top-level dir has no live counterpart to move aside."""
    live, new = tmp_path / "live", tmp_path / "new"
    live.mkdir()
    _live_tree(new, {"brand_new": "new"})

    update_cmd._commit_staged_replacements(_stage_all(live, new, ["brand_new"]))

    assert (live / "brand_new" / "version.txt").read_text() == "new"

def test_staging_clears_leftovers_from_an_interrupted_run(tmp_path):
    live, new = tmp_path / "live", tmp_path / "new"
    _live_tree(live, {"agent": "old"})
    _live_tree(new, {"agent": "new"})
    stale = Path(f"{live / 'agent'}.hermes-update-staging")
    stale.mkdir()
    (stale / "junk.txt").write_text("from a previous crash")

    update_cmd._commit_staged_replacements(_stage_all(live, new, ["agent"]))

    assert (live / "agent" / "version.txt").read_text() == "new"
    assert not (live / "agent" / "junk.txt").exists()

# ---------------------------------------------------------------------------
# Shared venv helpers (#76105)
# ---------------------------------------------------------------------------

def test_venv_helpers_accept_str_and_path():
    assert venv_python_path("/opt/x/venv") == venv_python_path(Path("/opt/x/venv"))

# ---------------------------------------------------------------------------
# Top-level FILES must be atomic too (#76104 review, C1)
# ---------------------------------------------------------------------------

def test_top_level_files_are_swapped_atomically(tmp_path):
    """The repo root holds 20 first-party modules (run_agent.py, cli.py,
    hermes_constants.py, ...). Covering only directories would leave exactly
    the bug class this PR closes."""
    live, new = tmp_path / "live", tmp_path / "new"
    live.mkdir()
    new.mkdir()
    (live / "run_agent.py").write_text("old")
    (new / "run_agent.py").write_text("new")

    staged = [
        (
            update_cmd._stage_replacement(
                str(new / "run_agent.py"), str(live / "run_agent.py")
            ),
            str(live / "run_agent.py"),
        )
    ]
    update_cmd._commit_staged_replacements(staged)

    assert (live / "run_agent.py").read_text() == "new"
    assert not [p for p in os.listdir(live) if "hermes-update" in p]

def test_file_swap_failure_restores_the_original_file(tmp_path, monkeypatch):
    """A mid-swap failure must not leave a stale-or-corrupt root module."""
    live, new = tmp_path / "live", tmp_path / "new"
    live.mkdir()
    new.mkdir()
    for name in ("cli.py", "run_agent.py"):
        (live / name).write_text("old")
        (new / name).write_text("new")

    staged = [
        (update_cmd._stage_replacement(str(new / n), str(live / n)), str(live / n))
        for n in ("cli.py", "run_agent.py")
    ]

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):  # a root file swaps in by os.replace over its hardlinked backup
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated AV interference")
        return real_replace(src, dst)

    monkeypatch.setattr(update_cmd.os, "replace", flaky_replace)
    with pytest.raises(OSError):
        update_cmd._commit_staged_replacements(staged)
    monkeypatch.undo()

    versions = {n: (live / n).read_text() for n in ("cli.py", "run_agent.py")}
    assert versions == {"cli.py": "old", "run_agent.py": "old"}, (
        f"mixed/corrupt root modules after rollback: {versions}"
    )

def test_failed_staging_leaves_no_orphaned_copies(tmp_path, monkeypatch):
    """#76104 review C2: orphaned staging dirs make the retry we recommend
    fail harder than the original attempt (less free space each time)."""
    live, new = tmp_path / "live", tmp_path / "new"
    _live_tree(live, {"agent": "old", "tools": "old", "gateway": "old"})
    _live_tree(new, {"agent": "new", "tools": "new", "gateway": "new"})

    real_copytree = update_cmd.shutil.copytree
    calls = {"n": 0}

    def flaky_copytree(src, dst, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError(28, "No space left on device")
        return real_copytree(src, dst, *a, **kw)

    monkeypatch.setattr(update_cmd.shutil, "copytree", flaky_copytree)

    staged: list[tuple[str, str]] = []
    with pytest.raises(OSError):
        try:
            for n in ("agent", "tools", "gateway"):
                staged.append(
                    (
                        update_cmd._stage_replacement(
                            str(new / n), str(live / n)
                        ),
                        str(live / n),
                    )
                )
        except Exception:
            update_cmd._discard_staged(staged)
            raise
    monkeypatch.undo()

    leftovers = [p for p in os.listdir(live) if "hermes-update" in p]
    assert leftovers == [], f"orphaned staging copies: {leftovers}"
    # And nothing live was touched.
    for n in ("agent", "tools", "gateway"):
        assert (live / n / "version.txt").read_text() == "old"

def test_venv_helpers_honour_an_explicit_platform_verdict():
    """Callers must be able to override the platform check (#76107 CI).

    The suite exercises Windows paths on Linux CI by patching predicates like
    `hermes_main._is_windows`. A helper that reads `sys.platform`
    unconditionally silently drops those paths out of coverage -- and broke
    `test_verify_core_dependencies.py::test_uses_virtual_env_from_environment`,
    which patches `_is_windows` and then asserts on a `Scripts/python.exe`
    path.
    """
    v = Path("/opt/proj/venv")
    assert venv_bin_dir(v, windows=True).name == "Scripts"
    assert venv_bin_dir(v, windows=False).name == "bin"
    assert venv_python_path(v, windows=True).name == "python.exe"
    assert venv_python_path(v, windows=False).name == "python"
    # Halves must stay consistent under an explicit verdict.
    for flag in (True, False):
        assert venv_python_path(v, windows=flag).parent == venv_bin_dir(
            v, windows=flag
        )

# ---------------------------------------------------------------------------
# Crash between "move dst aside" and "move staging in" (Phase 2 review HIGH)
# ---------------------------------------------------------------------------

def test_staging_restores_backup_when_dst_is_missing(tmp_path, monkeypatch):
    """A previous run that died mid-swap leaves dst missing and the backup as
    the ONLY copy of that entry. On retry, _stage_replacement must restore
    the backup to dst BEFORE clearing leftovers — otherwise a staging failure
    right after (disk exhaustion is likeliest exactly then) leaves a hole in
    the install with nothing to roll back to."""
    live, new = tmp_path / "live", tmp_path / "new"
    live.mkdir()
    _live_tree(new, {"agent": "new"})
    # Simulate the crashed state: dst gone, backup holds the old tree.
    backup = live / "agent.hermes-update-old"
    backup.mkdir()
    (backup / "version.txt").write_text("old")

    # Staging fails (disk full) on the fresh copy.
    def boom(src, dst, *a, **kw):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(update_cmd.shutil, "copytree", boom)
    with pytest.raises(OSError):
        update_cmd._stage_replacement(str(new / "agent"), str(live / "agent"))
    monkeypatch.undo()

    # The old tree must have been restored to dst before the failure.
    assert (live / "agent" / "version.txt").read_text() == "old"
    assert not backup.exists()

    # And a clean retry completes the update normally.
    staged = _stage_all(live, new, ["agent"])
    update_cmd._commit_staged_replacements(staged)
    assert (live / "agent" / "version.txt").read_text() == "new"
    assert not [p for p in os.listdir(live) if "hermes-update" in p]

def test_commit_failure_plus_discard_leaves_no_staging_litter(tmp_path, monkeypatch):
    """Phase-2 failure must not orphan staging copies for unswapped entries.

    _update_via_zip calls _discard_staged when _commit_staged_replacements
    raises. The rollback restores every swapped entry, but staging copies for
    the not-yet-swapped entries (potentially most of a full tree) would
    otherwise survive — and the retry's up-front free-space check runs BEFORE
    the lazy per-entry leftover cleanup, so the litter makes the retry fail
    harder than the original attempt. This pins the combination: rollback +
    discard leaves the old tree intact and ZERO update litter."""
    live, new = tmp_path / "live", tmp_path / "new"
    _live_tree(live, {"agent": "old", "tools": "old", "gateway": "old"})
    _live_tree(new, {"agent": "new", "tools": "new", "gateway": "new"})
    staged = _stage_all(live, new, ["agent", "tools", "gateway"])

    real_rename = os.rename
    calls = {"n": 0}

    def flaky_rename(src, dst):
        calls["n"] += 1
        if calls["n"] == 4:  # first entry fully swapped, second breaks
            raise OSError("simulated AV interference")
        return real_rename(src, dst)

    monkeypatch.setattr(update_cmd.os, "rename", flaky_rename)
    with pytest.raises(OSError):
        try:
            update_cmd._commit_staged_replacements(staged)
        except OSError:
            # Mirrors the _update_via_zip wiring.
            update_cmd._discard_staged(staged)
            raise
    monkeypatch.undo()

    # Old tree intact...
    for n in ("agent", "tools", "gateway"):
        assert (live / n / "version.txt").read_text() == "old"
    # ...and zero litter of any kind (staging OR backup).
    litter = [p for p in os.listdir(live) if "hermes-update" in p]
    assert litter == [], f"orphaned update litter: {litter}"


@pytest.mark.parametrize("hardlinks", [True, False])
def test_root_files_never_go_missing_mid_swap(tmp_path, monkeypatch, hardlinks):
    """Every launcher imports ``hermes_constants``/``hermes_bootstrap`` before anything else, and
    ``hermes_bootstrap`` is what reaches the restore after a killed swap: a kill between any two
    filesystem steps must find every root file present (old or new bytes), only directories moved.
    That holds on filesystems without hardlinks too (FAT32/exFAT/SMB)."""
    if not hardlinks:
        def no_link(*_a, **_k):
            raise OSError(1, "Operation not permitted")
        monkeypatch.setattr(update_cmd_zip.os, "link", no_link)
    live, new = tmp_path / "live", tmp_path / "new"
    for side, text in ((live, "old"), (new, "new")):
        (side / "hermes_cli").mkdir(parents=True)
        (side / "hermes_cli" / "main.py").write_text(text, encoding="utf-8")
        for name in ("hermes_constants.py", "hermes_bootstrap.py"):
            (side / name).write_text(text, encoding="utf-8")
    names = ("hermes_constants.py", "hermes_cli", "hermes_bootstrap.py")
    staged = [(update_cmd._stage_replacement(str(new / n), str(live / n)), str(live / n)) for n in names]
    missing: list[str] = []

    def probing(real):
        def step(src, dst):
            real(src, dst)
            missing.extend(n for n in ("hermes_constants.py", "hermes_bootstrap.py") if not (live / n).is_file())
        return step

    monkeypatch.setattr(update_cmd.os, "rename", probing(os.rename))
    monkeypatch.setattr(update_cmd.os, "replace", probing(os.replace))
    update_cmd._commit_staged_replacements(staged)
    monkeypatch.undo()
    assert not missing, f"root modules absent mid-swap: {missing}"
    assert {n: (live / n).read_text(encoding="utf-8-sig") for n in ("hermes_constants.py", "hermes_bootstrap.py")} == {
        "hermes_constants.py": "new", "hermes_bootstrap.py": "new"}
    assert not [p for p in os.listdir(live) if "hermes-update" in p]


# ---------------------------------------------------------------------------
# ZIP swap owner lock and staging journal (_early_recovery / _journaled_stage_and_swap)
# ---------------------------------------------------------------------------

_LOCK_ROLE = r'''
import json, os, sys, time
from pathlib import Path
from hermes_cli import _early_recovery as er
role, work = sys.argv[1], Path(sys.argv[2])
live, lock_path = work / "live", work / "live" / ".hermes-update-zip-swap.lock"
def wait_for(name):
    end = time.time() + 30
    while not (work / name).exists():
        if time.time() > end:
            raise SystemExit(f"{role}: no {name}")
        time.sleep(0.01)
def mark(name, owned):
    (work / name).write_text(json.dumps({"owned": bool(owned), "inode": os.stat(lock_path).st_ino
                                         if lock_path.exists() else None}))
if role == "a":  # owner whose release is caught right after its unlock: B already holds the inode
    real = er._lock_fd
    def lock_fd(fd, lock):
        done = real(fd, lock)
        if not lock:
            wait_for("b")
        return done
    er._lock_fd = lock_fd
    with er.zip_swap_owner_lock(live) as owned:
        mark("a", owned)
        wait_for("b-waiting"); time.sleep(0.3)
    (work / "a-done").touch()
elif role == "b":  # waiter that wins the lock as A lets go
    wait_for("a"); (work / "b-waiting").touch()
    with er.zip_swap_owner_lock(live, wait=20) as owned:
        mark("b", owned)
        wait_for("c")
else:  # newcomer after A's release completed
    wait_for("a-done")
    with er.zip_swap_owner_lock(live) as owned:
        mark("c", owned)
'''


def test_zip_swap_lock_is_one_inode_across_a_release(tmp_path):
    """Three real processes: A releases while B waits; B wins. C must then be refused: a release that
    unlinks the lock path lets C lock a fresh inode next to B's (two "exclusive" owners)."""
    import json
    import subprocess
    import sys

    (tmp_path / "live").mkdir()
    repo = os.path.realpath(Path(update_cmd.__file__).parent.parent)
    procs = [subprocess.Popen([sys.executable, "-c", _LOCK_ROLE, role, str(tmp_path)], cwd=repo,
                              env={**os.environ, "PYTHONPATH": repo}) for role in "abc"]
    assert [p.wait(timeout=90) for p in procs] == [0, 0, 0]
    a, b, c = (json.loads((tmp_path / r).read_text()) for r in "abc")
    assert a["owned"] and b["owned"]
    assert not c["owned"], f"two processes held the ZIP swap lock at once (inodes {b['inode']} and {c['inode']})"
    assert a["inode"] == b["inode"] == c["inode"]


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="POSIX permission bits; root ignores them")
def test_zip_swap_lock_refuses_admission_without_a_lock(tmp_path):
    """A root where the lock file cannot be created grants nothing: no lock, no swap, and a named reason."""
    from hermes_cli._early_recovery import zip_swap_owner_lock

    live = tmp_path / "live"
    live.mkdir()
    live.chmod(0o555)
    try:
        with zip_swap_owner_lock(live) as owned:
            assert not owned, "admitted to the ZIP swap without holding any lock"
            assert owned.reason.startswith("cannot open the ZIP swap lock")
        with pytest.raises(RuntimeError, match="no ZIP swap lock"):
            update_cmd_zip._journaled_stage_and_swap(str(tmp_path), [], live, None)
    finally:
        live.chmod(0o755)


def _unreadable_release(tmp_path: Path, *, read_only_dir: bool) -> tuple[Path, Path, list[Path]]:
    live, extracted = tmp_path / "live", tmp_path / "extracted"
    live.mkdir()
    extracted.mkdir()
    (live / "keep.txt").write_text("live data")
    (extracted / "first.txt").write_text("new first")
    (extracted / "second").mkdir()
    (extracted / "second" / "good.txt").write_text("new copied data")
    locked = []
    if read_only_dir:  # copytree copies its mode: the partial stage gets a directory nobody can empty
        ro = extracted / "second" / "aaa_ro"
        ro.mkdir()
        (ro / "x.txt").write_text("x")
        locked.append(ro)
    blocked = extracted / "second" / "zzz_no_read.txt"
    blocked.write_text("unreadable")
    locked.append(blocked)
    for path in locked:
        path.chmod(0o555 if path.is_dir() else 0)
    return live, extracted, locked


_POSIX_MODES = pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                                  reason="POSIX permission bits; root ignores them")


@_POSIX_MODES
def test_unreadable_source_file_leaves_no_partial_stage(tmp_path):
    """The copy of ``second`` dies on an unreadable file after copying the rest: that partial staging
    tree is the failing entry's and must be dropped like the finished ones (nothing live changes)."""
    from hermes_cli._early_recovery import ZIP_SWAP_JOURNAL, restore_interrupted_zip_swap

    live, extracted, locked = _unreadable_release(tmp_path, read_only_dir=False)
    try:
        with pytest.raises(OSError):
            update_cmd_zip._journaled_stage_and_swap(str(extracted), ["first.txt", "second"], live, None)
    finally:
        for path in locked:
            path.chmod(0o755)
    assert not list(live.glob("*.hermes-update-staging")), "a partial stage leaked past the failed copy"
    assert not (live / ZIP_SWAP_JOURNAL).exists()
    assert restore_interrupted_zip_swap(live) is False
    assert sorted(p.name for p in live.iterdir()) == [".hermes-update-zip-swap.lock", "keep.txt"]


@_POSIX_MODES
def test_a_stage_cleanup_that_cannot_finish_keeps_the_journal_for_recovery(tmp_path):
    """When the updater cannot remove its partial stage, the journal is that stage's only record: it
    stays, and the next launch's recovery removes the stage, then the journal."""
    from hermes_cli._early_recovery import ZIP_SWAP_JOURNAL, restore_interrupted_zip_swap

    live, extracted, locked = _unreadable_release(tmp_path, read_only_dir=True)
    try:
        with pytest.raises(OSError):
            update_cmd_zip._journaled_stage_and_swap(str(extracted), ["first.txt", "second"], live, None)
    finally:
        for path in locked:
            path.chmod(0o755)
    assert (live / "second.hermes-update-staging").exists(), "harness: the cleanup was expected to fail"
    assert (live / ZIP_SWAP_JOURNAL).exists(), "the journal went while its staging path was still on disk"
    assert restore_interrupted_zip_swap(live) is False  # nothing live moved: no relaunch
    assert not list(live.glob("*.hermes-update-staging")) and not (live / ZIP_SWAP_JOURNAL).exists()
    assert (live / "keep.txt").read_text() == "live data"


def _swap_or_recover(extracted: Path, entries: list[str], live: Path) -> None:
    """``_download_and_swap_zip``'s wiring: a failed swap is settled from its journal right after."""
    from hermes_cli._early_recovery import restore_interrupted_zip_swap

    try:
        update_cmd_zip._journaled_stage_and_swap(str(extracted), entries, live, None)
    except Exception:
        restore_interrupted_zip_swap(live)


@_POSIX_MODES
def test_a_stale_backup_never_stands_in_for_the_live_entry(tmp_path, monkeypatch):
    """A leftover ``<entry>.hermes-update-old`` this swap did not make (a pre-journal crash holding a
    read-only directory) must be removed or refuse the swap; it must never become the live entry."""
    from hermes_cli import update_cmd_commit

    monkeypatch.setattr(update_cmd_commit, "arm_commit_obligations", lambda *a, **k: None)
    live, extracted = tmp_path / "live", tmp_path / "extracted"
    (live / "pkg").mkdir(parents=True)
    (live / "pkg" / "m.py").write_text("LIVE_OLD", encoding="utf-8")
    (extracted / "pkg").mkdir(parents=True)
    (extracted / "pkg" / "m.py").write_text("NEW", encoding="utf-8")
    stale = live / "pkg.hermes-update-old" / "ro"
    stale.mkdir(parents=True)
    (stale / "f").write_text("stale remnant", encoding="utf-8")
    stale.chmod(0o555)
    try:
        _swap_or_recover(extracted, ["pkg"], live)
    finally:
        for path in live.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)
    module = live / "pkg" / "m.py"
    assert module.is_file() and module.read_text(encoding="utf-8-sig") in ("LIVE_OLD", "NEW"), sorted(
        str(p.relative_to(live)) for p in live.rglob("*"))


def test_a_failed_swap_keeps_a_file_the_user_made_at_a_never_installed_entry(tmp_path, monkeypatch):
    """The swap fails before ``brand_new`` is installed; a file the user created there meanwhile is
    theirs, and neither the rollback nor the journal recovery may delete it."""
    from hermes_cli import update_cmd_commit

    live, extracted = tmp_path / "live", tmp_path / "extracted"
    for side, text in ((live, "old"), (extracted, "new")):
        (side / "a").mkdir(parents=True)
        (side / "a" / "v.txt").write_text(text, encoding="utf-8")
    (extracted / "brand_new").write_text("new entry", encoding="utf-8")
    user_file = live / "brand_new"
    monkeypatch.setattr(update_cmd_commit, "arm_commit_obligations",
                        lambda *a, **k: user_file.write_text("USER NOTE", encoding="utf-8"))
    real_rename = os.rename

    def refuse_moving_a(src, dst):  # Windows AV holding ``a`` open
        if os.fspath(src) == str(live / "a") and os.fspath(dst).endswith(".hermes-update-old"):
            raise PermissionError(13, "in use")
        return real_rename(src, dst)

    monkeypatch.setattr(update_cmd_zip.os, "rename", refuse_moving_a)
    _swap_or_recover(extracted, ["a", "brand_new"], live)
    monkeypatch.undo()
    assert user_file.is_file() and user_file.read_text(encoding="utf-8-sig") == "USER NOTE"
    assert (live / "a" / "v.txt").read_text(encoding="utf-8-sig") == "old"
    assert not [p.name for p in live.iterdir() if "hermes-update-staging" in p.name or p.name.endswith("-old")]

def test_a_journal_that_cannot_be_dropped_after_the_commit_never_fails_the_update(tmp_path, monkeypatch):
    """The swap committed: an AV scan holding the journal must not turn it into a reported failure
    that disarms the new tree's completion obligations (F19)."""
    from hermes_cli import update_cmd_commit

    live, extracted = tmp_path / "live", tmp_path / "extracted"
    _live_tree(live, {"payload": "old"})
    _live_tree(extracted, {"payload": "new"})
    real = Path.unlink

    def held(self, *args, **kwargs):
        if self.name == update_cmd_zip.ZIP_SWAP_JOURNAL:
            raise PermissionError(13, "being used by another process", str(self))
        return real(self, *args, **kwargs)

    update_cmd_commit.reset_for_tests()
    monkeypatch.setattr(Path, "unlink", held)
    try:
        update_cmd_zip._journaled_stage_and_swap(str(extracted), ["payload"], live, "b" * 40)
        assert update_cmd_commit.commit_obligations_armed()
    finally:
        update_cmd_commit.reset_for_tests()
    assert (live / "payload" / "version.txt").read_text(encoding="utf-8-sig") == "new"
