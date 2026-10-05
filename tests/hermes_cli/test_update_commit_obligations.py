"""The commit point's obligations: armed before the tree moves, handed back only when still ours.

``arm_commit_obligations`` owes the completion tail and the host fleet restart before git (or the
ZIP swap) writes a file; ``disarm_commit_obligations`` puts them back after a failure that left the
tree at its start commit. The host record is shared by every install of the OS user.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from hermes_cli import update_cmd_commit as commit
from hermes_cli.update_host_obligation import host_obligation_path


@pytest.fixture(autouse=True)
def _fresh_commit_point(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    commit.reset_for_tests()
    yield
    commit.reset_for_tests()


@pytest.fixture
def root(tmp_path) -> Path:
    install = tmp_path / "install"
    install.mkdir()
    return install


def test_arm_refuses_when_no_store_accepts_the_fleet_restart(root):
    """Both the host record and the per-home breadcrumb unwritable: the move must not start (F26)."""
    from hermes_cli.update_cmd_fleet import _fleet_restart_pending_marker_path

    host_obligation_path().mkdir(parents=True)  # a directory where the record goes
    _fleet_restart_pending_marker_path().mkdir(parents=True)

    with pytest.raises(OSError):
        commit.arm_commit_obligations(root, "b" * 40)


def test_disarm_leaves_another_installs_newer_host_record(root):
    """A failed run hands back only what it armed: another install's record written since stays (F27)."""
    host = host_obligation_path()
    host.parent.mkdir(parents=True, exist_ok=True)
    host.write_text("OLD", encoding="utf-8")
    commit.arm_commit_obligations(root, "a" * 40)
    host.write_text("OTHER-INSTALL", encoding="utf-8")

    commit.disarm_commit_obligations()

    assert host.read_text(encoding="utf-8-sig") == "OTHER-INSTALL"


def test_plain_arm_and_disarm_restores_what_the_run_found(root):
    host = host_obligation_path()
    host.parent.mkdir(parents=True, exist_ok=True)
    host.write_text("OLD", encoding="utf-8")
    commit.arm_commit_obligations(root, "a" * 40)
    assert host.read_text(encoding="utf-8-sig") != "OLD"

    commit.disarm_commit_obligations()

    assert host.read_text(encoding="utf-8-sig") == "OLD"


def test_an_unreadable_record_refuses_the_arm_and_survives(root, monkeypatch):
    """A read error is not absence: guessing 'none' would delete the record on disarm (F28)."""
    host = host_obligation_path()
    host.parent.mkdir(parents=True, exist_ok=True)
    host.write_text("OLD", encoding="utf-8")
    real = Path.read_bytes

    def flaky(self):
        if self == host:
            raise PermissionError(13, "sharing violation", str(self))
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", flaky)
    with pytest.raises(OSError):
        commit.arm_commit_obligations(root, "a" * 40)
    monkeypatch.setattr(Path, "read_bytes", real)
    commit.disarm_commit_obligations()

    assert host.read_text(encoding="utf-8-sig") == "OLD"


@pytest.mark.platforms("posix")  # unprivileged symlinks
def test_disarm_never_writes_through_a_planted_restore_alias(root, tmp_path):
    """A same-user link at a temp name must not redirect the restore (N05)."""
    host = host_obligation_path()
    host.parent.mkdir(parents=True, exist_ok=True)
    host.write_text("ORIGINAL", encoding="utf-8")
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("PRECIOUS", encoding="utf-8")
    commit.arm_commit_obligations(root, "a" * 40)
    for alias in (host.with_name(host.name + ".restore"), host.with_name(f".{host.name}.{os.getpid()}.restore")):
        alias.symlink_to(sentinel)

    commit.disarm_commit_obligations()

    assert sentinel.read_text(encoding="utf-8-sig") == "PRECIOUS"
    assert not host.is_symlink() and host.read_text(encoding="utf-8-sig") == "ORIGINAL"


def _git(cwd: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
                          encoding="utf-8", env=env).stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_a_refused_pull_after_the_branch_switch_owes_the_head_the_switch_landed_on(tmp_path, monkeypatch):
    """CP0 switched a parked feature branch (X) to main (A); CP1's arm for B then failed. The
    obligation used to stay on B, which the checkout at A never contains, while disarm refused
    (HEAD != X): an undischargeable debt (review C1). It must name A, and the refusal must not
    claim the checkout was untouched."""
    import hermes_cli.main as hermes_main
    from hermes_cli import update_cmd
    from hermes_cli._early_recovery import interrupted_pull_marker
    from hermes_cli.update_host_obligation import read_host_obligation

    up, clone = tmp_path / "up", tmp_path / "clone"
    up.mkdir()
    _git(up, "init", "-q", "-b", "main")
    (up / "f").write_text("A", encoding="utf-8")
    _git(up, "add", "-A")
    _git(up, "commit", "-qm", "A")
    a = _git(up, "rev-parse", "HEAD")
    _git(tmp_path, "clone", "-q", str(up), str(clone))
    _git(clone, "checkout", "-q", "-b", "feat")
    (clone / "g").write_text("X", encoding="utf-8")
    _git(clone, "add", "-A")
    _git(clone, "commit", "-qm", "X")
    x = _git(clone, "rev-parse", "HEAD")
    (up / "f").write_text("B", encoding="utf-8")
    _git(up, "commit", "-qam", "B")
    b = _git(up, "rev-parse", "HEAD")
    _git(clone, "fetch", "-q", "origin")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", clone)
    monkeypatch.setattr(commit, "_owns_live_checkout", lambda _root: False)
    monkeypatch.chdir(clone)

    commit.record_run_start(["git"], clone)
    switched = update_cmd._switch_branch_at_commit_point(["git"], "main", "origin/main", pre=x, stash=None)
    assert switched.returncode == 0 and _git(clone, "rev-parse", "HEAD") == a
    interrupted_pull_marker(clone).mkdir()  # CP1's marker cannot be written
    reason = commit.arm_commit_point(["git"], clone, b, pre=a, target=b, stash=None)

    assert reason and "the checkout was not changed" not in reason and a[:10] in reason
    assert (read_host_obligation() or {}).get("expected_sha") == a
