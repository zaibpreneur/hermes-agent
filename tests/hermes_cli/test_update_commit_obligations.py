"""The commit point's obligations: armed before the tree moves, handed back only when still ours.

``arm_commit_obligations`` owes the completion tail and the host fleet restart before git (or the
ZIP swap) writes a file; ``disarm_commit_obligations`` puts them back after a failure that left the
tree at its start commit. The host record is shared by every install of the OS user.
"""

from __future__ import annotations

import os
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

    assert host.read_text(encoding="utf-8") == "OTHER-INSTALL"


def test_plain_arm_and_disarm_restores_what_the_run_found(root):
    host = host_obligation_path()
    host.parent.mkdir(parents=True, exist_ok=True)
    host.write_text("OLD", encoding="utf-8")
    commit.arm_commit_obligations(root, "a" * 40)
    assert host.read_text(encoding="utf-8") != "OLD"

    commit.disarm_commit_obligations()

    assert host.read_text(encoding="utf-8") == "OLD"


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

    assert host.read_text(encoding="utf-8") == "OLD"

