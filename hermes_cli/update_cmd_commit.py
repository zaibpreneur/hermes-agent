"""The single commit point of a source ``hermes update``: everything the tree move needs armed first.

Before git (or the ZIP swap) writes the first file of the new code, three things are durable:

* the interrupted-pull marker (``.git/hermes-update-pull``) naming the commit git moves to, for EVERY
  tree move (branch switch, upstream fork pull, the pull itself) so a launch after any exit that left
  a torn tree puts the old one back (``_early_recovery.restore_interrupted_pull``);
* the source-completion tail (``venv_sync.arm_completion``) and the host fleet-restart obligation,
  so a kill after the move but before the completion child starts still leaves the tail owed to the
  next launch (``venv_sync.prepare_launch``) and the restart owed to the next ``hermes update``.

Failures before the move are no-ops: ``disarm_commit_obligations`` puts both obligations back the way
this run found them once the tree is verified at its pre-update commit again.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from hermes_cli._early_recovery import interrupted_pull_marker, restore_interrupted_pull

# What this run found before it armed anything: {path: bytes or None}. None = nothing armed yet.
_armed_snapshot: Optional[dict[Path, Optional[bytes]]] = None
# (git_cmd, (HEAD, branch)) when this run reached its checkout phase.
_run_start: Optional[tuple[list, tuple[str, str]]] = None


def _owns_live_checkout(root: Path) -> bool:
    from hermes_cli.update_cmd import _m

    return _m()._pytest_owns_live_checkout(root)


def _obligation_paths(root: Path) -> list[Path]:
    from hermes_cli.update_cmd_fleet import _fleet_restart_pending_marker_path
    from hermes_cli.update_host_obligation import host_obligation_path
    from hermes_cli.venv_sync import completion_pending_path

    paths = [completion_pending_path(root), _fleet_restart_pending_marker_path()]
    host = host_obligation_path()
    if host is not None:
        paths.append(host)
    return paths


def arm_commit_obligations(root: Path, expected_sha: str) -> None:
    """Owe the completion tail and the fleet restart for ``expected_sha`` BEFORE the tree moves.

    Idempotent within a run (the first call snapshots what to restore on a no-op failure). An
    unwritable install state raises: nothing has moved yet, and moving without the obligation is
    exactly the tail-never-runs state this exists to prevent.
    """
    global _armed_snapshot
    from hermes_cli.update_cmd_fleet import _write_fleet_restart_pending_marker
    from hermes_cli.venv_sync import arm_completion

    root = Path(root)
    if _owns_live_checkout(root):
        return
    if _armed_snapshot is None:
        snapshot: dict[Path, Optional[bytes]] = {}
        for path in _obligation_paths(root):
            try:
                snapshot[path] = path.read_bytes()
            except OSError:
                snapshot[path] = None
        _armed_snapshot = snapshot
    arm_completion(root)
    _write_fleet_restart_pending_marker(expected_sha=expected_sha or "")


def disarm_commit_obligations() -> None:
    """Restore both obligations to what this run found: the tree never left its pre-update commit."""
    global _armed_snapshot
    snapshot, _armed_snapshot = _armed_snapshot, None
    for path, data in (snapshot or {}).items():
        try:
            if data is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(data)
        except OSError:
            pass  # an owed tail/restart left armed is a retry, never a lost obligation


def commit_obligations_armed() -> bool:
    return _armed_snapshot is not None


def arm_tree_move(git_cmd, root: Path, *, pre: str | None, target: str, stash: str | None) -> Path:
    """Write the interrupted-pull marker for one git tree move (pre -> target)."""
    marker = interrupted_pull_marker(root)
    marker.write_text(f"pid={os.getpid()}\npre={pre or ''}\ntarget={target}\nstash={stash or ''}\n",
                      encoding="utf-8")
    return marker


def settle_failed_tree_move(root: Path) -> bool:
    """Git exited without finishing a tree move: put back what it wrote, keep the marker if we can't.

    True when the tree is verified whole again (at the pre-move commit, or at the target when git got
    there anyway); False leaves the marker for the next launch's ``restore_interrupted_pull``.
    """
    restore_interrupted_pull(Path(root), after_failure=True)
    return not interrupted_pull_marker(Path(root)).is_file()


def target_syntax_error(git_cmd, root: Path, target_ref: str, relpaths) -> tuple[str, str] | None:
    """``(path, error)`` for the first startup-critical file that does not compile at ``target_ref``.

    Read from the object store, never written to the tree: this runs BEFORE HEAD moves, so a broken
    release is refused with the install untouched (the post-pull rollback stays as the backstop).
    """
    for rel in relpaths:
        shown = subprocess.run([*git_cmd, "show", f"{target_ref}:{rel}"], cwd=str(root), capture_output=True,
                               stdin=subprocess.DEVNULL, timeout=120)
        if shown.returncode != 0:
            continue  # absent at the target (or unreadable): the post-pull guard has the last word
        try:
            compile(shown.stdout, rel, "exec", dont_inherit=True)
        except (SyntaxError, ValueError) as exc:
            return rel, f"{type(exc).__name__}: {exc}"
    return None


def head_and_branch(git_cmd, root: Path) -> tuple[str, str]:
    def out(*args: str) -> str:
        cp = subprocess.run([*git_cmd, *args], cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace",
                            stdin=subprocess.DEVNULL, timeout=60)
        return cp.stdout.strip() if cp.returncode == 0 else ""

    return out("rev-parse", "HEAD"), out("rev-parse", "--abbrev-ref", "HEAD")


def checkout_untouched(git_cmd, root: Path, start: tuple[str, str] | None) -> bool:
    """True when the checkout is exactly where this run found it: same HEAD, same branch, no
    autostash taken, no tree move armed. Only then may a failed git update fall back to ZIP."""
    from hermes_cli.update_cmd_stash import _unrestored_autostash_notice

    if start is None or not start[0]:
        return False
    if _unrestored_autostash_notice() is not None or interrupted_pull_marker(root).is_file():
        return False
    return head_and_branch(git_cmd, root) == start


def record_run_start(git_cmd, root: Path) -> None:
    global _run_start
    _run_start = (list(git_cmd), head_and_branch(git_cmd, root))


def run_checkout_untouched(root: Path) -> bool:
    """``checkout_untouched`` for this run; True before the checkout phase (nothing can have moved)."""
    if _run_start is None:
        return True
    return checkout_untouched(_run_start[0], Path(root), _run_start[1])


def preflight_refusal(git_cmd, root: Path, target_ref: str, critical_files) -> str | None:
    """Why this update must not start, checked BEFORE the first tree move; ``None`` to proceed.

    * a venv owned by another OS user (the completion child used to refuse only after the swap);
    * a target whose startup-critical modules do not compile (rollback used to be the only guard).
    """
    root = Path(root)
    if not _owns_live_checkout(root):
        try:
            from hermes_cli.venv_sync import refuse_foreign_owned_venv

            refuse_foreign_owned_venv(root)
        except ImportError:
            pass
        except Exception as exc:  # pm's refusal carries its own remediation text
            return f"✗ {exc}"
    broken = target_syntax_error(git_cmd, root, target_ref, critical_files)
    if broken is not None:
        path, error = broken
        return (f"✗ The update target has a syntax error in a critical file:\n  {path}\n    "
                + "\n    ".join(error.splitlines()[:6]))
    return None


def reset_for_tests() -> None:
    global _armed_snapshot, _run_start
    _armed_snapshot = None
    _run_start = None
